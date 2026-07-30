from __future__ import annotations

import csv
import json
import math
import util_modules as ut
import src
import torch
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from ocir_reproduction.protocols import MAIN_PROTOCOL_V1
from pipelines import solver_base


FINAL_PREDICTION_COLUMNS = [
    "dataset",
    "source_partition",
    "unit",
    "end_cycle",
    "provided_rul",
    "capped_target_rul",
    "raw_pred_rul",
    "provided_eval_pred_rul",
    "capped_eval_pred_rul",
    "error_provided",
    "error_capped",
    "abs_error_provided",
    "abs_error_capped",
    "method",
    "model_seed",
    "window_length",
    "rul_cap",
    "main_protocol_version",
    "split_id",
    "validation_view",
    "train_sampler",
    "preprocessing_version",
]

class RulPipeline(solver_base.Solver):
    def __init__(self, config, 
                 logger,
                 encoder = None,
                 shared_encoder_layer = None):
        super(RulPipeline, self).__init__(config, logger)
        self.rul_cap = MAIN_PROTOCOL_V1["labels"]["rul_cap"]
        self.rul_validate_every_epochs = getattr(self, "rul_validate_every_epochs", 1)
        self._validate_rul_validation_config()
        self.best_val_engine_rmse = float("inf")
        self.best_rul_epoch = None
        self.best_rul_state_dict = None
        
        self.build_dataset()
        self.build_model(encoder, shared_encoder_layer)
        
    def __call__(self, validation = True):
        self.validation = validation
        if self.required_training:
            self.train()
        else:
            self.training_log = None
            self.logger.info(
                "Pre-trained RUL model loaded; skipping automatic official-test evaluation. "
                "Call evaluate_test_terminal() explicitly after final freeze."
            )
        self.rul_predictor.train(False)
        return self.rul_predictor

    def _validate_rul_validation_config(self):
        if isinstance(self.rul_validate_every_epochs, bool) or not isinstance(self.rul_validate_every_epochs, int):
            raise ValueError("rul_validate_every_epochs must be a positive integer")
        if self.rul_validate_every_epochs < 1:
            raise ValueError("rul_validate_every_epochs must be a positive integer")

    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_mse')
                            )
        Loss_MSE = ut.Value_averager()
        
        opt_rul = self.get_optimizers()
        
        last_validated_epoch = None
        
        for epoch in tqdm(range(self.rul_epochs), desc = "Training RUL estimator: "):
            self.set_training_sampler_epoch(epoch)
            self.rul_predictor.train(True)
            for i, (x,y, _, tidx) in enumerate(self.training_data): # TODO 
                x = x.to(self.device)
                y = y.to(self.device)
                tidx = tidx.to(self.device)

                loss_rul = self.rul_predictor.Loss_RUL(y, x, tidx)
                opt_rul[0].zero_grad()
                loss_rul.backward()
                for m in reversed(opt_rul):
                    if m: m.step()
                
                # Records
                Loss_MSE.update(loss_rul.item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            Loss_MSE.avg)
            self.logger.info(f"epoch[{epoch+1}/{self.rul_epochs}], Loss RUL:{Loss_MSE.avg: .4f}")
            
            if self.validation and (((epoch + 1) % self.rul_validate_every_epochs) == 0):
                metrics = self.evaluate_validation(epoch + 1)
                self.maybe_update_best_checkpoint(epoch + 1, metrics)
                last_validated_epoch = epoch + 1
        
        if self.validation and last_validated_epoch != self.rul_epochs:
            metrics = self.evaluate_validation(self.rul_epochs)
            self.maybe_update_best_checkpoint(self.rul_epochs, metrics)

        if self.validation:
            self.restore_best_checkpoint()
        else:
            self.logger.warning(
                "RUL validation disabled; saving last epoch checkpoint without validation selection."
            )

        self.logger.info("Training a RUL is done. Saving the model ...")
        ut.save_model(self.rul_predictor, path_ = self.model_save_path, name = "rul")

    def predict_loader(self, loader):
        self.rul_predictor.train(False)
        raw_predictions = []
        targets = []
        with torch.no_grad():
            for x, y, _, tidx in loader:
                x = x.to(self.device)
                tidx = tidx.to(self.device)
                raw_pred = self.rul_predictor(x, tidx)
                raw_predictions.append(raw_pred.detach().cpu())
                targets.append(y.detach().cpu())

        if not raw_predictions:
            raise ValueError("cannot evaluate an empty RUL DataLoader")

        return torch.cat(raw_predictions, dim=0), torch.cat(targets, dim=0)

    def get_loader_units(self, loader):
        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "all_metadata"):
            raise ValueError("RUL evaluation loader dataset must provide all_metadata()")

        metadata = dataset.all_metadata()
        try:
            units = [int(item.unit) for item in metadata]
        except AttributeError as exc:
            raise ValueError("RUL evaluation metadata must contain unit") from exc

        return torch.tensor(units, dtype=torch.int64)

    def compute_rul_metrics(self, raw_pred, target, units=None, clip_predictions=True):
        raw_pred = raw_pred.detach().cpu().float()
        target = target.detach().cpu().float()

        if raw_pred.shape != target.shape:
            raise ValueError(f"raw_pred shape {tuple(raw_pred.shape)} must equal target shape {tuple(target.shape)}")
        if raw_pred.ndim != 2 or raw_pred.shape[1] != 1:
            raise ValueError(f"RUL predictions and targets must have shape [N, 1], got {tuple(raw_pred.shape)}")

        pred = torch.clamp(raw_pred, min=0.0, max=float(self.rul_cap)) if clip_predictions else raw_pred
        error = (pred - target).view(-1)
        raw_error = (raw_pred - target).view(-1)
        instances_num = error.numel()
        if instances_num == 0:
            raise ValueError("cannot compute RUL metrics on empty tensors")

        window_rmse = torch.sqrt(torch.mean(error ** 2)).item()
        window_mae = torch.mean(torch.abs(error)).item()
        raw_window_rmse = torch.sqrt(torch.mean(raw_error ** 2)).item()
        score_terms = torch.where(
            error < 0,
            torch.exp(-error / 13.0) - 1.0,
            torch.exp(error / 10.0) - 1.0,
        )
        nasa_score = torch.sum(score_terms).item()

        engine_rmse = None
        if units is not None:
            units = units.detach().cpu().view(-1).to(torch.int64)
            if units.numel() != instances_num:
                raise ValueError(
                    f"unit count {units.numel()} must equal prediction count {instances_num}"
                )
            engine_mses = []
            for unit in torch.unique(units, sorted=True):
                unit_error = error[units == unit]
                if unit_error.numel() == 0:
                    raise ValueError(f"unit {int(unit.item())} has no validation errors")
                engine_mses.append(torch.mean(unit_error ** 2))
            engine_rmse = torch.sqrt(torch.mean(torch.stack(engine_mses))).item()

        return {
            "window_rmse": float(window_rmse),
            "window_mae": float(window_mae),
            "raw_window_rmse": float(raw_window_rmse),
            "nasa_score": float(nasa_score),
            "engine_rmse": None if engine_rmse is None else float(engine_rmse),
        }

    def evaluate_validation(self, epoch):
        if self.val_data is None:
            raise ValueError("self.val_data is required for RUL validation")

        raw_pred, target = self.predict_loader(self.val_data)
        units = self.get_loader_units(self.val_data)
        if units.numel() != raw_pred.shape[0]:
            raise ValueError(
                f"validation metadata rows {units.numel()} must equal predictions {raw_pred.shape[0]}"
            )

        metrics = self.compute_rul_metrics(
            raw_pred=raw_pred,
            target=target,
            units=units,
            clip_predictions=True,
        )
        if metrics["engine_rmse"] is None:
            raise ValueError("validation metrics must include engine_rmse")

        validation_metrics = {
            "val_window_rmse": metrics["window_rmse"],
            "val_window_mae": metrics["window_mae"],
            "val_raw_rmse": metrics["raw_window_rmse"],
            "val_nasa_score": metrics["nasa_score"],
            "val_engine_rmse": metrics["engine_rmse"],
        }
        self.logger.info(
            "Epoch (%s) - val_engine_rmse: %.5f, val_window_rmse: %.5f, "
            "val_window_mae: %.5f, val_raw_rmse: %.5f, val_nasa_score: %.5f",
            epoch,
            validation_metrics["val_engine_rmse"],
            validation_metrics["val_window_rmse"],
            validation_metrics["val_window_mae"],
            validation_metrics["val_raw_rmse"],
            validation_metrics["val_nasa_score"],
        )
        return validation_metrics

    def maybe_update_best_checkpoint(self, epoch, metrics):
        current_val_engine_rmse = metrics["val_engine_rmse"]
        if current_val_engine_rmse < self.best_val_engine_rmse:
            self.best_val_engine_rmse = current_val_engine_rmse
            self.best_rul_epoch = epoch
            self.best_rul_state_dict = deepcopy(self.rul_predictor.state_dict())
            self.logger.info(
                "New best RUL checkpoint: epoch=%s, val_engine_rmse=%.5f",
                epoch,
                self.best_val_engine_rmse,
            )

    def restore_best_checkpoint(self):
        if self.best_rul_state_dict is None:
            self.logger.warning("No RUL validation checkpoint was selected; keeping current model parameters.")
            self.rul_predictor.train(False)
            return

        self.rul_predictor.load_state_dict(self.best_rul_state_dict, strict=True)
        self.rul_predictor.train(False)
        self.logger.info(
            "Restored best RUL checkpoint: epoch=%s, val_engine_rmse=%.5f",
            self.best_rul_epoch,
            self.best_val_engine_rmse,
        )

    def get_loader_metadata(self, loader):
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise ValueError("RUL evaluation loader must provide dataset")
        if not hasattr(dataset, "all_metadata"):
            raise ValueError("RUL evaluation dataset must provide all_metadata()")

        metadata = dataset.all_metadata()
        if not metadata:
            raise ValueError("RUL evaluation metadata must be non-empty")
        if hasattr(dataset, "__len__") and len(metadata) != len(dataset):
            raise ValueError(f"metadata length {len(metadata)} must equal dataset length {len(dataset)}")
        return metadata

    def validate_test_terminal_metadata(self, metadata, prediction_count):
        if isinstance(prediction_count, bool) or not isinstance(prediction_count, int) or prediction_count <= 0:
            raise ValueError("prediction_count must be a positive integer")
        if len(metadata) != prediction_count:
            raise ValueError(f"metadata length {len(metadata)} must equal prediction count {prediction_count}")

        seen_units = set()
        previous_unit = None
        for index, item in enumerate(metadata):
            for field in ("source_partition", "unit", "end_cycle", "provided_rul", "capped_target_rul"):
                if not hasattr(item, field):
                    raise ValueError(f"metadata item {index} is missing {field}")

            if item.source_partition != "test":
                raise ValueError(f"metadata item {index} source_partition must be 'test', got {item.source_partition!r}")
            if item.provided_rul is None:
                raise ValueError(f"metadata item {index} provided_rul must not be None")
            if item.capped_target_rul is None:
                raise ValueError(f"metadata item {index} capped_target_rul must not be None")

            unit = int(item.unit)
            if unit in seen_units:
                raise ValueError(f"test terminal metadata contains duplicated unit {unit}")
            if previous_unit is not None and unit <= previous_unit:
                raise ValueError("test terminal metadata units must be sorted ascending without duplicates")
            seen_units.add(unit)
            previous_unit = unit

            provided_rul = float(item.provided_rul)
            capped_target_rul = float(item.capped_target_rul)
            expected_capped = min(provided_rul, float(self.rul_cap))
            if not math.isclose(capped_target_rul, expected_capped, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"metadata item {index} capped_target_rul must equal min(provided_rul, {self.rul_cap})"
                )

    def build_final_prediction_rows(self, raw_pred, metadata, method="ocir"):
        raw_pred = raw_pred.detach().cpu().float()
        if raw_pred.ndim != 2 or raw_pred.shape[1] != 1:
            raise ValueError(f"raw_pred must have shape [N, 1], got {tuple(raw_pred.shape)}")
        self.validate_test_terminal_metadata(metadata, int(raw_pred.shape[0]))

        protocol_version = MAIN_PROTOCOL_V1["version"]
        split_id = MAIN_PROTOCOL_V1["splits"]["split_id"]
        validation_view = MAIN_PROTOCOL_V1["validation_view"]["version"]
        train_sampler = MAIN_PROTOCOL_V1["training"]["sampler"]
        preprocessing_version = protocol_version

        rows = []
        for index, item in enumerate(metadata):
            raw_pred_rul = float(raw_pred[index, 0].item())
            provided_rul = float(item.provided_rul)
            capped_target_rul = float(item.capped_target_rul)
            provided_eval_pred_rul = max(raw_pred_rul, 0.0)
            capped_eval_pred_rul = min(max(raw_pred_rul, 0.0), float(self.rul_cap))
            error_provided = provided_eval_pred_rul - provided_rul
            error_capped = capped_eval_pred_rul - capped_target_rul

            rows.append(
                {
                    "dataset": item.dataset,
                    "source_partition": "test",
                    "unit": int(item.unit),
                    "end_cycle": int(item.end_cycle),
                    "provided_rul": provided_rul,
                    "capped_target_rul": capped_target_rul,
                    "raw_pred_rul": raw_pred_rul,
                    "provided_eval_pred_rul": provided_eval_pred_rul,
                    "capped_eval_pred_rul": capped_eval_pred_rul,
                    "error_provided": error_provided,
                    "error_capped": error_capped,
                    "abs_error_provided": abs(error_provided),
                    "abs_error_capped": abs(error_capped),
                    "method": method,
                    "model_seed": int(self.seed),
                    "window_length": int(self.window),
                    "rul_cap": int(self.rul_cap),
                    "main_protocol_version": protocol_version,
                    "split_id": split_id,
                    "validation_view": validation_view,
                    "train_sampler": train_sampler,
                    "preprocessing_version": preprocessing_version,
                }
            )
        return rows

    def compute_final_view_metrics(self, rows, view_name):
        if not rows:
            raise ValueError("rows must be non-empty")

        view_fields = {
            "provided_rul_eval_v1": ("provided_eval_pred_rul", "provided_rul"),
            "capped_target_eval_v1": ("capped_eval_pred_rul", "capped_target_rul"),
        }
        if view_name not in view_fields:
            raise ValueError(f"view_name must be one of {sorted(view_fields)}, got {view_name!r}")

        prediction_field, target_field = view_fields[view_name]
        errors = []
        for index, row in enumerate(rows):
            try:
                prediction = float(row[prediction_field])
                target = float(row[target_field])
            except KeyError as exc:
                raise ValueError(f"row {index} missing required field {exc.args[0]!r}") from exc
            errors.append(prediction - target)

        mse = sum(error ** 2 for error in errors) / len(errors)
        mae = sum(abs(error) for error in errors) / len(errors)
        nasa_score = sum(
            math.exp(-error / 13.0) - 1.0 if error < 0 else math.exp(error / 10.0) - 1.0
            for error in errors
        )
        return {
            "rmse": float(math.sqrt(mse)),
            "mae": float(mae),
            "nasa_score": float(nasa_score),
        }

    def build_final_metrics_payload(self, rows, method="ocir"):
        if not rows:
            raise ValueError("rows must be non-empty")

        datasets = {row["dataset"] for row in rows}
        if len(datasets) != 1:
            raise ValueError(f"final evaluation rows must contain one dataset, got {sorted(datasets)}")
        if any(row["source_partition"] != "test" for row in rows):
            raise ValueError("final evaluation rows must all have source_partition='test'")

        first = rows[0]
        provided_metrics = self.compute_final_view_metrics(rows, "provided_rul_eval_v1")
        capped_metrics = self.compute_final_view_metrics(rows, "capped_target_eval_v1")

        return {
            "dataset": first["dataset"],
            "source_partition": "test",
            "method": method,
            "model_seed": int(self.seed),
            "split_id": MAIN_PROTOCOL_V1["splits"]["split_id"],
            "split_seed": int(MAIN_PROTOCOL_V1["splits"]["split_seed"]),
            "window_length": int(self.window),
            "rul_cap": int(self.rul_cap),
            "label_mode": MAIN_PROTOCOL_V1["labels"]["mode"],
            "label_scale": MAIN_PROTOCOL_V1["labels"]["scale"],
            "label_normalization": MAIN_PROTOCOL_V1["labels"]["normalization"],
            "error_definition": "prediction_minus_target_per_evaluation_view",
            "main_protocol_version": MAIN_PROTOCOL_V1["version"],
            "validation_view": MAIN_PROTOCOL_V1["validation_view"]["version"],
            "train_sampler": MAIN_PROTOCOL_V1["training"]["sampler"],
            "preprocessing_version": first["preprocessing_version"],
            "num_test_units": len(rows),
            "evaluation_views": {
                "provided_rul_eval_v1": provided_metrics,
                "capped_target_eval_v1": capped_metrics,
            },
        }

    def write_final_evaluation_outputs(self, output_dir, rows, metrics_payload):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"

        if predictions_path.exists():
            self.logger.warning("Overwriting final evaluation predictions: %s", predictions_path)
        if metrics_path.exists():
            self.logger.warning("Overwriting final evaluation metrics: %s", metrics_path)

        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FINAL_PREDICTION_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def evaluate_test_terminal(self, output_dir=None, method="ocir", write_files=True):
        if self.testing_data is None:
            raise ValueError("self.testing_data is required for official test terminal evaluation")

        raw_pred, _target = self.predict_loader(self.testing_data)
        metadata = self.get_loader_metadata(self.testing_data)
        self.validate_test_terminal_metadata(metadata, int(raw_pred.shape[0]))
        rows = self.build_final_prediction_rows(raw_pred=raw_pred, metadata=metadata, method=method)
        metrics_payload = self.build_final_metrics_payload(rows, method=method)

        if write_files:
            if output_dir is None:
                output_dir = Path(self.model_save_path).parent
            self.write_final_evaluation_outputs(output_dir, rows, metrics_payload)

        provided_metrics = metrics_payload["evaluation_views"]["provided_rul_eval_v1"]
        capped_metrics = metrics_payload["evaluation_views"]["capped_target_eval_v1"]
        self.logger.info(
            "Official test provided_rul_eval_v1 - RMSE: %.5f, MAE: %.5f, NASA Score: %.5f",
            provided_metrics["rmse"],
            provided_metrics["mae"],
            provided_metrics["nasa_score"],
        )
        self.logger.info(
            "Official test capped_target_eval_v1 - RMSE: %.5f, MAE: %.5f, NASA Score: %.5f",
            capped_metrics["rmse"],
            capped_metrics["mae"],
            capped_metrics["nasa_score"],
        )
        return metrics_payload
        
    def build_model(self, encoder = None, shared_encoder_layer = None):
        def print_model(m, model_name):
            self.logger.info(f"Model: {model_name}")
            total_param = 0
            for name, param in m.named_parameters():
                num_params = param.numel()
                total_param += num_params
                self.logger.info(f"{name}: {num_params} parameters")
            
            self.logger.info(f"Total parameters in {model_name}: {total_param}")
            self.logger.info("")
            return total_param
        
        rul_predictor = src.RulEstimator(dz=self.dz, 
                                    pretrained_encoder=deepcopy(encoder.f_E), 
                                    shared_layer = deepcopy(encoder.shared_encoder_layers) if encoder.shared_encoder_layers is not None else None,
                                    device=self.device)  
            
        print_model(rul_predictor, "RUL predictior")
        self.rul_predictor, self.required_training = ut.load_model(rul_predictor, self.model_save_path, "rul")
        self.rul_predictor.to(self.device)
        
    def get_optimizers(self):
        opt_RUL, scheduler_RUL, wd_scheduler_RUL = ut.opt_constructor(False, #self.scheduler,
                                                                [self.rul_predictor.shared_layer, self.rul_predictor.encoder, self.rul_predictor.regressor],
                                                                # [self.rul_predictor.regressor],
                                                                lr = self.rul_lr,
                                                                warm_up = int(self.rul_epochs* self.ipe * self.warm_up),
                                                                fianl_step = int(self.rul_epochs* self.ipe),
                                                                start_lr = self.start_lr,
                                                                ref_lr = self.ref_lr,
                                                                final_lr = self.final_lr,
                                                                start_wd = self.start_wd,
                                                                final_wd = self.final_wd,
                                                                
                                                                ft_lr_rate = 1e-1)
        return [opt_RUL, scheduler_RUL, wd_scheduler_RUL]
