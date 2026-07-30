"""Executable main_protocol_v1 configuration.

The YAML file beside this module is the human-readable experiment contract.
This Python copy is dependency-free so the codebase can validate core protocol
constants before PyYAML is installed in the experiment environment.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MAIN_PROTOCOL_V1: dict[str, Any] = {
    "version": "main_protocol_v1",
    "datasets": ["FD001", "FD002", "FD003", "FD004"],
    "data": {
        "source": "NASA C-MAPSS turbofan degradation benchmark",
        "raw_file_pattern": {
            "train": "train_{dataset}.txt",
            "test": "test_{dataset}.txt",
            "rul": "RUL_{dataset}.txt",
        },
        "raw_columns": {
            "count": 26,
            "unit": "unit",
            "cycle": "cycle",
            "operating_settings": ["op1", "op2", "op3"],
            "sensors": [f"s{i}" for i in range(1, 22)],
        },
        "raw_data_policy": {
            "redistribute_raw_data": False,
            "raw_data_read_only": True,
            "manifest": "data/manifest/main_protocol_v1_manifest.json",
            "hash": "sha256",
        },
        "unit_key": ["dataset", "source_partition", "unit"],
    },
    "features": {
        "preserve_raw_sensors": True,
        "raw_sensor_dim": 21,
        "preserve_operating_settings": True,
        "operating_settings": ["op1", "op2", "op3"],
        "selected_sensors": [
            "s2",
            "s3",
            "s4",
            "s7",
            "s8",
            "s9",
            "s11",
            "s12",
            "s13",
            "s14",
            "s15",
            "s17",
            "s20",
            "s21",
        ],
        "selected_sensor_dim": 14,
        "ocir_model_uses_operating_settings": False,
    },
    "windowing": {
        "window_length": 30,
        "stride": 1,
        "short_sequence_padding": "left_repeat_first",
        "train_windows": "all_sliding_windows",
        "test_windows": "one_terminal_window_per_unit",
    },
    "labels": {
        "mode": "capped",
        "rul_cap": 125,
        "scale": "cycle",
        "normalization": "none",
        "sensitivity_rul_caps": [130],
        "train_label": "min(full_life_cycle - current_cycle, rul_cap)",
        "test_capped_target": "min(provided_rul, rul_cap)",
    },
    "splits": {
        "split_id": "split_v1_seed2026_engine_80_20",
        "split_seed": 2026,
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
        "split_by": "engine_unit",
        "save_path": "splits/cmapss/main_protocol_v1/{dataset}_split_seed2026.json",
    },
    "preprocessing": {
        "fit_scope": "train_units_only",
        "sensor_scaler": {
            "name": "global_z_score",
            "fit": "train selected sensors only",
            "transform": ["train", "validation", "test"],
        },
        "op_scaler": {
            "name": "op_z_score",
            "fit": "train op1-op3 only",
            "transform": ["train", "validation", "test"],
        },
        "pasa_kmeans": {
            "enabled_for": ["FD002", "FD004"],
            "k": 6,
            "fit": "train op1-op3 only",
            "predict": ["train", "validation", "test"],
        },
    },
    "validation_view": {
        "version": "validation_view_v1",
        "checkpoints": [0.5, 0.7, 0.9],
        "checkpoint_cycle": "max(window_length, floor(q * unit_length))",
        "target": "min(unit_length - checkpoint_cycle, rul_cap)",
        "prediction_postprocess": "clip(raw_prediction, 0, rul_cap)",
        "selection_metric": "val_engine_rmse",
        "save_path": "splits/cmapss/main_protocol_v1/{dataset}_validation_view_v1.csv",
    },
    "training": {
        "model_seeds": [1, 2, 3, 4, 5],
        "sampler": "engine_balanced_sampler_v1",
        "loss": "MSE on capped RUL",
        "rul_validate_every_epochs": 1,
        "official_test_access": "sealed_until_G4",
    },
    "evaluation": {
        "metrics": ["RMSE", "MAE", "NASA Score"],
        "error_definition": "prediction - target",
        "views": {
            "provided_rul_eval_v1": {
                "target": "provided_rul",
                "prediction": "max(raw_pred_rul, 0)",
                "primary": True,
            },
            "capped_target_eval_v1": {
                "target": "min(provided_rul, rul_cap)",
                "prediction": "clip(raw_pred_rul, 0, rul_cap)",
                "primary": False,
            },
        },
    },
    "outputs": {
        "experiment_root": "results/experiments/{exp_id}",
        "required_files": [
            "config.yaml",
            "metrics.json",
            "predictions.csv",
            "train_log.csv",
            "checkpoint.pt",
            "figures/",
        ],
        "prediction_columns": [
            "dataset",
            "source_partition",
            "unit",
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
        ],
    },
}


def get_main_protocol_v1() -> dict[str, Any]:
    return deepcopy(MAIN_PROTOCOL_V1)


def validate_main_protocol_v1(protocol: dict[str, Any] | None = None) -> list[str]:
    protocol = MAIN_PROTOCOL_V1 if protocol is None else protocol
    errors: list[str] = []

    if protocol.get("version") != "main_protocol_v1":
        errors.append("version must be main_protocol_v1")
    if protocol["features"]["selected_sensor_dim"] != len(protocol["features"]["selected_sensors"]):
        errors.append("selected_sensor_dim must match selected_sensors length")
    if protocol["features"]["selected_sensor_dim"] != 14:
        errors.append("OCIR main input must use 14 selected sensors")
    if protocol["windowing"]["window_length"] != 30:
        errors.append("window_length must be 30")
    if protocol["windowing"]["stride"] != 1:
        errors.append("stride must be 1")
    if protocol["labels"]["rul_cap"] != 125:
        errors.append("main rul_cap must be 125")
    if protocol["splits"]["split_seed"] != 2026:
        errors.append("split_seed must be 2026")
    if protocol["training"]["model_seeds"] != [1, 2, 3, 4, 5]:
        errors.append("model_seeds must be [1, 2, 3, 4, 5]")
    if protocol["training"].get("rul_validate_every_epochs") != 1:
        errors.append("rul_validate_every_epochs must default to 1")
    if protocol["validation_view"]["checkpoints"] != [0.5, 0.7, 0.9]:
        errors.append("validation checkpoints must be [0.5, 0.7, 0.9]")
    if set(protocol["evaluation"]["views"].keys()) != {"provided_rul_eval_v1", "capped_target_eval_v1"}:
        errors.append("evaluation views must contain provided and capped views")

    return errors
