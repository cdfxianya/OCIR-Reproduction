"""Run and archive main_protocol_v1 OCIR RUL experiments.

The dry-run path expands the protocol grid only. It does not create experiment
directories, instantiate pipelines, read processed data, or touch official test
objects.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir_reproduction.protocols import MAIN_PROTOCOL_V1
from pipelines import RlPipeline, RulPipeline
from pipelines.rul_estimation import FINAL_PREDICTION_COLUMNS


METHOD = "ocir"
TASK = "rul"
PROTOCOL_VERSION = "main_protocol_v1"
SPLIT_ID = MAIN_PROTOCOL_V1["splits"]["split_id"]
SPLIT_SEED = int(MAIN_PROTOCOL_V1["splits"]["split_seed"])
RUNNER_SUMMARY_COLUMNS = [
    "exp_id",
    "dataset",
    "model_seed",
    "status",
    "output_dir",
    "started_at_utc",
    "ended_at_utc",
    "duration_seconds",
    "error_type",
    "error_message",
]
TRAIN_LOG_COLUMNS = [
    "stage",
    "epoch",
    "iteration",
    "loss_rec",
    "loss_kl",
    "loss_klc",
    "loss_realD",
    "loss_fakeD",
    "loss_gen",
    "loss_Q",
    "loss_ccz",
    "loss_ccc",
    "loss_mse",
]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def format_timestamp_utc(timestamp: dt.datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def format_iso_utc(timestamp: dt.datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="./data/processed/main_protocol_v1")
    parser.add_argument("--results-root", default="./results/experiments")
    parser.add_argument("--datasets", nargs="+", default=None, choices=MAIN_PROTOCOL_V1["datasets"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpu-dev", default="0")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--rul-epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def dataset_c_type(dataset: str) -> str:
    return "continuous" if dataset in {"FD001", "FD003"} else "discrete"


def build_base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": TASK,
        "dataset": "FD001",
        "c_type": "continuous",
        "net": METHOD,
        "valid_split": 0.0,
        "seed": 1,
        "gpu_dev": str(args.gpu_dev),
        "n_epochs": int(args.n_epochs),
        "batch": int(args.batch),
        "patience": 10,
        "rul_epochs": int(args.rul_epochs),
        "rul_validate_every_epochs": int(MAIN_PROTOCOL_V1["training"]["rul_validate_every_epochs"]),
        "lr_": 5e-4,
        "rul_lr": 5e-4,
        "scheduler": 0,
        "alpha": 0.8,
        "kl_annealing": 0.15,
        "warm_up": 0.2,
        "start_lr": 1e-5,
        "ref_lr": 1.5e-4,
        "final_lr": 1e-5,
        "start_wd": 0.01,
        "final_wd": 0.0001,
        "model_save_path": "checkpoints",
        "plots_save_path": "plots",
        "his_save_path": "hist",
        "window": int(MAIN_PROTOCOL_V1["windowing"]["window_length"]),
        "H": 2,
        "hyper_lookback": 2,
        "time_embedding": False,
        "c_kl": 0,
        "dx": int(MAIN_PROTOCOL_V1["features"]["selected_sensor_dim"]),
        "dz": 20,
        "dc": int(MAIN_PROTOCOL_V1["preprocessing"]["pasa_kmeans"]["k"]),
        "d_model": 128,
        "num_heads": 2,
        "encoder_E": "transformer",
        "z_projection": "spc",
        "D_projection": "spc",
        "c_posterior_param": "soft",
        "extractor": METHOD,
        "conditional": 1,
        "data_path": str(args.data_path),
        "description": "main_protocol_v1_runner",
        "id_": 0,
        "log_path": "",
    }


def expand_experiment_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = args.datasets if args.datasets is not None else list(MAIN_PROTOCOL_V1["datasets"])
    seeds = args.model_seeds if args.model_seeds is not None else list(MAIN_PROTOCOL_V1["training"]["model_seeds"])
    allowed_seeds = set(int(seed) for seed in MAIN_PROTOCOL_V1["training"]["model_seeds"])
    bad_seeds = [seed for seed in seeds if int(seed) not in allowed_seeds]
    if bad_seeds:
        raise ValueError(f"model seeds must be in {sorted(allowed_seeds)}, got {bad_seeds}")
    return [{"dataset": dataset, "model_seed": int(seed)} for dataset in datasets for seed in seeds]


def _run_local_git(repo_root: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def collect_git_info(repo_root: pathlib.Path) -> dict[str, Any]:
    root_check = _run_local_git(repo_root, ["rev-parse", "--show-toplevel"])
    if root_check.returncode != 0:
        return {"commit": None, "short": "nogit", "dirty": None, "status_short": []}

    commit_result = _run_local_git(repo_root, ["rev-parse", "HEAD"])
    status_result = _run_local_git(repo_root, ["status", "--short"])
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_short = status_result.stdout.splitlines() if status_result.returncode == 0 else []
    return {
        "commit": commit,
        "short": commit[:7] if commit else "nogit",
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def build_exp_id(
    dataset: str,
    model_seed: int,
    git_info: dict[str, Any],
    timestamp_utc: dt.datetime,
) -> str:
    if dataset not in MAIN_PROTOCOL_V1["datasets"]:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if int(model_seed) not in MAIN_PROTOCOL_V1["training"]["model_seeds"]:
        raise ValueError(f"unsupported model_seed {model_seed!r}")
    git_short = git_info.get("short") or "nogit"
    return (
        f"{format_timestamp_utc(timestamp_utc)}_{PROTOCOL_VERSION}_{METHOD}_{TASK}_"
        f"{dataset}_seed{int(model_seed)}_split{SPLIT_SEED}_{git_short}"
    )


def setup_experiment_dirs(results_root: pathlib.Path, exp_id: str) -> dict[str, pathlib.Path]:
    exp_dir = results_root / exp_id
    if exp_dir.exists():
        raise FileExistsError(f"experiment directory already exists: {exp_dir}")

    paths = {
        "exp_dir": exp_dir,
        "figures_dir": exp_dir / "figures",
        "figures_rl_dir": exp_dir / "figures" / "rl",
        "figures_rul_dir": exp_dir / "figures" / "rul",
        "checkpoints_dir": exp_dir / "checkpoints",
        "raw_logs_dir": exp_dir / "raw_logs",
        "raw_logs_rl_dir": exp_dir / "raw_logs" / "rl",
        "raw_logs_rul_dir": exp_dir / "raw_logs" / "rul",
        "logs_dir": exp_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return paths


def make_run_config(base_config: dict[str, Any], dataset: str, model_seed: int, paths: dict[str, pathlib.Path]) -> dict[str, Any]:
    run_config = deepcopy(base_config)
    run_config.update(
        {
            "dataset": dataset,
            "seed": int(model_seed),
            "c_type": dataset_c_type(dataset),
            "model_save_path": str(paths["checkpoints_dir"]),
            "plots_save_path": str(paths["figures_dir"]),
            "his_save_path": str(paths["raw_logs_dir"]),
            "log_path": str(paths["exp_dir"]),
        }
    )
    return run_config


def make_experiment_config(
    base_config: dict[str, Any],
    dataset: str,
    model_seed: int,
    exp_id: str,
    paths: dict[str, pathlib.Path],
    git_info: dict[str, Any],
    argv: list[str],
) -> dict[str, Any]:
    run_config = make_run_config(base_config, dataset, model_seed, paths)
    return {
        "exp_id": exp_id,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "split_id": SPLIT_ID,
            "split_seed": SPLIT_SEED,
            "validation_view": MAIN_PROTOCOL_V1["validation_view"]["version"],
            "train_sampler": MAIN_PROTOCOL_V1["training"]["sampler"],
            "window_length": int(MAIN_PROTOCOL_V1["windowing"]["window_length"]),
            "rul_cap": int(MAIN_PROTOCOL_V1["labels"]["rul_cap"]),
            "label_scale": MAIN_PROTOCOL_V1["labels"]["scale"],
            "label_normalization": MAIN_PROTOCOL_V1["labels"]["normalization"],
        },
        "experiment": {
            "method": METHOD,
            "task": TASK,
            "dataset": dataset,
            "model_seed": int(model_seed),
            "data_path": run_config["data_path"],
            "output_dir": str(paths["exp_dir"]),
        },
        "training": {
            "n_epochs": int(run_config["n_epochs"]),
            "rul_epochs": int(run_config["rul_epochs"]),
            "batch": int(run_config["batch"]),
            "lr_": float(run_config["lr_"]),
            "rul_lr": float(run_config["rul_lr"]),
            "rul_validate_every_epochs": int(run_config["rul_validate_every_epochs"]),
        },
        "model": {
            "dx": int(run_config["dx"]),
            "dz": int(run_config["dz"]),
            "dc": int(run_config["dc"]),
            "d_model": int(run_config["d_model"]),
            "num_heads": int(run_config["num_heads"]),
            "encoder_E": run_config["encoder_E"],
            "z_projection": run_config["z_projection"],
            "D_projection": run_config["D_projection"],
            "c_type": run_config["c_type"],
        },
        "git": git_info,
        "command": {"argv": argv},
    }


def write_yaml(path: pathlib.Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def collect_runtime_info(
    exp_id: str,
    status: str,
    started_at_utc: str,
    git_info: dict[str, Any],
    argv: list[str],
    cwd: pathlib.Path,
    failure: dict[str, Any] | None = None,
    ended_at_utc: str | None = None,
    duration_seconds: float | None = None,
    gpu_dev: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu: dict[str, Any] = {
        "device_arg": gpu_dev,
        "selected_device": f"cuda:{gpu_dev}" if cuda_available and gpu_dev is not None else "cpu",
        "name": None,
        "count": 0,
        "memory_allocated_bytes": None,
        "memory_reserved_bytes": None,
        "max_memory_allocated_bytes": None,
    }
    if cuda_available:
        selected_index = int(gpu_dev) if gpu_dev is not None and str(gpu_dev).isdigit() else 0
        if selected_index < torch.cuda.device_count():
            gpu.update(
                {
                    "name": torch.cuda.get_device_name(selected_index),
                    "count": torch.cuda.device_count(),
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated(selected_index)),
                    "memory_reserved_bytes": int(torch.cuda.memory_reserved(selected_index)),
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(selected_index)),
                }
            )

    return {
        "exp_id": exp_id,
        "status": status,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "torch": {
            "version": None if torch is None else torch.__version__,
            "cuda_version": None if torch is None else torch.version.cuda,
            "cuda_available": cuda_available,
        },
        "gpu": gpu,
        "git": git_info,
        "command": {"argv": argv, "cwd": str(cwd)},
        "failure": failure,
    }


def setup_file_logger(log_path: pathlib.Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def build_stage_config(base_config: dict[str, Any], paths: dict[str, pathlib.Path], stage: str) -> dict[str, Any]:
    if stage not in {"RL", "RUL"}:
        raise ValueError("stage must be 'RL' or 'RUL'")
    config = deepcopy(base_config)
    suffix = stage.lower()
    config["plots_save_path"] = str(paths[f"figures_{suffix}_dir"])
    config["his_save_path"] = str(paths[f"raw_logs_{suffix}_dir"])
    config["model_save_path"] = str(paths["checkpoints_dir"])
    return config


def copy_standard_checkpoint(paths: dict[str, pathlib.Path]) -> pathlib.Path:
    source = paths["checkpoints_dir"] / "rul_cp.pth"
    target = paths["exp_dir"] / "checkpoint.pt"
    if not source.exists():
        raise FileNotFoundError(f"missing RUL checkpoint: {source}")
    shutil.copy2(source, target)
    return target


def normalize_train_logs(paths: dict[str, pathlib.Path]) -> pathlib.Path:
    output_path = paths["exp_dir"] / "train_log.csv"
    raw_sources = {
        "RL": paths["raw_logs_rl_dir"] / "training_log.csv",
        "RUL": paths["raw_logs_rul_dir"] / "training_log.csv",
    }
    rows: list[dict[str, str]] = []
    for stage, path in raw_sources.items():
        if not path.exists():
            raise FileNotFoundError(f"missing raw {stage} training log: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"raw {stage} training log has no header: {path}")
            for raw_row in reader:
                row = {column: "" for column in TRAIN_LOG_COLUMNS}
                row["stage"] = stage
                row["epoch"] = raw_row.get("epoch", "")
                row["iteration"] = raw_row.get("itr", raw_row.get("iteration", ""))
                for column in TRAIN_LOG_COLUMNS:
                    if column in {"stage", "epoch", "iteration"}:
                        continue
                    row[column] = raw_row.get(column, "")
                rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def record_final_test_access(
    repo_root: pathlib.Path,
    exp_id: str,
    dataset: str,
    model_seed: int,
    git_info: dict[str, Any],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    access_log_path = reports_dir / "test_access_log.md"
    if not access_log_path.exists():
        access_log_path.write_text(
            "# Final Test Access Log\n\n"
            "| timestamp_utc | exp_id | dataset | model_seed | method | protocol | reason | caller | output_dir | git_commit | git_dirty |\n"
            "|---|---|---|---:|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    row = [
        format_iso_utc(utc_now()),
        exp_id,
        dataset,
        str(int(model_seed)),
        METHOD,
        PROTOCOL_VERSION,
        "final_evaluation_after_training_and_best_checkpoint_restore",
        pathlib.Path(__file__).name,
        str(output_dir),
        str(git_info.get("commit")),
        str(git_info.get("dirty")),
    ]
    with access_log_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("| " + " | ".join(item.replace("|", "\\|") for item in row) + " |\n")
    return access_log_path


def merge_metrics_metadata(
    paths: dict[str, pathlib.Path],
    exp_id: str,
    git_info: dict[str, Any],
    runtime_payload: dict[str, Any],
    rul_pipeline: Any | None = None,
) -> pathlib.Path:
    metrics_path = paths["exp_dir"] / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "exp_id": exp_id,
            "output_dir": str(paths["exp_dir"]),
            "git_commit": git_info.get("commit"),
            "git_short": git_info.get("short"),
            "git_dirty": git_info.get("dirty"),
            "duration_seconds": runtime_payload.get("duration_seconds"),
        }
    )
    if rul_pipeline is not None:
        payload["best_rul_epoch"] = getattr(rul_pipeline, "best_rul_epoch", None)
        best_rmse = getattr(rul_pipeline, "best_val_engine_rmse", None)
        payload["best_val_engine_rmse"] = None if best_rmse is None else float(best_rmse)
    write_json(metrics_path, payload)
    return metrics_path


def validate_predictions_csv(path: pathlib.Path, config_payload: dict[str, Any]) -> int:
    if not path.exists():
        raise FileNotFoundError(f"missing predictions.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FINAL_PREDICTION_COLUMNS:
            raise ValueError(f"predictions.csv columns mismatch: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("predictions.csv must contain at least one prediction row")
    expected = config_payload["experiment"]
    for index, row in enumerate(rows):
        if row["dataset"] != expected["dataset"]:
            raise ValueError(f"predictions row {index} dataset mismatch")
        if int(row["model_seed"]) != int(expected["model_seed"]):
            raise ValueError(f"predictions row {index} model_seed mismatch")
        if row["main_protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"predictions row {index} protocol mismatch")
        if row["method"] != METHOD:
            raise ValueError(f"predictions row {index} method mismatch")
    return len(rows)


def validate_metrics_json(path: pathlib.Path, config_payload: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing metrics.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "evaluation_views",
        "provided_rul_eval_v1",
        "capped_target_eval_v1",
        "num_test_units",
        "dataset",
        "model_seed",
        "split_id",
        "split_seed",
        "window_length",
        "rul_cap",
        "label_scale",
        "label_normalization",
        "exp_id",
        "git_commit",
        "git_short",
        "git_dirty",
    ]
    missing = []
    for key in required:
        if key in {"provided_rul_eval_v1", "capped_target_eval_v1"}:
            if key not in payload.get("evaluation_views", {}):
                missing.append(f"evaluation_views.{key}")
        elif key not in payload:
            missing.append(key)
    if missing:
        raise ValueError(f"metrics.json missing required fields: {missing}")
    if payload["dataset"] != config_payload["experiment"]["dataset"]:
        raise ValueError("metrics.json dataset mismatch")
    if int(payload["model_seed"]) != int(config_payload["experiment"]["model_seed"]):
        raise ValueError("metrics.json model_seed mismatch")


def validate_train_log(path: pathlib.Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"missing train_log.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRAIN_LOG_COLUMNS:
            raise ValueError(f"train_log.csv columns mismatch: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("train_log.csv must contain at least one row")
    return len(rows)


def validate_experiment_outputs(paths: dict[str, pathlib.Path], config_payload: dict[str, Any]) -> None:
    required_paths = [
        paths["exp_dir"] / "config.yaml",
        paths["exp_dir"] / "runtime.json",
        paths["exp_dir"] / "metrics.json",
        paths["exp_dir"] / "predictions.csv",
        paths["exp_dir"] / "train_log.csv",
        paths["exp_dir"] / "checkpoint.pt",
        paths["figures_dir"],
        paths["figures_rl_dir"],
        paths["figures_rul_dir"],
        paths["checkpoints_dir"],
        paths["raw_logs_rl_dir"] / "training_log.csv",
        paths["raw_logs_rul_dir"] / "training_log.csv",
        paths["logs_dir"] / "log_all.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required experiment outputs: {missing}")

    runtime = json.loads((paths["exp_dir"] / "runtime.json").read_text(encoding="utf-8"))
    if runtime.get("status") != "success":
        raise ValueError("runtime.json status must be success for output validation")
    validate_metrics_json(paths["exp_dir"] / "metrics.json", config_payload)
    validate_predictions_csv(paths["exp_dir"] / "predictions.csv", config_payload)
    validate_train_log(paths["exp_dir"] / "train_log.csv")


def build_failure(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def run_single_experiment(
    base_config: dict[str, Any],
    dataset: str,
    model_seed: int,
    results_root: pathlib.Path,
    git_info: dict[str, Any],
    argv: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    timestamp = utc_now()
    started_at = format_iso_utc(timestamp)
    exp_id = build_exp_id(dataset, model_seed, git_info, timestamp)
    output_dir = results_root / exp_id
    summary = {
        "exp_id": exp_id,
        "dataset": dataset,
        "model_seed": int(model_seed),
        "status": "dry_run" if dry_run else "running",
        "output_dir": str(output_dir),
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "duration_seconds": None,
        "error_type": "",
        "error_message": "",
    }
    if dry_run:
        return summary

    start_monotonic = time.monotonic()
    paths: dict[str, pathlib.Path] | None = None
    logger: logging.Logger | None = None
    runtime_payload: dict[str, Any] | None = None
    config_payload: dict[str, Any] | None = None
    try:
        paths = setup_experiment_dirs(results_root, exp_id)
        config_payload = make_experiment_config(base_config, dataset, model_seed, exp_id, paths, git_info, argv)
        write_yaml(paths["exp_dir"] / "config.yaml", config_payload)
        runtime_payload = collect_runtime_info(
            exp_id=exp_id,
            status="running",
            started_at_utc=started_at,
            git_info=git_info,
            argv=argv,
            cwd=REPO_ROOT,
            gpu_dev=str(base_config.get("gpu_dev", "")),
        )
        write_json(paths["exp_dir"] / "runtime.json", runtime_payload)
        logger = setup_file_logger(paths["logs_dir"] / "log_all.txt", exp_id)

        run_config = make_run_config(base_config, dataset, model_seed, paths)
        rl_config = build_stage_config(run_config, paths, stage="RL")
        rul_config = build_stage_config(run_config, paths, stage="RUL")

        logger.info("Starting RL pipeline for %s seed %s", dataset, model_seed)
        rl_pipeline = RlPipeline(rl_config, logger)
        encoder = rl_pipeline()

        logger.info("Starting RUL pipeline for %s seed %s", dataset, model_seed)
        rul_pipeline = RulPipeline(rul_config, logger, encoder, None)
        rul_pipeline()

        copy_standard_checkpoint(paths)
        record_final_test_access(REPO_ROOT, exp_id, dataset, model_seed, git_info, paths["exp_dir"])
        rul_pipeline.evaluate_test_terminal(output_dir=paths["exp_dir"], method=METHOD, write_files=True)
        normalize_train_logs(paths)

        ended_at = format_iso_utc(utc_now())
        duration = time.monotonic() - start_monotonic
        runtime_payload = collect_runtime_info(
            exp_id=exp_id,
            status="success",
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            duration_seconds=duration,
            git_info=git_info,
            argv=argv,
            cwd=REPO_ROOT,
            gpu_dev=str(base_config.get("gpu_dev", "")),
        )
        merge_metrics_metadata(paths, exp_id, git_info, runtime_payload, rul_pipeline=rul_pipeline)
        write_json(paths["exp_dir"] / "runtime.json", runtime_payload)
        validate_experiment_outputs(paths, config_payload)

        summary.update(
            {
                "status": "success",
                "ended_at_utc": ended_at,
                "duration_seconds": duration,
            }
        )
    except Exception as exc:
        ended_at = format_iso_utc(utc_now())
        duration = time.monotonic() - start_monotonic
        failure = build_failure(exc)
        summary.update(
            {
                "status": "failed",
                "ended_at_utc": ended_at,
                "duration_seconds": duration,
                "error_type": failure["type"],
                "error_message": failure["message"],
            }
        )
        if paths is not None:
            failed_runtime = collect_runtime_info(
                exp_id=exp_id,
                status="failed",
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                duration_seconds=duration,
                git_info=git_info,
                argv=argv,
                cwd=REPO_ROOT,
                failure=failure,
                gpu_dev=str(base_config.get("gpu_dev", "")),
            )
            write_json(paths["exp_dir"] / "runtime.json", failed_runtime)
        if logger is not None:
            logger.exception("Experiment failed")
    finally:
        if logger is not None:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
    return summary


def write_runner_summary(results_root: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "runner_summary.json", {"experiments": rows})
    with (results_root / "runner_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNNER_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUNNER_SUMMARY_COLUMNS})


def run_experiment_grid(args: argparse.Namespace) -> int:
    base_config = build_base_config(args)
    git_info = collect_git_info(REPO_ROOT)
    results_root = pathlib.Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root

    rows = []
    for item in expand_experiment_grid(args):
        row = run_single_experiment(
            base_config=base_config,
            dataset=item["dataset"],
            model_seed=item["model_seed"],
            results_root=results_root,
            git_info=git_info,
            argv=[sys.executable, "-B", *sys.argv],
            dry_run=bool(args.dry_run),
        )
        rows.append(row)
        print(
            f"{row['dataset']},seed{row['model_seed']},{row['exp_id']},{row['output_dir']},{row['status']}"
        )
        if row["status"] == "failed" and args.fail_fast:
            break

    if not args.dry_run:
        write_runner_summary(results_root, rows)

    return 1 if any(row["status"] == "failed" for row in rows) else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_experiment_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
