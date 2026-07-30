"""Smoke-test experiment_runner_outputs_v1 without training or real data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_main_protocol_experiments import (
    FINAL_PREDICTION_COLUMNS,
    PROTOCOL_VERSION,
    TRAIN_LOG_COLUMNS,
    build_base_config,
    build_exp_id,
    build_official_test_access_approval,
    collect_runtime_info,
    copy_standard_checkpoint,
    expand_experiment_grid,
    make_experiment_config,
    merge_metrics_metadata,
    normalize_train_logs,
    setup_experiment_dirs,
    validate_experiment_outputs,
    write_json,
    write_yaml,
)


EXP_ID_RE = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z_main_protocol_v1_ocir_rul_FD00[1-4]_seed[1-5]_split2026_([0-9a-f]{7}|nogit)$"
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_reset_smoke_root(smoke_root: pathlib.Path) -> None:
    resolved = smoke_root.resolve()
    expected_parent = (REPO_ROOT / "results" / "experiments").resolve()
    if "_smoke_experiment_runner_outputs" not in resolved.parts:
        raise RuntimeError(f"refusing to reset non-smoke path: {resolved}")
    if expected_parent not in resolved.parents:
        raise RuntimeError(f"refusing to reset path outside results/experiments: {resolved}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=False)


def write_fake_raw_logs(paths: dict[str, pathlib.Path]) -> None:
    with (paths["raw_logs_rl_dir"] / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "itr",
                "loss_rec",
                "loss_kl",
                "loss_klc",
                "loss_realD",
                "loss_fakeD",
                "loss_gen",
                "loss_Q",
                "loss_ccz",
                "loss_ccc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "1",
                "itr": "0",
                "loss_rec": "0.1",
                "loss_kl": "0.2",
                "loss_klc": "0.3",
                "loss_realD": "0.4",
                "loss_fakeD": "0.5",
                "loss_gen": "0.6",
                "loss_Q": "0.7",
                "loss_ccz": "0.8",
                "loss_ccc": "0.9",
            }
        )
    with (paths["raw_logs_rul_dir"] / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "itr", "loss_mse"])
        writer.writeheader()
        writer.writerow({"epoch": "1", "itr": "0", "loss_mse": "1.23"})


def write_fake_predictions(paths: dict[str, pathlib.Path], dataset: str, model_seed: int) -> None:
    row = {
        "dataset": dataset,
        "source_partition": "test",
        "unit": "1",
        "end_cycle": "30",
        "provided_rul": "150.0",
        "capped_target_rul": "125.0",
        "raw_pred_rul": "120.0",
        "provided_eval_pred_rul": "120.0",
        "capped_eval_pred_rul": "120.0",
        "error_provided": "-30.0",
        "error_capped": "-5.0",
        "abs_error_provided": "30.0",
        "abs_error_capped": "5.0",
        "method": "ocir",
        "model_seed": str(model_seed),
        "window_length": "30",
        "rul_cap": "125",
        "main_protocol_version": PROTOCOL_VERSION,
        "split_id": "split_v1_seed2026_engine_80_20",
        "validation_view": "validation_view_v1",
        "train_sampler": "engine_balanced_sampler_v1",
        "preprocessing_version": PROTOCOL_VERSION,
    }
    with (paths["exp_dir"] / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def write_fake_metrics(paths: dict[str, pathlib.Path], dataset: str, model_seed: int) -> None:
    write_json(
        paths["exp_dir"] / "metrics.json",
        {
            "dataset": dataset,
            "source_partition": "test",
            "method": "ocir",
            "model_seed": model_seed,
            "split_id": "split_v1_seed2026_engine_80_20",
            "split_seed": 2026,
            "window_length": 30,
            "rul_cap": 125,
            "label_scale": "cycle",
            "label_normalization": "none",
            "num_test_units": 1,
            "evaluation_views": {
                "provided_rul_eval_v1": {"rmse": 30.0, "mae": 30.0, "nasa_score": 9.0},
                "capped_target_eval_v1": {"rmse": 5.0, "mae": 5.0, "nasa_score": 0.4},
            },
        },
    )


def main() -> int:
    args = argparse.Namespace(
        data_path="./data/processed/main_protocol_v1",
        results_root="./results/experiments",
        datasets=None,
        model_seeds=None,
        gpu_dev="0",
        n_epochs=50,
        rul_epochs=50,
        batch=128,
        fail_fast=False,
        dry_run=False,
        allow_official_test=True,
        g4_approval_id="G4-SMOKE-001",
        g4_approved_by="smoke-test",
        g4_approval_note="synthetic smoke only",
    )
    sealed_args = argparse.Namespace(**{**vars(args), "allow_official_test": False})
    try:
        build_official_test_access_approval(sealed_args)
    except PermissionError:
        pass
    else:
        raise AssertionError("non-dry-run official test access must require explicit G4 approval")
    dry_args = argparse.Namespace(**{**vars(args), "dry_run": True, "allow_official_test": False})
    assert_true(build_official_test_access_approval(dry_args) is None, "dry-run must not require official test approval")
    access_approval = build_official_test_access_approval(args)
    assert_true(access_approval["approval_id"] == "G4-SMOKE-001", "approval id mismatch")

    grid = expand_experiment_grid(args)
    assert_true(len(grid) == 20, "default grid must contain 20 experiments")
    assert_true([item["dataset"] for item in grid[:5]] == ["FD001"] * 5, "dataset-major grid order failed")
    assert_true([item["model_seed"] for item in grid[:5]] == [1, 2, 3, 4, 5], "seed order failed")
    assert_true(grid[-1] == {"dataset": "FD004", "model_seed": 5}, "last grid item mismatch")

    smoke_root = REPO_ROOT / "results" / "experiments" / "_smoke_experiment_runner_outputs"
    safe_reset_smoke_root(smoke_root)

    git_info = {"commit": "abcdef0123456789", "short": "abcdef0", "dirty": True, "status_short": [" M fake.py"]}
    timestamp = dt.datetime(2026, 7, 30, 10, 30, 12, tzinfo=dt.timezone.utc)
    exp_id = build_exp_id("FD001", 1, git_info, timestamp)
    assert_true(EXP_ID_RE.match(exp_id) is not None, f"exp_id does not match regex: {exp_id}")

    paths = setup_experiment_dirs(smoke_root, exp_id)
    try:
        setup_experiment_dirs(smoke_root, exp_id)
    except FileExistsError:
        pass
    else:
        raise AssertionError("setup_experiment_dirs must protect against overwriting exp_dir")

    base_config = build_base_config(args)
    config_payload = make_experiment_config(
        base_config=base_config,
        dataset="FD001",
        model_seed=1,
        exp_id=exp_id,
        paths=paths,
        git_info=git_info,
        argv=["python", "-B", "scripts/run_main_protocol_experiments.py"],
        access_approval=access_approval,
    )
    write_yaml(paths["exp_dir"] / "config.yaml", config_payload)
    (paths["logs_dir"] / "log_all.txt").write_text("fake log\n", encoding="utf-8")
    (paths["checkpoints_dir"] / "rul_cp.pth").write_bytes(b"fake checkpoint\n")
    copied = copy_standard_checkpoint(paths)
    assert_true(copied.name == "checkpoint.pt" and copied.exists(), "checkpoint copy failed")

    write_fake_raw_logs(paths)
    train_log_path = normalize_train_logs(paths)
    with train_log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert_true(reader.fieldnames == TRAIN_LOG_COLUMNS, "normalized train_log.csv columns mismatch")
        rows = list(reader)
    assert_true([row["stage"] for row in rows] == ["RL", "RUL"], "normalized train log stages mismatch")
    assert_true(rows[0]["loss_mse"] == "" and rows[1]["loss_rec"] == "", "missing metrics must be empty strings")

    write_fake_predictions(paths, "FD001", 1)
    write_fake_metrics(paths, "FD001", 1)
    runtime_payload = collect_runtime_info(
        exp_id=exp_id,
        status="success",
        started_at_utc="2026-07-30T10:30:12Z",
        ended_at_utc="2026-07-30T10:30:13Z",
        duration_seconds=1.0,
        git_info=git_info,
        argv=["python", "-B", "scripts/run_main_protocol_experiments.py"],
        cwd=REPO_ROOT,
        gpu_dev="0",
    )
    write_json(paths["exp_dir"] / "runtime.json", runtime_payload)
    merge_metrics_metadata(paths, exp_id, git_info, runtime_payload, rul_pipeline=None)
    validate_experiment_outputs(paths, config_payload)

    print("smoke_experiment_runner_outputs OK")
    print("grid=20")
    print(f"exp_id={exp_id}")
    print(f"smoke_dir={paths['exp_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
