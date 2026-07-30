"""Audit OCIR data/code alignment with the local main_protocol_v1.

This script is intentionally read-only. It checks two things:
1. Static OCIR source/script choices that affect protocol alignment.
2. Processed C-MAPSS files, when present.

It does not train a model and does not compute official-test metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir_reproduction.data.cmapss_protocol import (
    load_json,
    load_npz,
    validate_full_test_arrays,
    validate_manifest,
    validate_split_file,
    validate_test_terminal_arrays,
    validate_train_arrays,
    validate_validation_arrays,
)


DATASETS = ("FD001", "FD002", "FD003", "FD004")
MAIN_PROTOCOL = {
    "window": 30,
    "rul_cap": 125,
    "feature_dim": 14,
    "split_seed": 2026,
    "model_seeds": [1, 2, 3, 4, 5],
    "padding": "left_repeat_first",
    "test_view": "one terminal window per official test unit",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def add(checks: list[Check], name: str, status: str, detail: str) -> None:
    checks.append(Check(name=name, status=status, detail=detail))


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def method_body(content: str, name: str) -> str:
    match = re.search(
        rf"\n    def\s+{name}\s*\([^)]*\):(?P<body>.*?)(?=\n    def\s|\nclass\s|\Z)",
        content,
        flags=re.DOTALL,
    )
    return "" if match is None else match.group("body")


def audit_static(repo: Path, checks: list[Check]) -> None:
    main_py = repo / "main.py"
    if main_py.exists():
        main_text = text(main_py)
        default_window = re.search(r'--window".*?default\s*=\s*(\d+)', main_text, re.S)
        if default_window:
            value = int(default_window.group(1))
            status = "pass" if value == MAIN_PROTOCOL["window"] else "warn"
            add(
                checks,
                "main.py default window",
                status,
                f"default={value}; main_protocol_v1 expects {MAIN_PROTOCOL['window']}",
            )
        else:
            add(checks, "main.py default window", "warn", "could not find parser default")
    else:
        add(checks, "main.py exists", "fail", f"missing {main_py}")

    rul_scripts = sorted((repo / "scripts" / "rul").glob("*.sh"))
    if rul_scripts:
        mismatches = []
        matches = []
        for script in rul_scripts:
            content = text(script)
            windows = re.findall(r"^\s*W=(\d+)\s*$", content, flags=re.M)
            if not windows:
                continue
            values = sorted({int(v) for v in windows})
            if values == [MAIN_PROTOCOL["window"]]:
                matches.append(script.name)
            else:
                mismatches.append(f"{script.name}: W={values}")
        if mismatches:
            add(
                checks,
                "RUL script window",
                "fail",
                "; ".join(mismatches)
                + f"; main_protocol_v1 expects W={MAIN_PROTOCOL['window']}",
            )
        else:
            add(checks, "RUL script window", "pass", f"all detected scripts use W={MAIN_PROTOCOL['window']}: {matches}")
    else:
        add(checks, "RUL scripts", "warn", "no scripts/rul/*.sh files found")

    protocol_loader = repo / "ocir_reproduction" / "data" / "cmapss_protocol.py"
    if protocol_loader.exists():
        content = text(protocol_loader)
        add(
            checks,
            "protocol data loader exists",
            "pass",
            f"found {protocol_loader}",
        )
        add(
            checks,
            "protocol data loader interface",
            "pass" if "def load_main_protocol_cmapss" in content else "fail",
            "load_main_protocol_cmapss is available"
            if "def load_main_protocol_cmapss" in content
            else "missing load_main_protocol_cmapss",
        )
        add(
            checks,
            "protocol data arrays",
            "pass"
            if all(name in content for name in ("train_windows.npz", "validation_view_v1.npz", "test_terminal_windows.npz", "full_test_windows.npz"))
            else "fail",
            "loader reads the four main_protocol_v1 processed files"
            if all(name in content for name in ("train_windows.npz", "validation_view_v1.npz", "test_terminal_windows.npz", "full_test_windows.npz"))
            else "loader does not reference all required processed files",
        )
        add(
            checks,
            "protocol target bounds",
            "pass" if "y must satisfy 0 <= y <=" in content else "warn",
            "loader validates capped RUL target bounds"
            if "y must satisfy 0 <= y <=" in content
            else "could not verify capped target bound validation",
        )
        make_training_loader_start = content.find("def make_training_loader")
        make_eval_loader_start = content.find("\ndef make_eval_loader", make_training_loader_start)
        make_training_loader_body = (
            content[make_training_loader_start:make_eval_loader_start]
            if make_training_loader_start >= 0 and make_eval_loader_start > make_training_loader_start
            else ""
        )
        has_sampler_class = "class EngineBalancedSampler" in content
        has_sampler_loader = "sampler = EngineBalancedSampler" in make_training_loader_body and "sampler=sampler" in make_training_loader_body
        has_loader_shuffle_true = "shuffle=True" in make_training_loader_body
        add(
            checks,
            "engine_balanced_sampler_v1 loader",
            "pass" if has_sampler_class and has_sampler_loader and not has_loader_shuffle_true else "fail",
            "make_training_loader uses EngineBalancedSampler and does not use shuffle=True"
            if has_sampler_class and has_sampler_loader and not has_loader_shuffle_true
            else "make_training_loader must define/use EngineBalancedSampler and remove shuffle=True",
        )
    else:
        add(checks, "protocol data loader exists", "fail", f"missing {protocol_loader}")

    solver_base = repo / "pipelines" / "solver_base.py"
    if solver_base.exists():
        content = text(solver_base)
        add(
            checks,
            "training sampler epoch hook",
            "pass" if "def set_training_sampler_epoch" in content and ".set_epoch(epoch)" in content else "fail",
            "Solver.set_training_sampler_epoch delegates epoch to the training DataLoader sampler"
            if "def set_training_sampler_epoch" in content and ".set_epoch(epoch)" in content
            else "Solver must expose set_training_sampler_epoch and call sampler.set_epoch(epoch)",
        )
    else:
        add(checks, "training sampler epoch hook", "fail", f"missing {solver_base}")

    rl_pipeline = repo / "pipelines" / "representtion_learning.py"
    if rl_pipeline.exists():
        content = text(rl_pipeline)
        add(
            checks,
            "RL train sampler epoch",
            "pass" if "self.set_training_sampler_epoch(epoch)" in content else "fail",
            "RlPipeline.train calls set_training_sampler_epoch(epoch)"
            if "self.set_training_sampler_epoch(epoch)" in content
            else "RlPipeline.train must call self.set_training_sampler_epoch(epoch) at each epoch start",
        )
    else:
        add(checks, "RL pipeline exists", "fail", f"missing {rl_pipeline}")

    rul_pipeline = repo / "pipelines" / "rul_estimation.py"
    if rul_pipeline.exists():
        content = text(rul_pipeline)
        has_legacy_vali = re.search(r"def\s+vali\s*\(", content) is not None or "self.vali(" in content
        uses_legacy_metric = "ut.RUL_metric(" in content
        has_validation_entry = re.search(r"def\s+evaluate_validation\s*\(", content) is not None
        validation_uses_val_data = "self.val_data" in content and "predict_loader(self.val_data)" in content
        validation_uses_test = re.search(
            r"def\s+evaluate_validation\s*\([^)]*\):(?:(?!\n    def\s).)*self\.testing_data",
            content,
            flags=re.DOTALL,
        ) is not None
        validation_uses_full_test = re.search(
            r"def\s+evaluate_validation\s*\([^)]*\):(?:(?!\n    def\s).)*self\.full_test_set",
            content,
            flags=re.DOTALL,
        ) is not None

        add(
            checks,
            "RUL validation data",
            "pass"
            if (
                has_validation_entry
                and validation_uses_val_data
                and not has_legacy_vali
                and not uses_legacy_metric
                and not validation_uses_test
                and not validation_uses_full_test
            )
            else "fail",
            "RUL validation uses validation_view/self.val_data and avoids legacy official-test validation"
            if (
                has_validation_entry
                and validation_uses_val_data
                and not has_legacy_vali
                and not uses_legacy_metric
                and not validation_uses_test
                and not validation_uses_full_test
            )
            else (
                "RUL validation must define evaluate_validation(self.val_data), remove legacy vali/self.vali, "
                "avoid ut.RUL_metric, and keep official test out of validation"
            ),
        )
        add(
            checks,
            "RUL train sampler epoch",
            "pass" if "self.set_training_sampler_epoch(epoch)" in content else "fail",
            "RulPipeline.train calls set_training_sampler_epoch(epoch)"
            if "self.set_training_sampler_epoch(epoch)" in content
            else "RulPipeline.train must call self.set_training_sampler_epoch(epoch) at each epoch start",
        )

        final_eval_methods = (
            "get_loader_metadata",
            "validate_test_terminal_metadata",
            "build_final_prediction_rows",
            "compute_final_view_metrics",
            "build_final_metrics_payload",
            "write_final_evaluation_outputs",
        )
        missing_final_eval_methods = [
            name for name in final_eval_methods if re.search(rf"def\s+{name}\s*\(", content) is None
        ]
        add(
            checks,
            "final evaluation methods",
            "pass" if not missing_final_eval_methods else "fail",
            "RulPipeline exposes final evaluation helpers"
            if not missing_final_eval_methods
            else f"missing methods: {', '.join(missing_final_eval_methods)}",
        )

        evaluate_test_body = method_body(content, "evaluate_test_terminal")
        final_eval_terms = (
            "provided_rul_eval_v1",
            "capped_target_eval_v1",
            "predictions.csv",
            "metrics.json",
        )
        missing_final_eval_terms = [term for term in final_eval_terms if term not in content]
        add(
            checks,
            "final evaluation views and outputs",
            "pass" if evaluate_test_body and not missing_final_eval_terms else "fail",
            "RUL final evaluation writes predictions.csv/metrics.json and includes both views"
            if evaluate_test_body and not missing_final_eval_terms
            else f"missing final evaluation terms: {', '.join(missing_final_eval_terms)}",
        )
        forbidden_in_evaluate_test = [
            term for term in ("self.full_test_set", "self.val_data") if term in evaluate_test_body
        ]
        add(
            checks,
            "final evaluation official-test isolation",
            "pass" if evaluate_test_body and not forbidden_in_evaluate_test and "self.testing_data" in evaluate_test_body else "fail",
            "evaluate_test_terminal uses self.testing_data and does not access validation/full-test trajectory objects"
            if evaluate_test_body and not forbidden_in_evaluate_test and "self.testing_data" in evaluate_test_body
            else (
                "evaluate_test_terminal must use self.testing_data and avoid "
                f"{', '.join(forbidden_in_evaluate_test) if forbidden_in_evaluate_test else 'missing self.testing_data'}"
            ),
        )
        add(
            checks,
            "legacy RUL metric removed",
            "pass" if "ut.RUL_metric(" not in content else "fail",
            "pipelines/rul_estimation.py does not call ut.RUL_metric("
            if "ut.RUL_metric(" not in content
            else "pipelines/rul_estimation.py must not call ut.RUL_metric(",
        )
    else:
        add(checks, "RUL pipeline exists", "fail", f"missing {rul_pipeline}")

    runner = repo / "scripts" / "run_main_protocol_experiments.py"
    if runner.exists():
        content = text(runner)
        add(checks, "experiment runner exists", "pass", f"found {runner}")

        required_runner_terms = {
            "MAIN_PROTOCOL_V1 datasets": 'MAIN_PROTOCOL_V1["datasets"]',
            "MAIN_PROTOCOL_V1 model_seeds": 'MAIN_PROTOCOL_V1["training"]["model_seeds"]',
            "build_exp_id": "def build_exp_id",
            "run_single_experiment": "def run_single_experiment",
            "normalize_train_logs": "def normalize_train_logs",
            "copy_standard_checkpoint": "def copy_standard_checkpoint",
            "collect_git_info": "def collect_git_info",
            "collect_runtime_info": "def collect_runtime_info",
            "final evaluation call": "evaluate_test_terminal(output_dir=",
            "final test access record": "record_final_test_access(",
            "config.yaml": "config.yaml",
            "runtime.json": "runtime.json",
            "train_log.csv": "train_log.csv",
            "checkpoint.pt": "checkpoint.pt",
            "runner_summary.json": "runner_summary.json",
            "runner_summary.csv": "runner_summary.csv",
            "dry-run": "--dry-run",
        }
        missing_runner_terms = [name for name, term in required_runner_terms.items() if term not in content]
        add(
            checks,
            "experiment runner required hooks",
            "pass" if not missing_runner_terms else "fail",
            "runner exposes protocol grid, archive helpers, final evaluation, and summary outputs"
            if not missing_runner_terms
            else f"missing runner terms: {', '.join(missing_runner_terms)}",
        )

        forbidden_git_remote_terms = ("git pull", "git push", "git clone", '"pull"', '"push"', '"clone"')
        found_forbidden = [term for term in forbidden_git_remote_terms if term in content]
        add(
            checks,
            "experiment runner local-only git",
            "pass" if not found_forbidden else "fail",
            "runner does not contain remote git operations"
            if not found_forbidden
            else f"runner contains forbidden remote git operation terms: {', '.join(found_forbidden)}",
        )
    else:
        add(checks, "experiment runner exists", "fail", f"missing {runner}")


def audit_processed_dataset(dataset_dir: Path, dataset: str, checks: list[Check]) -> None:
    base = dataset_dir / dataset
    if not base.exists():
        add(checks, f"{dataset} processed directory", "skip", f"missing {base}")
        return

    required = {
        "manifest": base / "manifest.json",
        "split": base / f"split_seed{MAIN_PROTOCOL['split_seed']}.json",
        "train_windows": base / "train_windows.npz",
        "validation_view": base / "validation_view_v1.npz",
        "test_terminal_windows": base / "test_terminal_windows.npz",
        "full_test_windows": base / "full_test_windows.npz",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        add(checks, f"{dataset} required protocol files", "fail", f"missing: {', '.join(missing)}")
        return
    add(checks, f"{dataset} required protocol files", "pass", "all required .json/.npz files exist")

    try:
        manifest = load_json(required["manifest"])
        validate_manifest(
            manifest,
            dataset=dataset,
            window_length=MAIN_PROTOCOL["window"],
            rul_cap=MAIN_PROTOCOL["rul_cap"],
            split_seed=MAIN_PROTOCOL["split_seed"],
        )
        add(checks, f"{dataset} manifest", "pass", "manifest fields align with main_protocol_v1")
    except Exception as exc:
        add(checks, f"{dataset} manifest", "fail", str(exc))
        return

    try:
        split_info = load_json(required["split"])
        validate_split_file(split_info, dataset=dataset, split_seed=MAIN_PROTOCOL["split_seed"])
        add(
            checks,
            f"{dataset} split",
            "pass",
            f"train_units={len(split_info['train_units'])}, validation_units={len(split_info['validation_units'])}",
        )
    except Exception as exc:
        add(checks, f"{dataset} split", "fail", str(exc))
        return

    try:
        train_arrays = load_npz(required["train_windows"])
        train_n = validate_train_arrays(
            train_arrays,
            window_length=MAIN_PROTOCOL["window"],
            rul_cap=MAIN_PROTOCOL["rul_cap"],
        )
        add(checks, f"{dataset} train_windows.npz", "pass", f"N={train_n}, x={train_arrays['x'].shape}")
    except Exception as exc:
        add(checks, f"{dataset} train_windows.npz", "fail", str(exc))

    try:
        val_arrays = load_npz(required["validation_view"])
        val_n = validate_validation_arrays(
            val_arrays,
            window_length=MAIN_PROTOCOL["window"],
            rul_cap=MAIN_PROTOCOL["rul_cap"],
        )
        val_units = len(set(int(unit) for unit in val_arrays["unit"]))
        add(
            checks,
            f"{dataset} validation_view_v1.npz",
            "pass",
            f"N={val_n}, validation_units={val_units}, rows_per_unit=3",
        )
    except Exception as exc:
        add(checks, f"{dataset} validation_view_v1.npz", "fail", str(exc))

    try:
        test_arrays = load_npz(required["test_terminal_windows"])
        test_n = validate_test_terminal_arrays(
            test_arrays,
            window_length=MAIN_PROTOCOL["window"],
            rul_cap=MAIN_PROTOCOL["rul_cap"],
        )
        add(checks, f"{dataset} test_terminal_windows.npz", "pass", f"N_test_units={test_n}")
    except Exception as exc:
        add(checks, f"{dataset} test_terminal_windows.npz", "fail", str(exc))

    try:
        full_test_arrays = load_npz(required["full_test_windows"])
        full_test_n = validate_full_test_arrays(
            full_test_arrays,
            window_length=MAIN_PROTOCOL["window"],
            rul_cap=MAIN_PROTOCOL["rul_cap"],
        )
        add(checks, f"{dataset} full_test_windows.npz", "pass", f"N_full_test_windows={full_test_n}")
    except Exception as exc:
        add(checks, f"{dataset} full_test_windows.npz", "fail", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Path to OCIR-Reproduction repository")
    parser.add_argument(
        "--data-dir",
        default="./data/processed/main_protocol_v1",
        help="Processed main_protocol_v1 C-MAPSS directory",
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    data_dir = (repo / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    checks: list[Check] = []

    add(checks, "main_protocol_v1 target", "info", json.dumps(MAIN_PROTOCOL, ensure_ascii=False))
    audit_static(repo, checks)

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in targets:
        audit_processed_dataset(data_dir, dataset, checks)

    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "info": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1

    if args.json:
        print(json.dumps({"summary": counts, "checks": [asdict(c) for c in checks]}, ensure_ascii=False, indent=2))
    else:
        print("main_protocol_v1 alignment audit")
        print(f"repo: {repo}")
        print(f"data_dir: {data_dir}")
        print("summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for check in checks:
            print(f"[{check.status.upper()}] {check.name}: {check.detail}")

    return 1 if counts.get("fail", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
