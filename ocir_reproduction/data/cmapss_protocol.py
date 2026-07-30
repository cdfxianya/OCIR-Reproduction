"""Load processed C-MAPSS data that conforms to main_protocol_v1.

This module is intentionally an adapter only. It does not read raw NASA text
files, fit preprocessing transforms, generate windows, or compute metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ocir_reproduction.protocols import MAIN_PROTOCOL_V1


VALID_DATASETS = tuple(MAIN_PROTOCOL_V1["datasets"])
PROTOCOL_VERSION = MAIN_PROTOCOL_V1["version"]
SELECTED_SENSORS = tuple(MAIN_PROTOCOL_V1["features"]["selected_sensors"])
OPERATING_SETTINGS = tuple(MAIN_PROTOCOL_V1["features"]["operating_settings"])
WINDOW_LENGTH = MAIN_PROTOCOL_V1["windowing"]["window_length"]
RUL_CAP = MAIN_PROTOCOL_V1["labels"]["rul_cap"]
SPLIT_SEED = MAIN_PROTOCOL_V1["splits"]["split_seed"]
STRIDE = MAIN_PROTOCOL_V1["windowing"]["stride"]
PADDING = MAIN_PROTOCOL_V1["windowing"]["short_sequence_padding"]
OFFICIAL_TEST_USAGE = "terminal_window_only_for_main_eval"


@dataclass(frozen=True)
class WindowMetadata:
    dataset: str
    source_partition: str
    unit: int
    end_cycle: int
    index: int
    checkpoint_q: Optional[float] = None
    provided_rul: Optional[float] = None
    capped_target_rul: Optional[float] = None


class MainProtocolWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ocs: np.ndarray,
        tidx: np.ndarray,
        metadata: List[WindowMetadata],
    ) -> None:
        if len(metadata) != x.shape[0]:
            raise ValueError(f"metadata length {len(metadata)} does not match x rows {x.shape[0]}")

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.ocs = torch.as_tensor(ocs, dtype=torch.float32)
        self.tidx = torch.as_tensor(tidx, dtype=torch.int64)
        self.metadata = list(metadata)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.ocs[index], self.tidx[index]

    def get_metadata(self, index: int) -> WindowMetadata:
        return self.metadata[index]

    def all_metadata(self) -> List[WindowMetadata]:
        return list(self.metadata)


def resolve_dataset_dir(data_path: str | Path, dataset: str) -> Path:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {dataset!r}")

    dataset_dir = Path(data_path) / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"processed dataset directory does not exist: {dataset_dir}")
    return dataset_dir


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"npz file does not exist: {path}")

    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key].copy() for key in loaded.files}
    except Exception as exc:
        raise ValueError(f"failed to read npz file {path}: {exc}") from exc


def validate_manifest(
    manifest: Dict[str, Any],
    dataset: str,
    window_length: int,
    rul_cap: int,
    split_seed: int,
) -> None:
    _require_equal(manifest, "main_protocol_version", PROTOCOL_VERSION, "manifest")
    _require_equal(manifest, "dataset", dataset, "manifest")
    _require_equal(manifest, "window_length", window_length, "manifest")
    _require_equal(manifest, "window_length", WINDOW_LENGTH, "manifest")
    _require_equal(manifest, "stride", STRIDE, "manifest")
    _require_equal(manifest, "rul_cap", rul_cap, "manifest")
    _require_equal(manifest, "rul_cap", RUL_CAP, "manifest")
    _require_equal(manifest, "split_seed", split_seed, "manifest")
    _require_equal(manifest, "split_seed", SPLIT_SEED, "manifest")
    _require_equal(manifest, "selected_sensor_dim", len(SELECTED_SENSORS), "manifest")
    _require_equal(manifest, "raw_sensor_dim", MAIN_PROTOCOL_V1["features"]["raw_sensor_dim"], "manifest")
    _require_equal(manifest, "selected_sensors", list(SELECTED_SENSORS), "manifest")
    _require_equal(manifest, "operating_settings", list(OPERATING_SETTINGS), "manifest")
    _require_equal(manifest, "padding", PADDING, "manifest")
    _require_equal(manifest, "official_test_usage", OFFICIAL_TEST_USAGE, "manifest")


def validate_split_file(split_info: Dict[str, Any], dataset: str, split_seed: int) -> None:
    _require_equal(split_info, "main_protocol_version", PROTOCOL_VERSION, "split")
    _require_equal(split_info, "dataset", dataset, "split")
    _require_equal(split_info, "split_seed", split_seed, "split")
    _require_equal(split_info, "split_seed", SPLIT_SEED, "split")
    _require_equal(split_info, "split_by", MAIN_PROTOCOL_V1["splits"]["split_by"], "split")

    train_units = _require_int_list(split_info, "train_units", "split")
    validation_units = _require_int_list(split_info, "validation_units", "split")
    if not train_units:
        raise ValueError("split train_units must be non-empty")
    if not validation_units:
        raise ValueError("split validation_units must be non-empty")
    if train_units != sorted(train_units):
        raise ValueError("split train_units must be sorted ascending")
    if validation_units != sorted(validation_units):
        raise ValueError("split validation_units must be sorted ascending")
    overlap = sorted(set(train_units).intersection(validation_units))
    if overlap:
        raise ValueError(f"split train_units and validation_units overlap: {overlap}")


def require_arrays(arrays: Dict[str, np.ndarray], required: List[str], file_name: str) -> None:
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{file_name} missing required arrays: {missing}")


def validate_common_window_arrays(
    arrays: Dict[str, np.ndarray],
    file_name: str,
    window_length: int,
) -> int:
    require_arrays(arrays, ["x", "y", "ocs", "tidx", "unit", "end_cycle"], file_name)

    x = arrays["x"]
    y = arrays["y"]
    ocs = arrays["ocs"]
    tidx = arrays["tidx"]
    unit = arrays["unit"]
    end_cycle = arrays["end_cycle"]

    if x.dtype != np.float32:
        raise ValueError(f"{file_name}: x dtype must be float32, got {x.dtype}")
    if y.dtype != np.float32:
        raise ValueError(f"{file_name}: y dtype must be float32, got {y.dtype}")
    if ocs.dtype != np.float32:
        raise ValueError(f"{file_name}: ocs dtype must be float32, got {ocs.dtype}")
    if tidx.dtype != np.int64:
        raise ValueError(f"{file_name}: tidx dtype must be int64, got {tidx.dtype}")
    if unit.dtype != np.int64:
        raise ValueError(f"{file_name}: unit dtype must be int64, got {unit.dtype}")
    if end_cycle.dtype != np.int64:
        raise ValueError(f"{file_name}: end_cycle dtype must be int64, got {end_cycle.dtype}")

    if x.ndim != 3:
        raise ValueError(f"{file_name}: x must have shape [N, {window_length}, 14], got {x.shape}")
    n = x.shape[0]
    if n <= 0:
        raise ValueError(f"{file_name}: N must be > 0")
    expected_shapes = {
        "x": (n, window_length, len(SELECTED_SENSORS)),
        "y": (n, 1),
        "ocs": (n, window_length, 1),
        "tidx": (n, 1),
        "unit": (n,),
        "end_cycle": (n,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{file_name}: {key} shape must be {shape}, got {arrays[key].shape}")

    return n


def validate_train_arrays(arrays: Dict[str, np.ndarray], window_length: int, rul_cap: int) -> int:
    n = validate_common_window_arrays(arrays, "train_windows.npz", window_length)
    _validate_target_bounds(arrays["y"], rul_cap, "train_windows.npz")
    _validate_tidx_matches_end_cycle(arrays, "train_windows.npz")
    _validate_sorted_by_unit_end_cycle(arrays["unit"], arrays["end_cycle"], "train_windows.npz")
    return n


def validate_validation_arrays(arrays: Dict[str, np.ndarray], window_length: int, rul_cap: int) -> int:
    file_name = "validation_view_v1.npz"
    require_arrays(arrays, ["checkpoint_q"], file_name)
    n = validate_common_window_arrays(arrays, file_name, window_length)
    checkpoint_q = arrays["checkpoint_q"]
    if checkpoint_q.dtype != np.float32:
        raise ValueError(f"{file_name}: checkpoint_q dtype must be float32, got {checkpoint_q.dtype}")
    if checkpoint_q.shape != (n,):
        raise ValueError(f"{file_name}: checkpoint_q shape must be {(n,)}, got {checkpoint_q.shape}")

    allowed = np.array(MAIN_PROTOCOL_V1["validation_view"]["checkpoints"], dtype=np.float32)
    for value in checkpoint_q:
        if not np.any(np.isclose(value, allowed)):
            raise ValueError(f"{file_name}: checkpoint_q contains invalid value {value}")

    for unit in np.unique(arrays["unit"]):
        idx = np.where(arrays["unit"] == unit)[0]
        if idx.shape[0] != 3:
            raise ValueError(f"{file_name}: unit {int(unit)} must have exactly 3 rows, got {idx.shape[0]}")
        unit_q = arrays["checkpoint_q"][idx]
        if not _float_sets_equal(unit_q, allowed):
            raise ValueError(f"{file_name}: unit {int(unit)} checkpoint_q must be {allowed.tolist()}, got {unit_q.tolist()}")
        unit_end_cycle = arrays["end_cycle"][idx]
        if len(np.unique(unit_end_cycle)) != 3:
            raise ValueError(f"{file_name}: unit {int(unit)} end_cycle values must be unique")

    _validate_target_bounds(arrays["y"], rul_cap, file_name)
    _validate_tidx_matches_end_cycle(arrays, file_name)
    return n


def validate_test_terminal_arrays(arrays: Dict[str, np.ndarray], window_length: int, rul_cap: int) -> int:
    file_name = "test_terminal_windows.npz"
    _validate_test_target_arrays(arrays, window_length, rul_cap, file_name)

    unit = arrays["unit"]
    unique_units = np.unique(unit)
    if unique_units.shape[0] != unit.shape[0]:
        duplicated = _duplicated_values(unit)
        raise ValueError(f"{file_name}: each unit must appear exactly once; duplicates={duplicated}")
    _validate_sorted_by_unit_end_cycle(unit, arrays["end_cycle"], file_name)
    return unit.shape[0]


def validate_full_test_arrays(arrays: Dict[str, np.ndarray], window_length: int, rul_cap: int) -> int:
    file_name = "full_test_windows.npz"
    n = _validate_test_target_arrays(arrays, window_length, rul_cap, file_name)
    _validate_sorted_by_unit_end_cycle(arrays["unit"], arrays["end_cycle"], file_name)
    return n


def build_metadata(
    arrays: Dict[str, np.ndarray],
    dataset: str,
    source_partition: str,
) -> List[WindowMetadata]:
    n = arrays["unit"].shape[0]
    checkpoint_q = arrays.get("checkpoint_q")
    provided_rul = arrays.get("provided_rul")
    capped_target_rul = arrays.get("capped_target_rul")

    metadata: List[WindowMetadata] = []
    for index in range(n):
        metadata.append(
            WindowMetadata(
                dataset=dataset,
                source_partition=source_partition,
                unit=int(arrays["unit"][index]),
                end_cycle=int(arrays["end_cycle"][index]),
                index=index,
                checkpoint_q=None if checkpoint_q is None else float(checkpoint_q[index]),
                provided_rul=None if provided_rul is None else float(provided_rul[index, 0]),
                capped_target_rul=None if capped_target_rul is None else float(capped_target_rul[index, 0]),
            )
        )
    return metadata


def make_dataset(
    arrays: Dict[str, np.ndarray],
    metadata: List[WindowMetadata],
) -> MainProtocolWindowDataset:
    return MainProtocolWindowDataset(
        x=arrays["x"],
        y=arrays["y"],
        ocs=arrays["ocs"],
        tidx=arrays["tidx"],
        metadata=metadata,
    )


def get_units_from_dataset(dataset: MainProtocolWindowDataset) -> np.ndarray:
    if len(dataset) <= 0:
        raise ValueError("dataset must be non-empty")
    if not hasattr(dataset, "all_metadata"):
        raise ValueError("dataset must provide all_metadata()")

    metadata = dataset.all_metadata()
    if len(metadata) != len(dataset):
        raise ValueError(f"metadata length {len(metadata)} does not match dataset length {len(dataset)}")

    units: List[int] = []
    for index, item in enumerate(metadata):
        if not hasattr(item, "unit"):
            raise ValueError(f"metadata item {index} is missing unit")
        units.append(int(item.unit))

    return np.asarray(units, dtype=np.int64)


def compute_engine_balanced_weights(units: np.ndarray) -> torch.Tensor:
    units = np.asarray(units)
    if units.ndim != 1:
        raise ValueError(f"units must be a one-dimensional array, got shape {units.shape}")
    if units.shape[0] <= 0:
        raise ValueError("units must be non-empty")

    _, inverse, counts = np.unique(units.astype(np.int64), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    result = torch.as_tensor(weights, dtype=torch.double)
    if torch.any(result <= 0):
        raise ValueError("engine-balanced weights must all be > 0")
    return result


def summarize_engine_balanced_sampler(units: np.ndarray, weights: torch.Tensor) -> Dict[str, float]:
    units = np.asarray(units, dtype=np.int64)
    if units.ndim != 1 or units.shape[0] <= 0:
        raise ValueError("units must be a non-empty one-dimensional array")
    if weights.ndim != 1 or weights.shape[0] != units.shape[0]:
        raise ValueError(f"weights shape {tuple(weights.shape)} must match units shape {units.shape}")

    unique_units, counts = np.unique(units, return_counts=True)
    weights_np = weights.detach().cpu().numpy()
    unit_total_weights = np.asarray([weights_np[units == unit].sum() for unit in unique_units], dtype=np.float64)
    return {
        "num_windows": int(units.shape[0]),
        "num_units": int(unique_units.shape[0]),
        "min_windows_per_unit": int(counts.min()),
        "max_windows_per_unit": int(counts.max()),
        "min_unit_total_weight": float(unit_total_weights.min()),
        "max_unit_total_weight": float(unit_total_weights.max()),
    }


class EngineBalancedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: MainProtocolWindowDataset,
        model_seed: int,
        num_samples: Optional[int] = None,
    ) -> None:
        if isinstance(model_seed, bool) or not isinstance(model_seed, int):
            raise ValueError("model_seed must be an integer")
        if len(dataset) <= 0:
            raise ValueError("dataset must be non-empty")
        if num_samples is None:
            num_samples = len(dataset)
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        self.dataset = dataset
        self.model_seed = int(model_seed)
        self.num_samples = int(num_samples)
        self.epoch = 0
        self.replacement = True
        self.units = get_units_from_dataset(dataset)
        self.weights = compute_engine_balanced_weights(self.units)
        self.summary = summarize_engine_balanced_sampler(self.units, self.weights)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.model_seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=generator,
        )
        return iter(int(index) for index in indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def make_training_loader(
    dataset: MainProtocolWindowDataset,
    batch_size: int,
    model_seed: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    sampler = EngineBalancedSampler(dataset=dataset, model_seed=model_seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False)


def make_eval_loader(dataset: MainProtocolWindowDataset, batch_size: int) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def build_full_test_set(arrays: Dict[str, np.ndarray]) -> Dict[int, Dict[str, torch.Tensor]]:
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for unit in np.unique(arrays["unit"]):
        idx = np.where(arrays["unit"] == unit)[0]
        order = np.argsort(arrays["end_cycle"][idx], kind="stable")
        idx = idx[order]
        result[int(unit)] = {
            "X": torch.as_tensor(arrays["x"][idx], dtype=torch.float32),
            "Y": torch.as_tensor(arrays["y"][idx], dtype=torch.float32),
            "ocs": torch.as_tensor(arrays["ocs"][idx], dtype=torch.float32),
            "tidx": torch.as_tensor(arrays["tidx"][idx], dtype=torch.int64),
        }
    return result


def log_dataset_summary(
    logger: Any,
    dataset: str,
    train_n: int,
    val_n: int,
    test_n: int,
    full_test_n: int,
    sampler_summary: Optional[Dict[str, float]] = None,
) -> None:
    if logger is None:
        return
    logger.info(
        "Loaded main_protocol_v1 %s: train_windows=%d, validation_windows=%d, "
        "test_terminal_windows=%d, full_test_windows=%d",
        dataset,
        train_n,
        val_n,
        test_n,
        full_test_n,
    )
    if sampler_summary is not None:
        logger.info(
            "Train sampler engine_balanced_sampler_v1: units=%d, windows_per_unit=[%d, %d], "
            "unit_total_weight=[%.6f, %.6f]",
            int(sampler_summary["num_units"]),
            int(sampler_summary["min_windows_per_unit"]),
            int(sampler_summary["max_windows_per_unit"]),
            float(sampler_summary["min_unit_total_weight"]),
            float(sampler_summary["max_unit_total_weight"]),
        )


def load_main_protocol_cmapss(
    dataset: str,
    data_path: str,
    window_length: int,
    rul_cap: int,
    batch_size: int,
    split_seed: int,
    model_seed: int,
    logger: Any = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, Dict[str, torch.Tensor]]]:
    dataset_dir = resolve_dataset_dir(data_path, dataset)

    manifest = load_json(dataset_dir / "manifest.json")
    split_info = load_json(dataset_dir / "split_seed2026.json")

    validate_manifest(manifest, dataset, window_length, rul_cap, split_seed)
    validate_split_file(split_info, dataset, split_seed)

    train_arrays = load_npz(dataset_dir / "train_windows.npz")
    val_arrays = load_npz(dataset_dir / "validation_view_v1.npz")
    test_arrays = load_npz(dataset_dir / "test_terminal_windows.npz")
    full_test_arrays = load_npz(dataset_dir / "full_test_windows.npz")

    train_n = validate_train_arrays(train_arrays, window_length, rul_cap)
    val_n = validate_validation_arrays(val_arrays, window_length, rul_cap)
    test_n = validate_test_terminal_arrays(test_arrays, window_length, rul_cap)
    full_test_n = validate_full_test_arrays(full_test_arrays, window_length, rul_cap)

    train_dataset = make_dataset(train_arrays, build_metadata(train_arrays, dataset, "train"))
    val_dataset = make_dataset(val_arrays, build_metadata(val_arrays, dataset, "validation"))
    test_dataset = make_dataset(test_arrays, build_metadata(test_arrays, dataset, "test"))

    training_data = make_training_loader(train_dataset, batch_size, model_seed)
    val_data = make_eval_loader(val_dataset, batch_size)
    testing_data = make_eval_loader(test_dataset, batch_size)
    full_test_set = build_full_test_set(full_test_arrays)

    sampler_summary = getattr(training_data.sampler, "summary", None)
    log_dataset_summary(logger, dataset, train_n, val_n, test_n, full_test_n, sampler_summary=sampler_summary)
    return training_data, val_data, testing_data, full_test_set


def _require_equal(mapping: Dict[str, Any], key: str, expected: Any, label: str) -> None:
    if key not in mapping:
        raise ValueError(f"{label} missing required field {key!r}")
    actual = mapping[key]
    if actual != expected:
        raise ValueError(f"{label} field {key!r} must be {expected!r}, got {actual!r}")


def _require_int_list(mapping: Dict[str, Any], key: str, label: str) -> List[int]:
    if key not in mapping:
        raise ValueError(f"{label} missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{label} field {key!r} must be a list")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"{label} field {key!r} must contain only integers")
    return value


def _validate_target_bounds(y: np.ndarray, rul_cap: int, file_name: str) -> None:
    if np.any(y < 0) or np.any(y > rul_cap):
        raise ValueError(f"{file_name}: y must satisfy 0 <= y <= {rul_cap}")


def _validate_tidx_matches_end_cycle(arrays: Dict[str, np.ndarray], file_name: str) -> None:
    if not np.array_equal(arrays["tidx"][:, 0], arrays["end_cycle"]):
        raise ValueError(f"{file_name}: tidx[:, 0] must equal end_cycle")


def _validate_sorted_by_unit_end_cycle(unit: np.ndarray, end_cycle: np.ndarray, file_name: str) -> None:
    order = np.lexsort((end_cycle, unit))
    if not np.array_equal(order, np.arange(unit.shape[0])):
        raise ValueError(f"{file_name}: rows must be sorted by unit ascending, then end_cycle ascending")


def _float_sets_equal(values: np.ndarray, expected: np.ndarray) -> bool:
    values_sorted = np.sort(values.astype(np.float32))
    expected_sorted = np.sort(expected.astype(np.float32))
    return values_sorted.shape == expected_sorted.shape and np.allclose(values_sorted, expected_sorted)


def _duplicated_values(values: np.ndarray) -> List[int]:
    unique, counts = np.unique(values, return_counts=True)
    return [int(value) for value, count in zip(unique, counts) if count > 1]


def _validate_test_target_arrays(
    arrays: Dict[str, np.ndarray],
    window_length: int,
    rul_cap: int,
    file_name: str,
) -> int:
    require_arrays(arrays, ["provided_rul", "capped_target_rul"], file_name)
    n = validate_common_window_arrays(arrays, file_name, window_length)

    provided_rul = arrays["provided_rul"]
    capped_target_rul = arrays["capped_target_rul"]
    if provided_rul.dtype != np.float32:
        raise ValueError(f"{file_name}: provided_rul dtype must be float32, got {provided_rul.dtype}")
    if capped_target_rul.dtype != np.float32:
        raise ValueError(f"{file_name}: capped_target_rul dtype must be float32, got {capped_target_rul.dtype}")
    if provided_rul.shape != (n, 1):
        raise ValueError(f"{file_name}: provided_rul shape must be {(n, 1)}, got {provided_rul.shape}")
    if capped_target_rul.shape != (n, 1):
        raise ValueError(f"{file_name}: capped_target_rul shape must be {(n, 1)}, got {capped_target_rul.shape}")
    if not np.allclose(arrays["y"], capped_target_rul):
        raise ValueError(f"{file_name}: y must equal capped_target_rul")
    if not np.allclose(capped_target_rul, np.minimum(provided_rul, float(rul_cap))):
        raise ValueError(f"{file_name}: capped_target_rul must equal minimum(provided_rul, {rul_cap})")

    _validate_tidx_matches_end_cycle(arrays, file_name)
    return n
