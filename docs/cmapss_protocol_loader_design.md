# cmapss_protocol.py 数据加载器技术设计

本文档定义 `ocir_reproduction.data.cmapss_protocol` 的实现范围、函数框架、调用格式和返回值。该模块只负责读取已经按 `main_protocol_v1` 处理好的数据包，并转换为 OCIR 主过程需要的 PyTorch 数据对象。

本文档不定义 raw C-MAPSS 的清洗、标准化、KMeans、窗口生成过程；这些由数据处理交付方完成，交付格式见 `docs/main_protocol_v1_data_contract.md`。

## 1. 文件位置

必须实现以下文件：

```text
ocir_reproduction/data/__init__.py
ocir_reproduction/data/cmapss_protocol.py
```

`pipelines/solver_base.py` 将通过以下方式调用：

```python
from ocir_reproduction.data.cmapss_protocol import load_main_protocol_cmapss
```

## 2. 模块职责

`cmapss_protocol.py` 必须负责：

```text
定位数据集目录
读取 manifest.json
读取 split_seed2026.json
读取 train_windows.npz
读取 validation_view_v1.npz
读取 test_terminal_windows.npz
读取 full_test_windows.npz
校验文件字段、dtype、shape、协议常量
构建 PyTorch Dataset
构建 PyTorch DataLoader
构建兼容旧 RUL 可视化接口的 full_test_set
返回 training_data, val_data, testing_data, full_test_set
```

`cmapss_protocol.py` 禁止负责：

```text
读取 NASA raw txt
计算 SHA256
fit scaler
fit KMeans
生成窗口
训练模型
计算 official test metrics
写 predictions.csv
写 metrics.json
决定 official test 是否开封
```

## 3. 主入口函数

### 3.1 `load_main_protocol_cmapss`

函数签名：

```python
def load_main_protocol_cmapss(
    dataset: str,
    data_path: str,
    window_length: int,
    rul_cap: int,
    batch_size: int,
    split_seed: int,
    model_seed: int,
    logger=None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[int, dict[str, torch.Tensor]]]:
    ...
```

调用示例：

```python
training_data, val_data, testing_data, full_test_set = load_main_protocol_cmapss(
    dataset="FD001",
    data_path="data/processed/main_protocol_v1",
    window_length=30,
    rul_cap=125,
    batch_size=128,
    split_seed=2026,
    model_seed=1,
    logger=logger,
)
```

输入参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `dataset` | `str` | 必须是 `FD001`, `FD002`, `FD003`, `FD004` 之一 |
| `data_path` | `str` | 指向 `data/processed/main_protocol_v1` |
| `window_length` | `int` | 必须等于 `30` |
| `rul_cap` | `int` | 必须等于 `125` |
| `batch_size` | `int` | 必须大于 0 |
| `split_seed` | `int` | 必须等于 `2026` |
| `model_seed` | `int` | 用于训练 DataLoader 的随机生成器 |
| `logger` | 可选 | 若非空，写入加载摘要 |

返回值：

```python
(
    training_data,
    val_data,
    testing_data,
    full_test_set,
)
```

返回对象定义：

| 返回对象 | 类型 | 用途 |
|---|---|---|
| `training_data` | `torch.utils.data.DataLoader` | OCIR 表征学习和 RUL 训练 |
| `val_data` | `torch.utils.data.DataLoader` | internal validation，只含 `validation_view_v1` 样本 |
| `testing_data` | `torch.utils.data.DataLoader` | official test terminal windows，只能最终评估使用 |
| `full_test_set` | `dict[int, dict[str, torch.Tensor]]` | 兼容旧 OCIR 可视化接口，不用于主评估 |

主流程必须为：

```text
1. resolve_dataset_dir
2. load manifest.json
3. load split_seed2026.json
4. validate manifest
5. validate split file
6. load four npz files
7. validate four npz schemas
8. build metadata
9. build datasets
10. build dataloaders
11. build full_test_set
12. return four objects
```

## 4. 数据样本元信息

### 4.1 `WindowMetadata`

定义：

```python
@dataclass(frozen=True)
class WindowMetadata:
    dataset: str
    source_partition: str
    unit: int
    end_cycle: int
    index: int
    checkpoint_q: float | None = None
    provided_rul: float | None = None
    capped_target_rul: float | None = None
```

用途：

```text
保存窗口级元信息
训练时不进入模型
后续导出 predictions.csv 时使用
```

字段定义：

| 字段 | 类型 | 含义 |
|---|---|---|
| `dataset` | `str` | `FD001` 到 `FD004` |
| `source_partition` | `str` | `train`, `validation`, `test`, `test_full_windows` |
| `unit` | `int` | C-MAPSS 文件内 unit ID |
| `end_cycle` | `int` | 窗口最后一行对应的原始 cycle |
| `index` | `int` | 该样本在对应 `.npz` 文件中的行号 |
| `checkpoint_q` | `float | None` | validation checkpoint，取 `0.5`, `0.7`, `0.9` |
| `provided_rul` | `float | None` | official test provided RUL |
| `capped_target_rul` | `float | None` | `min(provided_rul, 125)` |

## 5. Dataset 类

### 5.1 `MainProtocolWindowDataset.__init__`

函数签名：

```python
class MainProtocolWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ocs: np.ndarray,
        tidx: np.ndarray,
        metadata: list[WindowMetadata],
    ) -> None:
        ...
```

输入要求：

```text
x.shape = [N, 30, 14], dtype float32
y.shape = [N, 1], dtype float32
ocs.shape = [N, 30, 1], dtype float32
tidx.shape = [N, 1], dtype int64
len(metadata) = N
```

目的：

```text
把协议化 npz 数组包装成 PyTorch Dataset
```

返回值：

```text
None
```

### 5.2 `MainProtocolWindowDataset.__len__`

函数签名：

```python
def __len__(self) -> int:
    ...
```

返回值：

```text
样本数 N
```

### 5.3 `MainProtocolWindowDataset.__getitem__`

函数签名：

```python
def __getitem__(
    self,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

返回值：

```text
x:    torch.float32 [30, 14]
y:    torch.float32 [1]
ocs:  torch.float32 [30, 1]
tidx: torch.int64   [1]
```

DataLoader batch 后必须变成：

```text
x:    torch.float32 [B, 30, 14]
y:    torch.float32 [B, 1]
ocs:  torch.float32 [B, 30, 1]
tidx: torch.int64   [B, 1]
```

### 5.4 `MainProtocolWindowDataset.get_metadata`

函数签名：

```python
def get_metadata(self, index: int) -> WindowMetadata:
    ...
```

目的：

```text
按样本 index 读取窗口元信息
```

返回值：

```text
WindowMetadata
```

### 5.5 `MainProtocolWindowDataset.all_metadata`

函数签名：

```python
def all_metadata(self) -> list[WindowMetadata]:
    ...
```

目的：

```text
返回所有窗口元信息
```

返回值：

```text
list[WindowMetadata]
```

## 6. 路径与文件读取函数

### 6.1 `resolve_dataset_dir`

函数签名：

```python
def resolve_dataset_dir(
    data_path: str | Path,
    dataset: str,
) -> Path:
    ...
```

调用示例：

```python
dataset_dir = resolve_dataset_dir("data/processed/main_protocol_v1", "FD001")
```

目的：

```text
检查 dataset 名称合法
返回 data_path / dataset
检查该目录存在
```

返回值：

```text
Path
```

异常：

```text
ValueError: dataset 不在 FD001-FD004 中
FileNotFoundError: dataset 目录不存在
```

### 6.2 `load_json`

函数签名：

```python
def load_json(path: Path) -> dict[str, Any]:
    ...
```

调用示例：

```python
manifest = load_json(dataset_dir / "manifest.json")
```

目的：

```text
读取 UTF-8 JSON 文件
确保顶层对象是 dict
```

返回值：

```text
dict[str, Any]
```

异常：

```text
FileNotFoundError: JSON 文件不存在
ValueError: JSON 顶层不是 object
json.JSONDecodeError: JSON 格式错误
```

### 6.3 `load_npz`

函数签名：

```python
def load_npz(path: Path) -> dict[str, np.ndarray]:
    ...
```

调用示例：

```python
arrays = load_npz(dataset_dir / "train_windows.npz")
```

目的：

```text
读取 npz 文件
立即复制为普通 dict
关闭 NpzFile 句柄
```

返回值：

```text
dict[str, np.ndarray]
```

异常：

```text
FileNotFoundError: npz 文件不存在
ValueError: 文件无法读取为 npz
```

## 7. 协议与 manifest 校验函数

### 7.1 `validate_manifest`

函数签名：

```python
def validate_manifest(
    manifest: dict[str, Any],
    dataset: str,
    window_length: int,
    rul_cap: int,
    split_seed: int,
) -> None:
    ...
```

目的：

```text
检查 manifest.json 与 main_protocol_v1 和调用参数一致
```

必须检查：

```text
manifest["main_protocol_version"] == "main_protocol_v1"
manifest["dataset"] == dataset
manifest["window_length"] == window_length == 30
manifest["stride"] == 1
manifest["rul_cap"] == rul_cap == 125
manifest["split_seed"] == split_seed == 2026
manifest["selected_sensor_dim"] == 14
manifest["selected_sensors"] 顺序严格等于协议 classic 14 sensors
manifest["raw_sensor_dim"] == 21
manifest["operating_settings"] == ["op1", "op2", "op3"]
manifest["padding"] == "left_repeat_first"
manifest["official_test_usage"] == "terminal_window_only_for_main_eval"
```

返回值：

```text
None
```

异常：

```text
ValueError: 任一字段缺失或不一致
```

### 7.2 `validate_split_file`

函数签名：

```python
def validate_split_file(
    split_info: dict[str, Any],
    dataset: str,
    split_seed: int,
) -> None:
    ...
```

目的：

```text
检查 split_seed2026.json 是否满足 engine-level split 约束
```

必须检查：

```text
split_info["main_protocol_version"] == "main_protocol_v1"
split_info["dataset"] == dataset
split_info["split_seed"] == split_seed == 2026
split_info["split_by"] == "engine_unit"
train_units 存在且非空
validation_units 存在且非空
train_units 与 validation_units 无交集
train_units 升序
validation_units 升序
```

返回值：

```text
None
```

异常：

```text
ValueError
```

## 8. `.npz` 通用校验函数

### 8.1 `require_arrays`

函数签名：

```python
def require_arrays(
    arrays: dict[str, np.ndarray],
    required: list[str],
    file_name: str,
) -> None:
    ...
```

目的：

```text
检查 npz 中是否包含所有必须数组
```

返回值：

```text
None
```

异常：

```text
ValueError: 缺少字段
```

### 8.2 `validate_common_window_arrays`

函数签名：

```python
def validate_common_window_arrays(
    arrays: dict[str, np.ndarray],
    file_name: str,
    window_length: int,
) -> int:
    ...
```

目的：

```text
检查所有窗口 npz 共有字段的 dtype 和 shape
```

必须检查：

```text
x 存在，dtype float32，shape [N, window_length, 14]
y 存在，dtype float32，shape [N, 1]
ocs 存在，dtype float32，shape [N, window_length, 1]
tidx 存在，dtype int64，shape [N, 1]
unit 存在，dtype int64，shape [N]
end_cycle 存在，dtype int64，shape [N]
N > 0
```

返回值：

```text
N: int
```

异常：

```text
ValueError
```

## 9. 各文件专用校验函数

### 9.1 `validate_train_arrays`

函数签名：

```python
def validate_train_arrays(
    arrays: dict[str, np.ndarray],
    window_length: int,
    rul_cap: int,
) -> int:
    ...
```

目的：

```text
检查 train_windows.npz
```

必须字段：

```text
x, y, ocs, tidx, unit, end_cycle
```

额外检查：

```text
0 <= y <= rul_cap
unit/end_cycle 按 unit 升序、同 unit 内 end_cycle 升序
tidx[:, 0] == end_cycle
```

返回值：

```text
N_train_windows: int
```

### 9.2 `validate_validation_arrays`

函数签名：

```python
def validate_validation_arrays(
    arrays: dict[str, np.ndarray],
    window_length: int,
    rul_cap: int,
) -> int:
    ...
```

目的：

```text
检查 validation_view_v1.npz
```

必须字段：

```text
x, y, ocs, tidx, unit, end_cycle, checkpoint_q
```

额外检查：

```text
checkpoint_q dtype float32
checkpoint_q.shape == [N]
checkpoint_q 只能包含 0.5, 0.7, 0.9
每个 unit 恰好 3 行
同一 unit 的 checkpoint_q 集合严格等于 {0.5, 0.7, 0.9}
同一 unit 的 end_cycle 互不重复
0 <= y <= rul_cap
tidx[:, 0] == end_cycle
```

返回值：

```text
N_validation_windows: int
```

### 9.3 `validate_test_terminal_arrays`

函数签名：

```python
def validate_test_terminal_arrays(
    arrays: dict[str, np.ndarray],
    window_length: int,
    rul_cap: int,
) -> int:
    ...
```

目的：

```text
检查 test_terminal_windows.npz
```

必须字段：

```text
x, y, ocs, tidx, unit, end_cycle, provided_rul, capped_target_rul
```

额外检查：

```text
provided_rul dtype float32，shape [N, 1]
capped_target_rul dtype float32，shape [N, 1]
y == capped_target_rul
capped_target_rul == minimum(provided_rul, rul_cap)
每个 unit 恰好 1 行
tidx[:, 0] == end_cycle
```

返回值：

```text
N_test_units: int
```

### 9.4 `validate_full_test_arrays`

函数签名：

```python
def validate_full_test_arrays(
    arrays: dict[str, np.ndarray],
    window_length: int,
    rul_cap: int,
) -> int:
    ...
```

目的：

```text
检查 full_test_windows.npz
```

必须字段：

```text
x, y, ocs, tidx, unit, end_cycle, provided_rul, capped_target_rul
```

额外检查：

```text
provided_rul dtype float32，shape [N, 1]
capped_target_rul dtype float32，shape [N, 1]
y == capped_target_rul
capped_target_rul == minimum(provided_rul, rul_cap)
同一 unit 内 end_cycle 升序
tidx[:, 0] == end_cycle
```

返回值：

```text
N_full_test_windows: int
```

## 10. metadata 构建函数

### 10.1 `build_metadata`

函数签名：

```python
def build_metadata(
    arrays: dict[str, np.ndarray],
    dataset: str,
    source_partition: str,
) -> list[WindowMetadata]:
    ...
```

目的：

```text
根据 npz 数组构建每个窗口的 WindowMetadata
```

调用示例：

```python
metadata = build_metadata(train_arrays, "FD001", "train")
```

字段来源：

```text
unit 来自 arrays["unit"]
end_cycle 来自 arrays["end_cycle"]
index 为行号
checkpoint_q 若 arrays 中存在则读取，否则 None
provided_rul 若 arrays 中存在则读取，否则 None
capped_target_rul 若 arrays 中存在则读取，否则 None
```

返回值：

```text
list[WindowMetadata]
```

## 11. Dataset 与 DataLoader 构建函数

### 11.1 `make_dataset`

函数签名：

```python
def make_dataset(
    arrays: dict[str, np.ndarray],
    metadata: list[WindowMetadata],
) -> MainProtocolWindowDataset:
    ...
```

目的：

```text
把已校验 npz 数组和 metadata 包装成 Dataset
```

返回值：

```text
MainProtocolWindowDataset
```

### 11.2 `make_training_loader`

函数签名：

```python
def make_training_loader(
    dataset: MainProtocolWindowDataset,
    batch_size: int,
    model_seed: int,
) -> DataLoader:
    ...
```

目的：

```text
构建 training_data
```

第一版行为：

```text
shuffle = True
drop_last = False
torch.Generator 使用 model_seed
```

返回值：

```text
torch.utils.data.DataLoader
```

备注：

```text
严格 engine-balanced sampler 后续可在此函数内部替换实现，不改变外部接口。
```

### 11.3 `make_eval_loader`

函数签名：

```python
def make_eval_loader(
    dataset: MainProtocolWindowDataset,
    batch_size: int,
) -> DataLoader:
    ...
```

目的：

```text
构建 val_data 或 testing_data
```

行为：

```text
shuffle = False
drop_last = False
```

返回值：

```text
torch.utils.data.DataLoader
```

## 12. full_test_set 构建函数

### 12.1 `build_full_test_set`

函数签名：

```python
def build_full_test_set(
    arrays: dict[str, np.ndarray],
) -> dict[int, dict[str, torch.Tensor]]:
    ...
```

目的：

```text
构建旧 RUL pipeline 当前需要的 full_test_set 结构
```

返回值格式：

```python
{
    unit_id: {
        "X": torch.float32 [N_unit_windows, 30, 14],
        "Y": torch.float32 [N_unit_windows, 1],
        "ocs": torch.float32 [N_unit_windows, 30, 1],
        "tidx": torch.int64 [N_unit_windows, 1],
    },
    ...
}
```

要求：

```text
key 为 int unit_id
按 unit 分组
同一 unit 内按 end_cycle 升序
```

## 13. 日志函数

### 13.1 `log_dataset_summary`

函数签名：

```python
def log_dataset_summary(
    logger,
    dataset: str,
    train_n: int,
    val_n: int,
    test_n: int,
    full_test_n: int,
) -> None:
    ...
```

目的：

```text
如果 logger 不为 None，记录数据加载摘要
```

返回值：

```text
None
```

日志内容至少包括：

```text
dataset
train window count
validation window count
test terminal unit/window count
full test window count
```

## 14. 完整调用流程伪代码

`load_main_protocol_cmapss` 必须按以下顺序执行：

```python
def load_main_protocol_cmapss(...):
    dataset_dir = resolve_dataset_dir(data_path, dataset)

    manifest = load_json(dataset_dir / "manifest.json")
    split_info = load_json(dataset_dir / "split_seed2026.json")

    validate_manifest(
        manifest=manifest,
        dataset=dataset,
        window_length=window_length,
        rul_cap=rul_cap,
        split_seed=split_seed,
    )
    validate_split_file(
        split_info=split_info,
        dataset=dataset,
        split_seed=split_seed,
    )

    train_arrays = load_npz(dataset_dir / "train_windows.npz")
    val_arrays = load_npz(dataset_dir / "validation_view_v1.npz")
    test_arrays = load_npz(dataset_dir / "test_terminal_windows.npz")
    full_test_arrays = load_npz(dataset_dir / "full_test_windows.npz")

    train_n = validate_train_arrays(train_arrays, window_length, rul_cap)
    val_n = validate_validation_arrays(val_arrays, window_length, rul_cap)
    test_n = validate_test_terminal_arrays(test_arrays, window_length, rul_cap)
    full_test_n = validate_full_test_arrays(full_test_arrays, window_length, rul_cap)

    train_dataset = make_dataset(
        train_arrays,
        build_metadata(train_arrays, dataset, "train"),
    )
    val_dataset = make_dataset(
        val_arrays,
        build_metadata(val_arrays, dataset, "validation"),
    )
    test_dataset = make_dataset(
        test_arrays,
        build_metadata(test_arrays, dataset, "test"),
    )

    training_data = make_training_loader(train_dataset, batch_size, model_seed)
    val_data = make_eval_loader(val_dataset, batch_size)
    testing_data = make_eval_loader(test_dataset, batch_size)
    full_test_set = build_full_test_set(full_test_arrays)

    log_dataset_summary(logger, dataset, train_n, val_n, test_n, full_test_n)

    return training_data, val_data, testing_data, full_test_set
```

## 15. 最小验收测试

实现完成后，以下代码必须通过：

```python
training_data, val_data, testing_data, full_test_set = load_main_protocol_cmapss(
    dataset="FD001",
    data_path="data/processed/main_protocol_v1",
    window_length=30,
    rul_cap=125,
    batch_size=8,
    split_seed=2026,
    model_seed=1,
)

x, y, ocs, tidx = next(iter(training_data))

assert x.dtype == torch.float32
assert y.dtype == torch.float32
assert ocs.dtype == torch.float32
assert tidx.dtype == torch.int64

assert x.shape[1:] == (30, 14)
assert y.shape[1:] == (1,)
assert ocs.shape[1:] == (30, 1)
assert tidx.shape[1:] == (1,)

assert isinstance(full_test_set, dict)
```

## 16. 与 OCIR 主过程的接口边界

OCIR 表征学习阶段读取：

```python
for x, y, ocs, tidx in training_data:
    loss_R, R = ocir.L_R(x, tidx)
```

RUL 训练阶段读取：

```python
for x, y, ocs, tidx in training_data:
    loss = rul_predictor.Loss_RUL(y, x, tidx)
```

validation/test 读取：

```python
for x, y, ocs, tidx in val_data:
    pred = rul_predictor(x, tidx)
```

因此本模块必须保证：

```text
DataLoader 每个 batch 只返回四个对象
四个对象顺序固定为 x, y, ocs, tidx
不得额外返回 metadata
metadata 只能通过 Dataset 方法读取
```

