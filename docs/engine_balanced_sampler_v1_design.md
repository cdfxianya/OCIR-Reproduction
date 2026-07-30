# engine_balanced_sampler_v1 技术文档

## 1. 实现目标

本文档规定 `main_protocol_v1` 中 `train_sampling_v1` 的模型侧实现。

协议目标：

```text
训练 DataLoader 每个 epoch 抽取 len(train_dataset) 个窗口样本。
抽样有放回。
每台 train engine 在期望上具有相同总采样概率。
同一 engine 内部的窗口等概率。
每个 epoch 的采样随机状态由 model_seed + epoch 派生。
```

本实现不改变 processed 数据格式。

本实现不修改：

```text
docs/main_protocol_v1_data_contract.md
train_windows.npz
validation_view_v1.npz
test_terminal_windows.npz
full_test_windows.npz
```

## 2. 当前问题

当前 `make_training_loader()` 使用：

```python
DataLoader(..., shuffle=True, generator=generator)
```

该行为等价于：

```text
所有 train window 等概率抽样或洗牌。
```

如果某台 engine 的窗口数更多，它在一个 epoch 中出现的次数也更多。

设：

```text
N_u = engine u 的窗口数
```

则当前 uniform-window 训练使 engine `u` 的训练影响大致与 `N_u` 成正比。

该行为不满足 `02_cmapss_data_source_and_alignment(1).md` 中的：

```text
engine 在期望上等权
```

## 3. 目标采样规则

对每个 train window `i`，其所属 engine 为：

```text
unit_i
```

对每台 engine `u`：

```text
N_u = count(unit_i == u)
```

每个属于 engine `u` 的窗口权重为：

```text
w_i = 1 / N_u
```

因此：

```text
sum(w_i for unit_i == u) = 1
```

所有 engine 的总权重相等。

每个 epoch 的样本数固定为：

```text
num_samples = len(train_dataset)
```

采样方式：

```text
replacement = True
```

第 `epoch` 个 epoch 的随机种子：

```text
sampler_seed = model_seed + epoch
```

其中：

```text
epoch 从 0 开始计数
```

## 4. 文件更改

需要修改：

```text
ocir_reproduction/data/cmapss_protocol.py
pipelines/solver_base.py
pipelines/representtion_learning.py
pipelines/rul_estimation.py
scripts/audit_main_protocol_alignment.py
ocir_protocol_alignment_checklist.md
```

不修改：

```text
docs/main_protocol_v1_data_contract.md
```

## 5. 新增类

### 5.1 `EngineBalancedSampler`

新增位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

函数框架：

```python
class EngineBalancedSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: MainProtocolWindowDataset,
        model_seed: int,
        num_samples: int | None = None,
    ) -> None:
        ...

    def set_epoch(self, epoch: int) -> None:
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...
```

构造调用：

```python
sampler = EngineBalancedSampler(
    dataset=train_dataset,
    model_seed=model_seed,
)
```

返回对象：

```text
EngineBalancedSampler
```

目的：

```text
为 train DataLoader 提供 engine-balanced、with-replacement、epoch-aware 的样本 index 生成器。
```

初始化行为：

```text
1. 从 dataset.all_metadata() 读取每条样本的 unit。
2. 统计每个 unit 的窗口数 N_u。
3. 计算每条样本的权重 w_i = 1 / N_u。
4. 保存 model_seed。
5. 默认 epoch = 0。
6. 如果 num_samples is None，则 num_samples = len(dataset)。
```

参数约束：

```text
dataset 必须是 MainProtocolWindowDataset 或提供等价 all_metadata() 接口。
len(dataset) 必须大于 0。
metadata 长度必须等于 len(dataset)。
model_seed 必须是 int，bool 不允许。
num_samples 若提供，必须是正整数。
```

异常：

```text
ValueError: dataset 为空、metadata 缺失、metadata 长度不一致、unit 缺失、num_samples 非正。
TypeError 或 ValueError: model_seed 非 int。
```

### 5.2 `EngineBalancedSampler.set_epoch`

函数框架：

```python
def set_epoch(self, epoch: int) -> None:
    ...
```

调用格式：

```python
sampler.set_epoch(epoch)
```

返回值：

```text
None
```

目的：

```text
设置当前 epoch，使 __iter__ 使用 model_seed + epoch 生成该 epoch 的采样序列。
```

参数约束：

```text
epoch 必须是非负整数。
bool 不允许。
```

异常：

```text
ValueError: epoch 不是非负整数。
```

### 5.3 `EngineBalancedSampler.__iter__`

函数框架：

```python
def __iter__(self):
    ...
```

调用格式：

```python
for index in sampler:
    ...
```

返回值：

```text
Iterator[int]
```

目的：

```text
生成当前 epoch 的训练样本 index 序列。
```

行为要求：

```text
1. 创建新的 torch.Generator。
2. generator.manual_seed(model_seed + current_epoch)。
3. 使用 torch.multinomial(weights, num_samples, replacement=True, generator=generator)。
4. 返回 Python int index iterator。
```

确定性要求：

```text
相同 dataset、model_seed、epoch -> index 序列完全一致。
相同 dataset、model_seed、不同 epoch -> index 序列应通常不同。
```

### 5.4 `EngineBalancedSampler.__len__`

函数框架：

```python
def __len__(self) -> int:
    ...
```

调用格式：

```python
len(sampler)
```

返回值：

```text
int
```

目的：

```text
返回每个 epoch 抽样的样本数。
```

行为要求：

```text
返回 num_samples。
默认等于 len(train_dataset)。
```

## 6. 新增函数

### 6.1 `get_units_from_dataset`

新增位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

函数框架：

```python
def get_units_from_dataset(dataset: MainProtocolWindowDataset) -> np.ndarray:
    ...
```

调用格式：

```python
units = get_units_from_dataset(train_dataset)
```

返回值：

```text
np.ndarray
shape: [N]
dtype: int64
```

目的：

```text
从 dataset metadata 中提取每个 window 所属的 engine unit。
```

行为要求：

```text
返回顺序必须与 dataset index 顺序一致。
metadata 长度必须等于 len(dataset)。
```

### 6.2 `compute_engine_balanced_weights`

新增位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

函数框架：

```python
def compute_engine_balanced_weights(units: np.ndarray) -> torch.DoubleTensor:
    ...
```

调用格式：

```python
weights = compute_engine_balanced_weights(units)
```

返回值：

```text
torch.DoubleTensor
shape: [N]
```

目的：

```text
根据每条样本所属 unit 计算 Weighted sampling 权重。
```

权重定义：

```text
weights[i] = 1 / count(units == units[i])
```

行为要求：

```text
units 必须是一维非空数组。
weights 中每个值必须大于 0。
同一 unit 的 total weight 必须等于 1。
```

### 6.3 `summarize_engine_balanced_sampler`

新增位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

函数框架：

```python
def summarize_engine_balanced_sampler(
    units: np.ndarray,
    weights: torch.Tensor,
) -> dict[str, int | float]:
    ...
```

调用格式：

```python
summary = summarize_engine_balanced_sampler(units, weights)
```

返回值：

```python
{
    "num_windows": int,
    "num_units": int,
    "min_windows_per_unit": int,
    "max_windows_per_unit": int,
    "min_unit_total_weight": float,
    "max_unit_total_weight": float,
}
```

目的：

```text
记录 sampler 的可审计摘要，证明 engine-level 总权重一致。
```

### 6.4 `set_training_sampler_epoch`

新增位置：

```text
pipelines/solver_base.py
```

函数框架：

```python
def set_training_sampler_epoch(self, epoch: int) -> None:
    ...
```

调用格式：

```python
self.set_training_sampler_epoch(epoch)
```

返回值：

```text
None
```

目的：

```text
在每个训练 epoch 开始前，将当前 epoch 显式传给 training DataLoader 的 sampler。
```

行为要求：

```text
1. 读取 sampler = self.training_data.sampler。
2. 如果 sampler 有 set_epoch 方法，则调用 sampler.set_epoch(epoch)。
3. 如果 sampler 没有 set_epoch 方法，则不报错，但写 logger warning。
```

参数约束：

```text
epoch 必须是非负整数。
```

## 7. 修改函数

### 7.1 `make_training_loader`

当前位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

当前函数框架：

```python
def make_training_loader(
    dataset: MainProtocolWindowDataset,
    batch_size: int,
    model_seed: int,
) -> DataLoader:
    ...
```

修改后函数框架：

```python
def make_training_loader(
    dataset: MainProtocolWindowDataset,
    batch_size: int,
    model_seed: int,
) -> DataLoader:
    ...
```

调用格式：

```python
training_data = make_training_loader(train_dataset, batch_size, model_seed)
```

调用格式不变。

返回值：

```text
torch.utils.data.DataLoader
```

返回值不变。

功能变化：

| 项 | 修改前 | 修改后 |
|---|---|---|
| 采样方式 | `shuffle=True` | `sampler=EngineBalancedSampler(...)` |
| 样本权重 | 所有 window 等权 | 每台 engine 总权重等权 |
| 是否有放回 | 否 | 是 |
| 每 epoch 样本数 | `len(dataset)` | `len(dataset)` |
| 随机种子 | DataLoader generator 连续推进 | `model_seed + epoch` |

修改后实现要求：

```python
sampler = EngineBalancedSampler(dataset=dataset, model_seed=model_seed)

return DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=False,
    drop_last=False,
)
```

禁止同时使用：

```python
shuffle=True
sampler=sampler
```

### 7.2 `log_dataset_summary`

当前位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

当前函数框架：

```python
def log_dataset_summary(
    logger: Any,
    dataset: str,
    train_n: int,
    val_n: int,
    test_n: int,
    full_test_n: int,
) -> None:
    ...
```

修改后函数框架：

```python
def log_dataset_summary(
    logger: Any,
    dataset: str,
    train_n: int,
    val_n: int,
    test_n: int,
    full_test_n: int,
    sampler_summary: dict[str, int | float] | None = None,
) -> None:
    ...
```

调用格式变化：

```python
log_dataset_summary(
    logger,
    dataset,
    train_n,
    val_n,
    test_n,
    full_test_n,
    sampler_summary=sampler_summary,
)
```

返回值：

```text
None
```

返回值不变。

功能变化：

```text
在原有数据量日志之外，记录 train sampler 名称、unit 数、窗口数范围和每台 unit 总权重范围。
```

### 7.3 `load_main_protocol_cmapss`

当前位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

当前函数框架：

```python
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
    ...
```

修改后函数框架：

```python
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
    ...
```

调用格式不变。

返回值不变。

功能变化：

```text
构建 training_data 时自动使用 engine_balanced_sampler_v1。
日志中记录 sampler_summary。
```

### 7.4 `RlPipeline.train`

当前位置：

```text
pipelines/representtion_learning.py
```

当前行为：

```python
for epoch in range(self.n_epochs):
    for i, batch in enumerate(self.training_data):
        ...
```

修改后行为：

```python
for epoch in range(self.n_epochs):
    self.set_training_sampler_epoch(epoch)
    for i, batch in enumerate(self.training_data):
        ...
```

调用格式不变。

返回值不变。

目的：

```text
使 RL 表征学习阶段每个 epoch 使用 model_seed + epoch 的 engine-balanced sampling。
```

### 7.5 `RulPipeline.train`

当前位置：

```text
pipelines/rul_estimation.py
```

当前行为：

```python
for epoch in range(self.rul_epochs):
    for i, batch in enumerate(self.training_data):
        ...
```

修改后行为：

```python
for epoch in range(self.rul_epochs):
    self.set_training_sampler_epoch(epoch)
    for i, batch in enumerate(self.training_data):
        ...
```

调用格式不变。

返回值不变。

目的：

```text
使 RUL supervised training 阶段每个 epoch 使用 model_seed + epoch 的 engine-balanced sampling。
```

## 8. 删除内容

### 8.1 删除训练 DataLoader 的 `shuffle=True`

删除位置：

```text
ocir_reproduction/data/cmapss_protocol.py
```

删除对象：

```python
generator = torch.Generator()
generator.manual_seed(int(model_seed))
return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
```

替换为：

```python
sampler = EngineBalancedSampler(dataset=dataset, model_seed=model_seed)
return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
```

删除前调用检查：

```text
rg -n "make_training_loader\(|shuffle=True|generator=generator" ocir_reproduction pipelines scripts docs main.py
```

当前实际代码调用：

```text
load_main_protocol_cmapss -> make_training_loader(train_dataset, batch_size, model_seed)
```

删除该路径后外部调用格式不变，因此不需要修改调用方参数。

不删除：

```text
make_training_loader
make_eval_loader
MainProtocolWindowDataset
WindowMetadata
```

## 9. 运行时调用关系

训练数据构造：

```text
load_main_protocol_cmapss
  -> make_dataset
  -> make_training_loader
      -> EngineBalancedSampler
      -> DataLoader(..., sampler=sampler)
```

每个 RL epoch：

```text
RlPipeline.train
  -> self.set_training_sampler_epoch(epoch)
  -> iterate self.training_data
      -> EngineBalancedSampler.__iter__
      -> torch.multinomial(..., seed=model_seed+epoch)
```

每个 RUL epoch：

```text
RulPipeline.train
  -> self.set_training_sampler_epoch(epoch)
  -> iterate self.training_data
      -> EngineBalancedSampler.__iter__
      -> torch.multinomial(..., seed=model_seed+epoch)
```

## 10. 验收标准

静态检查：

```text
ocir_reproduction/data/cmapss_protocol.py 中存在 class EngineBalancedSampler
make_training_loader 中不存在 shuffle=True
make_training_loader 中存在 sampler=EngineBalancedSampler
pipelines/solver_base.py 中存在 set_training_sampler_epoch
pipelines/representtion_learning.py 的 train loop 调用 self.set_training_sampler_epoch(epoch)
pipelines/rul_estimation.py 的 train loop 调用 self.set_training_sampler_epoch(epoch)
```

行为检查：

```text
sampler.__len__() == len(train_dataset)
同一 model_seed、同一 epoch 生成完全相同 index 序列
同一 model_seed、不同 epoch 生成不同 index 序列
每台 engine 的 total weight 相等
num_samples == len(train_dataset)
replacement == True
```

协议检查：

```text
python scripts/validate_main_protocol.py
python scripts/audit_main_protocol_alignment.py
```

synthetic fixture 检查：

```text
构造至少 2 台 train engine。
engine A 与 engine B 的窗口数不同。
检查 compute_engine_balanced_weights 后两台 engine 的权重和相同。
检查 EngineBalancedSampler 在 epoch=0 与 epoch=1 的 index 序列不同。
检查重新设置 epoch=0 后 index 序列与第一次 epoch=0 完全一致。
```

## 11. 非目标

本次不实现：

```text
life-stage-balanced sampling
uniform-window sensitivity experiment
训练标签直方图输出
按 life stage 的误差分析
final evaluation views
predictions.csv
metrics.json
```

