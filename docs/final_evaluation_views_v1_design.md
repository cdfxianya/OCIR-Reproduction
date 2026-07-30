# Final Evaluation 两个 Views 技术文档

## 1. 修复目标

本文档规定 `main_protocol_v1` 下 RUL final evaluation 的两个 official test evaluation views 的实现方案。

目标文件：

```text
pipelines/rul_estimation.py
scripts/audit_main_protocol_alignment.py
```

本修复必须解决以下歧义：

```text
official test 到底用哪个数据对象评估
raw_pred_rul 是否只生成一次
provided_rul_eval_v1 和 capped_target_eval_v1 分别用什么 target
两个 views 的 prediction 后处理分别是什么
metrics.json 与 predictions.csv 至少写哪些字段
full_test_set 是否参与 final evaluation
```

本修复完成后，最终 official test 主评估必须满足：

```text
每个 official test engine 恰好一条 terminal-window raw prediction
同一条 raw_pred_rul 确定性派生两个 evaluation view
两个 views 分别计算 RMSE、MAE、NASA Score
两个 views 分别写入 metrics.json
逐 engine 预测与误差写入 predictions.csv
```

## 2. 非目标

本修复不做以下事项：

```text
不修改 OCIR 主模型结构
不修改 RulEstimator 模型结构
不修改 train_windows.npz / validation_view_v1.npz / test_terminal_windows.npz / full_test_windows.npz 数据契约
不修改 docs/main_protocol_v1_data_contract.md
不实现 5-seed runner
不重构完整 results/experiments 输出管理
不跑真实 FD001-FD004
不执行任何下载、联网安装、git pull、git push 或 git clone
不把 full_test_windows.npz 用作 official test 主表指标
```

## 3. 协议依据

最高依据为最初指定的协议文档：

```text
H:\Users\zhang\Desktop\ml\OCIR-learning\02_cmapss_data_source_and_alignment(1).md
H:\Users\zhang\Desktop\ml\OCIR-learning\03_cmapss_research_problem_and_method(1).md
```

当前仓库内的直接依据为：

```text
ocir_reproduction/protocols/main_protocol_v1.py
ocir_reproduction/protocols/main_protocol_v1.yaml
docs/main_protocol_v1_data_contract.md
docs/rul_validation_fix_design.md
```

旧 OCIR 源代码只作为主过程参考，不作为 final evaluation target、prediction clipping 或 RUL 单位的依据。

`main_protocol_v1` 固定：

```text
main_protocol_version = main_protocol_v1
window_length = 30
rul_cap = 125
split_seed = 2026
label scale = cycle
label normalization = none
evaluation error_definition = prediction - target
evaluation views = provided_rul_eval_v1, capped_target_eval_v1
```

## 4. RUL 单位

RUL 标签、模型输出和评价预测值全部使用原始 cycle 单位。

强制要求：

```text
target 不归一化到 [0, 1]
raw_pred_rul 不假定为 [0, 1]
metric 里不得把 prediction 或 target 乘以 125
不得调用会隐式乘以 125 的旧 RUL_metric 实现
```

`test_terminal_windows.npz` 中：

```text
provided_rul      = NASA RUL_FDxxx.txt 提供的 terminal remaining cycles
capped_target_rul = min(provided_rul, 125)
y                 = capped_target_rul
```

## 5. Final Evaluation 数据对象

### 5.1 主评估对象

final evaluation 两个 views 只能使用：

```text
self.testing_data
```

其来源必须是：

```text
test_terminal_windows.npz
```

`self.testing_data` 必须是 `torch.utils.data.DataLoader`，且必须满足：

```text
shuffle = False
每个 official test unit 恰好一行
DataLoader batch 返回 x, y, ocs, tidx
loader.dataset.all_metadata() 与 DataLoader 样本顺序一致
```

每条记录来自 `test_terminal_windows.npz` 的 terminal window：

```text
x                 -> 模型输入窗口
y                 -> capped_target_rul
unit              -> official test namespace 下的 unit id
end_cycle         -> terminal observed cycle
provided_rul      -> NASA provided terminal RUL
capped_target_rul -> min(provided_rul, rul_cap)
```

### 5.2 禁止参与主评估的对象

final evaluation 主表指标不得使用：

```text
self.training_data
self.val_data
self.full_test_set
full_test_windows.npz
validation_view_v1.npz
train_windows.npz
```

`full_test_set` 只允许用于诊断和可视化，不参与 `metrics.json` 的两个 official test views，不参与 `predictions.csv` 的主评估行。

如果后续需要输出 full-test trajectory 可视化，必须建立单独函数和单独文件名，不能复用本文定义的 `predictions.csv` 和 `metrics.json` 主评估语义。

## 6. 两个 Evaluation Views 的严格定义

设 official test terminal 样本索引为 `i`，该样本属于 dataset `d`、source partition `test`、unit `u_i`。

模型只允许对每个样本执行一次前向预测：

```text
raw_pred_rul_i = RulEstimator(x_i, tidx_i)
```

`raw_pred_rul_i` 是模型原始输出，单位为 cycle，不做归一化反变换。

### 6.1 `provided_rul_eval_v1`

评价 target：

```text
target_i = provided_rul_i
```

评价 prediction：

```text
prediction_i = max(raw_pred_rul_i, 0)
```

误差：

```text
error_i = prediction_i - target_i
```

该 view 是主结论 view，保留 NASA 数据包提供的 terminal RUL 参考值。

### 6.2 `capped_target_eval_v1`

评价 target：

```text
target_i = capped_target_rul_i
```

其中：

```text
capped_target_rul_i = min(provided_rul_i, rul_cap)
rul_cap = 125
```

评价 prediction：

```text
prediction_i = clip(raw_pred_rul_i, 0, rul_cap)
```

误差：

```text
error_i = prediction_i - target_i
```

该 view 是强制辅助 view，与 capped training label 位于同一目标空间。

### 6.3 两个 Views 的一致性约束

两个 views 必须从同一条 `raw_pred_rul_i` 派生。

禁止：

```text
为 provided_rul_eval_v1 单独跑一次模型
为 capped_target_eval_v1 单独跑一次模型
为两个 views 使用不同 checkpoint
为两个 views 使用不同输入窗口
根据 official test 结果决定是否 clip
根据 official test 结果调整 target、后处理或 view 主次
```

## 7. Metric 公式

每个 evaluation view 独立计算 RMSE、MAE、NASA Score。

对 view `v`，设样本数为 `N`：

$$
e_i^{(v)} = \hat{y}_i^{(v)} - y_i^{(v)}
$$

其中：

```text
hat_y_i^(v) = 当前 view 的评价 prediction
y_i^(v)     = 当前 view 的评价 target
```

RMSE：

$$
RMSE^{(v)} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(e_i^{(v)})^2}
$$

MAE：

$$
MAE^{(v)} = \frac{1}{N}\sum_{i=1}^{N}|e_i^{(v)}|
$$

NASA Score 的单样本项：

$$
score_i^{(v)} =
\begin{cases}
\exp(-e_i^{(v)} / 13) - 1, & e_i^{(v)} < 0 \\
\exp(e_i^{(v)} / 10) - 1, & e_i^{(v)} \ge 0
\end{cases}
$$

NASA Score 总分：

$$
NASA\ Score^{(v)} = \sum_{i=1}^{N} score_i^{(v)}
$$

误差方向固定为：

```text
error = prediction - target
```

因此：

```text
error > 0 表示 prediction 高于该 view 的 target
error < 0 表示 prediction 低于该 view 的 target
```

NASA Score 不取均值，按样本求和。

## 8. 新增内容

以下函数应新增为 `RulPipeline` 的实例方法，除非实现代理发现更小范围的纯函数更符合现有代码；若改为纯函数，函数名、输入、输出语义必须保持等价。

### 8.1 `get_loader_metadata`

函数框架：

```python
def get_loader_metadata(self, loader) -> list:
    ...
```

调用格式：

```python
metadata = self.get_loader_metadata(self.testing_data)
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `loader` | `torch.utils.data.DataLoader` | 必须有 `dataset`，且 `dataset` 必须提供 `all_metadata()` |

返回值：

```text
metadata: list[WindowMetadata]
```

目的：

```text
按 DataLoader 样本顺序读取 metadata，用于获取 dataset、source_partition、unit、end_cycle、provided_rul、capped_target_rul。
```

行为要求：

```text
如果 loader.dataset 不存在，抛出 ValueError
如果 loader.dataset 没有 all_metadata()，抛出 ValueError
如果 metadata 为空，抛出 ValueError
返回顺序必须与 shuffle=False 的 evaluation DataLoader 一致
```

### 8.2 `validate_test_terminal_metadata`

函数框架：

```python
def validate_test_terminal_metadata(self, metadata: list, prediction_count: int) -> None:
    ...
```

调用格式：

```python
self.validate_test_terminal_metadata(metadata, raw_pred.shape[0])
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `metadata` | `list[WindowMetadata]` | 每项必须包含 test terminal evaluation 所需字段 |
| `prediction_count` | `int` | 必须等于 `len(metadata)` |

返回值：

```text
None
```

目的：

```text
在计算 official test views 前验证 metadata 与 raw prediction 一一对应，防止字段缺失或样本数量错位。
```

行为要求：

```text
prediction_count 必须为正整数
len(metadata) 必须等于 prediction_count
每个 metadata.source_partition 必须等于 "test"
每个 metadata.provided_rul 必须不是 None
每个 metadata.capped_target_rul 必须不是 None
每个 metadata.capped_target_rul 必须等于 min(metadata.provided_rul, self.rul_cap)
同一个 unit 不得重复出现
unit 必须按升序排列
```

若任一条件不满足，必须抛出 `ValueError`。

### 8.3 `build_final_prediction_rows`

函数框架：

```python
def build_final_prediction_rows(
    self,
    raw_pred: torch.Tensor,
    metadata: list,
    method: str = "ocir",
) -> list[dict]:
    ...
```

调用格式：

```python
rows = self.build_final_prediction_rows(
    raw_pred=raw_pred,
    metadata=metadata,
    method="ocir",
)
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `raw_pred` | `torch.Tensor` | shape `[N, 1]`，CPU 或 GPU 均可，但函数内部必须转 CPU float |
| `metadata` | `list[WindowMetadata]` | 长度必须等于 `N` |
| `method` | `str` | 默认 `"ocir"` |

返回值：

```text
rows: list[dict]
```

目的：

```text
从同一批 raw_pred_rul 和 test terminal metadata 派生 predictions.csv 的逐 engine 行。
```

每个 row 必须至少包含：

```text
dataset
source_partition
unit
end_cycle
provided_rul
capped_target_rul
raw_pred_rul
provided_eval_pred_rul
capped_eval_pred_rul
error_provided
error_capped
abs_error_provided
abs_error_capped
method
model_seed
window_length
rul_cap
main_protocol_version
split_id
validation_view
train_sampler
preprocessing_version
```

字段取值规则：

```text
dataset                 = metadata.dataset
source_partition        = "test"
unit                    = metadata.unit
end_cycle               = metadata.end_cycle
provided_rul            = metadata.provided_rul
capped_target_rul       = metadata.capped_target_rul
raw_pred_rul            = raw_pred[i, 0]
provided_eval_pred_rul  = max(raw_pred_rul, 0)
capped_eval_pred_rul    = min(max(raw_pred_rul, 0), self.rul_cap)
error_provided          = provided_eval_pred_rul - provided_rul
error_capped            = capped_eval_pred_rul - capped_target_rul
abs_error_provided      = abs(error_provided)
abs_error_capped        = abs(error_capped)
method                  = method
model_seed              = self.seed
window_length           = self.window
rul_cap                 = self.rul_cap
main_protocol_version   = "main_protocol_v1"
split_id                = MAIN_PROTOCOL_V1["splits"]["split_id"]
validation_view         = MAIN_PROTOCOL_V1["validation_view"]["version"]
train_sampler           = MAIN_PROTOCOL_V1["training"]["sampler"]
preprocessing_version   = "main_protocol_v1"
```

如果实现代理能从 manifest 中读取更具体的 preprocessing version，可以替换 `"main_protocol_v1"`；否则不得留空。

### 8.4 `compute_final_view_metrics`

函数框架：

```python
def compute_final_view_metrics(self, rows: list[dict], view_name: str) -> dict[str, float]:
    ...
```

调用格式：

```python
provided_metrics = self.compute_final_view_metrics(rows, "provided_rul_eval_v1")
capped_metrics = self.compute_final_view_metrics(rows, "capped_target_eval_v1")
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `rows` | `list[dict]` | 必须由 `build_final_prediction_rows()` 生成 |
| `view_name` | `str` | 只能是 `"provided_rul_eval_v1"` 或 `"capped_target_eval_v1"` |

返回值：

```python
{
    "rmse": float,
    "mae": float,
    "nasa_score": float,
}
```

目的：

```text
按指定 final evaluation view 计算 RMSE、MAE、NASA Score。
```

view 字段映射：

| `view_name` | prediction 字段 | target 字段 | error 字段 |
|---|---|---|---|
| `provided_rul_eval_v1` | `provided_eval_pred_rul` | `provided_rul` | `error_provided` |
| `capped_target_eval_v1` | `capped_eval_pred_rul` | `capped_target_rul` | `error_capped` |

行为要求：

```text
rows 不得为空
view_name 不在允许集合内时抛出 ValueError
必须使用 error = prediction - target
不得重新调用模型
不得乘以 125
NASA Score 必须按样本求和
```

### 8.5 `build_final_metrics_payload`

函数框架：

```python
def build_final_metrics_payload(
    self,
    rows: list[dict],
    method: str = "ocir",
) -> dict:
    ...
```

调用格式：

```python
metrics_payload = self.build_final_metrics_payload(rows, method="ocir")
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `rows` | `list[dict]` | 必须由 `build_final_prediction_rows()` 生成 |
| `method` | `str` | 默认 `"ocir"` |

返回值：

```text
metrics_payload: dict
```

目的：

```text
生成 metrics.json 的完整 JSON 对象。
```

返回对象必须至少包含：

```json
{
  "dataset": "FD001",
  "source_partition": "test",
  "method": "ocir",
  "model_seed": 1,
  "split_id": "split_v1_seed2026_engine_80_20",
  "split_seed": 2026,
  "window_length": 30,
  "rul_cap": 125,
  "label_mode": "capped",
  "label_scale": "cycle",
  "label_normalization": "none",
  "error_definition": "prediction_minus_target_per_evaluation_view",
  "main_protocol_version": "main_protocol_v1",
  "validation_view": "validation_view_v1",
  "train_sampler": "engine_balanced_sampler_v1",
  "preprocessing_version": "main_protocol_v1",
  "num_test_units": 100,
  "evaluation_views": {
    "provided_rul_eval_v1": {
      "rmse": 0.0,
      "mae": 0.0,
      "nasa_score": 0.0
    },
    "capped_target_eval_v1": {
      "rmse": 0.0,
      "mae": 0.0,
      "nasa_score": 0.0
    }
  }
}
```

说明：

```text
num_test_units 必须等于 rows 的数量
dataset 必须来自 rows 中唯一 dataset
source_partition 必须固定为 "test"
evaluation_views 必须同时包含两个 views
```

本次不强制写入 `git_commit`、`split_sha256`、`nominal_seed`、`nominal_checkpoint_sha256`，这些留给后续完整 experiment output 管理实现；如果实现代理能无副作用获取，可以作为可选字段追加。

### 8.6 `write_final_evaluation_outputs`

函数框架：

```python
def write_final_evaluation_outputs(
    self,
    output_dir,
    rows: list[dict],
    metrics_payload: dict,
) -> None:
    ...
```

调用格式：

```python
self.write_final_evaluation_outputs(output_dir, rows, metrics_payload)
```

参数：

| 参数 | 类型 | 约束 |
|---|---|---|
| `output_dir` | `str | pathlib.Path` | 不存在时必须创建 |
| `rows` | `list[dict]` | 写入 `predictions.csv` |
| `metrics_payload` | `dict` | 写入 `metrics.json` |

返回值：

```text
None
```

目的：

```text
把 final evaluation 主评估结果写入磁盘。
```

输出文件：

```text
output_dir/predictions.csv
output_dir/metrics.json
```

写入要求：

```text
predictions.csv 使用 UTF-8
predictions.csv 必须包含本文第 8.3 节列出的所有字段
metrics.json 使用 UTF-8
metrics.json 使用 ensure_ascii=False
metrics.json 使用 indent=2
```

若 `predictions.csv` 或 `metrics.json` 已存在，允许覆盖，但必须在 logger 中记录覆盖路径。

## 9. 修改内容

### 9.1 修改 `evaluate_test_terminal`

当前调用格式：

```python
def evaluate_test_terminal(self):
    ...
```

修改后调用格式：

```python
def evaluate_test_terminal(
    self,
    output_dir=None,
    method: str = "ocir",
    write_files: bool = True,
) -> dict:
    ...
```

返回值变化：

修改前返回：

```python
{
    "test_window_rmse": float,
    "test_window_mae": float,
    "test_nasa_score": float,
}
```

修改后返回：

```text
metrics_payload: dict
```

其中 `metrics_payload["evaluation_views"]` 必须同时包含：

```text
provided_rul_eval_v1
capped_target_eval_v1
```

功能变化：

| 行为 | 修改前 | 修改后 |
|---|---|---|
| 使用数据 | `self.testing_data` | 不变，只能使用 `self.testing_data` |
| 预测次数 | 一次 | 一次 |
| target | 默认使用 batch 里的 `y`，即 capped target | 同时使用 metadata 中的 `provided_rul` 和 `capped_target_rul` |
| prediction 后处理 | 单一 `clip(raw_pred,0,125)` | provided view 用 `max(raw_pred,0)`；capped view 用 `clip(raw_pred,0,125)` |
| metrics | 单一 capped-style RMSE/MAE/NASA | 两个 views 分别 RMSE/MAE/NASA |
| 文件输出 | 无 | 可写 `predictions.csv` 和 `metrics.json` |

实现流程必须为：

```text
1. 确认 self.testing_data 不是 None
2. raw_pred, _ = self.predict_loader(self.testing_data)
3. metadata = self.get_loader_metadata(self.testing_data)
4. self.validate_test_terminal_metadata(metadata, raw_pred.shape[0])
5. rows = self.build_final_prediction_rows(raw_pred, metadata, method=method)
6. metrics_payload = self.build_final_metrics_payload(rows, method=method)
7. 如果 write_files=True，则确定 output_dir 并调用 write_final_evaluation_outputs
8. logger 记录两个 views 的 RMSE、MAE、NASA Score
9. 返回 metrics_payload
```

`output_dir` 规则：

```text
如果调用方显式传入 output_dir，使用该目录
如果 output_dir is None，使用 Path(self.model_save_path).parent
```

该默认值只用于兼容当前 `Logss/.../checkpoints` 结构；后续完整 experiment runner 可以显式传入 `results/experiments/{exp_id}`。

禁止：

```text
evaluate_test_terminal 不得访问 self.full_test_set
evaluate_test_terminal 不得访问 self.val_data
evaluate_test_terminal 不得执行训练
evaluate_test_terminal 不得修改 checkpoint
evaluate_test_terminal 不得根据 metrics 更新 best checkpoint
```

### 9.2 修改 `compute_rul_metrics`

当前调用格式：

```python
def compute_rul_metrics(self, raw_pred, target, units=None, clip_predictions=True):
    ...
```

修改后调用格式：

```python
def compute_rul_metrics(self, raw_pred, target, units=None, clip_predictions=True):
    ...
```

调用格式不变。

返回值不变：

```python
{
    "window_rmse": float,
    "window_mae": float,
    "raw_window_rmse": float,
    "nasa_score": float,
    "engine_rmse": float | None,
}
```

功能变化：

```text
不要求必须修改该函数。
如果实现代理复用该函数计算 capped_target_eval_v1，必须保证不影响 validation 现有行为。
provided_rul_eval_v1 不能直接用 clip_predictions=True，因为 provided view 只做 max(raw_pred,0)，不做上界 125 裁剪。
```

允许两种实现方式：

```text
方式 A：保持 compute_rul_metrics 不变，两个 final views 使用 compute_final_view_metrics 计算。
方式 B：给 compute_rul_metrics 增加更通用的 postprocess 参数，但不得破坏现有 evaluate_validation 调用。
```

推荐方式 A，影响面更小。

### 9.3 修改 `scripts/audit_main_protocol_alignment.py`

必须新增静态检查，确认 `pipelines/rul_estimation.py` 满足：

```text
存在 build_final_prediction_rows
存在 compute_final_view_metrics
存在 build_final_metrics_payload
存在 write_final_evaluation_outputs
evaluate_test_terminal 中出现 provided_rul_eval_v1
evaluate_test_terminal 中出现 capped_target_eval_v1
evaluate_test_terminal 中不出现 self.full_test_set
evaluate_test_terminal 中不出现 self.val_data
evaluate_test_terminal 或其调用链写入 predictions.csv
evaluate_test_terminal 或其调用链写入 metrics.json
pipelines/rul_estimation.py 中不出现 ut.RUL_metric(
```

audit 不需要计算真实 official test metrics。

## 10. 删除内容

本修复不要求删除整个函数。

必须删除或替换的旧行为：

```text
evaluate_test_terminal 只返回 test_window_rmse/test_window_mae/test_nasa_score 的单 view 行为
evaluate_test_terminal 只用 y 作为唯一 official test target 的行为
```

删除或替换前必须检查调用：

```text
rg -n "evaluate_test_terminal\\(" pipelines src ocir_reproduction scripts main.py
rg -n "test_window_rmse|test_window_mae|test_nasa_score" pipelines src ocir_reproduction scripts docs
```

如果除 `pipelines/rul_estimation.py` 内部以外存在外部调用，必须保证外部调用能接受新的返回结构，或同步修改调用方。

保留：

```text
predict_loader
get_loader_units
compute_rul_metrics
evaluate_validation
maybe_update_best_checkpoint
restore_best_checkpoint
```

## 11. `predictions.csv` 输出格式

`predictions.csv` 每个 dataset、method、model_seed、official test unit 恰好一行。

本次 final evaluation 最低列集合固定为：

| 列名 | 类型 | 含义 |
|---|---|---|
| `dataset` | string | `FD001` 到 `FD004` |
| `source_partition` | string | 固定为 `test` |
| `unit` | int | official test unit id |
| `end_cycle` | int | terminal observed cycle |
| `provided_rul` | float | `RUL_FDxxx.txt` 提供的 terminal RUL |
| `capped_target_rul` | float | `min(provided_rul, rul_cap)` |
| `raw_pred_rul` | float | 模型原始输出，cycle 单位 |
| `provided_eval_pred_rul` | float | `max(raw_pred_rul, 0)` |
| `capped_eval_pred_rul` | float | `clip(raw_pred_rul, 0, rul_cap)` |
| `error_provided` | float | `provided_eval_pred_rul - provided_rul` |
| `error_capped` | float | `capped_eval_pred_rul - capped_target_rul` |
| `abs_error_provided` | float | `abs(error_provided)` |
| `abs_error_capped` | float | `abs(error_capped)` |
| `method` | string | 默认 `ocir` |
| `model_seed` | int | `self.seed` |
| `window_length` | int | `self.window`，必须为 30 |
| `rul_cap` | int | 必须为 125 |
| `main_protocol_version` | string | `main_protocol_v1` |
| `split_id` | string | `split_v1_seed2026_engine_80_20` |
| `validation_view` | string | `validation_view_v1` |
| `train_sampler` | string | `engine_balanced_sampler_v1` |
| `preprocessing_version` | string | 默认 `main_protocol_v1` |

列顺序必须与上表一致。

## 12. `metrics.json` 输出格式

`metrics.json` 必须是单个 JSON object。

最低字段集合：

```json
{
  "dataset": "FD001",
  "source_partition": "test",
  "method": "ocir",
  "model_seed": 1,
  "split_id": "split_v1_seed2026_engine_80_20",
  "split_seed": 2026,
  "window_length": 30,
  "rul_cap": 125,
  "label_mode": "capped",
  "label_scale": "cycle",
  "label_normalization": "none",
  "error_definition": "prediction_minus_target_per_evaluation_view",
  "main_protocol_version": "main_protocol_v1",
  "validation_view": "validation_view_v1",
  "train_sampler": "engine_balanced_sampler_v1",
  "preprocessing_version": "main_protocol_v1",
  "num_test_units": 100,
  "evaluation_views": {
    "provided_rul_eval_v1": {
      "rmse": 0.0,
      "mae": 0.0,
      "nasa_score": 0.0
    },
    "capped_target_eval_v1": {
      "rmse": 0.0,
      "mae": 0.0,
      "nasa_score": 0.0
    }
  }
}
```

禁止：

```text
不得只输出 capped_target_eval_v1
不得只输出 provided_rul_eval_v1
不得把两个 views 的样本混合后计算一个总 RMSE
不得把 NASA Score 除以 num_test_units 后仍命名为 nasa_score
```

## 13. Smoke Test 设计

实现后必须新增或执行一个不依赖真实 FD001-FD004 的 synthetic smoke test。

最小 synthetic case：

```text
rul_cap = 125
raw_pred_rul = [-10, 50, 200]
provided_rul = [20, 60, 150]
capped_target_rul = [20, 60, 125]
```

期望派生：

```text
provided_eval_pred_rul = [0, 50, 200]
capped_eval_pred_rul   = [0, 50, 125]
error_provided         = [-20, -10, 50]
error_capped           = [-20, -10, 0]
```

期望指标：

provided view：

$$
RMSE = \sqrt{((-20)^2 + (-10)^2 + 50^2) / 3}
$$

$$
MAE = (20 + 10 + 50) / 3
$$

$$
NASA\ Score = (\exp(20/13)-1) + (\exp(10/13)-1) + (\exp(50/10)-1)
$$

capped view：

$$
RMSE = \sqrt{((-20)^2 + (-10)^2 + 0^2) / 3}
$$

$$
MAE = (20 + 10 + 0) / 3
$$

$$
NASA\ Score = (\exp(20/13)-1) + (\exp(10/13)-1) + (\exp(0/10)-1)
$$

smoke test 必须验证：

```text
两个 views 使用同一 raw_pred_rul
provided view 不做上界 125 裁剪
capped view 做上界 125 裁剪
误差方向为 prediction - target
NASA Score 分母方向正确：低估用 13，高估用 10
predictions.csv 包含 3 行
metrics.json 同时包含两个 views
```

## 14. Audit 检查标准

静态 audit 必须至少检查：

```text
pipelines/rul_estimation.py 存在 build_final_prediction_rows
pipelines/rul_estimation.py 存在 compute_final_view_metrics
pipelines/rul_estimation.py 存在 build_final_metrics_payload
pipelines/rul_estimation.py 存在 write_final_evaluation_outputs
pipelines/rul_estimation.py 中存在 provided_rul_eval_v1
pipelines/rul_estimation.py 中存在 capped_target_eval_v1
pipelines/rul_estimation.py 中存在 predictions.csv
pipelines/rul_estimation.py 中存在 metrics.json
pipelines/rul_estimation.py 中不存在 ut.RUL_metric(
evaluate_test_terminal 函数体中不存在 self.full_test_set
evaluate_test_terminal 函数体中不存在 self.val_data
```

processed-data audit 保持现有行为：

```text
真实 FD001-FD004 未交付时允许 skip
synthetic fixture 可用于验证 test_terminal_windows.npz 一 unit 一行和 target 字段一致性
```

## 15. 验收命令

实现代理必须在不联网、不下载的前提下执行：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\validate_main_protocol.py
H:\Conda\envs\ocir-repro\python.exe -B scripts\audit_main_protocol_alignment.py
```

如果 Windows 权限导致 `py_compile` 无法写 `__pycache__`，使用 AST 语法检查替代：

```text
H:\Conda\envs\ocir-repro\python.exe -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py') if '.git' not in p.parts]; print('AST syntax OK')"
```

必须额外执行 final evaluation smoke test。smoke test 可以是临时内联脚本，但不得下载数据。

验收通过条件：

```text
协议检查通过
audit fail=0
smoke test 数值与第 13 节手算结果一致
smoke test 生成的 predictions.csv 和 metrics.json schema 正确
evaluate_test_terminal 不访问 full_test_set
evaluate_test_terminal 不访问 val_data
```

## 16. 实现顺序

实现代理必须按以下顺序执行：

```text
1. 检查 evaluate_test_terminal 外部调用
2. 新增 get_loader_metadata
3. 新增 validate_test_terminal_metadata
4. 新增 build_final_prediction_rows
5. 新增 compute_final_view_metrics
6. 新增 build_final_metrics_payload
7. 新增 write_final_evaluation_outputs
8. 修改 evaluate_test_terminal
9. 更新 audit_main_protocol_alignment.py
10. 运行静态检查
11. 运行 synthetic smoke test
12. 更新 ocir_protocol_alignment_checklist.md 中 final evaluation 相关状态
```

## 17. 完成后的对齐状态

完成本文档对应实现后，以下清单项可以从“未对齐”或“部分完成”推进：

```text
Test 主评估样本
Evaluation views
Prediction 后处理
Metrics
predictions schema
metrics schema
```

以下清单项仍不得因此打勾：

```text
5 seeds
输出目录完整 results/experiments/{exp_id}
真实 FD001-FD004 processed 数据包验收
Final test access log
Git commit 记录
```
