# RUL Validation 修复技术文档

## 1. 修复范围

本文档规定 `RulPipeline` 中 RUL validation 的修复方案。

目标文件：

```text
pipelines/rul_estimation.py
```

本修复只改变 RUL 训练过程中的 validation、checkpoint selection、final model save 行为。

本修复不改变：

```text
OCIR 主模型结构
RulEstimator 模型结构
train_windows.npz / validation_view_v1.npz / test_terminal_windows.npz / full_test_windows.npz 的数据格式
load_main_protocol_cmapss 的返回接口
```

本修复的最高依据是最初的数据与方法协议文档：

```text
02_cmapss_data_source_and_alignment(1).md
03_cmapss_research_problem_and_method(1).md
```

旧 OCIR 源代码只作为主模型过程参考，不作为 RUL 标签单位的依据。

RUL 标签单位固定为：

```text
cycle
```

RUL 标签不得归一化到 `[0, 1]`。

## 2. 当前问题

当前 `RulPipeline.vali()` 在非 `full_test` 分支中实际使用 `self.testing_data`：

```python
if (self.val_data is None) or (epoch == "evaluation"):
    data = self.testing_data
else:
    data = self.testing_data # self.val_data # just use self.testing_data
```

该行为违反 `main_protocol_v1` 的 official test 封存约束。

训练阶段禁止使用以下对象做 validation、checkpoint selection、epoch selection、early stopping 或超参数选择：

```text
self.testing_data
self.full_test_set
test_terminal_windows.npz
full_test_windows.npz
RUL_FDxxx.txt 派生出来的任何 official test target
```

训练阶段唯一允许用于 RUL validation 的数据对象是：

```text
self.val_data
```

其来源必须是：

```text
validation_view_v1.npz
```

## 3. 修复后数据角色

RUL 阶段必须使用以下数据分工：

| 对象 | 来源文件 | 训练阶段用途 | 是否参与梯度更新 |
|---|---|---|---|
| `self.training_data` | `train_windows.npz` | 拟合 RUL 模型参数 | 是 |
| `self.val_data` | `validation_view_v1.npz` | 选择 best epoch checkpoint | 否 |
| `self.testing_data` | `test_terminal_windows.npz` | 最终 official test 评估 | 否 |
| `self.full_test_set` | `full_test_windows.npz` | 诊断和可视化 | 否 |

训练过程不得自动调用 `self.testing_data` 或 `self.full_test_set`。

## 4. Validation 输入

`self.val_data` 必须是 `torch.utils.data.DataLoader`。

每个 batch 的返回格式必须为：

```python
x, y, ocs, tidx
```

字段要求：

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `x` | `torch.float32` | `[B, 30, 14]` | train-only 标准化后的传感器窗口 |
| `y` | `torch.float32` | `[B, 1]` | capped RUL，单位为 cycle |
| `ocs` | `torch.float32` | `[B, 30, 1]` | 操作工况 cluster id |
| `tidx` | `torch.int64` | `[B, 1]` | 当前窗口终点 cycle |

`self.val_data.dataset` 必须提供：

```python
all_metadata()
```

其返回值必须与 DataLoader 的样本顺序一致。

Validation DataLoader 必须满足：

```text
shuffle = False
```

## 5. Validation 样本定义

`validation_view_v1.npz` 中每台 validation engine 必须恰好有 3 个窗口：

```text
checkpoint_q = 0.5
checkpoint_q = 0.7
checkpoint_q = 0.9
```

每条 validation 样本的 target 必须为：

```text
y = min(unit_length - end_cycle, 125)
```

其中：

```text
rul_cap = 125
```

Validation 样本不允许来自 official test。

## 6. Prediction 后处理

Validation 指标计算前必须对模型原始输出执行裁剪：

```python
pred = torch.clamp(raw_pred, min=0.0, max=125.0)
```

`raw_pred` 不得先乘以 `125`。

`target` 不得先乘以 `125`。

原因：

```text
main_protocol_v1 中 y 已经是 capped RUL 原始 cycle 单位。
```

因此 RUL validation 不得调用旧的：

```python
ut.RUL_metric(pred, target)
```

该函数内部会将 `pred` 和 `target` 乘以 `125`，不符合本协议的 RUL 单位定义。

## 7. Selection Metric

RUL checkpoint selection 的唯一指标为：

```text
val_engine_rmse
```

选择方向：

```text
minimize
```

Tie-break 规则：

```text
当两个 epoch 的 val_engine_rmse 完全相等时，保留更早的 epoch。
```

实现上必须只在严格小于时更新 best checkpoint：

```python
if current_val_engine_rmse < best_val_engine_rmse:
    update_best_checkpoint()
```

不得使用 `<=` 更新。

## 8. Validation Frequency

Validation 频率必须由显式参数控制：

```text
rul_validate_every_epochs
```

默认值：

```text
rul_validate_every_epochs = 1
```

含义：

```text
每隔 rul_validate_every_epochs 个 epoch 执行一次 evaluate_validation。
```

默认情况下，每个 epoch 都执行一次 validation。

参数约束：

```text
rul_validate_every_epochs 必须是正整数
```

如果未在 config 中提供该参数，代码必须使用默认值 `1`。

如果提供的值小于 `1` 或不是整数，必须抛出：

```python
ValueError
```

Validation 检查点规则：

```text
当 (epoch + 1) % rul_validate_every_epochs == 0 时执行 validation
训练结束后必须确保最终 epoch 至少被 validation 一次
```

如果最终 epoch 已经在循环中 validation 过，不得重复 validation 同一个 epoch。

该参数决定：

```text
哪些 epoch 有资格被 val_engine_rmse 选为 best checkpoint
```

该参数不得根据 official test 结果修改。

## 9. val_engine_rmse 定义

设 validation prediction 后处理后的误差为：

```text
error_i = pred_i - target_i
```

对每台 validation engine `u`，取该 engine 的所有 validation 样本误差。

在 `validation_view_v1` 中，每台 engine 应恰好有 3 个误差：

```text
q = 0.5, 0.7, 0.9
```

该 engine 的 MSE 定义为：

```text
engine_mse_u = mean(error_i^2 for samples belonging to engine u)
```

整个 validation set 的 engine-level RMSE 定义为：

```text
val_engine_rmse = sqrt(mean(engine_mse_u over validation engines))
```

该指标不是：

```text
每个 engine 取最低 RMSE
每个 checkpoint_q 取最低 RMSE
每个样本独立加权后直接选点
official test RMSE
training loss
```

该指标选择的是：

```text
训练 epoch
```

不选择：

```text
engine
window
checkpoint_q
test sample
```

## 10. 新增函数

以下函数应新增为 `RulPipeline` 的实例方法。

### 10.1 `predict_loader`

函数框架：

```python
def predict_loader(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

调用格式：

```python
raw_pred, target = self.predict_loader(self.val_data)
```

返回值：

```text
raw_pred: torch.Tensor, shape [N, 1], device CPU
target:   torch.Tensor, shape [N, 1], device CPU
```

目的：

```text
对指定 DataLoader 执行 RUL predictor 前向推理，并按 DataLoader 顺序收集 raw prediction 与 target。
```

行为要求：

```text
调用前将 self.rul_predictor 置为 eval mode
使用 torch.no_grad()
x 和 tidx 移动到 self.device
y 只作为 target 收集
不计算 loss
不执行 backward
不执行 optimizer step
不画图
不保存文件
```

### 10.2 `get_loader_units`

函数框架：

```python
def get_loader_units(self, loader) -> torch.Tensor:
    ...
```

调用格式：

```python
units = self.get_loader_units(self.val_data)
```

返回值：

```text
units: torch.Tensor, shape [N], dtype torch.int64, device CPU
```

目的：

```text
读取 loader.dataset.all_metadata() 中的 unit 字段，并生成与 predict_loader 输出顺序一致的 unit tensor。
```

行为要求：

```text
如果 loader.dataset 不存在 all_metadata()，必须抛出 ValueError
如果 metadata 数量与 prediction 数量不一致，调用方必须抛出 ValueError
该函数只用于 shuffle=False 的 evaluation loader
```

### 10.3 `compute_rul_metrics`

函数框架：

```python
def compute_rul_metrics(
    self,
    raw_pred: torch.Tensor,
    target: torch.Tensor,
    units: torch.Tensor | None = None,
    clip_predictions: bool = True,
) -> dict[str, float]:
    ...
```

调用格式：

```python
metrics = self.compute_rul_metrics(
    raw_pred=raw_pred,
    target=target,
    units=units,
    clip_predictions=True,
)
```

返回值：

```python
{
    "window_rmse": float,
    "window_mae": float,
    "raw_window_rmse": float,
    "nasa_score": float,
    "engine_rmse": float | None,
}
```

目的：

```text
在 capped RUL 原始 cycle 单位下计算 RUL evaluation metrics。
```

行为要求：

```text
raw_pred.shape 必须等于 target.shape
raw_pred.shape 必须为 [N, 1]
target.shape 必须为 [N, 1]
如果 clip_predictions=True，则 pred = clamp(raw_pred, 0, self.rul_cap)
不得调用 ut.RUL_metric
不得将 pred 或 target 乘以 125
```

NASA score 定义：

```text
error = pred - target
如果 error < 0: score_i = exp(-error / 13) - 1
如果 error >= 0: score_i = exp(error / 10) - 1
nasa_score = sum(score_i)
```

### 10.4 `evaluate_validation`

函数框架：

```python
def evaluate_validation(self, epoch) -> dict[str, float]:
    ...
```

调用格式：

```python
metrics = self.evaluate_validation(epoch + 1)
```

返回值：

```python
{
    "val_window_rmse": float,
    "val_window_mae": float,
    "val_raw_rmse": float,
    "val_nasa_score": float,
    "val_engine_rmse": float,
}
```

目的：

```text
在 self.val_data 上执行 RUL validation，并返回用于 checkpoint selection 的指标。
```

行为要求：

```text
必须使用 self.val_data
不得使用 self.testing_data
不得使用 self.full_test_set
必须调用 predict_loader(self.val_data)
必须调用 get_loader_units(self.val_data)
必须调用 compute_rul_metrics(..., units=units, clip_predictions=True)
必须将 metrics 写入 logger
不得保存模型
不得画 official test 图
```

如果 `self.val_data is None`，必须抛出：

```python
ValueError
```

### 10.5 `maybe_update_best_checkpoint`

函数框架：

```python
def maybe_update_best_checkpoint(self, epoch, metrics: dict[str, float]) -> None:
    ...
```

调用格式：

```python
self.maybe_update_best_checkpoint(epoch + 1, metrics)
```

返回值：

```text
None
```

目的：

```text
根据 metrics["val_engine_rmse"] 判断是否更新当前 RUL best checkpoint。
```

行为要求：

```text
如果 self.best_val_engine_rmse 尚未初始化，视为正无穷
只在 metrics["val_engine_rmse"] 严格小于 self.best_val_engine_rmse 时更新
更新时必须 deepcopy(self.rul_predictor.state_dict())
更新时必须记录 epoch
不得访问 self.testing_data
不得访问 self.full_test_set
```

更新的实例属性：

```python
self.best_val_engine_rmse
self.best_rul_epoch
self.best_rul_state_dict
```

### 10.6 `restore_best_checkpoint`

函数框架：

```python
def restore_best_checkpoint(self) -> None:
    ...
```

调用格式：

```python
self.restore_best_checkpoint()
```

返回值：

```text
None
```

目的：

```text
训练结束后，将 self.rul_predictor 恢复到 validation 最优 epoch 的参数。
```

行为要求：

```text
如果 self.best_rul_state_dict 存在，则 load_state_dict
如果 self.best_rul_state_dict 不存在，则保持当前模型参数不变，并写入 logger warning
恢复后必须将 self.rul_predictor 置为 eval mode
```

### 10.7 `evaluate_test_terminal`

本小节只保留“official test 必须显式调用，不能用于 validation/model selection”的约束。
final evaluation 的返回结构、两个 views、`predictions.csv` 和 `metrics.json` 输出规则，以 `docs/final_evaluation_views_v1_design.md` 为准。

函数框架：

```python
def evaluate_test_terminal(
    self,
    output_dir=None,
    method: str = "ocir",
    write_files: bool = True,
    access_approval=None,
) -> dict:
    ...
```

调用格式：

```python
metrics = self.evaluate_test_terminal()
```

返回值：

```text
metrics_payload: dict
```

目的：

```text
为最终 official test 阶段保留显式评估入口。
```

行为要求：

```text
必须使用 self.testing_data
不得在 train() 中调用
不得在 __call__() 中自动调用
只有实验进入最终报告阶段时才允许人工显式调用
必须先校验 G4 access_approval，未授权时不得访问 self.testing_data
必须按 final_evaluation_views_v1 同时输出 provided_rul_eval_v1 与 capped_target_eval_v1
```

## 11. 修改函数

### 11.1 `__call__`

当前调用格式：

```python
def __call__(self, validation=True):
    ...
```

修改后调用格式：

```python
def __call__(self, validation=True):
    ...
```

调用格式不变。

当前返回值：

```text
self.rul_predictor
```

修改后返回值：

```text
self.rul_predictor
```

返回值不变。

功能变化：

| 行为 | 修改前 | 修改后 |
|---|---|---|
| `self.required_training=True` | 调用 `self.train()` | 调用 `self.train()` |
| `self.required_training=False` | 自动调用 `self.vali("evaluation")` 和 `self.vali("full_test")` | 不自动执行 official test 或 full test |
| 返回模型 | 是 | 是 |

修改后禁止出现：

```python
self.vali("evaluation")
self.vali("full_test")
self.vali("full_test", [5, 10, 15, 20])
```

### 11.2 `train`

当前调用格式：

```python
def train(self):
    ...
```

修改后调用格式：

```python
def train(self):
    ...
```

调用格式不变。

当前返回值：

```text
None
```

修改后返回值：

```text
None
```

返回值不变。

功能变化：

| 行为 | 修改前 | 修改后 |
|---|---|---|
| 训练 loss | `Loss_RUL(y, x, tidx)` | 不变 |
| periodic validation | `self.vali(epoch+1)` | `evaluate_validation(epoch+1)` |
| checkpoint selection | 无，保存最后 epoch | 用 `val_engine_rmse` 保存 best epoch |
| 训练结束 official test | 自动调用 | 禁止自动调用 |
| 保存模型 | 保存最后 epoch | 保存 validation best epoch |

修改后训练循环中的 validation 逻辑必须为：

```python
if self.validation and ((epoch + 1) % self.rul_validate_every_epochs) == 0:
    metrics = self.evaluate_validation(epoch + 1)
    self.maybe_update_best_checkpoint(epoch + 1, metrics)
```

训练结束后，如果最终 epoch 尚未 validation，必须补一次：

```python
if self.validation and (self.rul_epochs % self.rul_validate_every_epochs) != 0:
    metrics = self.evaluate_validation(self.rul_epochs)
    self.maybe_update_best_checkpoint(self.rul_epochs, metrics)
```

训练结束保存模型前必须调用：

```python
self.restore_best_checkpoint()
```

然后调用：

```python
ut.save_model(self.rul_predictor, path_=self.model_save_path, name="rul")
```

修改后 `train()` 中禁止出现：

```python
self.vali("evaluation")
self.vali("full_test")
self.testing_data
self.full_test_set
```

## 12. 删除函数

### 12.1 删除 `RulPipeline.vali`

删除对象：

```python
def vali(self, epoch, specific_instance=[]):
    ...
```

删除原因：

```text
该函数混合了 validation、official test terminal evaluation、full test visualization 三种职责。
该函数在 validation 分支中实际使用 self.testing_data。
该函数会导致 official test 在训练阶段泄漏。
```

删除前调用检查：

```text
rg -n "\.vali\(|def vali" pipelines src ocir_reproduction scripts
```

已知结果：

```text
RulPipeline.vali 的调用只出现在 pipelines/rul_estimation.py 内部。
pipelines/representtion_learning.py 中也存在 vali，但属于 RlPipeline，不属于本次删除对象。
```

删除前必须先完成：

```text
替换 __call__ 中所有 self.vali(...) 调用
替换 train 中所有 self.vali(...) 调用
确认其他文件不存在 RulPipeline.vali 的外部调用
```

删除后必须再次执行：

```text
rg -n "\.vali\(|def vali" pipelines src ocir_reproduction scripts
```

验收条件：

```text
pipelines/rul_estimation.py 中不得出现 self.vali 或 def vali
pipelines/representtion_learning.py 中的 vali 可以保留
```

## 13. 保留函数

### 13.1 保留 `ut.RUL_metric`

保留对象：

```python
ut.RUL_metric(pred, target)
```

保留原因：

```text
该函数可能仍被历史代码或非 protocol 实验使用。
```

限制：

```text
RUL validation 修复后的 pipelines/rul_estimation.py 不得调用该函数。
```

## 14. 日志要求

每次 validation 必须至少记录：

```text
epoch
val_window_rmse
val_window_mae
val_raw_rmse
val_nasa_score
val_engine_rmse
best_rul_epoch
best_val_engine_rmse
```

日志不得包含 official test 指标，除非显式调用 `evaluate_test_terminal()`。

## 15. 验收标准

代码实现后，必须满足以下静态检查：

```text
pipelines/rul_estimation.py 中不存在 def vali
pipelines/rul_estimation.py 中不存在 self.vali(
train() 中不存在 self.testing_data
train() 中不存在 self.full_test_set
__call__() 中不存在 self.testing_data
__call__() 中不存在 self.full_test_set
evaluate_validation() 中存在 self.val_data
evaluate_validation() 中不存在 self.testing_data
evaluate_validation() 中不存在 self.full_test_set
pipelines/rul_estimation.py 中不存在 ut.RUL_metric(
```

必须满足以下运行检查：

```text
python -m py_compile pipelines/rul_estimation.py
python scripts/validate_main_protocol.py
```

如果存在 synthetic processed fixture，则必须额外检查：

```text
RulPipeline.evaluate_validation() 可以在 self.val_data 上完成一次 forward
compute_rul_metrics() 返回 val_engine_rmse
val_engine_rmse 为有限 float
```

## 16. 实现顺序

必须按以下顺序实现：

```text
1. 新增 predict_loader
2. 新增 get_loader_units
3. 新增 compute_rul_metrics
4. 新增 evaluate_validation
5. 新增 maybe_update_best_checkpoint
6. 新增 restore_best_checkpoint
7. 新增 evaluate_test_terminal
8. 修改 train
9. 修改 __call__
10. 删除 RulPipeline.vali
11. 更新 scripts/audit_main_protocol_alignment.py
12. 执行静态检查和运行检查
```
