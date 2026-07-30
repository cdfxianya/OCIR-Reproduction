# main_protocol_v1 实现总记录

本文档是 `OCIR-Reproduction` 当前协议化复现工作的总记录，用来合并并串联此前分散的技术文档、实现决策和验收结果。

本文档不替代已经提交给数据处理负责人的 `docs/main_protocol_v1_data_contract.md`。数据契约仍以该文件为准；本文档记录模型侧、runner 侧和审计侧为了配合该契约所做的实现过程。

## 1. 最高依据

本仓库当前实现不追求逐行严格复现旧 OCIR 仓库，而是按用户最开始指定的协议文档执行。

最高依据：

```text
02_cmapss_data_source_and_alignment(1).md
03_cmapss_research_problem_and_method(1).md
```

仓库内直接依据：

```text
ocir_reproduction/protocols/main_protocol_v1.py
ocir_reproduction/protocols/main_protocol_v1.yaml
docs/main_protocol_v1_data_contract.md
```

旧 OCIR 源代码的角色：

```text
只作为 OCIR 主模型和主训练过程参考
不作为数据契约、RUL 标签单位、official test 使用方式、evaluation views 或输出归档结构的最高依据
```

## 2. 分阶段技术文档

本总记录合并以下阶段性文档的结论：

| 文档 | 作用 | 当前状态 |
|---|---|---|
| `docs/main_protocol_v1_data_contract.md` | processed 数据交付契约 | 保留独立，不在本文中改写 |
| `docs/cmapss_protocol_loader_design.md` | 协议化数据加载器设计 | 已实现 |
| `docs/rul_validation_fix_design.md` | RUL validation 与 checkpoint 选择修复设计 | 已实现，并已同步 G4 gate 约束 |
| `docs/engine_balanced_sampler_v1_design.md` | engine-balanced sampler 设计 | 已实现 |
| `docs/final_evaluation_views_v1_design.md` | official test 两个 evaluation views 设计 | 已实现，并已同步 G4 gate 约束 |
| `docs/experiment_runner_outputs_v1_design.md` | 统一 runner 和标准输出目录设计 | 已实现，并已升级为 G4 approval gate |

## 3. 最终协议常量

当前主协议固定为：

```text
protocol_version = main_protocol_v1
datasets = FD001, FD002, FD003, FD004
window_length = 30
selected_sensor_dim = 14
rul_cap = 125
label_scale = cycle
label_normalization = none
split_seed = 2026
split_id = split_v1_seed2026_engine_80_20
validation_view = validation_view_v1
validation_checkpoints = 0.5, 0.7, 0.9
train_sampler = engine_balanced_sampler_v1
model_seeds = 1, 2, 3, 4, 5
method = ocir
task = rul
```

RUL 标签和预测：

```text
使用原始 cycle 单位
不归一化到 [0, 1]
metrics 中不得乘以 125
训练 label 使用 capped RUL
official test final evaluation 同时输出 provided 和 capped 两个 views
```

## 4. 当前目录与模块

核心新增/改造目录：

```text
ocir_reproduction/
  data/
    cmapss_protocol.py
  protocols/
    main_protocol_v1.py
    main_protocol_v1.yaml
docs/
scripts/
pipelines/
src/
util_modules/
```

核心入口：

```text
scripts/validate_main_protocol.py
scripts/audit_main_protocol_alignment.py
scripts/run_main_protocol_experiments.py
scripts/smoke_experiment_runner_outputs.py
main.py
```

`main.py` 保留为单次开发入口；标准协议实验入口是：

```text
scripts/run_main_protocol_experiments.py
```

## 5. 数据契约和数据加载层

### 5.1 数据目录

模型侧期望的数据目录是：

```text
data/processed/main_protocol_v1/
  FD001/
    manifest.json
    split_seed2026.json
    train_windows.npz
    validation_view_v1.npz
    test_terminal_windows.npz
    full_test_windows.npz
  FD002/
  FD003/
  FD004/
```

### 5.2 数据加载入口

实现文件：

```text
ocir_reproduction/data/cmapss_protocol.py
```

主入口：

```python
load_main_protocol_cmapss(
    dataset,
    data_path,
    window_length,
    rul_cap,
    batch_size,
    split_seed,
    model_seed,
    logger=None,
)
```

返回：

```text
training_data
val_data
testing_data
full_test_set
```

模型侧只读取 processed 数据，不负责：

```text
读取 NASA raw txt
fit scaler
fit KMeans
生成 window
计算 raw 数据 hash
证明 train-only preprocessing 的 fit scope
```

这些由数据处理交付方负责，并由 `manifest.json`、`split_seed2026.json` 和审计脚本共同验收。

### 5.3 数据对象角色

| 对象 | 来源文件 | 用途 |
|---|---|---|
| `training_data` | `train_windows.npz` | RL/RUL 训练 |
| `val_data` | `validation_view_v1.npz` | RUL validation 和 best epoch selection |
| `testing_data` | `test_terminal_windows.npz` | final official test evaluation |
| `full_test_set` | `full_test_windows.npz` | 诊断和可视化，不参与 official test 主表指标 |

## 6. RUL Validation 修复

### 6.1 原问题

旧 RUL pipeline 的 validation 逻辑会在非 full-test 分支中使用 `self.testing_data`，导致 official test 泄漏风险。

禁止行为：

```text
训练阶段使用 self.testing_data
训练阶段使用 self.full_test_set
用 official test 选择 epoch
用 official test 做 early stopping
自动调用 official test evaluation
```

### 6.2 新行为

`RulPipeline.evaluate_validation()` 只使用：

```text
self.val_data
```

validation 指标：

```text
val_window_rmse
val_window_mae
val_raw_rmse
val_nasa_score
val_engine_rmse
```

checkpoint 选择：

```text
使用 val_engine_rmse 选择 best epoch
严格小于当前 best 时更新
tie 保留更早 epoch
训练结束后 restore best checkpoint
```

validation 频率：

```text
rul_validate_every_epochs
默认值 = 1
默认每个 epoch check 一次
```

## 7. engine_balanced_sampler_v1

### 7.1 目标

训练采样必须做到 engine-level 等权，而不是 window-level 等权。

如果某台 engine 有更多窗口，普通 `shuffle=True` 会让它在一个 epoch 中出现更多次。协议要求每台 train engine 在期望上具有相同总采样权重。

### 7.2 实现

实现文件：

```text
ocir_reproduction/data/cmapss_protocol.py
```

核心类：

```python
EngineBalancedSampler
```

每个 window 的权重：

```text
weights[i] = 1 / count(unit == unit[i])
```

因此每个 engine 的总权重相等：

```text
sum(weights for one unit) = 1
```

采样规则：

```text
replacement = True
num_samples = len(train_dataset)
epoch seed = model_seed + epoch
```

训练循环接入：

```text
RlPipeline.train() 每个 epoch 调用 self.set_training_sampler_epoch(epoch)
RulPipeline.train() 每个 epoch 调用 self.set_training_sampler_epoch(epoch)
```

`make_training_loader()` 已从 `shuffle=True` 改为：

```python
DataLoader(..., sampler=EngineBalancedSampler(...), shuffle=False)
```

## 8. Final Evaluation 两个 Views

### 8.1 主评估对象

final official test evaluation 只能使用：

```text
self.testing_data
```

其来源必须是：

```text
test_terminal_windows.npz
```

`full_test_set` / `full_test_windows.npz` 不参与 official test 主表指标。

### 8.2 两个 views

每个 official test unit 只前向一次，得到：

```text
raw_pred_rul
```

两个 views 从同一个 `raw_pred_rul` 派生。

`provided_rul_eval_v1`：

```text
target = provided_rul
prediction = max(raw_pred_rul, 0)
error = prediction - target
```

`capped_target_eval_v1`：

```text
target = capped_target_rul
prediction = clip(raw_pred_rul, 0, rul_cap)
error = prediction - target
```

指标：

```text
RMSE
MAE
NASA Score
```

NASA Score 按样本求和，不取均值。

### 8.3 输出

`evaluate_test_terminal()` 输出：

```text
predictions.csv
metrics.json
```

`predictions.csv` 每个 official test unit 一行，包含：

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

`metrics.json` 至少包含两个 views：

```text
evaluation_views.provided_rul_eval_v1
evaluation_views.capped_target_eval_v1
```

## 9. Official Test G4 Gate

### 9.1 目的

official test 必须保持 sealed，直到最终报告阶段。G4 gate 的目的不是改变指标，而是防止 official test 被无意或反复用作隐性 validation set。

### 9.2 强制规则

`RulPipeline.evaluate_test_terminal()` 必须收到 G4 approval payload，否则抛出：

```text
PermissionError
```

runner 非 dry-run 默认封存 official test。必须显式传入：

```text
--allow-official-test
--g4-approval-id
--g4-approved-by
```

可选：

```text
--g4-approval-note
```

dry-run 不需要 approval，因为 dry-run 不创建实验目录、不读数据、不实例化 pipeline、不访问 official test。

### 9.3 访问日志

在调用 final evaluation 前，runner 写入：

```text
reports/test_access_log.md
```

最低记录字段：

```text
timestamp_utc
exp_id
dataset
model_seed
method
protocol
approval_stage
approval_id
approved_by
reason
caller
output_dir
git_commit
git_dirty
approval_note
```

`reason` 固定为：

```text
final_evaluation_after_training_and_best_checkpoint_restore
```

## 10. 统一实验 Runner 和输出目录

### 10.1 Runner

标准 runner：

```text
scripts/run_main_protocol_experiments.py
```

默认展开：

```text
FD001 seed1
FD001 seed2
FD001 seed3
FD001 seed4
FD001 seed5
FD002 seed1
FD002 seed2
FD002 seed3
FD002 seed4
FD002 seed5
FD003 seed1
FD003 seed2
FD003 seed3
FD003 seed4
FD003 seed5
FD004 seed1
FD004 seed2
FD004 seed3
FD004 seed4
FD004 seed5
```

默认同一 split：

```text
split_seed = 2026
split_id = split_v1_seed2026_engine_80_20
```

dry-run：

```text
python -B scripts/run_main_protocol_experiments.py --dry-run
```

dry-run 行为：

```text
展开 20 个实验
打印 dataset, seed, exp_id, output_dir
不创建实验目录
不实例化 pipeline
不读取 processed data
不访问 official test
```

### 10.2 exp_id

格式：

```text
{timestamp_utc}_main_protocol_v1_ocir_rul_{dataset}_seed{model_seed}_split2026_{git_short}
```

示例：

```text
20260730T104303Z_main_protocol_v1_ocir_rul_FD001_seed1_split2026_8467170
```

### 10.3 标准输出目录

每个实验目录：

```text
results/experiments/{exp_id}/
  config.yaml
  runtime.json
  metrics.json
  predictions.csv
  train_log.csv
  checkpoint.pt
  figures/
    rl/
    rul/
  checkpoints/
    OCIR_cp.pth
    rul_cp.pth
  raw_logs/
    rl/
      training_log.csv
    rul/
      training_log.csv
  logs/
    log_all.txt
```

checkpoint 规则：

```text
source = checkpoints/rul_cp.pth
target = checkpoint.pt
```

`checkpoint.pt` 是 RUL predictor 的最终标准 checkpoint。

### 10.4 runtime 和 git 信息

`runtime.json` 记录：

```text
status
started_at_utc
ended_at_utc
duration_seconds
Python version
Python executable
torch version
CUDA version
CUDA available
GPU name
GPU memory
git commit
git short
git dirty
git status_short
argv
cwd
failure
```

runner 不自动 commit，不执行：

```text
git pull
git push
git clone
git fetch
```

只允许本地读取 git 状态。

## 11. 审计脚本

主审计脚本：

```text
scripts/audit_main_protocol_alignment.py
```

当前审计范围包括：

```text
main.py 默认 window
协议数据 loader 是否存在
协议数据文件读取和数组校验
engine_balanced_sampler_v1 是否接入
RL/RUL epoch sampler hook
RUL validation 是否只用 validation_view
RUL final evaluation 两个 views
official test isolation
旧 ut.RUL_metric 是否移除
experiment runner 是否存在
runner 是否有 5-seed grid、输出归档、G4 gate、dry-run
runner 是否不包含 remote git 操作
processed 数据目录是否存在
```

真实数据未交付时，FD001-FD004 processed 目录检查允许 skip。

## 12. 验收记录

### 12.1 协议检查

命令：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\validate_main_protocol.py
```

通过输出：

```text
main_protocol_v1 OK
datasets: FD001, FD002, FD003, FD004
window_length: 30
rul_cap: 125
split_seed: 2026
model_seeds: [1, 2, 3, 4, 5]
```

### 12.2 audit

命令：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\audit_main_protocol_alignment.py
```

最近通过状态：

```text
pass=17
warn=1
fail=0
skip=4
info=1
```

说明：

```text
warn=1: 仓库没有 scripts/rul/*.sh
skip=4: FD001-FD004 processed 数据目录尚未交付
```

### 12.3 runner smoke

命令：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\smoke_experiment_runner_outputs.py
```

通过输出：

```text
smoke_experiment_runner_outputs OK
grid=20
```

smoke 不实例化模型，不读取真实 processed data。

### 12.4 dry-run

命令：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\run_main_protocol_experiments.py --dry-run
```

结果：

```text
输出 20 个 dataset/seed 实验预览
不创建真实 exp 目录
不读数据
不访问 official test
```

### 12.5 G4 gate

检查内容：

```text
非 dry-run 且没有 --allow-official-test / --g4-approval-id / --g4-approved-by 时
runner 在训练、建目录、读数据前抛出 PermissionError
```

通过信息：

```text
G4 gate check OK: non-dry-run without approval is blocked before experiment creation
```


## 13. 修改全程时间线

### 阶段 1：建立工作仓库和环境

完成内容：

```text
克隆 cdfxianya/OCIR-Reproduction 作为工作仓库
确认 conda 环境 ocir-repro
改用 requirements.txt，不再使用 environment.yml
验证 Python、torch、CUDA、numpy、pandas、scikit-learn 等依赖
```

### 阶段 2：明确复用边界

决策：

```text
不严格逐行复现旧 OCIR
严格按最初 02/03 md 协议实现
只复用 OCIR 主过程代码
新建协议化数据层
数据处理交给外部负责人
模型侧只读取 processed protocol data
```

保留：

```text
src/
pipelines/
util_modules/
```

新增：

```text
ocir_reproduction/
docs/
scripts/
```

### 阶段 3：数据契约与 loader

完成内容：

```text
写 main_protocol_v1_data_contract.md
实现 ocir_reproduction.data.cmapss_protocol.load_main_protocol_cmapss
实现 manifest/split/npz 校验
实现 train/validation/test/full_test 数据对象
实现 processed-data audit
```


### 阶段 4：RUL validation 修复

完成内容：

```text
移除旧 self.testing_data validation 行为
训练期间只使用 validation_view_v1
实现 val_engine_rmse best checkpoint 选择
实现 rul_validate_every_epochs，默认每 epoch check 一次
恢复 best checkpoint 后保存 RUL 模型
```

### 阶段 5：RUL 标签单位确认

完成内容：

```text
确认 main_protocol_v1 下 y 和 raw prediction 都是 cycle 单位
不使用 [0, 1] 归一化
metric 中不乘以 125
旧 OCIR 中乘 125 的 RUL_metric 不作为本协议依据
```

### 阶段 6：engine_balanced_sampler_v1

完成内容：

```text
写 engine_balanced_sampler_v1_design.md
实现 EngineBalancedSampler
替换 training DataLoader shuffle=True
RL/RUL 每个 epoch 调用 set_training_sampler_epoch(epoch)
更新 audit
smoke 验证 engine 总权重相等、同 seed/epoch 可复现、不同 epoch 序列变化
```

### 阶段 7：final evaluation 两个 views

完成内容：

```text
写 final_evaluation_views_v1_design.md
实现 official test terminal final evaluation
实现 provided_rul_eval_v1
实现 capped_target_eval_v1
实现 predictions.csv
实现 metrics.json
确保 full_test_set 不参与主表指标
更新 audit
synthetic smoke 验证数值
```

### 阶段 8：runner 和输出归档

完成内容：

```text
写 experiment_runner_outputs_v1_design.md
实现 scripts/run_main_protocol_experiments.py
实现 scripts/smoke_experiment_runner_outputs.py
实现 results/experiments/{exp_id}
实现 config.yaml
实现 runtime.json
实现 train_log.csv 规整
实现 checkpoint.pt 复制
实现 runner_summary.json/csv
实现 dry-run 20-grid 展开
实现 git/runtime/GPU/time 记录机制
```

### 阶段 9：G4 official test gate

完成内容：

```text
为 evaluate_test_terminal 增加 access_approval 参数
未收到 G4 approval payload 时抛出 PermissionError
runner 增加 --allow-official-test
runner 增加 --g4-approval-id
runner 增加 --g4-approved-by
runner 增加 --g4-approval-note
访问 official test 前写 reports/test_access_log.md
更新文档、audit 和 smoke
验证无 approval 时非 dry-run 会被阻断
```

### 阶段 10：Git 恢复点

已创建commit：

```text
8467170 Add main protocol OCIR reproduction scaffold
2e264b5 Add G4 gate for official test evaluation
```

## 14. 当前未完成项

除等待数据交付之外，当前主路径尚未完成：

```text
真实 FD001 single seed 端到端运行
真实 FD001-FD004 × 5 seeds 完整运行
真实硬件/时间/显存统计结论
baselines 阶段是否进入和如何实现
```

需要等数据后执行：

```text
processed-data audit
FD001 single seed smoke
FD001-FD004 × 5 seeds
runner_summary 汇总验收
metrics/predictions 真实结果验收
```

