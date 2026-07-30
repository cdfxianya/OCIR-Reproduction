# experiment_runner_outputs_v1 技术文档

## 1. 目标

本文档规定 `experiment_runner_outputs_v1` 的实现方案：为 `main_protocol_v1` 增加统一实验 runner 和标准输出目录。

本任务完成后，仓库必须具备一个可审计的协议实验入口，能够按默认配置展开：

```text
datasets = FD001, FD002, FD003, FD004
model_seeds = 1, 2, 3, 4, 5
task = rul
method = ocir
protocol = main_protocol_v1
```

每个 dataset 和 seed 必须生成一个独立实验目录：

```text
results/experiments/{exp_id}
```

每个完整实验目录必须至少包含：

```text
config.yaml
runtime.json
metrics.json
predictions.csv
train_log.csv
checkpoint.pt
figures/
```

## 2. 非目标

本任务不做以下事项：

```text
不修改 docs/main_protocol_v1_data_contract.md
不重新定义 processed data 格式
不实现 baselines
不跑真实 FD001-FD004
不下载数据
不执行 git pull
不执行 git push
不执行 git clone
不自动 commit
不改变 OCIR 主模型结构
不改变 RUL validation 的 selection metric
不改变 final evaluation 两个 views 的语义
```

`main.py` 可以继续作为单次开发入口存在，不要求删除旧 `Logss/` 兼容逻辑。

## 3. 协议依据

最高依据是用户最开始指定的两个协议文档：

```text
H:\Users\zhang\Desktop\ml\OCIR-learning\02_cmapss_data_source_and_alignment(1).md
H:\Users\zhang\Desktop\ml\OCIR-learning\03_cmapss_research_problem_and_method(1).md
```

当前仓库内的直接依据是：

```text
ocir_reproduction/protocols/main_protocol_v1.py
ocir_reproduction/protocols/main_protocol_v1.yaml
docs/rul_validation_fix_design.md
docs/final_evaluation_views_v1_design.md
ocir_reproduction/data/cmapss_protocol.py
```

原 OCIR 源代码只作为主过程参考，不作为实验输出结构的最高依据。

## 4. 总体设计决策

必须新增脚本：

```text
scripts/run_main_protocol_experiments.py
```

该脚本是 `main_protocol_v1` 的标准实验 runner。

推荐不把标准 runner 塞进 `main.py`，原因如下：

```text
main.py 当前承担单次开发运行职责
标准 runner 需要管理 20 个实验、输出目录、运行审计、失败策略和最终评估
runner 需要拿到 RulPipeline 实例以调用 evaluate_test_terminal(output_dir=...)
直接复用 main.main(args, logger) 无法拿到 RulPipeline 实例
```

因此实现方式必须为：

```text
scripts/run_main_protocol_experiments.py 直接实例化 pipelines.RlPipeline 和 pipelines.RulPipeline
runner 不通过 main.main() 间接运行完整实验
runner 可以复用 main.py 的默认参数定义，但不能依赖 main.main() 返回 pipeline
```

`main.py` 不需要修改调用格式。实现代理如需减少重复参数定义，可以新增纯函数 `build_arg_parser()`，但这不是本任务的强制要求。

## 5. exp_id 严格规则

### 5.1 必须包含的信息

`exp_id` 必须同时包含以下字段：

```text
protocol
method
task
dataset
model_seed
split_seed
git commit short hash
timestamp
```

### 5.2 格式

`exp_id` 必须按以下格式生成：

```text
{timestamp_utc}_{protocol}_{method}_{task}_{dataset}_seed{model_seed}_split{split_seed}_{git_short}
```

示例：

```text
20260730T103012Z_main_protocol_v1_ocir_rul_FD001_seed1_split2026_a1b2c3d
```

### 5.3 字段规则

| 字段 | 规则 |
|---|---|
| `timestamp_utc` | UTC 时间，格式固定为 `YYYYMMDDTHHMMSSZ` |
| `protocol` | 固定为 `main_protocol_v1` |
| `method` | 固定为 `ocir` |
| `task` | 默认固定为 `rul` |
| `dataset` | 只能是 `FD001`、`FD002`、`FD003`、`FD004` |
| `model_seed` | 默认只能是 `1`、`2`、`3`、`4`、`5` |
| `split_seed` | 固定为 `2026` |
| `git_short` | 当前 commit 的前 7 位；如果仓库无 commit，则使用 `nogit` |

### 5.4 字符约束

`exp_id` 必须匹配：

```text
^[0-9]{8}T[0-9]{6}Z_main_protocol_v1_ocir_rul_FD00[1-4]_seed[1-5]_split2026_([0-9a-f]{7}|nogit)$
```

### 5.5 覆盖策略

runner 创建实验目录前必须检查：

```text
results/experiments/{exp_id}
```

如果目录已经存在，必须抛出 `FileExistsError`，不得覆盖。

本协议不提供 `--overwrite`。重复实验必须生成新的 timestamp，从而生成新的 `exp_id`。

## 6. 标准输出目录树

每个完整实验目录必须是：

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

其中：

| 路径 | 必需 | 含义 |
|---|---:|---|
| `config.yaml` | 是 | 运行前固化的实验配置 |
| `runtime.json` | 是 | 运行环境、git 状态、开始结束时间、失败信息 |
| `metrics.json` | 是 | final evaluation 两个 views 的指标和实验元数据 |
| `predictions.csv` | 是 | official test terminal 逐 engine 预测 |
| `train_log.csv` | 是 | 标准化训练日志 |
| `checkpoint.pt` | 是 | 标准命名的最终 RUL checkpoint |
| `figures/` | 是 | 图像输出根目录 |
| `checkpoints/` | 是 | 旧代码保存的原始 checkpoint |
| `raw_logs/` | 是 | 旧 Logger 产生的原始 CSV |
| `logs/log_all.txt` | 是 | Python logger 文本日志 |

### 6.1 runtime 信息的明确选择

运行环境信息必须写入单独文件：

```text
runtime.json
```

不把完整 runtime 塞进 `config.yaml`，原因是：

```text
config.yaml 表示计划运行配置，运行开始前即可写入
runtime.json 表示实际运行结果，包含开始时间、结束时间、耗时、异常和硬件信息
```

`metrics.json` 可以包含少量关键 runtime 引用字段，例如 `exp_id`、`git_commit`、`git_dirty`、`duration_seconds`，但完整运行环境必须以 `runtime.json` 为准。

## 7. config.yaml schema

`config.yaml` 必须使用 UTF-8 写入。

最低 schema：

```yaml
exp_id: "20260730T103012Z_main_protocol_v1_ocir_rul_FD001_seed1_split2026_a1b2c3d"
protocol:
  version: "main_protocol_v1"
  split_id: "split_v1_seed2026_engine_80_20"
  split_seed: 2026
  validation_view: "validation_view_v1"
  train_sampler: "engine_balanced_sampler_v1"
  window_length: 30
  rul_cap: 125
  label_scale: "cycle"
  label_normalization: "none"
experiment:
  method: "ocir"
  task: "rul"
  dataset: "FD001"
  model_seed: 1
  data_path: "./data/processed/main_protocol_v1"
  output_dir: "results/experiments/{exp_id}"
training:
  n_epochs: 50
  rul_epochs: 50
  batch: 128
  lr_: 0.0005
  rul_lr: 0.0005
  rul_validate_every_epochs: 1
model:
  dx: 14
  dz: 20
  dc: 6
  d_model: 128
  num_heads: 2
  encoder_E: "transformer"
  z_projection: "spc"
  D_projection: "spc"
  c_type: "continuous"
git:
  commit: "a1b2c3d4..."
  short: "a1b2c3d"
  dirty: true
  status_short: [" M pipelines/rul_estimation.py"]
command:
  argv: ["python", "-B", "scripts/run_main_protocol_experiments.py"]
```

字段约束：

```text
protocol.version 必须是 main_protocol_v1
experiment.method 必须是 ocir
experiment.task 必须是 rul
experiment.dataset 必须属于 FD001 到 FD004
experiment.model_seed 必须属于 1 到 5
protocol.split_seed 必须是 2026
protocol.window_length 必须是 30
protocol.rul_cap 必须是 125
model.dx 必须是 14
```

## 8. runtime.json schema

`runtime.json` 必须使用 UTF-8、`ensure_ascii=False`、`indent=2` 写入。

最低 schema：

```json
{
  "exp_id": "20260730T103012Z_main_protocol_v1_ocir_rul_FD001_seed1_split2026_a1b2c3d",
  "status": "success",
  "started_at_utc": "2026-07-30T10:30:12Z",
  "ended_at_utc": "2026-07-30T11:15:42Z",
  "duration_seconds": 2730.0,
  "python": {
    "version": "3.9.25",
    "executable": "H:\\Conda\\envs\\ocir-repro\\python.exe"
  },
  "torch": {
    "version": "2.4.0+cu121",
    "cuda_version": "12.1",
    "cuda_available": true
  },
  "gpu": {
    "device_arg": "0",
    "selected_device": "cuda:0",
    "name": "NVIDIA GeForce RTX 4060 Ti",
    "count": 1,
    "memory_allocated_bytes": 0,
    "memory_reserved_bytes": 0,
    "max_memory_allocated_bytes": 0
  },
  "git": {
    "commit": "a1b2c3d4...",
    "short": "a1b2c3d",
    "dirty": true,
    "status_short": [" M pipelines/rul_estimation.py"]
  },
  "command": {
    "argv": ["python", "-B", "scripts/run_main_protocol_experiments.py"],
    "cwd": "H:\\Users\\zhang\\Desktop\\ml\\OCIR-Reproduction"
  },
  "failure": null
}
```

失败时：

```text
status 必须是 failed
ended_at_utc 必须写入
duration_seconds 必须写入
failure 必须是 object
failure.type 必须是异常类名
failure.message 必须是异常消息
failure.traceback 必须是字符串
```

## 9. metrics.json 归档规则

`RulPipeline.evaluate_test_terminal(output_dir=exp_dir, method="ocir", write_files=True)` 已经负责写入：

```text
predictions.csv
metrics.json
```

runner 必须在 final evaluation 写出 `metrics.json` 后，将实验级字段合并进 `metrics.json`。

合并后 `metrics.json` 最低额外字段：

```json
{
  "exp_id": "20260730T103012Z_main_protocol_v1_ocir_rul_FD001_seed1_split2026_a1b2c3d",
  "output_dir": "results/experiments/{exp_id}",
  "git_commit": "a1b2c3d4...",
  "git_short": "a1b2c3d",
  "git_dirty": true,
  "duration_seconds": 2730.0,
  "best_rul_epoch": 17,
  "best_val_engine_rmse": 12.34
}
```

不得删除或改写 final evaluation 已定义的核心字段：

```text
evaluation_views.provided_rul_eval_v1
evaluation_views.capped_target_eval_v1
num_test_units
dataset
model_seed
split_id
split_seed
window_length
rul_cap
label_scale
label_normalization
```

## 10. predictions.csv 归档规则

`predictions.csv` 由 `evaluate_test_terminal()` 生成。

runner 不得重新计算 prediction，不得重新调用模型生成第二套 official test prediction。

runner 只允许做以下操作：

```text
检查文件存在
检查列集合和列顺序
检查行数大于 0
检查 dataset/model_seed/protocol 字段与 config.yaml 一致
```

如果检查失败，当前实验必须标记为 failed。

## 11. train_log.csv 最低 schema

标准顶层文件：

```text
results/experiments/{exp_id}/train_log.csv
```

必须由 runner 从原始训练日志规整得到，不要求大改 pipeline。

原始日志来源：

```text
results/experiments/{exp_id}/raw_logs/rl/training_log.csv
results/experiments/{exp_id}/raw_logs/rul/training_log.csv
```

标准 `train_log.csv` 必须使用宽表 schema：

```text
stage,epoch,iteration,loss_rec,loss_kl,loss_klc,loss_realD,loss_fakeD,loss_gen,loss_Q,loss_ccz,loss_ccc,loss_mse
```

字段含义：

| 字段 | 含义 |
|---|---|
| `stage` | `RL` 或 `RUL` |
| `epoch` | 1-based epoch |
| `iteration` | batch iteration index，沿用现有 logger 的 `itr` |
| `loss_rec` | RL reconstruction loss，RUL 行为空 |
| `loss_kl` | RL KL loss，RUL 行为空 |
| `loss_klc` | RL condition KL loss，RUL 行为空 |
| `loss_realD` | RL discriminator real loss，RUL 行为空 |
| `loss_fakeD` | RL discriminator fake loss，RUL 行为空 |
| `loss_gen` | RL generator loss，RUL 行为空 |
| `loss_Q` | RL Q loss，RUL 行为空 |
| `loss_ccz` | RL condition consistency z loss，RUL 行为空 |
| `loss_ccc` | RL condition consistency c loss，RUL 行为空 |
| `loss_mse` | RUL MSE loss，RL 行为空 |

规整规则：

```text
如果 RL raw log 存在，则读取并写 stage=RL
如果 RUL raw log 存在，则读取并写 stage=RUL
缺失的 metric 列必须写空字符串
不得把 RL 和 RUL 的不同 header 直接拼接成顶层 train_log.csv
如果某个 raw log 缺失，当前实验必须标记 failed
```

## 12. checkpoint 保存和复制规则

当前旧保存函数为：

```python
ut.save_model(model, path_=self.model_save_path, name="OCIR")
ut.save_model(model, path_=self.model_save_path, name="rul")
```

实际文件名为：

```text
OCIR_cp.pth
rul_cp.pth
```

runner 必须把 pipeline 的 `model_save_path` 设置为：

```text
results/experiments/{exp_id}/checkpoints
```

训练完成后必须存在：

```text
results/experiments/{exp_id}/checkpoints/OCIR_cp.pth
results/experiments/{exp_id}/checkpoints/rul_cp.pth
```

标准顶层 checkpoint 必须是 RUL predictor 的最终 checkpoint：

```text
results/experiments/{exp_id}/checkpoint.pt
```

复制规则：

```text
source = checkpoints/rul_cp.pth
target = checkpoint.pt
```

`checkpoint.pt` 必须在 RUL training 完成、`restore_best_checkpoint()` 已执行、`ut.save_model(..., name="rul")` 已写出后复制。

如果 `checkpoints/rul_cp.pth` 不存在，当前实验必须标记为 failed。

## 13. 5-seed runner 行为

### 13.1 默认展开

不提供任何 dataset 或 seed 参数时，runner 必须展开 20 个实验：

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

### 13.2 同一 split

所有 seed 必须使用同一 split：

```text
split_seed = 2026
split_id = split_v1_seed2026_engine_80_20
```

runner 不得按 model seed 重新划分 train/validation units。

### 13.3 可选筛选

runner 可以提供参数缩小运行范围：

```text
--datasets FD001 FD003
--model-seeds 1 2
```

但默认值必须来自 `MAIN_PROTOCOL_V1`：

```text
MAIN_PROTOCOL_V1["datasets"]
MAIN_PROTOCOL_V1["training"]["model_seeds"]
```

### 13.4 失败策略

默认失败策略：

```text
一个实验失败后，记录失败 runtime.json，然后继续后续实验
runner 最终退出码为 1，只要任意实验失败
```

可选参数：

```text
--fail-fast
```

启用后：

```text
第一个实验失败后立即停止后续实验
当前失败实验必须写 runtime.json
runner 退出码为 1
```

### 13.5 汇总文件

runner 结束后必须写：

```text
results/experiments/runner_summary.json
results/experiments/runner_summary.csv
```

最低汇总字段：

```text
exp_id,dataset,model_seed,status,output_dir,started_at_utc,ended_at_utc,duration_seconds,error_type,error_message
```

## 14. official test 调用规则

训练阶段禁止访问 official test。

runner 必须按以下顺序执行单个完整实验：

```text
1. 创建 exp_dir
2. 写 config.yaml
3. 写 runtime.json 初始状态 running
4. 运行 RL pipeline
5. 运行 RUL pipeline
6. RUL pipeline 内部完成 validation best checkpoint restore
7. 确认 checkpoints/rul_cp.pth 存在
8. 复制 checkpoints/rul_cp.pth 到 checkpoint.pt
9. 记录 final test access log
10. 调用 rul_pipeline.evaluate_test_terminal(output_dir=exp_dir, method="ocir", write_files=True)
11. 合并 metrics.json 的实验级元数据
12. 规整 train_log.csv
13. 写 runtime.json 最终状态 success
```

第 10 步之前不得调用：

```text
self.testing_data
self.full_test_set
evaluate_test_terminal()
```

### 14.1 G4 approval gate

因为本 runner 的完整输出要求包含 `metrics.json` 和 `predictions.csv`，而这两个文件来自 official test terminal evaluation，所以 official test 默认必须处于 sealed 状态。

非 dry-run 运行必须显式提供：

```text
--allow-official-test
--g4-approval-id
--g4-approved-by
```

可选提供：

```text
--g4-approval-note
```

如果没有提供完整 G4 approval 参数，runner 必须在创建实验目录、训练模型或读取 processed data 之前失败。

`--dry-run` 不需要 G4 approval，因为 dry-run 不创建实验目录、不实例化 pipeline、不读取 processed data、不访问 official test。

### 14.2 final test access log

在调用 `evaluate_test_terminal()` 前，runner 必须向 `reports/test_access_log.md` 追加一条访问记录。

访问记录最低字段：

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

## 15. 新增文件和函数

### 15.1 新增 `scripts/run_main_protocol_experiments.py`

用途：

```text
统一运行 main_protocol_v1 的 OCIR RUL experiments，生成标准输出目录和审计文件。
```

#### `parse_args`

函数框架：

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ...
```

调用格式：

```python
args = parse_args()
```

返回值：

```text
argparse.Namespace
```

必须支持的参数：

```text
--data-path
--results-root
--datasets
--model-seeds
--gpu-dev
--n-epochs
--rul-epochs
--batch
--fail-fast
--dry-run
```

默认值必须与 `MAIN_PROTOCOL_V1` 或 `main.py` 当前默认值一致。

#### `build_base_config`

函数框架：

```python
def build_base_config(args: argparse.Namespace) -> dict:
    ...
```

调用格式：

```python
base_config = build_base_config(args)
```

返回值：

```text
dict，可直接传给 pipelines.RlPipeline / pipelines.RulPipeline
```

目的：

```text
生成与 main.py 单次运行等价的基础参数，但不创建旧 Logss 目录。
```

必须包含当前 pipeline 需要的全部字段，例如：

```text
task,dataset,c_type,net,valid_split,seed,gpu_dev,n_epochs,batch,patience,
rul_epochs,rul_validate_every_epochs,lr_,rul_lr,scheduler,alpha,kl_annealing,
warm_up,start_lr,ref_lr,final_lr,start_wd,final_wd,model_save_path,
plots_save_path,his_save_path,window,H,hyper_lookback,time_embedding,c_kl,
dx,dz,dc,d_model,num_heads,encoder_E,z_projection,D_projection,
c_posterior_param,extractor,conditional,data_path,description,id_,log_path
```

#### `expand_experiment_grid`

函数框架：

```python
def expand_experiment_grid(args: argparse.Namespace) -> list[dict]:
    ...
```

调用格式：

```python
grid = expand_experiment_grid(args)
```

返回值：

```text
list[dict]，每个 dict 至少包含 dataset 和 model_seed
```

目的：

```text
按照 datasets 和 model_seeds 展开实验组合，默认生成 20 个实验。
```

顺序必须固定为：

```text
先 dataset，后 seed
```

#### `collect_git_info`

函数框架：

```python
def collect_git_info(repo_root: pathlib.Path) -> dict:
    ...
```

调用格式：

```python
git_info = collect_git_info(repo_root)
```

返回值：

```text
dict，包含 commit、short、dirty、status_short
```

目的：

```text
记录当前代码状态，不自动 commit。
```

实现约束：

```text
只能执行本地 git 命令
不得执行 git pull
不得执行 git push
不得执行 git fetch
不得执行 git clone
```

如果当前目录不是 git 仓库：

```text
commit = null
short = "nogit"
dirty = null
status_short = []
```

#### `build_exp_id`

函数框架：

```python
def build_exp_id(
    dataset: str,
    model_seed: int,
    git_info: dict,
    timestamp_utc: datetime.datetime,
) -> str:
    ...
```

调用格式：

```python
exp_id = build_exp_id(dataset, model_seed, git_info, timestamp_utc)
```

返回值：

```text
str
```

目的：

```text
按第 5 节规则生成唯一、可读、可审计的实验 ID。
```

#### `setup_experiment_dirs`

函数框架：

```python
def setup_experiment_dirs(results_root: pathlib.Path, exp_id: str) -> dict[str, pathlib.Path]:
    ...
```

调用格式：

```python
paths = setup_experiment_dirs(results_root, exp_id)
```

返回值：

```text
dict[str, pathlib.Path]
```

必须返回的 keys：

```text
exp_dir
figures_dir
figures_rl_dir
figures_rul_dir
checkpoints_dir
raw_logs_dir
raw_logs_rl_dir
raw_logs_rul_dir
logs_dir
```

目的：

```text
创建标准目录树，并阻止覆盖已有 exp_dir。
```

#### `make_experiment_config`

函数框架：

```python
def make_experiment_config(
    base_config: dict,
    dataset: str,
    model_seed: int,
    exp_id: str,
    paths: dict[str, pathlib.Path],
    git_info: dict,
    argv: list[str],
) -> dict:
    ...
```

调用格式：

```python
config_payload = make_experiment_config(...)
```

返回值：

```text
dict，用于写 config.yaml
```

目的：

```text
固化单个实验的计划配置。
```

#### `write_yaml`

函数框架：

```python
def write_yaml(path: pathlib.Path, payload: dict) -> None:
    ...
```

调用格式：

```python
write_yaml(paths["exp_dir"] / "config.yaml", config_payload)
```

返回值：

```text
None
```

目的：

```text
以 UTF-8 写出 config.yaml。
```

#### `write_json`

函数框架：

```python
def write_json(path: pathlib.Path, payload: dict) -> None:
    ...
```

调用格式：

```python
write_json(paths["exp_dir"] / "runtime.json", runtime_payload)
```

返回值：

```text
None
```

目的：

```text
以 UTF-8、ensure_ascii=False、indent=2 写出 JSON。
```

#### `collect_runtime_info`

函数框架：

```python
def collect_runtime_info(
    exp_id: str,
    status: str,
    started_at_utc: str,
    git_info: dict,
    argv: list[str],
    cwd: pathlib.Path,
    failure: dict | None = None,
) -> dict:
    ...
```

调用格式：

```python
runtime_payload = collect_runtime_info(...)
```

返回值：

```text
dict
```

目的：

```text
收集 Python、torch、CUDA、GPU、git、命令行和失败信息。
```

#### `setup_file_logger`

函数框架：

```python
def setup_file_logger(log_path: pathlib.Path, logger_name: str) -> logging.Logger:
    ...
```

调用格式：

```python
logger = setup_file_logger(paths["logs_dir"] / "log_all.txt", exp_id)
```

返回值：

```text
logging.Logger
```

目的：

```text
为单个实验创建独立 logger。
```

实现约束：

```text
不得让不同实验共享同一个 FileHandler
重复调用时必须避免重复添加 handler
```

#### `build_stage_config`

函数框架：

```python
def build_stage_config(base_config: dict, paths: dict[str, pathlib.Path], stage: str) -> dict:
    ...
```

调用格式：

```python
rl_config = build_stage_config(run_config, paths, stage="RL")
rul_config = build_stage_config(run_config, paths, stage="RUL")
```

返回值：

```text
dict
```

目的：

```text
为 RL 和 RUL 分别设置 plots_save_path 和 his_save_path，避免两个 stage 写入同一个 raw training_log.csv。
```

路径规则：

```text
RL plots_save_path = figures/rl
RL his_save_path = raw_logs/rl
RUL plots_save_path = figures/rul
RUL his_save_path = raw_logs/rul
model_save_path = checkpoints
```

#### `run_single_experiment`

函数框架：

```python
def run_single_experiment(
    base_config: dict,
    dataset: str,
    model_seed: int,
    results_root: pathlib.Path,
    git_info: dict,
    argv: list[str],
    dry_run: bool = False,
) -> dict:
    ...
```

调用格式：

```python
summary_row = run_single_experiment(...)
```

返回值：

```text
dict，单个实验的 summary row
```

目的：

```text
执行单个 dataset/seed 实验，生成标准输出目录。
```

正常运行必须直接调用：

```python
rl_pipeline = pipelines.RlPipeline(rl_config, logger)
encoder = rl_pipeline()
rul_pipeline = pipelines.RulPipeline(rul_config, logger, encoder, None)
rul_pipeline()
```

随后必须调用：

```python
rul_pipeline.evaluate_test_terminal(
    output_dir=paths["exp_dir"],
    method="ocir",
    write_files=True,
)
```

#### `copy_standard_checkpoint`

函数框架：

```python
def copy_standard_checkpoint(paths: dict[str, pathlib.Path]) -> pathlib.Path:
    ...
```

调用格式：

```python
checkpoint_path = copy_standard_checkpoint(paths)
```

返回值：

```text
pathlib.Path，指向 checkpoint.pt
```

目的：

```text
把 checkpoints/rul_cp.pth 复制为顶层 checkpoint.pt。
```

#### `normalize_train_logs`

函数框架：

```python
def normalize_train_logs(paths: dict[str, pathlib.Path]) -> pathlib.Path:
    ...
```

调用格式：

```python
train_log_path = normalize_train_logs(paths)
```

返回值：

```text
pathlib.Path，指向 train_log.csv
```

目的：

```text
把 RL 和 RUL 的原始 training_log.csv 转换为标准顶层 train_log.csv。
```

#### `record_final_test_access`

函数框架：

```python
def record_final_test_access(
    repo_root: pathlib.Path,
    exp_id: str,
    dataset: str,
    model_seed: int,
    git_info: dict,
    output_dir: pathlib.Path,
    access_approval: dict,
) -> pathlib.Path:
    ...
```

调用格式：

```python
access_log_path = record_final_test_access(...)
```

返回值：

```text
pathlib.Path，指向 reports/test_access_log.md
```

目的：

```text
在 official test final evaluation 前追加 G4 approval 审计记录。
```

#### `validate_experiment_outputs`

函数框架：

```python
def validate_experiment_outputs(paths: dict[str, pathlib.Path], config_payload: dict) -> None:
    ...
```

调用格式：

```python
validate_experiment_outputs(paths, config_payload)
```

返回值：

```text
None
```

目的：

```text
检查标准输出目录是否完整，检查 metrics/predictions/train_log/checkpoint/config/runtime 的最低 schema。
```

#### `run_experiment_grid`

函数框架：

```python
def run_experiment_grid(args: argparse.Namespace) -> int:
    ...
```

调用格式：

```python
exit_code = run_experiment_grid(args)
```

返回值：

```text
int，0 表示全部成功，1 表示至少一个实验失败
```

目的：

```text
按固定顺序执行全部 dataset/seed 组合，写 runner_summary.json 和 runner_summary.csv。
```

### 15.2 新增 `scripts/smoke_experiment_runner_outputs.py`

用途：

```text
在不依赖真实 FD001-FD004 processed 数据的情况下，验证 runner 的目录、schema、归档函数和 grid 展开。
```

#### `main`

函数框架：

```python
def main() -> int:
    ...
```

调用格式：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\smoke_experiment_runner_outputs.py
```

返回值：

```text
0 表示 smoke 通过，非 0 表示失败
```

目的：

```text
构造 fake legacy checkpoint、fake raw logs、fake metrics.json、fake predictions.csv，调用 runner 的归档和校验函数。
```

该 smoke 不得实例化真实模型，不得读取真实 processed data。

## 16. 修改内容

### 16.1 `scripts/audit_main_protocol_alignment.py`

必须新增静态检查：

```text
存在 scripts/run_main_protocol_experiments.py
runner 引用 MAIN_PROTOCOL_V1["datasets"]
runner 引用 MAIN_PROTOCOL_V1["training"]["model_seeds"]
runner 定义 build_exp_id
runner 定义 run_single_experiment
runner 定义 normalize_train_logs
runner 定义 copy_standard_checkpoint
runner 定义 collect_git_info
runner 定义 collect_runtime_info
runner 调用 RulPipeline.evaluate_test_terminal
runner 调用 record_final_test_access
runner 写 config.yaml
runner 写 runtime.json
runner 写 train_log.csv
runner 写 checkpoint.pt
runner 写 runner_summary.json
runner 写 runner_summary.csv
runner 不包含 git pull
runner 不包含 git push
runner 不包含 git clone
```

audit 不需要运行真实训练。

### 16.2 `ocir_protocol_alignment_checklist.md`

实现完成并验收后，允许把以下项标为 checked：

```text
5 seeds
输出目录
Git commit 记录
硬件/时间审计 的记录机制部分
Final test access log 与 G4 approval gate
```

不得因为本任务把以下项标为 checked：

```text
processed 数据包
FD001-FD004 真实数据验收
Baselines
硬件/时间审计 的真实 20-run 结果
```

## 17. 删除内容

本任务不要求删除任何文件或函数。

实现前必须检查：

```text
rg -n "run_main_protocol_experiments|runner_summary|checkpoint.pt|train_log.csv" .
```

如果已有同名脚本或同名函数，必须先阅读并复用或扩展，不得盲目新增重复入口。

不得删除：

```text
main.py
docs/main_protocol_v1_data_contract.md
ocir_protocol_alignment_checklist.md
```

## 18. dry-run 行为

runner 必须支持：

```text
--dry-run
```

`--dry-run` 行为：

```text
展开实验 grid
生成将要使用的 exp_id 预览
不创建实验目录
不实例化 pipeline
不读取 processed data
不访问 official test
不写 metrics.json
不写 predictions.csv
退出码为 0
```

dry-run 输出必须包含：

```text
dataset
model_seed
exp_id
output_dir
```

dry-run 用于确认 5 seeds 展开逻辑，不用于验收标准输出目录。

## 19. smoke test 设计

必须新增或执行不依赖真实数据的 smoke test。

推荐新增：

```text
scripts/smoke_experiment_runner_outputs.py
```

smoke 必须验证：

```text
默认 grid 展开为 20 个实验
FD001 到 FD004 顺序正确
每个 dataset 内 seed 顺序为 1 到 5
exp_id 匹配第 5 节 regex
setup_experiment_dirs 会阻止覆盖
fake checkpoints/rul_cp.pth 可复制为 checkpoint.pt
fake raw_logs/rl/training_log.csv 和 raw_logs/rul/training_log.csv 可合并为标准 train_log.csv
fake metrics.json 可合并 exp_id/git/runtime 字段
fake predictions.csv schema 可通过 validate_experiment_outputs
runtime.json success schema 可通过 validate_experiment_outputs
```

smoke fixture 可以写入：

```text
data/processed/_smoke_experiment_runner_outputs/
```

或：

```text
results/experiments/_smoke_experiment_runner_outputs/
```

该路径已被 `.gitignore` 规则覆盖时，不应提交 fixture 产物。

smoke 不得调用：

```text
pipelines.RlPipeline
pipelines.RulPipeline
load_main_protocol_cmapss
evaluate_test_terminal
```

## 20. 验收命令

实现代理必须在不联网、不下载的前提下运行：

```text
H:\Conda\envs\ocir-repro\python.exe -B scripts\validate_main_protocol.py
H:\Conda\envs\ocir-repro\python.exe -B scripts\audit_main_protocol_alignment.py
H:\Conda\envs\ocir-repro\python.exe -B scripts\smoke_experiment_runner_outputs.py
H:\Conda\envs\ocir-repro\python.exe -B scripts\run_main_protocol_experiments.py --dry-run
```

如果 Windows 权限导致 `py_compile` 写 `__pycache__` 失败，使用 AST 检查：

```text
H:\Conda\envs\ocir-repro\python.exe -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py') if '.git' not in p.parts]; print('AST syntax OK')"
```

通过标准：

```text
validate_main_protocol.py 输出 main_protocol_v1 OK
audit_main_protocol_alignment.py 输出 fail=0
smoke_experiment_runner_outputs.py 退出码为 0
run_main_protocol_experiments.py --dry-run 展开 20 个实验
dry-run 不创建 results/experiments/{exp_id}
无下载、无联网、无 git pull/push/clone
docs/main_protocol_v1_data_contract.md 无 diff
```

## 21. 实现顺序

实现代理必须按以下顺序执行：

```text
1. 检查是否已有 runner 相关入口
2. 新增 scripts/run_main_protocol_experiments.py 的纯函数骨架
3. 实现 exp_id、grid、git、runtime、目录创建、yaml/json 写入
4. 实现 dry-run
5. 实现单实验 pipeline 调用，但先不要跑真实数据
6. 实现 checkpoint 复制
7. 实现 train_log 规整
8. 实现 final test access G4 approval gate 和访问记录
9. 实现 metrics.json 元数据合并和输出校验
10. 新增 scripts/smoke_experiment_runner_outputs.py
11. 更新 audit 静态检查
12. 更新 ocir_protocol_alignment_checklist.md
13. 运行验收命令
```

## 22. Obsidian 兼容要求

本文档和后续修改的 Markdown 必须满足：

```text
UTF-8 编码
不要在 Markdown 表格里放 display math
不要使用 legacy inline math delimiter 作为公式分隔符
不要使用 legacy display math delimiter 作为公式分隔符
display math 如需使用，必须用独立行 $$
```

本任务文档不需要 display math。

## 23. 实现前需要用户确认的问题

无必须确认问题。

本文档已明确：为了让完整 runner 生成 `metrics.json` 和 `predictions.csv`，实现必须在 final evaluation 前验证 G4 approval 并写入 `reports/test_access_log.md` 记录。
