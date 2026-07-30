# main_protocol_v1 数据交付技术规范

本文档定义 OCIR-Reproduction 项目需要的数据处理交付物。数据处理负责人只需要按本文档生成数据包；模型侧会通过 `load_main_protocol_cmapss(...)` 读取这些文件。

本文档中的要求均为强制要求，除非明确标为“可选”。

## 1. 协议版本

数据包必须声明并遵守：

```text
main_protocol_version = main_protocol_v1
```

适用数据集：

```text
FD001, FD002, FD003, FD004
```

主协议固定参数：

```text
window_length = 30
stride = 1
rul_cap = 125
split_seed = 2026
selected_sensors = s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21
```

## 2. 原始数据要求

原始 C-MAPSS 文件必须来自 NASA 官方数据包。公开代码仓库不得包含原始 NASA 数据文件。

每个数据集必须使用以下三个文件：

```text
train_FD001.txt, test_FD001.txt, RUL_FD001.txt
train_FD002.txt, test_FD002.txt, RUL_FD002.txt
train_FD003.txt, test_FD003.txt, RUL_FD003.txt
train_FD004.txt, test_FD004.txt, RUL_FD004.txt
```

`train_FDxxx.txt` 和 `test_FDxxx.txt` 每行必须有 26 列：

```text
unit, cycle, op1, op2, op3, s1, s2, ..., s21
```

必须校验：

```text
每个 train/test 文件存在
每个 train/test 文件每行 26 列
每个 RUL 文件存在
test unit 数量等于 RUL 文件行数
test unit 按 unit ID 升序与 RUL 文件逐行对应
```

## 3. 输出目录结构

数据处理结果必须输出到：

```text
data/processed/main_protocol_v1/
```

目录结构必须为：

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
    manifest.json
    split_seed2026.json
    train_windows.npz
    validation_view_v1.npz
    test_terminal_windows.npz
    full_test_windows.npz
  FD003/
    manifest.json
    split_seed2026.json
    train_windows.npz
    validation_view_v1.npz
    test_terminal_windows.npz
    full_test_windows.npz
  FD004/
    manifest.json
    split_seed2026.json
    train_windows.npz
    validation_view_v1.npz
    test_terminal_windows.npz
    full_test_windows.npz
```

## 4. 统一数组规范

所有 `.npz` 文件必须使用 NumPy `np.savez_compressed` 或等价方式保存。

所有数值数组必须满足：

```text
x dtype = float32
y dtype = float32
ocs dtype = float32
tidx dtype = int64
unit dtype = int64
end_cycle dtype = int64
```

统一 shape 约定：

```text
x:    [N, 30, 14]
y:    [N, 1]
ocs:  [N, 30, d_ocs]
tidx: [N, 1]
```

其中：

```text
N = 当前文件中的窗口样本数
d_ocs = 1
```

`source_partition` 不要求存为 `.npz` 字符串数组；它必须写入对应 `manifest.json`，并由文件名唯一确定。

## 5. 模型输入 x

`x` 必须是 classic 14 sensors 的窗口：

```text
s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21
```

顺序必须严格等于上面的顺序。

`x` 必须经过 train-only global z-score 标准化：

```text
mean, std 只能由 split 后的 train units 计算
validation/test 只能 transform，不能重新 fit
```

标准化必须只作用于 selected 14 sensors，不得把 `op1-op3` 拼入 OCIR 主模型输入。

最终模型输入维度必须为：

```text
x.shape[-1] = 14
```

## 6. 操作设置与 ocs

OCIR 主模型不直接接收 `op1-op3`，但数据处理必须保留操作设置审计信息。

每个数据集的 `manifest.json` 必须记录：

```text
preserve_operating_settings = true
operating_settings = [op1, op2, op3]
op_preprocessing = train_only_op_z_score
```

`ocs` 是给 OCIR 分析和可视化使用的工况标签窗口，不作为 OCIR 主模型输入。

`ocs` 必须满足：

```text
ocs.shape = [N, 30, 1]
```

FD001 和 FD003：

```text
ocs 全部填 0.0
```

FD002 和 FD004：

```text
先用 split 后的 train units 的 op1-op3 拟合 StandardScaler
再用缩放后的 train op1-op3 拟合 KMeans(k=6)
train/validation/test 都只能使用该 KMeans 的 predict 结果
不得在 validation/test 上 fit 或 fit_predict
ocs 中保存每个 cycle 的 regime id，取值为 0 到 5，dtype 保存为 float32
```

## 7. tidx

`tidx` 表示窗口最后一个 cycle 的时间索引。

要求：

```text
tidx.shape = [N, 1]
tidx dtype = int64
tidx[i, 0] = 当前窗口最后一行对应的原始 cycle
```

cycle 使用 C-MAPSS 原始 cycle 编号，从 1 开始，不改为 0-based。

## 8. Unit key

任何样本的唯一发动机身份必须由三元组定义：

```text
dataset, source_partition, unit
```

`.npz` 文件中至少必须保存：

```text
unit: [N]
```

`dataset` 和 `source_partition` 由目录名和文件名确定。

禁止把 train unit 1 与 test unit 1 当成同一台发动机。

## 9. Train/Validation split

只允许在 `train_FDxxx.txt` 内部按 engine unit 做 train/validation 划分。

固定规则：

```text
split_seed = 2026
train_fraction = 0.8
validation_fraction = 0.2
split_by = engine unit
```

划分必须满足：

```text
train units 与 validation units 不重叠
同一 engine 的所有 cycle 只能出现在一个 split 中
所有 model seeds 共用同一个 split
```

必须保存：

```text
split_seed2026.json
```

文件格式必须为：

```json
{
  "main_protocol_version": "main_protocol_v1",
  "dataset": "FD001",
  "split_seed": 2026,
  "split_by": "engine_unit",
  "train_fraction": 0.8,
  "validation_fraction": 0.2,
  "train_units": [1, 2, 3],
  "validation_units": [4, 5]
}
```

`train_units` 和 `validation_units` 中的 unit ID 必须升序保存。

## 10. train_windows.npz

训练窗口文件必须包含以下数组：

```text
x
y
ocs
tidx
unit
end_cycle
```

shape 必须为：

```text
x:         float32 [N_train_windows, 30, 14]
y:         float32 [N_train_windows, 1]
ocs:       float32 [N_train_windows, 30, 1]
tidx:      int64   [N_train_windows, 1]
unit:      int64   [N_train_windows]
end_cycle: int64   [N_train_windows]
```

训练窗口生成规则：

```text
只使用 split_seed2026.json 中的 train_units
每台 train engine 使用所有长度为 30 的滑动窗口
stride = 1
不使用 official test
```

每个窗口的 target：

```text
y[i, 0] = min(T_u - end_cycle[i], 125)
```

其中 `T_u` 是该 train engine 的完整 run-to-failure 长度。

窗口必须按以下稳定顺序保存：

```text
unit 升序
同一 unit 内 end_cycle 升序
```

## 11. validation_view_v1.npz

validation 文件必须包含以下数组：

```text
x
y
ocs
tidx
unit
end_cycle
checkpoint_q
```

shape 必须为：

```text
x:            float32 [N_val_units * 3, 30, 14]
y:            float32 [N_val_units * 3, 1]
ocs:          float32 [N_val_units * 3, 30, 1]
tidx:         int64   [N_val_units * 3, 1]
unit:         int64   [N_val_units * 3]
end_cycle:    int64   [N_val_units * 3]
checkpoint_q: float32 [N_val_units * 3]
```

每台 validation engine 必须恰好生成 3 个窗口：

```text
q = 0.5
q = 0.7
q = 0.9
```

对每个 validation unit：

```text
t_q = max(30, floor(q * T_u))
y_q = min(T_u - t_q, 125)
```

要求：

```text
end_cycle = t_q
tidx = t_q
同一 unit 的三个 t_q 必须互不重复
每个窗口以 t_q 为最后一行
如果 t_q < 30 不应发生；若发生，协议检查必须失败
```

保存顺序必须为：

```text
unit 升序
同一 unit 内 checkpoint_q 按 0.5, 0.7, 0.9 顺序
```

## 12. test_terminal_windows.npz

official test 主评估文件必须包含以下数组：

```text
x
y
ocs
tidx
unit
end_cycle
provided_rul
capped_target_rul
```

shape 必须为：

```text
x:                 float32 [N_test_units, 30, 14]
y:                 float32 [N_test_units, 1]
ocs:               float32 [N_test_units, 30, 1]
tidx:              int64   [N_test_units, 1]
unit:              int64   [N_test_units]
end_cycle:         int64   [N_test_units]
provided_rul:      float32 [N_test_units, 1]
capped_target_rul: float32 [N_test_units, 1]
```

生成规则：

```text
每个 official test unit 恰好一行
x 是该 test unit 的最后一个 terminal window
end_cycle 是该 test unit 已观测到的最后一个 cycle
tidx = end_cycle
provided_rul 来自 RUL_FDxxx.txt
capped_target_rul = min(provided_rul, 125)
y = capped_target_rul
```

如果某台 test engine 的观测长度小于 30，必须使用 `left_repeat_first` padding：

```text
在序列左侧重复该 engine 的第一行，直到长度达到 30
```

保存顺序必须为：

```text
test unit 升序
```

## 13. full_test_windows.npz

`full_test_windows.npz` 只用于可视化和诊断，不用于 official test 主表指标。

文件必须包含：

```text
x
y
ocs
tidx
unit
end_cycle
provided_rul
capped_target_rul
```

shape：

```text
x:                 float32 [N_full_test_windows, 30, 14]
y:                 float32 [N_full_test_windows, 1]
ocs:               float32 [N_full_test_windows, 30, 1]
tidx:              int64   [N_full_test_windows, 1]
unit:              int64   [N_full_test_windows]
end_cycle:         int64   [N_full_test_windows]
provided_rul:      float32 [N_full_test_windows, 1]
capped_target_rul: float32 [N_full_test_windows, 1]
```

对 test unit 的任意窗口，设：

```text
terminal_cycle = 该 test unit 的最后观测 cycle
provided_rul_terminal = RUL_FDxxx.txt 中该 unit 的 provided RUL
raw_remaining_at_window = provided_rul_terminal + terminal_cycle - end_cycle
```

则：

```text
provided_rul = raw_remaining_at_window
capped_target_rul = min(raw_remaining_at_window, 125)
y = capped_target_rul
```

保存顺序必须为：

```text
unit 升序
同一 unit 内 end_cycle 升序
```

## 14. manifest.json

每个 dataset 目录下必须保存 `manifest.json`。

必须包含以下字段：

```json
{
  "main_protocol_version": "main_protocol_v1",
  "dataset": "FD001",
  "window_length": 30,
  "stride": 1,
  "rul_cap": 125,
  "split_seed": 2026,
  "selected_sensors": ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
  "selected_sensor_dim": 14,
  "raw_sensor_dim": 21,
  "operating_settings": ["op1", "op2", "op3"],
  "x_preprocessing": "train_only_global_z_score",
  "op_preprocessing": "train_only_op_z_score",
  "ocs_definition": "zero_single_regime_for_FD001_FD003_or_train_only_kmeans_k6_for_FD002_FD004",
  "padding": "left_repeat_first",
  "official_test_usage": "terminal_window_only_for_main_eval",
  "files": {
    "train_windows": "train_windows.npz",
    "validation_view": "validation_view_v1.npz",
    "test_terminal_windows": "test_terminal_windows.npz",
    "full_test_windows": "full_test_windows.npz"
  },
  "raw_files": {
    "train": {
      "name": "train_FD001.txt",
      "sha256": "REQUIRED",
      "num_columns": 26,
      "num_units": 100
    },
    "test": {
      "name": "test_FD001.txt",
      "sha256": "REQUIRED",
      "num_columns": 26,
      "num_units": 100
    },
    "rul": {
      "name": "RUL_FD001.txt",
      "sha256": "REQUIRED",
      "num_rows": 100
    }
  }
}
```

`sha256` 必须是真实 SHA256 字符串，不得填占位值。

## 15. 禁止事项

以下行为禁止：

```text
不得上传 NASA 原始数据到代码仓库
不得用 official test 拟合 scaler、KMeans、PCA、特征选择或任何可学习预处理
不得在 validation/test 上 fit 或 fit_predict KMeans
不得把 op1-op3 拼入 OCIR 主模型输入 x
不得让同一 train engine 同时出现在 train 和 validation
不得把 official test 的所有窗口作为主评估样本
不得为 provided view 和 capped view 分别生成两次模型预测
不得根据 official test 结果调整 split、window、scaler、KMeans、sensor list 或后处理
```

## 16. 验收标准

数据包交付后，模型侧会按以下标准验收。

每个 dataset 必须满足：

```text
manifest.json 存在且字段完整
split_seed2026.json 存在且 train/validation unit 不重叠
train_windows.npz 存在且 x.shape[-2:] = [30, 14]
validation_view_v1.npz 存在且每个 validation unit 恰好 3 行
test_terminal_windows.npz 存在且每个 test unit 恰好 1 行
full_test_windows.npz 存在但不用于主评估
所有 x/y/ocs/tidx/unit/end_cycle shape 与本文档一致
所有 dtype 与本文档一致
所有 target 定义与本文档一致
```

最终 OCIR DataLoader 将只使用以下 batch 接口：

```python
x, y, ocs, tidx
```

严格要求：

```text
x:    torch.float32 [B, 30, 14]
y:    torch.float32 [B, 1]
ocs:  torch.float32 [B, 30, 1]
tidx: torch.int64   [B, 1]
```
