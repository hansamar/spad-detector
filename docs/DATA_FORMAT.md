# SPAD Detector 数据格式说明

## 概述

spad-detector 输出三类数据格式，对应不同的科研和工程用途：

| 格式 | 文件 | Shape | dtype | 用途 |
|------|------|-------|-------|------|
| **count_cube** | `counts.bin` | `[n_frames, roi_h, roi_w]` | uint16 | 科研主输出：每帧每像素光子计数 |
| **tdc_frame_cube** | `tdc_frame_cube.bin` | `[n_frames, roi_h, roi_w]` | uint16 | ToF/TDC 可视化：每帧每像素首个 TDC bin |
| **event_list** | `events.npz` | N 条事件记录 | mixed | 事件级扩展：含时间、像素、TDC bin、来源 |
| **bundle** | `bundle.zip` | 全部上述文件 | — | 推荐下载格式 |

> **重要概念区分**：`n_frames` 是主仿真时间帧数（对应 `observation_time_s × sample_rate_hz`），**不是** ToF/TDC bin 数量。
> ToF 信息存储在 `tdc_frame_cube` 的每个元素值中，或 `event_list` 的 `event_tof_bins` 字段中。

---

## 1. count_cube — 科研主数据

### 文件结构

```
counts.bin           — uint16 二进制数组，frame-major 布局
counts.metadata.json — 配套元数据
counts.summary.json  — 仿真摘要
```

### .bin 布局

```
索引: index = frame × roi_h × roi_w + row × roi_w + col
形状: [n_frames, roi_h, roi_w]
类型: uint16
含义: counts[frame, row, col] = 该帧该像素记录到的 photon counts
```

### metadata.json 关键字段

| 字段 | 含义 |
|------|------|
| `shape` | `[n_frames, roi_h, roi_w]` |
| `dtype` | `uint16` |
| `layout` | `frame-major` |
| `frame_duration_us` | 每帧积分时间 (µs) |
| `sample_rate_hz` | 等效帧率 |
| `observation_time_s` | 总观测时间 |
| `random_seed` | 仿真随机种子 |
| `detector_preset` | 探测器预设方案名称 |
| `pde` | 光子探测效率 |
| `dead_time_ns` | 死时间 (ns) |

---

## 2. tdc_frame_cube — ToF/TDC 兼容输出

### 文件结构

```
tdc_frame_cube.bin           — uint16 二进制数组
tdc_frame_cube.metadata.json — 配套元数据
```

### .bin 布局

```
索引: index = frame × roi_h × roi_w + row × roi_w + col
形状: [n_frames, roi_h, roi_w]
类型: uint16
含义: tdc[frame, row, col] = 该帧该像素检测到的 TDC bin（1-based）

空像素: 0（有效 TDC bin 范围为 1..tdcMaxCount）
有效范围: 1 ≤ tdc_bin ≤ tdcMaxCount
```

### 碰撞处理

当同一帧同一像素有多个事件时，使用 `collision_policy` 处理：

- `first_event`（默认）：保留第一个事件
- `min_tof_bin`：保留最小 TDC bin

策略在 metadata 中记录。

### metadata.json 额外字段

| 字段 | 含义 |
|------|------|
| `empty_pixel_value` | 空像素填充值 |
| `valid_tdc_range` | `[1, tdcMaxCount]` |
| `collision_policy` | 碰撞处理策略 |
| `source` | `generated_from_event_list` |

---

## 3. event_list — 事件级扩展

### 文件结构

```
events.npz            — NumPy 压缩存档
events.metadata.json  — 配套元数据
```

### .npz 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `event_times_s` | float32 | 事件时间 (s)，从仿真起点算起 |
| `event_frame_index` | int32 | 所属帧索引 |
| `event_row` | uint16 | 像素行 |
| `event_col` | uint16 | 像素列 |
| `event_pixel` | int32 | 展平像素索引 `row × roi_w + col` |
| `event_tof_bins` | uint16 | TDC bin（1-based） |
| `event_source` | uint8 | 事件来源编码 |

### event_source 编码

| 值 | 含义 | 说明 |
|----|------|------|
| 0 | unknown | 无法区分来源 |
| 1 | signal | 目标反射信号 |
| 2 | background | 场景杂散光 |
| 3 | dark | 暗计数 |
| 4 | afterpulse | 后脉冲（可实现时添加） |
| 5 | crosstalk | 串扰（可实现时添加） |

当前实现中 signal/background/dark 按期望值比例采样分配。afterpulse 和 crosstalk 在详细事件模型可用前标记为 unknown。

---

## 4. bundle — 完整产物包

```
bundle.zip
├── counts.bin
├── metadata.json
├── summary.json
├── tdc_frame_cube.bin
├── tdc_frame_cube.metadata.json
├── events.npz
├── events.metadata.json
├── truth.npz
├── truth.metadata.json
└── manifest.json
```

`truth.npz` 保存未降采样的仿真真值序列，包括机械频率、像面中心、距离、可见性、背景与投影尺寸。它只用于仿真评估，不作为算法输入。`truth.metadata.json` 记录字段单位、shape、采样率和随机种子。

---

## 5. 距离诊断量说明

以下量从 `tdc_bin_width_ns` 和 `tdc_max_count` 计算：

```
c = 299792458 m/s

tdc_bin_width_s = tdc_bin_width_ns × 1e-9

range_bin_m = c × tdc_bin_width_s / 2

max_unambiguous_range_m = c × tdc_max_count × tdc_bin_width_s / 2

timing_jitter_range_sigma_m = c × timing_jitter_ns × 1e-9 / 2
```

### 示范

若 `tdc_bin_width_ns = 0.256`、`tdc_max_count = 8191`：

- range_bin_m ≈ 0.0384 m (3.84 cm)
- max_unambiguous_range_m ≈ 314.5 m

---

## 6. 常见误区

1. ❌ **把 n_frames 当作 ToF bin 数** — n_frames 是主时间帧数，由 `observation_time_s × sample_rate_hz` 决定
2. ❌ **把 frame_duration_us 当作 TDC bin 宽度** — frame_duration_us 是每帧积分时间，TDC bin 宽度是 `time_resolution_ps`
3. ❌ **在没有 metadata.json 的情况下直接读取 .bin** — 没有 shape/dtype 信息无法正确 reshape
4. ❌ **将 tdc_frame_cube 作为 photon-counting 主数据** — 每个 frame-pixel 只能存一个 TDC 值，丢失了光子计数信息
5. ❌ **将 CW random ToF 解释为真实距离** — CW 模式下的 ToF bin 是随机合成值

---

## 与旧版 SPAD-Simulator 的区别

旧版 `SPAD-Simulator` 的 `.bin` 格式是 TDC-only（每 pixel 存一个 ToF bin），适合交互式演示。

新版 `spad-detector` 的默认主输出是 photon-count cube（每 pixel 存光子计数），更适合科研定量分析。

TDC frame cube 作为可选导出保留了与旧格式的兼容性，但不应作为主数据格式。

---

## 数据契约与完整性校验

新导出数据使用 `spad-dataset` 1.0.0 契约。每个 metadata 文件包含：

- `schema.name` 与 `schema.version`：数据契约名称和版本。
- `format`：`count_cube`、`tdc_frame_cube` 或 `event_list`，用于确定数值语义。
- `dtype`、`shape`、`layout`：二进制解码所需的完整结构信息。
- `sample_rate_hz`、`tdc_bin_width_ns`、`empty_pixel_value`：时间采样和空事件定义。
- `random_seed`、`assumptions`、`warnings`：复现实验所需的随机性与模型边界。
- `artifact.bytes` 与 `artifact.sha256`：载荷大小和 SHA-256 完整性校验值。

`bundle.zip` 额外包含 `manifest.json`，集中列出 `counts.bin`、`tdc_frame_cube.bin`、`events.npz` 和 `truth.npz` 中实际存在的载荷及其角色、大小和 SHA-256。算法程序应先读取 metadata 和 manifest，完成格式与完整性校验后再运行。
