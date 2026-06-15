# spad-detector 项目完善任务书


# 1. 项目定位

`spad-detector` 是原 `SPAD-Simulator` 的升级版。原项目更偏前端交互式 ToF/TDC 仿真器，新项目更偏后端驱动的 photon-counting 系统级仿真平台。

当前新项目的主数据格式是：

```text
counts[n_frames, roi_h, roi_w]
dtype = uint16
meaning = 每一帧、每个像素的 photon counts
```

原项目的 `.bin` 数据格式是：

```text
tdc[frame, row, col]
dtype = uint16
meaning = 每一帧、每个像素的 ToF/TDC bin
empty_value = tdcMaxCount + 2
```

两者代表不同的数据范式。新项目不应把原 `.bin` 格式恢复为唯一主格式，而应新增兼容导出模式，使平台同时支持科研用 photon-count cube 和 ToF/TDC 交互用 frame cube。

---

# 2. 总体设计原则

## 2.1 后端作为唯一科研仿真核心

所有正式仿真结果必须由 Python 后端产生。Angular/TypeScript 前端不得作为正式物理模型的权威来源。

允许保留前端快速 preview，但 preview 必须标记为 preview，不得与后端输出混用为正式结果。

## 2.2 默认输出保持 count cube

现有默认输出不破坏：

```text
count_cube:
  dtype = uint16
  shape = [n_frames, roi_h, roi_w]
  value = photon counts per frame per pixel
```

任何兼容旧项目的 ToF/TDC 输出都应作为可选导出，不应替代默认 count cube。

## 2.3 所有 raw .bin 必须配套 metadata

不再允许只导出裸 `.bin` 而没有 shape、dtype、单位、format 信息。所有 `.bin` 导出必须至少配套：

```text
metadata.json
summary.json
README 或 data_format 说明
```

## 2.4 明确区分 frame time 与 ToF/TDC time

`n_frames` 表示主仿真时间帧数量，不是 ToF bin 数量。

`frameDurationUs` 表示每帧积分时间或主时间步长。

`timeResolutionPs` 表示 TDC/ToF bin 宽度。

文档、UI 和 metadata 中必须清楚区分这几个概念。

---

# 3. 需要新增或完善的核心功能

## 3.1 数据导出模式统一

新增统一的数据导出枚举：

```text
ExportFormat:
  count_cube
  tdc_frame_cube
  event_list
  bundle
```

各格式定义如下。

### 3.1.1 count_cube

默认科研主输出。

```text
filename:
  counts.bin

dtype:
  uint16

shape:
  [n_frames, roi_h, roi_w]

layout:
  frame-major
  index = frame * roi_h * roi_w + row * roi_w + col

meaning:
  counts[frame,row,col] = 该帧该像素记录到的 photon counts

empty value:
  none
```

该模式保持现有行为，但必须新增 metadata sidecar。

### 3.1.2 tdc_frame_cube

兼容原 `SPAD-Simulator` 的 ToF/TDC frame cube。

```text
filename:
  tdc_frame_cube.bin

dtype:
  uint16

shape:
  [n_frames, roi_h, roi_w]

layout:
  frame-major
  index = frame * roi_h * roi_w + row * roi_w + col

meaning:
  tdc[frame,row,col] = 该帧该像素检测到的 TDC bin

empty value:
  tdcMaxCount + 2
```

有效值范围：

```text
1 <= tdc_bin <= tdcMaxCount
```

无检测时填：

```text
empty_pixel_value = tdcMaxCount + 2
```

如果同一帧同一像素有多个事件，默认 collision policy 使用：

```text
collision_policy = "first_event"
```

并在 metadata 中记录。可以预留以下策略，但默认只实现一个：

```text
first_event
min_tof_bin
max_signal_priority
```

注意：`tdc_frame_cube` 每个 frame-pixel 只能保存一个 TDC 值，因此不适合作为多光子 photon-counting 主数据。它只能作为 ToF/TDC 可视化和旧项目兼容格式。

### 3.1.3 event_list

事件表输出，用于保留多个事件、事件来源和 ToF 信息。

建议字段：

```text
event_times_s: float32[N]
event_frame_index: int32[N]
event_row: uint16[N]
event_col: uint16[N]
event_pixel: int32[N]
event_tof_bins: uint16[N]
event_source: uint8[N]
```

`event_pixel` 的展平规则：

```text
event_pixel = row * roi_w + col
```

`event_source` 编码建议：

```text
0 = unknown
1 = signal
2 = background
3 = dark
4 = afterpulse
5 = crosstalk
```

如果当前模型无法可靠区分 afterpulse/crosstalk 来源，可以先实现：

```text
1 = signal
2 = background
3 = dark
0 = unknown
```

并在 metadata 中说明 source granularity。

### 3.1.4 bundle

一次性导出完整结果包：

```text
spad_simulation_<job_id>.zip
  counts.bin
  metadata.json
  summary.json
  preview_counts.png 或 preview_counts.npy
  optional:
    tdc_frame_cube.bin
    events.npz
```

`bundle` 应作为推荐下载格式。

---

# 4. metadata.json 规范

每个导出结果必须包含 metadata。建议结构如下：

```json
{
  "format": "count_cube",
  "dtype": "uint16",
  "shape": [20000, 32, 32],
  "layout": "frame-major",
  "index_rule": "index = frame * roi_h * roi_w + row * roi_w + col",

  "n_frames": 20000,
  "roi_h": 32,
  "roi_w": 32,

  "frame_duration_us": 20.0,
  "sample_rate_hz": 50000.0,
  "observation_time_s": 0.4,

  "time_resolution_ps": 256.0,
  "tdc_bin_width_ns": 0.256,
  "tdc_max_count": 8191,
  "empty_pixel_value": null,

  "range_bin_m": 0.0384,
  "max_unambiguous_range_m": 314.5,

  "detector_preset": "PF32",
  "pde": 0.274,
  "dead_time_ns": 10.0,
  "timing_jitter_ns": 0.05,

  "random_seed": 1234,
  "simulation_mode": "frame",
  "export_created_utc": "YYYY-MM-DDTHH:MM:SSZ",

  "version": {
    "project": "spad-detector",
    "git_commit": "<commit_sha_if_available>"
  }
}
```

对于 `tdc_frame_cube`，metadata 必须额外包含：

```json
{
  "format": "tdc_frame_cube",
  "empty_pixel_value": 8193,
  "valid_tdc_range": [1, 8191],
  "collision_policy": "first_event",
  "source": "generated_from_event_list"
}
```

对于 `event_list`，metadata 必须额外包含：

```json
{
  "format": "event_list",
  "fields": {
    "event_times_s": "float32 seconds",
    "event_frame_index": "int32",
    "event_row": "uint16",
    "event_col": "uint16",
    "event_pixel": "int32, row * roi_w + col",
    "event_tof_bins": "uint16",
    "event_source": "uint8"
  },
  "event_source_encoding": {
    "0": "unknown",
    "1": "signal",
    "2": "background",
    "3": "dark",
    "4": "afterpulse",
    "5": "crosstalk"
  }
}
```

---

# 5. ToF/TDC 距离诊断

前端和后端 summary 均应提供 ToF/TDC 诊断量。

计算公式：

```text
tdc_bin_width_s = timeResolutionPs * 1e-12

range_bin_m = c * tdc_bin_width_s / 2

max_unambiguous_range_m = c * tdcMaxCount * tdc_bin_width_s / 2

timing_jitter_range_sigma_m = c * timing_jitter_s / 2
```

其中：

```text
c = 299792458 m/s
```

UI 中应显示：

```text
TDC bin width
TDC bit depth 或 tdcMaxCount
range bin / distance resolution
maximum unambiguous range
timing jitter equivalent range error
frame duration
sample rate
total observation time
```

文案中明确：

```text
frameDurationUs 控制主仿真帧积分时间。
timeResolutionPs 控制 ToF/TDC bin 宽度。
nFrames 是主时间帧数，不是 ToF bin 数。
```

---

# 6. 后端实现任务

## 6.1 保持现有 API 兼容

不得破坏已有接口：

```text
POST /api/simulate
POST /api/simulate/summary
POST /api/simulate/jobs
GET  /api/simulate/jobs/{job_id}
GET  /api/simulate/jobs/{job_id}/download
```

现有下载接口可继续默认返回 count cube `.bin`，但应新增推荐的 bundle 下载方式。

建议新增或扩展：

```text
GET /api/simulate/jobs/{job_id}/download?format=count_cube
GET /api/simulate/jobs/{job_id}/download?format=tdc_frame_cube
GET /api/simulate/jobs/{job_id}/download?format=event_list
GET /api/simulate/jobs/{job_id}/download?format=bundle
GET /api/simulate/jobs/{job_id}/metadata
```

如果不方便改动现有 endpoint，可以新增：

```text
GET /api/simulate/jobs/{job_id}/artifacts/{artifact_name}
```

## 6.2 增加导出工具模块

建议新增：

```text
backend/exporters.py
```

或类似模块，集中处理：

```text
write_count_cube_bin(...)
write_tdc_frame_cube_bin(...)
write_event_npz(...)
write_metadata_json(...)
write_summary_json(...)
write_bundle_zip(...)
```

不要把导出逻辑散落在路由函数中。

## 6.3 增加数据格式模型

在后端 schema 中新增：

```text
ExportFormat
ArtifactMetadata
EventSource
EventList
```

请求参数可增加：

```json
{
  "export_format": "bundle",
  "include_tdc_frame_cube": true,
  "include_event_list": true
}
```

对于同步 API，默认不返回大体积 event list，避免 JSON 过大。大体积结果应通过异步任务 artifacts 下载。

## 6.4 事件生成逻辑

当前 frame-level count cube 是主输出。event_list 不应只是对总 counts 盲目展开，而应尽量保留事件来源和 ToF 信息。

推荐实现路径：

1. 在仿真过程中保留各分量期望值：

   ```text
   expected_signal
   expected_background
   expected_dark
   optional expected_afterpulse
   optional expected_crosstalk
   ```

2. 当用户请求 event output 时，对各分量分别采样事件数量。

3. 对 signal 事件，根据 range / ToF 模型生成 `event_tof_bins`。

4. 对 background / dark 事件，可按配置生成随机 ToF bin，并将 source 标为 background 或 dark。

5. 对无法区分来源的事件，source 标为 unknown。

6. 生成事件后再派生：

   ```text
   event_times_s
   event_frame_index
   event_row
   event_col
   event_pixel
   event_tof_bins
   event_source
   ```

7. 由 event_list 派生 `tdc_frame_cube`，而不是从纯 count cube 反推。

如果短期内无法完成分量级事件生成，可先实现 `synthetic_event_list_from_counts`，但必须在 metadata 中写：

```json
{
  "event_generation": "synthetic_from_frame_counts",
  "warning": "Event timestamps and TDC bins are synthesized from sampled frame counts and do not represent full event-level TCSPC transport."
}
```

## 6.5 tdc_frame_cube 生成规则

从 event_list 生成：

```text
tdc = full([n_frames, roi_h, roi_w], empty_pixel_value, dtype=uint16)
```

对每个事件：

```text
f = event_frame_index[k]
r = event_row[k]
c = event_col[k]
bin = event_tof_bins[k]
```

若当前 tdc[f,r,c] 为空，则写入 bin。

若已有值，根据 collision policy 处理：

```text
first_event:
  保留已有值

min_tof_bin:
  tdc[f,r,c] = min(existing, bin)
```

默认使用 `first_event`，并记录在 metadata。

## 6.6 大数据内存控制

对大规模 `n_frames * roi_h * roi_w` 输出，必须注意内存。

要求：

```text
不要在不必要时同时持有 counts、tdc_cube、event_list 多份超大数组。
异步任务中优先写 artifact 到磁盘。
event_list 可设置最大事件数限制。
超限时返回明确错误或 warning。
```

新增参数：

```text
max_event_count
include_event_list
include_tdc_frame_cube
```

当事件数量超过限制时，保留 count cube 和 metadata，跳过 event_list，并写 warning。

---

# 7. 前端实现任务

## 7.1 输出格式选择

在仿真参数或下载区域增加导出格式选择：

```text
Download count cube
Download TDC frame cube
Download event list
Download full bundle
```

默认推荐：

```text
Download full bundle
```

当前默认仿真仍使用 frame/count 模式。

## 7.2 ToF/TDC 诊断面板

新增一个小面板，显示：

```text
Frame duration
Sample rate
Number of frames
Total observation time
TDC bin width
TDC max count
Range bin
Max unambiguous range
Timing jitter equivalent range error
```

面板旁加入说明：

```text
nFrames 是主时间帧数，不是 ToF bin 数。
timeResolutionPs 是 TDC bin 宽度。
frameDurationUs 是每帧积分时间。
```

## 7.3 自定义叶片轮廓可复现

保留 custom blade silhouette 功能，并完善：

```text
显示采样点数量
允许导出 custom_shape.json
记录 custom_shape_x/y/intensity 到 metadata
确保采样是确定性的
```

如果当前前端已经使用 stride 或固定顺序采样，应保留。不要恢复随机抽样。

## 7.4 快速 preview 与正式仿真区分

如果保留前端快速预览，UI 必须标注：

```text
Preview only
Research output is generated by backend simulation
```

不要让前端 TypeScript 物理模型与后端 Python 物理模型同时作为正式输出。

---

# 8. 文档任务

## 8.1 新增 DATA_FORMAT.md

新增：

```text
docs/DATA_FORMAT.md
```

内容包括：

```text
count_cube 格式
tdc_frame_cube 格式
event_list 格式
metadata.json 字段说明
.bin 展平顺序
Python 读取示例
MATLAB 读取示例
常见误区
```

必须明确说明：

```text
count_cube 的 n_frames 不是 ToF。
tdc_frame_cube 的 n_frames 也不是 ToF。
ToF 信息在每个元素值或 event_tof_bins 中。
```

## 8.2 新增读取示例

新增：

```text
examples/read_count_cube.py
examples/read_tdc_frame_cube.py
examples/read_event_list.py
examples/read_count_cube_matlab.m
examples/read_tdc_frame_cube_matlab.m
```

Python 示例：

```python
import json
import numpy as np

with open("metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])

data = np.fromfile("counts.bin", dtype=dtype)
cube = data.reshape(shape)

print(cube.shape, cube.dtype)
```

TDC 示例：

```python
empty = meta["empty_pixel_value"]
valid = cube != empty
tof_bins = cube[valid]
```

## 8.3 README 更新

README 中增加：

```text
Output formats
How to choose count cube / TDC cube / event list
How to read exported .bin
Difference from legacy SPAD-Simulator
Frame duration vs TDC time resolution
```

不要把原项目描述为正式物理模型来源。原项目只作为 legacy data format 和交互设计来源。

---

# 9. 测试任务

新增或完善测试。

## 9.1 后端单元测试

测试内容：

```text
count_cube .bin 写入和读取 shape 一致
metadata.json 字段完整
tdc_frame_cube sentinel 正确
tdc_frame_cube 有效值在 [1, tdcMaxCount]
event_list 字段长度一致
event_pixel = row * roi_w + col
bundle zip 包含必需文件
```

## 9.2 API 测试

测试：

```text
GET download?format=count_cube
GET download?format=tdc_frame_cube
GET download?format=event_list
GET download?format=bundle
GET metadata
```

确保不存在请求格式时返回清晰错误。

## 9.3 前端测试

测试：

```text
导出格式选择控件存在
ToF/TDC 诊断计算正确
nFrames / frameDurationUs / timeResolutionPs 显示不混淆
下载按钮调用正确 endpoint
```

## 9.4 回归测试

现有仿真默认行为不得破坏：

```text
默认 /api/simulate 仍能返回 summary
默认 job download 仍可获得 count cube
现有 frame 模式参数仍能运行
已有 README 示例仍可执行或同步更新
```

---

# 10. 明确不应恢复的旧功能

以下旧项目行为不应作为新项目主逻辑恢复。

## 10.1 不恢复原 `.bin` 作为唯一主格式

原 `.bin` 每个 frame-pixel 只能存一个 ToF/TDC 值，不适合作为 photon-counting 主数据。新项目主输出应继续是 count cube。

## 10.2 不把 CW 随机 ToF 作为测距结果

如果 CW 模式下需要事件时间戳，只能标记为 background timestamp 或 ungated timestamp，不得解释为真实距离测量。

## 10.3 不把信号和噪声混写且无 source 标签

event_list 中应尽量提供 event_source。无法区分来源时用 unknown，并在 metadata 中说明。

## 10.4 不恢复发射光束角与接收 FOV 绑定

发射端 divergence 和接收端 FOV 是独立物理参数，应保持分离。

## 10.5 不把 TypeScript 前端物理模型作为科研输出

前端快速模型只用于 preview。正式结果必须来自后端 Python 仿真。

---

# 11. 推荐实施顺序

## Phase 1：数据格式与 metadata

1. 新增 `ExportFormat`。
2. 新增 metadata 生成函数。
3. 修改异步下载流程，生成 `counts.bin + metadata.json + summary.json`。
4. 新增 `DATA_FORMAT.md`。
5. 新增 Python 读取示例。

完成标准：

```text
用户下载 count cube 后，可以只凭 metadata 正确 reshape。
```

## Phase 2：ToF/TDC 兼容导出

1. 新增 `tdc_frame_cube` 生成函数。
2. 新增 sentinel 规则：`empty = tdcMaxCount + 2`。
3. 新增 collision policy。
4. 新增下载 endpoint 或 query 参数。
5. 新增读取示例和测试。

完成标准：

```text
可以导出与原 SPAD-Simulator 逻辑兼容的 uint16 TDC frame cube。
```

## Phase 3：event_list

1. 新增事件字段。
2. 增加 `event_tof_bins`。
3. 增加 `event_source`。
4. 支持 event list artifact 下载。
5. 由 event list 派生 tdc frame cube。

完成标准：

```text
event_list 可以表达同一 frame-pixel 内的多个事件。
tdc_frame_cube 可以由 event_list 稳定派生。
```

## Phase 4：前端 UI 与诊断

1. 增加导出格式选择。
2. 增加 ToF/TDC 诊断面板。
3. 增加 frame duration / TDC resolution 解释。
4. 下载 full bundle。

完成标准：

```text
用户不会再把 nFrames 误解为 ToF bin。
```

## Phase 5：文档、测试和清理

1. 更新 README。
2. 增加 MATLAB 示例。
3. 增加 API 测试。
4. 清理重复或误导性旧逻辑。
5. 确保默认仿真流程不破坏。

完成标准：

```text
pytest / frontend build / existing tests 全部通过。
```

---

# 12. 验收标准

项目完成后必须满足以下条件。

## 12.1 数据格式验收

```text
count_cube:
  可下载
  有 metadata
  可用 Python 正确 reshape
  shape = [n_frames, roi_h, roi_w]

tdc_frame_cube:
  可下载
  有 metadata
  empty_pixel_value = tdcMaxCount + 2
  有效 TDC bin 在合法范围内

event_list:
  字段齐全
  长度一致
  event_pixel 与 row/col 一致
  event_tof_bins 存在
  event_source 存在
```

## 12.2 功能验收

```text
默认仿真流程不变
默认主数据仍是 count cube
用户可以选择 legacy TDC 输出
用户可以下载 bundle
前端显示 ToF/TDC 诊断量
自定义形状输入可复现
```

## 12.3 文档验收

```text
README 说明三类输出格式
DATA_FORMAT.md 完整
Python 示例可运行
MATLAB 示例可运行或语法清楚
明确说明 n_frames 不是 ToF
```

## 12.4 科研一致性验收

```text
后端 Python 是正式仿真来源
前端 preview 不作为科研输出
信号、背景、暗计数在 summary 或 metadata 中保持可解释
不引入发射 FOV 与接收 FOV 绑定的旧错误
不把 CW 随机 ToF 解释为真实测距
```

---

# 13. 建议的最终目录结构

```text
backend/
  exporters.py
  schemas.py
  simulation/
    ...
docs/
  DATA_FORMAT.md
examples/
  read_count_cube.py
  read_tdc_frame_cube.py
  read_event_list.py
  read_count_cube_matlab.m
  read_tdc_frame_cube_matlab.m
tests/
  test_export_count_cube.py
  test_export_tdc_cube.py
  test_event_list.py
  test_metadata.py
```

---

# 14. 提交要求

每个 phase 尽量单独提交，commit message 使用清晰动词：

```text
Add metadata sidecars for binary exports
Add legacy TDC frame cube export
Add event list artifact export
Add TDC diagnostics to simulation UI
Document binary data formats
Add export format regression tests
```

不要在一个提交里同时做大规模重构、UI 修改和物理模型修改。

---

# 15. 最终交付物

完成后应交付：

```text
1. 后端支持 count_cube / tdc_frame_cube / event_list / bundle 导出
2. 所有 .bin 都有 metadata.json
3. docs/DATA_FORMAT.md
4. Python + MATLAB 读取示例
5. 前端 ToF/TDC 诊断面板
6. 前端导出格式选择
7. event_tof_bins 和 event_source 字段
8. 自定义形状输入可复现并进入 metadata
9. 单元测试和 API 回归测试
10. README 更新
```

最终平台应清楚区分两类数据：

```text
科研主数据：
  photon count cube
  counts[n_frames, roi_h, roi_w]

兼容与 ToF 可视化数据：
  TDC frame cube
  tdc[n_frames, roi_h, roi_w]

事件级扩展数据：
  event list with time, pixel, ToF bin, source
```

默认保持 count cube 为主，ToF/TDC 和 event list 作为可选增强输出。
