"""集中处理仿真产物导出：count_cube / tdc_frame_cube / event_list / bundle。

所有 .bin 导出均配套 metadata.json 和 summary.json sidecar。
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT_MS = 299_792_458.0
DATASET_SCHEMA_NAME = "spad-dataset"
DATASET_SCHEMA_VERSION = "1.0.0"


class ExportFormat(str, Enum):
    """仿真数据导出格式枚举。"""

    count_cube = "count_cube"
    tdc_frame_cube = "tdc_frame_cube"
    event_list = "event_list"
    bundle = "bundle"


class EventSource(int, Enum):
    """事件来源编码。"""

    unknown = 0
    signal = 1
    background = 2
    dark = 3
    afterpulse = 4
    crosstalk = 5


def _utc_now_iso() -> str:
    """返回 UTC ISO 时间字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_descriptor(path: Path, *, role: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.name,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _metadata_with_artifact(metadata: dict, path: Path, *, role: str) -> dict:
    return {**metadata, "artifact": _artifact_descriptor(path, role=role)}


def frame_tof_diagnostics(
    tdc_bin_width_ns: float,
    tdc_max_count: int,
    timing_jitter_ns: float,
) -> dict:
    """计算 TDC/ToF 距离诊断量。

    Args:
        tdc_bin_width_ns: TDC bin 宽度 (ns)
        tdc_max_count: TDC 最大计数值
        timing_jitter_ns: 定时抖动 (ns)

    Returns:
        dict: range_bin_m, max_unambiguous_range_m, timing_jitter_range_sigma_m
    """
    bin_width_s = tdc_bin_width_ns * 1e-9
    range_bin_m = SPEED_OF_LIGHT_MS * bin_width_s / 2.0
    max_range_m = SPEED_OF_LIGHT_MS * tdc_max_count * bin_width_s / 2.0
    jitter_range_m = SPEED_OF_LIGHT_MS * timing_jitter_ns * 1e-9 / 2.0
    return {
        "range_bin_m": range_bin_m,
        "max_unambiguous_range_m": max_range_m,
        "timing_jitter_range_sigma_m": jitter_range_m,
    }


def build_metadata(
    *,
    format: ExportFormat | str,
    n_frames: int,
    roi_h: int,
    roi_w: int,
    sample_rate_hz: float,
    observation_time_s: float,
    dtype: str = "uint16",
    detector_preset: str = "",
    quantum_efficiency: float = 0.0,
    dead_time_ns: float = 0.0,
    timing_jitter_ns: float = 0.0,
    tdc_bin_width_ns: float = 0.0,
    tdc_max_count: int = 0,
    empty_pixel_value: int | None = None,
    collision_policy: str | None = None,
    random_seed: int | None = None,
    simulation_mode: str = "frame",
    event_generation: str | None = None,
    event_warning: str | None = None,
    event_source_encoding: dict | None = None,
    event_fields: dict | None = None,
    warnings: list[str] | None = None,
    assumptions: list[str] | None = None,
    git_commit: str | None = None,
    custom_shape: dict | None = None,
) -> dict:
    """构建标准化的 metadata.json 负载。

    返回值可直接 json.dump 写入文件。
    """
    diag = frame_tof_diagnostics(
        tdc_bin_width_ns=tdc_bin_width_ns,
        tdc_max_count=tdc_max_count,
        timing_jitter_ns=timing_jitter_ns,
    )

    meta: dict = {
        "schema": {
            "name": DATASET_SCHEMA_NAME,
            "version": DATASET_SCHEMA_VERSION,
        },
        "format": format if isinstance(format, str) else format.value,
        "dtype": dtype,
        "shape": [n_frames, roi_h, roi_w],
        "layout": "frame-major",
        "index_rule": "index = frame * roi_h * roi_w + row * roi_w + col",
        "n_frames": n_frames,
        "roi_h": roi_h,
        "roi_w": roi_w,
        "frame_duration_us": (1.0 / sample_rate_hz) * 1e6 if sample_rate_hz > 0 else 0.0,
        "sample_rate_hz": sample_rate_hz,
        "observation_time_s": observation_time_s,
        "time_resolution_ps": tdc_bin_width_ns * 1e3,
        "tdc_bin_width_ns": tdc_bin_width_ns,
        "tdc_max_count": tdc_max_count,
        "empty_pixel_value": empty_pixel_value,
        "range_bin_m": diag["range_bin_m"],
        "max_unambiguous_range_m": diag["max_unambiguous_range_m"],
        "timing_jitter_range_sigma_m": diag["timing_jitter_range_sigma_m"],
        "detector_preset": detector_preset,
        "pde": quantum_efficiency,
        "dead_time_ns": dead_time_ns,
        "timing_jitter_ns": timing_jitter_ns,
        "random_seed": random_seed,
        "simulation_mode": simulation_mode,
        "export_created_utc": _utc_now_iso(),
        "warnings": warnings or [],
        "assumptions": assumptions or [],
        "version": {
            "project": "spad-detector",
            "git_commit": git_commit or "",
        },
    }

    if format == ExportFormat.tdc_frame_cube:
        meta["valid_tdc_range"] = [1, tdc_max_count]
        meta["collision_policy"] = collision_policy or "first_event"
        meta["source"] = event_generation or "generated_from_event_list"

    if format == ExportFormat.event_list:
        meta["event_generation"] = event_generation or ""
        if event_warning:
            meta["event_generation_warning"] = event_warning
        if event_fields:
            meta["fields"] = event_fields
        if event_source_encoding:
            meta["event_source_encoding"] = event_source_encoding

    if custom_shape is not None:
        meta["custom_shape"] = custom_shape

    return meta


def write_count_cube_bin(
    output_path: str | Path,
    counts: np.ndarray,
    *,
    metadata: dict,
    summary: dict | None = None,
) -> None:
    """写入 count_cube .bin + metadata.json (+ summary.json)。

    Args:
        output_path: .bin 文件路径（不含扩展名，会自动追加 .bin）
        counts: uint16 [n_frames, roi_h, roi_w]
        metadata: 由 build_metadata() 生成的元数据 dict
        summary: 可选的 summary dict
    """
    bin_path = Path(output_path).with_suffix(".bin")
    meta_path = Path(output_path).with_suffix(".metadata.json")
    summary_path = Path(output_path).with_suffix(".summary.json")

    np.asarray(counts, dtype=np.uint16).tofile(str(bin_path))
    metadata = _metadata_with_artifact(metadata, bin_path, role="photon_count_cube")
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_tdc_frame_cube_bin(
    output_path: str | Path,
    tdc_cube: np.ndarray,
    *,
    metadata: dict,
    summary: dict | None = None,
) -> None:
    """写入 tdc_frame_cube .bin + metadata.json (+ summary.json)。

    Args:
        output_path: 输出文件基路径
        tdc_cube: uint16 [n_frames, roi_h, roi_w]
        metadata: 由 build_metadata() 生成的元数据 dict（format=tdc_frame_cube）
        summary: 可选的 summary dict
    """
    bin_path = Path(output_path).with_suffix(".bin")
    meta_path = Path(output_path).with_suffix(".metadata.json")
    summary_path = Path(output_path).with_suffix(".summary.json")

    np.asarray(tdc_cube, dtype=np.uint16).tofile(str(bin_path))
    metadata = _metadata_with_artifact(metadata, bin_path, role="tdc_frame_cube")
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_event_npz(
    output_path: str | Path,
    *,
    event_times_s: np.ndarray,
    event_frame_index: np.ndarray,
    event_row: np.ndarray,
    event_col: np.ndarray,
    event_pixel: np.ndarray,
    event_tof_bins: np.ndarray,
    event_source: np.ndarray,
    metadata: dict,
    summary: dict | None = None,
) -> None:
    """写入 event_list .npz + metadata.json (+ summary.json)。

    Args:
        output_path: 输出文件基路径
        *_*: 各字段数组，长度必须一致
        metadata: 由 build_metadata() 生成的元数据 dict（format=event_list）
        summary: 可选的 summary dict
    """
    npz_path = Path(output_path).with_suffix(".npz")
    meta_path = Path(output_path).with_suffix(".metadata.json")
    summary_path = Path(output_path).with_suffix(".summary.json")

    np.savez_compressed(
        str(npz_path),
        event_times_s=event_times_s,
        event_frame_index=event_frame_index,
        event_row=event_row,
        event_col=event_col,
        event_pixel=event_pixel,
        event_tof_bins=event_tof_bins,
        event_source=event_source,
    )
    metadata = _metadata_with_artifact(metadata, npz_path, role="event_list")
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_bundle_zip(
    output_path: str | Path,
    *,
    counts: np.ndarray,
    metadata: dict,
    summary: dict,
    tdc_cube: np.ndarray | None = None,
    tdc_metadata: dict | None = None,
    events: dict[str, np.ndarray] | None = None,
    events_metadata: dict | None = None,
) -> None:
    """写入完整 bundle .zip，包含所有可用产物。

    Args:
        output_path: .zip 文件路径
        counts: uint16 count_cube
        metadata: count_cube metadata
        summary: simulation summary
        tdc_cube: 可选 tdc_frame_cube
        tdc_metadata: 可选 tdc metadata
        events: 可选 event 字段 dict（key→array）
        events_metadata: 可选 events metadata
    """
    zip_path = Path(output_path).with_suffix(".zip")

    with tempfile.TemporaryDirectory(prefix="spad_bundle_") as tmp:
        tmp_root = Path(tmp)

        counts_path = tmp_root / "counts.bin"
        counts_path.write_bytes(
            np.asarray(counts, dtype=np.uint16).tobytes()
        )
        bundle_metadata = _metadata_with_artifact(metadata, counts_path, role="photon_count_cube")
        (tmp_root / "metadata.json").write_text(
            json.dumps(bundle_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (tmp_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if tdc_cube is not None:
            tdc_path = tmp_root / "tdc_frame_cube.bin"
            tdc_path.write_bytes(
                np.asarray(tdc_cube, dtype=np.uint16).tobytes()
            )
            if tdc_metadata:
                bundle_tdc_metadata = _metadata_with_artifact(tdc_metadata, tdc_path, role="tdc_frame_cube")
                (tmp_root / "tdc_frame_cube.metadata.json").write_text(
                    json.dumps(bundle_tdc_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        if events is not None:
            events_path = tmp_root / "events.npz"
            np.savez_compressed(
                str(events_path),
                **{k: v for k, v in events.items()},
            )
            if events_metadata:
                bundle_events_metadata = _metadata_with_artifact(events_metadata, events_path, role="event_list")
                (tmp_root / "events.metadata.json").write_text(
                    json.dumps(bundle_events_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        payload_names = ["counts.bin", "tdc_frame_cube.bin", "events.npz"]
        manifest = {
            "schema": {
                "name": DATASET_SCHEMA_NAME,
                "version": DATASET_SCHEMA_VERSION,
            },
            "created_utc": _utc_now_iso(),
            "artifacts": [
                _artifact_descriptor(tmp_root / name, role={
                    "counts.bin": "photon_count_cube",
                    "tdc_frame_cube.bin": "tdc_frame_cube",
                    "events.npz": "event_list",
                }[name])
                for name in payload_names
                if (tmp_root / name).exists()
            ],
        }
        (tmp_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(tmp_root.iterdir()):
                zf.write(str(f), f.name)


def generate_tdc_frame_cube_from_event_list(
    event_frame_index: np.ndarray,
    event_row: np.ndarray,
    event_col: np.ndarray,
    event_tof_bins: np.ndarray,
    n_frames: int,
    roi_h: int,
    roi_w: int,
    empty_pixel_value: int,
    collision_policy: str = "first_event",
) -> np.ndarray:
    """从 event_list 派生 tdc_frame_cube。

    Args:
        event_frame_index: int32 [N]
        event_row: uint16 [N]
        event_col: uint16 [N]
        event_tof_bins: uint16 [N]
        n_frames: 帧数
        roi_h: ROI 高度
        roi_w: ROI 宽度
        empty_pixel_value: 空像素填充值。推荐使用 0；有效 TDC bin 从 1 开始。
        collision_policy: 碰撞处理策略 ("first_event" | "min_tof_bin")

    Returns:
        tdc_cube: uint16 [n_frames, roi_h, roi_w]
    """
    tdc = np.full((n_frames, roi_h, roi_w), empty_pixel_value, dtype=np.uint16)

    if collision_policy == "first_event":
        for k in range(len(event_frame_index)):
            f = int(event_frame_index[k])
            r = int(event_row[k])
            c = int(event_col[k])
            if tdc[f, r, c] == empty_pixel_value:
                tdc[f, r, c] = event_tof_bins[k]
    elif collision_policy == "min_tof_bin":
        for k in range(len(event_frame_index)):
            f = int(event_frame_index[k])
            r = int(event_row[k])
            c = int(event_col[k])
            existing = int(tdc[f, r, c])
            new_bin = int(event_tof_bins[k])
            if existing == empty_pixel_value:
                tdc[f, r, c] = new_bin
            else:
                tdc[f, r, c] = min(existing, new_bin)
    else:
        raise ValueError(f"不支持的 collision_policy: {collision_policy}")

    return tdc


def generate_tdc_frame_cube_from_counts(
    *,
    counts: np.ndarray,
    tdc_bin_width_ns: float,
    tdc_max_count: int,
    timing_jitter_ns: float,
    signal_cube: np.ndarray | None,
    bg_expected_cube: np.ndarray | None,
    dark_expected_cube: np.ndarray | None,
    truth_range_series: np.ndarray | None,
    rng: np.random.Generator,
    empty_pixel_value: int,
    fallback_range_m: float | None = None,
) -> np.ndarray:
    """直接从 frame-level count cube 生成 tdc_frame_cube。

    用于 event_list 因事件数过大被跳过时，避免 TDC 导出产物缺失。
    """
    counts = np.asarray(counts)
    if counts.ndim != 3:
        raise ValueError("counts must have shape [n_frames, roi_h, roi_w]")

    n_frames, roi_h, roi_w = counts.shape
    tdc = np.full((n_frames, roi_h, roi_w), empty_pixel_value, dtype=np.uint16)
    active = counts > 0
    if not np.any(active):
        return tdc

    bin_width_s = tdc_bin_width_ns * 1e-9 if tdc_bin_width_ns > 0 else 0.0
    if truth_range_series is not None and len(truth_range_series) > 0 and bin_width_s > 0:
        ranges = np.asarray(truth_range_series, dtype=np.float64).ravel()
        if ranges.size < n_frames:
            ranges = np.pad(ranges, (0, n_frames - ranges.size), mode="edge")
        tof_bins = np.rint((2.0 * ranges[:n_frames] / SPEED_OF_LIGHT_MS) / bin_width_s).astype(np.int64)
    elif fallback_range_m is not None and fallback_range_m > 0 and bin_width_s > 0:
        fallback_bin = int(np.rint((2.0 * float(fallback_range_m) / SPEED_OF_LIGHT_MS) / bin_width_s))
        tof_bins = np.full(n_frames, fallback_bin, dtype=np.int64)
    else:
        raise ValueError("signal TDC generation requires truth_range_series or a positive fallback_range_m")
    tof_bins = np.clip(tof_bins, 1, max(1, int(tdc_max_count)))

    signal_like = active
    if signal_cube is not None and bg_expected_cube is not None and dark_expected_cube is not None:
        signal = np.asarray(signal_cube, dtype=np.float64)
        bg = np.asarray(bg_expected_cube, dtype=np.float64)
        dark = np.asarray(dark_expected_cube, dtype=np.float64)
        total = signal + bg + dark
        frac_signal = np.divide(signal, total, out=np.zeros_like(signal, dtype=np.float64), where=total > 0)
        signal_like = active & (rng.random(active.shape) < frac_signal)

    noise_like = active & ~signal_like
    if np.any(noise_like):
        tdc[noise_like] = rng.integers(1, max(2, int(tdc_max_count) + 1), size=int(np.count_nonzero(noise_like)), dtype=np.uint16)

    for frame_idx in range(n_frames):
        frame_mask = signal_like[frame_idx]
        if not np.any(frame_mask):
            continue
        base_bin = int(tof_bins[frame_idx])
        if timing_jitter_ns > 0 and tdc_bin_width_ns > 0:
            sigma_bins = max(timing_jitter_ns / tdc_bin_width_ns, 0.25)
            values = np.rint(base_bin + rng.normal(0.0, sigma_bins, size=int(np.count_nonzero(frame_mask)))).astype(np.int64)
            values = np.clip(values, 1, max(1, int(tdc_max_count))).astype(np.uint16)
            tdc[frame_idx][frame_mask] = values
        else:
            tdc[frame_idx][frame_mask] = np.uint16(base_bin)

    return tdc


def generate_synthetic_event_list(
    *,
    counts: np.ndarray,
    dt: float,
    timing_jitter_ns: float,
    tdc_bin_width_ns: float,
    tdc_max_count: int,
    signal_cube: np.ndarray,
    bg_expected_cube: np.ndarray,
    dark_expected_cube: np.ndarray,
    truth_range_series: np.ndarray | None,
    rng: np.random.Generator,
    roi_h: int,
    roi_w: int,
) -> dict[str, np.ndarray]:
    """从 frame-level 数据合成 event_list（含 event_source 标记）。

    当前实现为 synthetic 模式：各分量按期望比例使用 multinomial 采样，signal 事件
    分配 ToF bin（基于每帧 range），background/dark 事件分配随机 bin。

    返回的 dict 包含所有 event 字段数组。
    """
    n_frames, h, w = counts.shape
    event_times: list[np.ndarray] = []
    event_sources: list[np.ndarray] = []
    event_tof: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    event_cols: list[np.ndarray] = []

    # 计算各分量在每帧每像素中的期望比例
    total_per_frame_pixel = signal_cube + bg_expected_cube + dark_expected_cube
    total_safe = np.where(total_per_frame_pixel > 0, total_per_frame_pixel, 1.0)
    frac_signal = signal_cube / total_safe
    frac_bg = bg_expected_cube / total_safe
    frac_dark = dark_expected_cube / total_safe

    bin_width_s = tdc_bin_width_ns * 1e-9 if tdc_bin_width_ns > 0 else 0.0

    # 逐帧 ToF 时间（基于每帧 range，支持动态目标）
    if truth_range_series is not None and len(truth_range_series) > 0:
        ranges = np.asarray(truth_range_series, dtype=np.float64).ravel()
        tof_s_per_frame = 2.0 * ranges / SPEED_OF_LIGHT_MS
    else:
        tof_s_per_frame = np.zeros(n_frames, dtype=np.float64)

    for frame_idx in range(n_frames):
        frame_counts = counts[frame_idx]
        active_pixels = np.argwhere(frame_counts > 0)
        if active_pixels.size == 0:
            continue

        for row, col in active_pixels:
            count = int(frame_counts[row, col])
            if count <= 0:
                continue

            # 按期望比例分配事件来源（multinomial 保证总和等于 count）
            probs = np.array([
                float(frac_signal[frame_idx, row, col]),
                float(frac_bg[frame_idx, row, col]),
                float(frac_dark[frame_idx, row, col]),
            ])
            s = float(probs.sum())
            if not np.isfinite(s) or s <= 0.0:
                probs = np.array([0.0, 0.0, 1.0])  # 退化到 unknown（标记为 dark）
            else:
                probs = probs / s
            n_signal, n_bg, n_dark = rng.multinomial(count, probs)

            # 按帧级 range 计算该帧的期望 ToF bin（带边界保护）
            range_idx = min(frame_idx, len(tof_s_per_frame) - 1)
            frame_tof_bin = int(round(tof_s_per_frame[range_idx] / bin_width_s)) if bin_width_s > 0 else 1
            frame_tof_bin = max(1, min(frame_tof_bin, tdc_max_count))

            # signal events: ToF bin 围绕期望值，抖动由 timing_jitter_ns 决定
            if n_signal > 0:
                times_signal = frame_idx * dt + np.sort(rng.uniform(0.0, dt, size=n_signal))
                if bin_width_s > 0:
                    sigma_bins = max(timing_jitter_ns / tdc_bin_width_ns, 0.25)
                    signal_bins = (frame_tof_bin + rng.normal(0.0, sigma_bins, size=n_signal)).round().astype(np.int32)
                    signal_bins = np.clip(signal_bins, 1, tdc_max_count)
                else:
                    signal_bins = np.ones(n_signal, dtype=np.int32)

                if timing_jitter_ns > 0:
                    times_signal = times_signal + rng.normal(0.0, timing_jitter_ns * 1e-9, size=n_signal)

                upper = float(np.nextafter(np.float32((frame_idx + 1) * dt), np.float32(frame_idx * dt)))
                times_signal = np.clip(times_signal, frame_idx * dt, upper)

                event_times.append(times_signal.astype(np.float32))
                event_sources.append(np.full(n_signal, EventSource.signal.value, dtype=np.uint8))
                event_tof.append(signal_bins.astype(np.uint16))
                event_rows.append(np.full(n_signal, row, dtype=np.uint16))
                event_cols.append(np.full(n_signal, col, dtype=np.uint16))

            # background events: 随机 ToF bin
            if n_bg > 0:
                times_bg = frame_idx * dt + np.sort(rng.uniform(0.0, dt, size=n_bg))
                bg_bins = rng.integers(1, tdc_max_count + 1, size=n_bg, dtype=np.int32)

                if timing_jitter_ns > 0:
                    times_bg = times_bg + rng.normal(0.0, timing_jitter_ns * 1e-9, size=n_bg)

                upper = float(np.nextafter(np.float32((frame_idx + 1) * dt), np.float32(frame_idx * dt)))
                times_bg = np.clip(times_bg, frame_idx * dt, upper)

                event_times.append(times_bg.astype(np.float32))
                event_sources.append(np.full(n_bg, EventSource.background.value, dtype=np.uint8))
                event_tof.append(bg_bins.astype(np.uint16))
                event_rows.append(np.full(n_bg, row, dtype=np.uint16))
                event_cols.append(np.full(n_bg, col, dtype=np.uint16))

            # dark events: 随机 ToF bin
            if n_dark > 0:
                times_dark = frame_idx * dt + np.sort(rng.uniform(0.0, dt, size=n_dark))
                dark_bins = rng.integers(1, tdc_max_count + 1, size=n_dark, dtype=np.int32)

                if timing_jitter_ns > 0:
                    times_dark = times_dark + rng.normal(0.0, timing_jitter_ns * 1e-9, size=n_dark)

                upper = float(np.nextafter(np.float32((frame_idx + 1) * dt), np.float32(frame_idx * dt)))
                times_dark = np.clip(times_dark, frame_idx * dt, upper)

                event_times.append(times_dark.astype(np.float32))
                event_sources.append(np.full(n_dark, EventSource.dark.value, dtype=np.uint8))
                event_tof.append(dark_bins.astype(np.uint16))
                event_rows.append(np.full(n_dark, row, dtype=np.uint16))
                event_cols.append(np.full(n_dark, col, dtype=np.uint16))

    if not event_times:
        return {
            "event_times_s": np.zeros(0, dtype=np.float32),
            "event_frame_index": np.zeros(0, dtype=np.int32),
            "event_row": np.zeros(0, dtype=np.uint16),
            "event_col": np.zeros(0, dtype=np.uint16),
            "event_pixel": np.zeros(0, dtype=np.int32),
            "event_tof_bins": np.zeros(0, dtype=np.uint16),
            "event_source": np.zeros(0, dtype=np.uint8),
        }

    all_times = np.concatenate(event_times)
    all_sources = np.concatenate(event_sources)
    all_tof = np.concatenate(event_tof)
    all_rows = np.concatenate(event_rows)
    all_cols = np.concatenate(event_cols)

    # 按时间排序
    order = np.argsort(all_times)
    all_times = all_times[order]
    all_sources = all_sources[order]
    all_tof = all_tof[order]
    all_rows = all_rows[order]
    all_cols = all_cols[order]

    # 计算 frame_index from times
    frame_index = np.clip(
        (all_times / dt).astype(np.int32), 0, n_frames - 1
    )
    all_pixels = (all_rows.astype(np.int32) * roi_w + all_cols.astype(np.int32)).astype(np.int32)

    return {
        "event_times_s": all_times.astype(np.float32),
        "event_frame_index": frame_index.astype(np.int32),
        "event_row": all_rows.astype(np.uint16),
        "event_col": all_cols.astype(np.uint16),
        "event_pixel": all_pixels,
        "event_tof_bins": all_tof.astype(np.uint16),
        "event_source": all_sources.astype(np.uint8),
    }
