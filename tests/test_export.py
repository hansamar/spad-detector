"""测试数据导出、metadata 生成和格式符合性。"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.exporters import (
    ExportFormat,
    EventSource,
    build_metadata,
    generate_synthetic_event_list,
    generate_tdc_frame_cube_from_event_list,
    write_count_cube_bin,
    write_tdc_frame_cube_bin,
    write_event_npz,
    write_bundle_zip,
    frame_tof_diagnostics,
    SPEED_OF_LIGHT_MS,
)


def test_frame_tof_diagnostics() -> None:
    diag = frame_tof_diagnostics(
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        timing_jitter_ns=0.05,
    )
    assert np.isclose(diag["range_bin_m"], 0.0384, atol=0.001)
    assert np.isclose(diag["max_unambiguous_range_m"], 314.5, atol=0.5)
    assert diag["timing_jitter_range_sigma_m"] > 0


def test_build_metadata_count_cube() -> None:
    meta = build_metadata(
        format=ExportFormat.count_cube,
        n_frames=100,
        roi_h=32,
        roi_w=32,
        sample_rate_hz=50000,
        observation_time_s=0.002,
        detector_preset="PF32",
        quantum_efficiency=0.274,
        dead_time_ns=10.0,
        timing_jitter_ns=0.05,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        random_seed=42,
    )
    assert meta["format"] == "count_cube"
    assert meta["shape"] == [100, 32, 32]
    assert meta["dtype"] == "uint16"
    assert meta["layout"] == "frame-major"
    assert meta["empty_pixel_value"] is None
    assert meta["random_seed"] == 42
    assert meta["pde"] == 0.274
    # JSON serializable
    json.dumps(meta)


def test_build_metadata_tdc_cube() -> None:
    meta = build_metadata(
        format=ExportFormat.tdc_frame_cube,
        n_frames=50,
        roi_h=16,
        roi_w=16,
        sample_rate_hz=1000,
        observation_time_s=0.05,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        empty_pixel_value=8193,
        collision_policy="first_event",
        random_seed=7,
    )
    assert meta["format"] == "tdc_frame_cube"
    assert meta["empty_pixel_value"] == 8193
    assert meta["valid_tdc_range"] == [1, 8191]
    assert meta["collision_policy"] == "first_event"
    assert meta["source"] == "generated_from_event_list"


def test_build_metadata_event_list() -> None:
    meta = build_metadata(
        format=ExportFormat.event_list,
        n_frames=50,
        roi_h=16,
        roi_w=16,
        sample_rate_hz=1000,
        observation_time_s=0.05,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        event_generation="synthetic_from_frame_counts",
        event_warning="warning text",
        event_fields={"a": "b"},
        event_source_encoding={"1": "signal"},
        random_seed=7,
    )
    assert meta["format"] == "event_list"
    assert meta["event_generation"] == "synthetic_from_frame_counts"
    assert meta["event_generation_warning"] == "warning text"
    assert meta["fields"] == {"a": "b"}


def test_count_cube_readback_shape() -> None:
    """验证 count_cube .bin 写入和读回 shape 一致。"""
    counts = np.random.poisson(5, size=(20, 8, 8)).astype(np.uint16)
    meta = build_metadata(
        format=ExportFormat.count_cube,
        n_frames=20,
        roi_h=8,
        roi_w=8,
        sample_rate_hz=1000,
        observation_time_s=0.02,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        random_seed=1,
    )

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "counts"
        write_count_cube_bin(base, counts, metadata=meta)
        assert base.with_suffix(".bin").exists()
        assert base.with_suffix(".metadata.json").exists()

        # 读回
        loaded = np.fromfile(str(base.with_suffix(".bin")), dtype=np.uint16)
        cube = loaded.reshape(meta["shape"])
        assert cube.shape == (20, 8, 8)
        assert cube.dtype == np.uint16
        assert np.array_equal(cube, counts)


def test_tdc_cube_sentinel() -> None:
    """验证 tdc_frame_cube sentinel 正确。"""
    empty_val = 8193
    tdc = np.full((2, 4, 4), empty_val, dtype=np.uint16)
    tdc[0, 1, 2] = 100
    tdc[1, 3, 0] = 500

    meta = build_metadata(
        format=ExportFormat.tdc_frame_cube,
        n_frames=2,
        roi_h=4,
        roi_w=4,
        sample_rate_hz=100,
        observation_time_s=0.02,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        empty_pixel_value=empty_val,
        collision_policy="first_event",
        random_seed=1,
    )

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "tdc"
        write_tdc_frame_cube_bin(base, tdc, metadata=meta)
        assert base.with_suffix(".bin").exists()

        loaded = np.fromfile(str(base.with_suffix(".bin")), dtype=np.uint16).reshape(2, 4, 4)
        assert int(loaded[0, 1, 2]) == 100
        assert int(loaded[1, 3, 0]) == 500
        assert np.sum(loaded == empty_val) == 30  # 32 pixels total, 2 are non-empty


def test_event_list_field_lengths() -> None:
    """验证 event_list 字段长度一致。"""
    rng = np.random.default_rng(42)
    counts = rng.poisson(2, size=(5, 4, 4)).astype(np.uint16)
    signal_cube = np.full((5, 4, 4), 1.0, dtype=np.float32)
    bg_cube = np.full((5, 4, 4), 0.5, dtype=np.float32)
    dark_cube = np.full((5, 4, 4), 0.5, dtype=np.float32)

    events = generate_synthetic_event_list(
        counts=counts,
        dt=0.001,
        timing_jitter_ns=0.05,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        signal_cube=signal_cube,
        bg_expected_cube=bg_cube,
        dark_expected_cube=dark_cube,
        truth_range_series=np.full(5, 50.0, dtype=np.float64),
        rng=rng,
        roi_h=4,
        roi_w=4,
    )

    n = len(events["event_times_s"])
    assert n > 0, "should have at least some events"
    for field in ("event_frame_index", "event_row", "event_col", "event_pixel", "event_tof_bins", "event_source"):
        assert len(events[field]) == n, f"field {field} length mismatch: {len(events[field])} != {n}"


def test_event_pixel_consistency() -> None:
    """验证 event_pixel = row × roi_w + col。"""
    n = 10
    roi_w = 8
    rows = np.random.randint(0, 4, size=n, dtype=np.uint16)
    cols = np.random.randint(0, roi_w, size=n, dtype=np.uint16)
    pixels = (rows.astype(np.int32) * roi_w + cols.astype(np.int32)).astype(np.int32)

    for k in range(n):
        assert int(pixels[k]) == int(rows[k]) * roi_w + int(cols[k])


def test_tdc_derived_from_events() -> None:
    """验证从 event_list 派生 tdc_frame_cube。"""
    n_frames, roi_h, roi_w = 3, 4, 4
    empty_val = 8193

    # 构造简单的 event list
    event_frame = np.array([0, 0, 1, 2], dtype=np.int32)
    event_row = np.array([0, 1, 2, 3], dtype=np.uint16)
    event_col = np.array([0, 1, 2, 3], dtype=np.uint16)
    event_bins = np.array([100, 200, 300, 400], dtype=np.uint16)

    tdc = generate_tdc_frame_cube_from_event_list(
        event_frame_index=event_frame,
        event_row=event_row,
        event_col=event_col,
        event_tof_bins=event_bins,
        n_frames=n_frames,
        roi_h=roi_h,
        roi_w=roi_w,
        empty_pixel_value=empty_val,
        collision_policy="first_event",
    )

    assert tdc.shape == (3, 4, 4)
    assert int(tdc[0, 0, 0]) == 100
    assert int(tdc[0, 1, 1]) == 200
    assert int(tdc[1, 2, 2]) == 300
    assert int(tdc[2, 3, 3]) == 400
    assert np.sum(tdc == empty_val) == 44  # 48 total - 4 events


def test_tdc_collision_first_event() -> None:
    """验证 first_event collision policy 保留第一个。"""
    empty_val = 8193
    # 两个事件落在同一 pixel (0, 0, 0)
    event_frame = np.array([0, 0], dtype=np.int32)
    event_row = np.array([0, 0], dtype=np.uint16)
    event_col = np.array([0, 0], dtype=np.uint16)
    event_bins = np.array([100, 50], dtype=np.uint16)  # 先 100，后 50（更小）

    tdc = generate_tdc_frame_cube_from_event_list(
        event_frame_index=event_frame,
        event_row=event_row,
        event_col=event_col,
        event_tof_bins=event_bins,
        n_frames=1,
        roi_h=1,
        roi_w=1,
        empty_pixel_value=empty_val,
        collision_policy="first_event",
    )
    assert int(tdc[0, 0, 0]) == 100  # 保留第一个


def test_tdc_collision_min_tof() -> None:
    """验证 min_tof_bin collision policy。"""
    empty_val = 8193
    event_frame = np.array([0, 0], dtype=np.int32)
    event_row = np.array([0, 0], dtype=np.uint16)
    event_col = np.array([0, 0], dtype=np.uint16)
    event_bins = np.array([100, 50], dtype=np.uint16)

    tdc = generate_tdc_frame_cube_from_event_list(
        event_frame_index=event_frame,
        event_row=event_row,
        event_col=event_col,
        event_tof_bins=event_bins,
        n_frames=1,
        roi_h=1,
        roi_w=1,
        empty_pixel_value=empty_val,
        collision_policy="min_tof_bin",
    )
    assert int(tdc[0, 0, 0]) == 50  # 保留更小的


def test_tdc_valid_range() -> None:
    """验证 TDC bin 值在 [1, tdcMaxCount] 范围。"""
    rng = np.random.default_rng(99)
    tdc_max = 8191
    counts = rng.poisson(5, size=(10, 4, 4)).astype(np.uint16)
    sig = np.full((10, 4, 4), 2.0, dtype=np.float32)
    bg = np.full((10, 4, 4), 1.5, dtype=np.float32)
    dark = np.full((10, 4, 4), 1.5, dtype=np.float32)

    events = generate_synthetic_event_list(
        counts=counts,
        dt=0.001,
        timing_jitter_ns=0,
        tdc_bin_width_ns=0.256,
        tdc_max_count=tdc_max,
        signal_cube=sig,
        bg_expected_cube=bg,
        dark_expected_cube=dark,
        truth_range_series=np.full(10, 50.0, dtype=np.float64),
        rng=rng,
        roi_h=4,
        roi_w=4,
    )

    # 验证 TDC bin 在有效范围内
    tof = events["event_tof_bins"]
    assert len(tof) > 0
    assert np.all(tof >= 1), f"min tof bin: {tof.min()}"
    assert np.all(tof <= tdc_max), f"max tof bin: {tof.max()}"

    # 验证事件总数与 counts 一致（multinomial）
    assert len(events["event_times_s"]) == int(np.sum(counts)), \
        f"event count {len(events['event_times_s'])} != counts sum {int(np.sum(counts))}"

    # 验证所有字段长度一致
    n = len(events["event_times_s"])
    for field in ("event_frame_index", "event_row", "event_col", "event_pixel", "event_tof_bins", "event_source"):
        assert len(events[field]) == n, f"{field} length mismatch"


def test_bundle_zip_contents() -> None:
    """验证 bundle.zip 包含必需文件。"""
    rng = np.random.default_rng(1)
    counts = rng.poisson(3, size=(4, 4, 4)).astype(np.uint16)
    meta = build_metadata(
        format=ExportFormat.count_cube,
        n_frames=4,
        roi_h=4,
        roi_w=4,
        sample_rate_hz=100,
        observation_time_s=0.04,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        random_seed=1,
    )
    summary = {"n_frames": 4}

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "bundle"
        write_bundle_zip(zip_path, counts=counts, metadata=meta, summary=summary)
        assert zip_path.with_suffix(".zip").exists()

        with zipfile.ZipFile(str(zip_path.with_suffix(".zip")), "r") as zf:
            names = zf.namelist()
            assert "counts.bin" in names
            assert "metadata.json" in names
            assert "summary.json" in names


def test_export_format_enum() -> None:
    assert ExportFormat("count_cube") == ExportFormat.count_cube
    assert ExportFormat("bundle") == ExportFormat.bundle
    try:
        ExportFormat("invalid")
        assert False, "should have raised"
    except ValueError:
        pass


def test_event_source_encoding() -> None:
    """验证 event_source 值在合法范围。"""
    rng = np.random.default_rng(77)
    counts = rng.poisson(5, size=(5, 4, 4)).astype(np.uint16)
    sig = np.full((5, 4, 4), 2.0, dtype=np.float32)
    bg = np.full((5, 4, 4), 1.0, dtype=np.float32)
    dark = np.full((5, 4, 4), 2.0, dtype=np.float32)

    events = generate_synthetic_event_list(
        counts=counts,
        dt=0.001,
        timing_jitter_ns=0,
        tdc_bin_width_ns=0.256,
        tdc_max_count=8191,
        signal_cube=sig,
        bg_expected_cube=bg,
        dark_expected_cube=dark,
        truth_range_series=np.full(5, 50.0, dtype=np.float64),
        rng=rng,
        roi_h=4,
        roi_w=4,
    )

    sources = events["event_source"]
    assert len(sources) > 0
    unique_sources = set(int(s) for s in sources)
    # 仅 signal(1), background(2), dark(3) 应出现
    for s in unique_sources:
        assert s in (EventSource.signal.value, EventSource.background.value, EventSource.dark.value)


if __name__ == "__main__":
    tests = [
        name for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for test_name in tests:
        try:
            globals()[test_name]()
            print(f"PASS {test_name}")
        except Exception as exc:
            print(f"FAIL {test_name}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
