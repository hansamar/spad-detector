from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from backend.convert import params_from_request, result_to_summary_response
from backend.exporters import (
    ExportFormat,
    build_metadata,
    generate_synthetic_event_list,
    generate_tdc_frame_cube_from_counts,
    generate_tdc_frame_cube_from_event_list,
    write_count_cube_bin,
    write_tdc_frame_cube_bin,
    write_event_npz,
    write_bundle_zip,
)
from backend.models import SimulateJobStatusResponse, SimulateRequest, SimulateSummaryResponse
from sim.active_imaging_sim import simulate_active_spad


ARTIFACT_DIR = Path(os.environ.get("SPAD_SIM_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "output" / "backend_jobs"))
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("SPAD_JOB_WORKERS", "1")))
_lock = threading.Lock()
_jobs: dict[str, dict] = {}

# 每个异步任务产生的产物文件列表
_ARTIFACT_FILES = {
    ExportFormat.count_cube: "counts.bin",
    ExportFormat.tdc_frame_cube: "tdc_frame_cube.bin",
    ExportFormat.event_list: "events.npz",
    ExportFormat.bundle: "bundle.zip",
}


def _now() -> float:
    return time.time()


def _response_from_record(record: dict) -> SimulateJobStatusResponse:
    download_url = None
    if record["status"] == "completed" and record.get("artifacts_ready", False):
        download_url = f"/api/simulate/jobs/{record['job_id']}/download"
    return SimulateJobStatusResponse(
        job_id=record["job_id"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        summary=record.get("summary"),
        result=record.get("result"),
        error=record.get("error"),
        download_url=download_url,
    )


def create_simulation_job(req: SimulateRequest) -> SimulateJobStatusResponse:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "summary": None,
        "result": None,
        "error": None,
        "artifacts_dir": str(ARTIFACT_DIR / f"{job_id}_artifacts"),
        "artifacts_ready": False,
    }
    with _lock:
        _jobs[job_id] = record
    _executor.submit(_run_job, job_id, req)
    return _response_from_record(record)


def get_simulation_job(job_id: str) -> SimulateJobStatusResponse | None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            return None
        return _response_from_record(dict(record))


def _get_artifact_path(job_id: str, format: ExportFormat | str) -> Path | None:
    """根据格式返回产物文件路径。"""
    with _lock:
        record = _jobs.get(job_id)
        if record is None or record["status"] != "completed":
            return None
        artifacts_dir = Path(record["artifacts_dir"])

    if isinstance(format, str):
        format = ExportFormat(format)

    filename = _ARTIFACT_FILES.get(format)
    if filename is None:
        return None
    path = artifacts_dir / filename
    return path if path.exists() else None


def get_simulation_job_artifact(job_id: str, format: str | None = None) -> Path | None:
    """获取产物文件路径。默认返回 count_cube .bin。

    指定格式不存在时返回 None，调用方应据此报错，不应静默回退。
    """
    try:
        fmt = ExportFormat(format) if format else ExportFormat.count_cube
    except ValueError:
        return None
    return _get_artifact_path(job_id, fmt)


def get_simulation_job_metadata(job_id: str) -> Path | None:
    """获取 metadata.json 文件路径。"""
    with _lock:
        record = _jobs.get(job_id)
        if record is None or record["status"] != "completed":
            return None
        path = Path(record["artifacts_dir"]) / "counts.metadata.json"
    return path if path.exists() else None


def _set_status(job_id: str, **updates) -> None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            return
        record.update(updates)
        record["updated_at"] = _now()


def _run_job(job_id: str, req: SimulateRequest) -> None:
    _set_status(job_id, status="running")
    try:
        params = params_from_request(req)
        persist_artifacts = bool(req.persist_artifacts) if req.persist_artifacts is not None else True
        params.summary_only = not persist_artifacts
        if persist_artifacts:
            params.save_truth_series = True
        result = simulate_active_spad(params)
        summary = result_to_summary_response(result, None)
        if not persist_artifacts:
            _set_status(job_id, status="completed", summary=summary, result=None, error=None, artifacts_ready=False)
            return

        with _lock:
            record = _jobs[job_id]
            artifacts_dir = Path(record["artifacts_dir"])

        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # ── 从结果中提取关键数据 ──
            counts = np.asarray(result["counts"], dtype=np.uint16)
            n_frames = int(result["n_frames"])
            roi_h = int(result["roi_h"])
            roi_w = int(result["roi_w"])
            sample_rate_hz = float(result["sample_rate_hz"])
            obs_time = float(n_frames / sample_rate_hz) if sample_rate_hz > 0 else 0.0

            signal_cube = np.asarray(result["signal_cube"], dtype=np.float32)
            bg_cube = np.asarray(result["bg_expected_cube"], dtype=np.float32)
            dark_cube = np.asarray(result["dark_expected_cube"], dtype=np.float32)

            detector_summary = result.get("detector_summary", {})
            qe = float(detector_summary.get("quantum_efficiency", 0.0))
            tdc_bin_ns = float(detector_summary.get("tdc_bin_width_ns", 0.0))
            tdc_max = int(detector_summary.get("max_count_per_frame", 0))
            timing_jitter_ns = float(params.spad.timing_jitter_ns)
            dead_time_ns = float(params.spad.dead_time_ns)
            seed = int(result.get("seed", 0))
            preset = str(result.get("detector_preset", ""))
            dt = 1.0 / sample_rate_hz if sample_rate_hz > 0 else 0.0

            # ── 构建 metadata ──
            summary_dict = summary.model_dump(mode="json")

            # 自定义形状信息
            custom_shape_meta = None
            if params.target.custom_shape_x is not None and len(params.target.custom_shape_x) > 0:
                custom_shape_meta = {
                    "enabled": True,
                    "num_points": len(params.target.custom_shape_x),
                    "aspect_ratio": params.target.custom_shape_aspect_ratio,
                    "sampling": "deterministic_stride",
                }

            count_metadata = build_metadata(
                format=ExportFormat.count_cube,
                n_frames=n_frames,
                roi_h=roi_h,
                roi_w=roi_w,
                sample_rate_hz=sample_rate_hz,
                observation_time_s=obs_time,
                dtype="uint16",
                detector_preset=preset,
                quantum_efficiency=qe,
                dead_time_ns=dead_time_ns,
                timing_jitter_ns=timing_jitter_ns,
                tdc_bin_width_ns=tdc_bin_ns,
                tdc_max_count=tdc_max,
                random_seed=seed,
                simulation_mode=str(result.get("output_mode", "frame")),
                warnings=list(result.get("warnings", [])),
                assumptions=list(result.get("assumptions", [])),
                custom_shape=custom_shape_meta,
            )

            # ── 可选生成 event_list ──
            want_events = bool(req.include_event_list) if req.include_event_list is not None else False
            event_dict = None
            event_metadata = None
            max_events = req.max_event_count if req.max_event_count is not None else 10_000_000
            event_list_limit_warning = None

            if want_events:
                total_observed = int(np.sum(counts))
                if total_observed > max_events:
                    # 超限：跳过事件生成，写入 warning
                    event_list_limit_warning = (
                        f"event_list skipped: {total_observed} events exceeds max_event_count ({max_events})"
                    )
                    count_metadata["warnings"].append(event_list_limit_warning)
                else:
                    rng = np.random.default_rng(seed + 1)
                    event_dict = generate_synthetic_event_list(
                        counts=counts,
                        dt=dt,
                        timing_jitter_ns=timing_jitter_ns,
                        tdc_bin_width_ns=tdc_bin_ns,
                        tdc_max_count=tdc_max,
                        signal_cube=signal_cube,
                        bg_expected_cube=bg_cube,
                        dark_expected_cube=dark_cube,
                        truth_range_series=result.get("truth_range_series"),
                        rng=rng,
                        roi_h=roi_h,
                        roi_w=roi_w,
                    )

                    event_metadata = build_metadata(
                        format=ExportFormat.event_list,
                        n_frames=n_frames,
                        roi_h=roi_h,
                        roi_w=roi_w,
                        sample_rate_hz=sample_rate_hz,
                        observation_time_s=obs_time,
                        dtype="mixed",
                        detector_preset=preset,
                        quantum_efficiency=qe,
                        dead_time_ns=dead_time_ns,
                        timing_jitter_ns=timing_jitter_ns,
                        tdc_bin_width_ns=tdc_bin_ns,
                        tdc_max_count=tdc_max,
                        random_seed=seed,
                        simulation_mode=str(result.get("output_mode", "frame")),
                        event_generation="synthetic_from_frame_counts",
                        event_warning="Event timestamps and TDC bins are synthesized from sampled frame counts and do not represent full event-level TCSPC transport.",
                        event_fields={
                            "event_times_s": "float32 seconds since simulation start",
                            "event_frame_index": "int32",
                            "event_row": "uint16",
                            "event_col": "uint16",
                            "event_pixel": "int32, row * roi_w + col",
                            "event_tof_bins": "uint16",
                            "event_source": "uint8",
                        },
                        event_source_encoding={
                            "0": "unknown",
                            "1": "signal",
                            "2": "background",
                            "3": "dark",
                            "4": "afterpulse",
                            "5": "crosstalk",
                        },
                        warnings=list(result.get("warnings", [])),
                    )

                    write_event_npz(
                        artifacts_dir / "events",
                        **event_dict,
                        metadata=event_metadata,
                        summary=summary_dict,
                    )

            # ── 可选生成 tdc_frame_cube ──
            want_tdc = bool(req.include_tdc_frame_cube) if req.include_tdc_frame_cube is not None else False
            tdc_cube = None
            tdc_metadata = None

            if want_tdc:
                empty_val = 0
                tdc_warnings = list(result.get("warnings", []))
                if event_dict is not None:
                    tdc_cube = generate_tdc_frame_cube_from_event_list(
                        event_frame_index=event_dict["event_frame_index"],
                        event_row=event_dict["event_row"],
                        event_col=event_dict["event_col"],
                        event_tof_bins=event_dict["event_tof_bins"],
                        n_frames=n_frames,
                        roi_h=roi_h,
                        roi_w=roi_w,
                        empty_pixel_value=empty_val,
                        collision_policy="first_event",
                    )
                    event_generation = "derived_from_event_list"
                else:
                    tdc_cube = generate_tdc_frame_cube_from_counts(
                        counts=counts,
                        tdc_bin_width_ns=tdc_bin_ns,
                        tdc_max_count=tdc_max,
                        timing_jitter_ns=timing_jitter_ns,
                        signal_cube=signal_cube,
                        bg_expected_cube=bg_cube,
                        dark_expected_cube=dark_cube,
                        truth_range_series=result.get("truth_range_series"),
                        fallback_range_m=float(params.target.target_range_m),
                        rng=np.random.default_rng(seed + 2),
                        empty_pixel_value=empty_val,
                    )
                    event_generation = "direct_from_frame_counts"
                    tdc_warnings.append(
                        "tdc_frame_cube generated directly from frame counts because event_list was not materialized"
                    )
                    if event_list_limit_warning is not None:
                        tdc_warnings.append(event_list_limit_warning)

                tdc_metadata = build_metadata(
                    format=ExportFormat.tdc_frame_cube,
                    n_frames=n_frames,
                    roi_h=roi_h,
                    roi_w=roi_w,
                    sample_rate_hz=sample_rate_hz,
                    observation_time_s=obs_time,
                    dtype="uint16",
                    detector_preset=preset,
                    quantum_efficiency=qe,
                    dead_time_ns=dead_time_ns,
                    timing_jitter_ns=timing_jitter_ns,
                    tdc_bin_width_ns=tdc_bin_ns,
                    tdc_max_count=tdc_max,
                    empty_pixel_value=empty_val,
                    collision_policy="first_event",
                    random_seed=seed,
                    simulation_mode=str(result.get("output_mode", "frame")),
                    event_generation=event_generation,
                    warnings=tdc_warnings,
                )

                write_tdc_frame_cube_bin(
                    artifacts_dir / "tdc_frame_cube",
                    tdc_cube,
                    metadata=tdc_metadata,
                    summary=summary_dict,
                )

            # ── 写入 bundle（仅包含已生成的产物）──
            write_bundle_zip(
                artifacts_dir / "bundle",
                counts=counts,
                metadata=count_metadata,
                summary=summary_dict,
                tdc_cube=tdc_cube,
                tdc_metadata=tdc_metadata,
                events=event_dict,
                events_metadata=event_metadata,
            )

            # ── 写入 count_cube（在 event/TDC 之后，确保所有 warning 已追加到 metadata）──
            write_count_cube_bin(
                artifacts_dir / "counts",
                counts,
                metadata=count_metadata,
                summary=summary_dict,
            )

        except OSError as exc:
            _set_status(job_id, status="failed", error=f"写入产物文件失败: {exc}")
            return
        _set_status(job_id, status="completed", summary=summary, result=None, error=None, artifacts_ready=True)
    except Exception as exc:
        _set_status(job_id, status="failed", error=str(exc))
