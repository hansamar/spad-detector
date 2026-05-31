from __future__ import annotations

import os
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from backend.convert import params_from_request, result_to_summary_response
from backend.models import SimulateJobStatusResponse, SimulateRequest, SimulateSummaryResponse
from sim.active_imaging_sim import simulate_active_spad


ARTIFACT_DIR = Path(os.environ.get("SPAD_SIM_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "output" / "backend_jobs"))
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("SPAD_JOB_WORKERS", "1")))
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _job_download_url(job_id: str, status: str) -> str | None:
    return f"/api/simulate/jobs/{job_id}/download" if status == "completed" else None


def _response_from_record(record: dict) -> SimulateJobStatusResponse:
    return SimulateJobStatusResponse(
        job_id=record["job_id"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        summary=record.get("summary"),
        result=record.get("result"),
        error=record.get("error"),
        download_url=_job_download_url(record["job_id"], record["status"]),
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
        "artifact_path": str(ARTIFACT_DIR / f"{job_id}.bin"),
        "summary_path": str(ARTIFACT_DIR / f"{job_id}.summary.json"),
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


def get_simulation_job_artifact(job_id: str) -> Path | None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None or record["status"] != "completed":
            return None
        path = Path(record["artifact_path"])
    return path if path.exists() else None


def _set_status(job_id: str, **updates) -> None:
    with _lock:
        record = _jobs[job_id]
        record.update(updates)
        record["updated_at"] = _now()


def _run_job(job_id: str, req: SimulateRequest) -> None:
    _set_status(job_id, status="running")
    try:
        params, scenario_info = params_from_request(req)
        result = simulate_active_spad(params)
        result["scenario_id"] = None
        summary = result_to_summary_response(result, scenario_info)

        with _lock:
            record = _jobs[job_id]
            artifact_path = record["artifact_path"]
            summary_path = record["summary_path"]

        np.asarray(result["counts"], dtype=np.uint16).tofile(artifact_path)
        Path(summary_path).write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _set_status(job_id, status="completed", summary=summary, result=None, error=None)
    except Exception as exc:
        _set_status(job_id, status="failed", error=str(exc))
