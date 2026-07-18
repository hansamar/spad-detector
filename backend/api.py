from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.capabilities import detect_compute_capabilities
from backend.convert import params_from_request, result_to_response, result_to_summary_response
from backend.exporters import ExportFormat
from backend.jobs import create_simulation_job, get_simulation_job, get_simulation_job_artifact, get_simulation_job_metadata
from backend.models import SimulateJobStatusResponse, SimulateRequest, SimulateResponse, SimulateSummaryResponse
from sim.active_imaging_sim import simulate_active_spad

router = APIRouter(prefix="/api")


@router.get("/capabilities")
async def get_capabilities():
    return detect_compute_capabilities()


@router.post("/simulate", response_model=SimulateResponse)
async def run_simulation(req: SimulateRequest):
    try:
        params = params_from_request(req)
        params.summary_only = False
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(simulate_active_spad, params)
        return result_to_response(result, None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate/summary", response_model=SimulateSummaryResponse)
async def run_simulation_summary(req: SimulateRequest):
    try:
        params = params_from_request(req)
        params.summary_only = True
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(simulate_active_spad, params)
        return result_to_summary_response(result, None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate/jobs", response_model=SimulateJobStatusResponse)
async def create_simulation_run_job(req: SimulateRequest):
    return create_simulation_job(req)


@router.get("/simulate/jobs/{job_id}", response_model=SimulateJobStatusResponse)
async def get_simulation_run_job(job_id: str):
    status = get_simulation_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@router.get("/simulate/jobs/{job_id}/metadata")
async def get_simulation_run_job_metadata(job_id: str):
    """返回 completed job 的 metadata.json。"""
    path = get_simulation_job_metadata(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="metadata not found or job not completed")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"metadata_{job_id}.json",
    )


@router.get("/simulate/jobs/{job_id}/download")
async def download_simulation_run_job(
    job_id: str,
    format: str | None = Query(default=None, alias="format"),
):
    """按格式下载仿真产物。

    Args:
        job_id: 任务 ID
        format: 导出格式 (count_cube | tdc_frame_cube | event_list | bundle)，默认 count_cube

    返回相应的 .bin / .npz / .zip 文件。
    """
    artifact = get_simulation_job_artifact(job_id, format=format)
    if artifact is None:
        valid_formats = [f.value for f in ExportFormat]
        raise HTTPException(
            status_code=404,
            detail=f"artifact not found for format '{format or 'count_cube'}' — job may not be completed or format not supported. Valid: {valid_formats}",
        )
    # 确定文件名和 media type
    media_type = "application/octet-stream"
    suffix = artifact.suffix
    if suffix == ".json":
        media_type = "application/json"
    elif suffix == ".zip":
        media_type = "application/zip"
    elif suffix == ".npz":
        media_type = "application/octet-stream"

    filename = artifact.name
    return FileResponse(
        artifact,
        media_type=media_type,
        filename=filename,
    )
