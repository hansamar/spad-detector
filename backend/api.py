from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.capabilities import detect_compute_capabilities
from backend.convert import params_from_request, result_to_response, result_to_summary_response
from backend.jobs import create_simulation_job, get_simulation_job, get_simulation_job_artifact
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
        result = await run_in_threadpool(simulate_active_spad, params)
        return result_to_response(result, None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate/summary", response_model=SimulateSummaryResponse)
async def run_simulation_summary(req: SimulateRequest):
    try:
        params = params_from_request(req)
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


@router.get("/simulate/jobs/{job_id}/download")
async def download_simulation_run_job(job_id: str):
    artifact = get_simulation_job_artifact(job_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="completed artifact not found")
    return FileResponse(
        artifact,
        media_type="application/octet-stream",
        filename=f"spad_simulation_{job_id}.bin",
    )
