import secrets
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.analysis import AnalyzeResponse, JobResponse

router = APIRouter(prefix="/api/v1")


async def authenticate(request: Request, x_api_key: str | None = Header(None)) -> None:
    expected = request.app.state.settings.api_key
    if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Missing or invalid X-API-Key")


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(authenticate)])
async def analyze(request: Request, file: Annotated[UploadFile, File()]) -> AnalyzeResponse:
    settings = request.app.state.settings
    if Path(file.filename or "").suffix.lower() != ".mov":
        await file.close()
        raise HTTPException(415, "Only .mov recordings are accepted")
    job_id = str(uuid4())
    path = settings.upload_dir / f"{job_id}.mov"
    queued = False
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        async with aiofiles.open(path, "xb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "Upload exceeds configured size limit")
                await output.write(chunk)
        if total == 0:
            raise HTTPException(422, "Empty recording")
        await run_in_threadpool(request.app.state.jobs.enqueue, job_id, path)
        queued = True
        return AnalyzeResponse(job_id=job_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Upload storage or job queue unavailable") from exc
    finally:
        await file.close()
        if not queued:
            path.unlink(missing_ok=True)


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(authenticate)],
)
async def job_status(request: Request, job_id: UUID) -> JobResponse:
    try:
        result = await run_in_threadpool(request.app.state.jobs.status, str(job_id))
    except Exception as exc:
        raise HTTPException(503, "Job store unavailable") from exc
    if result is None:
        raise HTTPException(404, "Job not found or expired")
    return result


@router.get("/jobs/{job_id}/details", dependencies=[Depends(authenticate)])
async def job_details(request: Request, job_id: UUID) -> dict:
    try:
        result = await run_in_threadpool(request.app.state.jobs.details, str(job_id))
    except Exception as exc:
        raise HTTPException(503, "Job store unavailable") from exc
    if result is None:
        raise HTTPException(404, "Job not found or expired")
    return result
