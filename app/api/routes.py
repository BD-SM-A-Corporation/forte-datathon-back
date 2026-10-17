import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models import Job, JobStatus, Verdict, Violation
from app.pipeline.evidence import EvidenceRenderer
from app.pipeline.rules_engine import rules_engine
from app.pipeline.worker import process_video_job
from app.schemas import (
    AnalyzeInitiateResponse,
    JobResponse,
    RuleItem,
    ViolationResponse,
)

router = APIRouter()


async def verify_api_key_optional(x_api_key: Optional[str] = Header(None)):
    """
    Validates X-API-Key if provided, or allows requests from UI frontend.
    For strict jury endpoints, verify_api_key_strict is used.
    """
    if x_api_key and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key header",
        )
    return x_api_key


async def verify_api_key_strict(x_api_key: Optional[str] = Header(None)):
    """Strict verification for jury-only endpoints."""
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
    return x_api_key


@router.post("/api/v1/analyze", response_model=AnalyzeInitiateResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(verify_api_key_optional),
):
    """
    Accepts video upload (up to ~100MB+ directly to FastAPI, bypassing Vercel limits),
    stores it in storage/uploads, and triggers asynchronous analysis.
    """
    job_id = str(uuid.uuid4())
    upload_dir = settings.BASE_DIR / settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_extension = Path(file.filename or "video.mp4").suffix or ".mp4"
    saved_filename = f"{job_id}{file_extension}"
    saved_path = upload_dir / saved_filename

    # Save uploaded file asynchronously
    async with aiofiles.open(saved_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await out_file.write(chunk)

    # Create job in database
    new_job = Job(
        id=job_id,
        filename=file.filename or "video.mp4",
        file_path=str(saved_path),
        status=JobStatus.PENDING,
        verdict=Verdict.PENDING,
        progress=0,
    )
    db.add(new_job)
    await db.commit()

    # Dispatch background worker
    background_tasks.add_task(process_video_job, job_id)

    return AnalyzeInitiateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video uploaded successfully and queued for compliance analysis.",
    )


@router.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(verify_api_key_optional),
):
    """
    Returns job status, processing progress, final verdict (violation / no_violation),
    and detailed violation records with evidence frames.
    """
    stmt = select(Job).options(selectinload(Job.violations)).where(Job.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    violations_response = []
    for v in job.violations:
        b64 = None
        if v.frame_path and os.path.exists(v.frame_path):
            try:
                import cv2

                img = cv2.imread(v.frame_path)
                if img is not None:
                    b64 = EvidenceRenderer.frame_to_base64(img)
            except Exception:
                pass

        violations_response.append(
            ViolationResponse(
                id=v.id,
                event_type=v.event_type,
                title=v.title,
                description=v.description,
                timestamp_sec=v.timestamp_sec,
                end_timestamp_sec=v.end_timestamp_sec,
                confidence=v.confidence,
                frame_url=f"/api/v1/frames/{job_id}/{v.id}" if v.frame_path else None,
                frame_b64=b64,
                rule_source=v.rule_source,
                rule_section=v.rule_section,
                rule_url=v.rule_url,
            )
        )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        verdict=job.verdict,
        progress=job.progress,
        filename=job.filename,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
        violations_count=len(violations_response),
        violations=violations_response,
    )


@router.get("/api/v1/rules", response_model=List[RuleItem])
async def list_regulatory_rules():
    """Lists verified NBRK regulatory rules and legal sources."""
    return rules_engine.get_all_rules()


@router.get("/api/v1/frames/{job_id}/{violation_id}")
async def get_evidence_frame(
    job_id: str,
    violation_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Serves the JPEG keyframe evidence for a specific violation."""
    stmt = select(Violation).where(
        Violation.id == violation_id, Violation.job_id == job_id
    )
    result = await db.execute(stmt)
    violation = result.scalar_one_or_none()

    if (
        not violation
        or not violation.frame_path
        or not os.path.exists(violation.frame_path)
    ):
        raise HTTPException(status_code=404, detail="Evidence frame not found")

    return FileResponse(violation.frame_path, media_type="image/jpeg")


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "forte-datathon-backend"}
