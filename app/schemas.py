from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.models import JobStatus, Verdict


class RuleItem(BaseModel):
    event_type: str
    title: str
    description: str
    source: str
    section: str
    url: str
    verified: bool = True
    severity: str = "medium"


class ViolationResponse(BaseModel):
    id: Optional[int] = None
    event_type: str
    title: str
    description: str
    timestamp_sec: float
    end_timestamp_sec: Optional[float] = None
    confidence: float
    frame_url: Optional[str] = None
    frame_b64: Optional[str] = None
    rule_source: Optional[str] = None
    rule_section: Optional[str] = None
    rule_url: Optional[str] = None

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Verdict
    progress: int = 0
    filename: str
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    violations_count: int = 0
    violations: List[ViolationResponse] = []

    class Config:
        from_attributes = True


class AnalyzeInitiateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
