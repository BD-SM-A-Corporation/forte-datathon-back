from datetime import datetime
import enum
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from app.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, enum.Enum):
    VIOLATION = "violation"
    NO_VIOLATION = "no_violation"
    PENDING = "pending"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    verdict = Column(Enum(Verdict), default=Verdict.PENDING, nullable=False)
    progress = Column(Integer, default=0)  # 0 to 100%
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    violations = relationship("Violation", back_populates="job", cascade="all, delete-orphan")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    end_timestamp_sec = Column(Float, nullable=True)
    confidence = Column(Float, default=0.0)
    frame_path = Column(String(512), nullable=True)

    # Normative and legal metadata
    rule_source = Column(String(512), nullable=True)
    rule_section = Column(String(100), nullable=True)
    rule_url = Column(String(512), nullable=True)

    job = relationship("Job", back_populates="violations")
