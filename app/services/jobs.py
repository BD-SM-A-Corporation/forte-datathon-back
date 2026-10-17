from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.core.config import Settings
from app.schemas.analysis import JobResponse, Result


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection = Redis.from_url(settings.redis_url)
        self.queue = Queue(
            settings.queue_name,
            connection=self.connection,
            default_timeout=settings.processing_budget_sec,
        )

    def enqueue(self, job_id: str, path: Path) -> None:
        self.queue.enqueue(
            "app.workers.analysis.process_video",
            str(path.resolve()),
            job_id=job_id,
            job_timeout=self.settings.processing_budget_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.result_ttl_sec,
            ttl=self.settings.queue_ttl_sec,
        )

    def get(self, job_id: str) -> Job | None:
        try:
            return Job.fetch(job_id, connection=self.connection)
        except NoSuchJobError:
            return None

    def status(self, job_id: str) -> JobResponse | None:
        job = self.get(job_id)
        if job is None:
            return None
        status = job.get_status(refresh=True)
        if status == "finished":
            data = job.return_value()
            if not data or data.get("error"):
                return JobResponse(
                    job_id=job_id,
                    status="failed",
                    error=(data or {}).get("error", "Analysis did not produce a result"),
                )
            result = Result.model_validate(data["result"])
            return JobResponse(job_id=job_id, status="done", **result.model_dump())
        if status in {"failed", "stopped", "canceled"}:
            return JobResponse(
                job_id=job_id,
                status="failed",
                error="Worker failed or exceeded its processing budget; verdict unavailable",
            )
        return JobResponse(job_id=job_id, status="processing")

    def details(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        data = job.return_value() if job.get_status() == "finished" else None
        return {"job_id": job_id, "details": data.get("details") if data else None}
