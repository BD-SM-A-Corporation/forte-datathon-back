"""Remove orphan uploads after queue/result retention; run periodically on shared storage."""

import time

from app.core.config import Settings
from app.services.jobs import JobStore


def main() -> None:
    settings = Settings()
    store = JobStore(settings)
    age = settings.queue_ttl_sec + settings.processing_budget_sec + settings.result_ttl_sec
    for path in settings.upload_dir.glob("*.mov"):
        if time.time() - path.stat().st_mtime > age:
            job = store.get(path.stem)
            if job is None or job.get_status() in {"failed", "finished", "stopped", "canceled"}:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
