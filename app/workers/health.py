"""SimpleWorker has no heartbeat during inference; allow one processing budget."""

import socket
from datetime import UTC, datetime

from rq import Worker

from app.core.config import Settings
from app.services.jobs import JobStore


def main() -> None:
    settings = Settings()
    connection = JobStore(settings).connection
    connection.ping()
    now = datetime.now(UTC)
    healthy = any(
        w.hostname == socket.gethostname()
        and w.last_heartbeat is not None
        and (now - w.last_heartbeat).total_seconds() < settings.processing_budget_sec + 90
        for w in Worker.all(connection=connection)
    )
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
