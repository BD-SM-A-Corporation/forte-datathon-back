import json
import logging
import time


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "job_id": getattr(record, "job_id", None),
                "metrics": getattr(record, "metrics", None),
            },
            default=str,
        )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
