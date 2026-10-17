import logging
import time
from pathlib import Path

from rq import SimpleWorker, get_current_job

from app.core.config import Settings
from app.core.errors import AnalysisError
from app.core.logging import configure_logging
from app.services.inference import create_observer
from app.services.jobs import JobStore
from app.services.pipeline import Pipeline
from app.services.rules import RuleEngine

logger = logging.getLogger(__name__)
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        settings = Settings()
        rules = RuleEngine.load(settings.rules_path)
        _pipeline = Pipeline(
            settings,
            create_observer(settings, rule_guidance=rules.observation_guidance()),
            rules,
        )
    return _pipeline


def process_video(path: str) -> dict:
    started = time.monotonic()
    file = Path(path)
    job = get_current_job()
    job_id = job.id if job else None
    try:
        output = get_pipeline().run(file)
        return {"result": output.result.model_dump(), "details": output.details}
    except AnalysisError as exc:
        logger.warning("analysis_failed", extra={"job_id": job_id, "metrics": {"error": str(exc)}})
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- worker boundary must sanitize all client errors
        # No raw model output, uploaded content, filenames, or exception text in client errors.
        logger.error(
            "analysis_failed_unexpectedly",
            extra={"job_id": job_id, "metrics": {"exception_type": type(exc).__name__}},
        )
        return {"error": "Analysis failed; check worker configuration and runtime logs"}
    finally:
        file.unlink(missing_ok=True)
        logger.info(
            "job_finished",
            extra={"job_id": job_id, "metrics": {"processing_sec": time.monotonic() - started}},
        )


def main() -> None:
    configure_logging()
    settings = Settings()
    RuleEngine.load(
        settings.rules_path
    )  # Validate at process startup, including an empty catalogue.
    store = JobStore(settings)
    # SimpleWorker preserves local model/CUDA state. DMR owns its own model lifecycle.
    # Use one local worker per GPU; size DMR concurrency to the server's capacity.
    SimpleWorker([store.queue], connection=store.connection).work()


if __name__ == "__main__":
    main()
