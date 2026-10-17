"""Run from repository root: python -m scripts.benchmark recording.mov [--mock]."""

import argparse
import json
import platform
import time
from pathlib import Path

from app.core.config import Settings
from app.services.inference import MockObserver, create_observer
from app.services.pipeline import Pipeline
from app.services.rules import RuleEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--mock", action="store_true", help="Preprocessing timing only, no accuracy claim"
    )
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    settings = Settings()
    observer = MockObserver() if args.mock else create_observer(settings)
    rules = RuleEngine.load(settings.rules_path)
    # Mock benchmarking still requires a configured test/verified catalogue; never bypass verdict checks.
    pipeline = Pipeline(settings, observer, rules)
    runs = []
    for index in range(args.repeat):
        start = time.monotonic()
        output = pipeline.run(args.video)
        elapsed = time.monotonic() - start
        runs.append(
            {
                "run": index + 1,
                "wall_sec": elapsed,
                "within_300_sec": elapsed <= 300,
                **output.details["metrics"],
            }
        )
    print(
        json.dumps(
            {
                "mock": args.mock,
                "python": platform.python_version(),
                "backend": "mock" if args.mock else settings.inference_backend,
                "model": settings.model_name
                if settings.inference_backend == "dmr"
                else str(settings.qwen_model_path),
                "device": settings.model_device if settings.inference_backend == "local" else None,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
