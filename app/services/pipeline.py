import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import BudgetExceeded, VideoError
from app.schemas.analysis import Result, Violation
from app.services.events import consolidate
from app.services.evidence import extract_evidence
from app.services.inference import Observer
from app.services.motion import MotionDetector
from app.services.rules import RuleEngine
from app.services.video import (
    Frame,
    Interval,
    VideoMetadata,
    inspect_video,
    sample_frames,
    transform_frame,
    windows,
)

logger = logging.getLogger(__name__)


class IntervalProvider(Protocol):
    """Extension point for future local detectors or tracking; original-video seconds."""

    def intervals(self, path: Path, metadata: VideoMetadata) -> list[Interval]: ...


@dataclass
class PipelineResult:
    result: Result
    details: dict


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        observer: Observer,
        rules: RuleEngine,
        providers: tuple[IntervalProvider, ...] = (),
    ):
        self.settings, self.observer, self.rules = settings, observer, rules
        self.providers = providers

    def run(self, path: Path) -> PipelineResult:
        started = time.monotonic()
        deadline = started + self.settings.processing_budget_sec

        def check() -> None:
            if time.monotonic() >= deadline:
                raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")

        metadata = inspect_video(path, self.settings)
        check()
        self.rules.require_configured()
        motion = MotionDetector(self.settings)
        events = []
        calls = 0
        # Only sampled, resized frames are cached on disk; RAM holds at most one window.
        with tempfile.TemporaryDirectory(prefix="forte-frames-") as directory:
            cache: dict[float, Path] = {}
            baseline: list[float] = []

            def save(frame: Frame) -> None:
                if frame.timestamp_sec not in cache:
                    target = Path(directory) / f"{len(cache)}.npy"
                    np.save(target, transform_frame(frame, self.settings).image)
                    cache[frame.timestamp_sec] = target

            for frame in sample_frames(path, metadata, self.settings.base_sample_fps, check=check):
                motion.add(frame)
                baseline.append(frame.timestamp_sec)
                save(frame)
            if not baseline:
                raise VideoError("No sampled video frames")

            def analyze(interval: Interval, timestamps: list[float]) -> None:
                nonlocal calls
                check()
                selected = [t for t in timestamps if interval.start_sec <= t <= interval.end_sec]
                if not selected:
                    return
                if len(selected) > self.settings.max_frames_per_window:
                    raise VideoError("Sampling exceeds configured frame cap")
                frames = [Frame(t, np.load(cache[t], allow_pickle=False)) for t in selected]
                events.extend(self.observer.observe(frames, deadline))
                calls += 1
                check()

            for interval in windows(
                metadata.duration_sec,
                self.settings.window_duration_sec,
                self.settings.window_overlap_sec,
            ):
                analyze(interval, baseline)
            motion_intervals = motion.result()
            candidates = [
                Interval(max(0, e.start_sec - 0.5), min(metadata.duration_sec, e.end_sec + 0.5))
                for e in events
            ]
            candidates += sorted(motion_intervals, key=lambda i: -i.motion_score)
            for provider in self.providers:
                candidates.extend(provider.intervals(path, metadata))
                check()
            refinements: list[Interval] = []
            detail_length = min(
                self.settings.window_duration_sec,
                (self.settings.max_frames_per_window - 1) / self.settings.detailed_sample_fps,
            )
            for candidate in candidates:
                if not 0 <= candidate.start_sec <= candidate.end_sec <= metadata.duration_sec:
                    raise VideoError("Candidate interval outside recording")
                for sub in windows(
                    candidate.end_sec - candidate.start_sec,
                    detail_length,
                    min(self.settings.window_overlap_sec, detail_length / 2),
                ):
                    interval = Interval(
                        candidate.start_sec + sub.start_sec, candidate.start_sec + sub.end_sec
                    )
                    if interval not in refinements:
                        refinements.append(interval)
            refinements = refinements[: self.settings.max_refinement_windows]
            detailed: list[float] = []
            if refinements:
                for frame in sample_frames(
                    path, metadata, self.settings.detailed_sample_fps, check=check
                ):
                    if any(i.start_sec <= frame.timestamp_sec <= i.end_sec for i in refinements):
                        save(frame)
                        detailed.append(frame.timestamp_sec)
                for interval in refinements:
                    analyze(interval, detailed)

        events = consolidate(events, metadata.duration_sec, self.settings.event_merge_gap_sec)
        matches = [(event, rule) for event in events for rule in self.rules.match(event)]
        evidence = extract_evidence(
            path, metadata, [t for event, _ in matches for t in event.evidence_timestamps], check
        )
        violations = [
            Violation(
                type=rule.name,
                timecode_sec=event.start_sec,
                description=f"{event.description} Potential rule match: {rule.explanation_template}",
                evidence_frames=[evidence[t] for t in event.evidence_timestamps],
                confidence=event.confidence,
                uncertainties=event.uncertainties,
            )
            for event, rule in matches
        ]
        check()
        elapsed = time.monotonic() - started
        result = Result(verdict="violation" if violations else "no_violation", violations=violations)
        details = {
            "observations": [e.model_dump() for e in events],
            "matches": [
                {"observation": event.model_dump(), "rule": rule.model_dump(mode="json")}
                for event, rule in matches
            ],
            "context_matches": [
                {"observation": event.model_dump(), "rule": rule.model_dump(mode="json")}
                for event in events
                for rule in self.rules.context(event)
            ],
            "regulatory_assessment": "potential_matches" if matches else "no_supported_match",
            "review_required": True,
            "assessment": result.assessment.model_dump(),
            "motion_intervals": [asdict(i) for i in motion_intervals],
            "metrics": {
                "processing_sec": elapsed,
                "video_duration_sec": metadata.duration_sec,
                "inference_windows": calls,
                "baseline_frames": len(baseline),
                "refinement_windows": len(refinements),
            },
            "scope": "No-match verdicts apply only to the configured rules and sampled visible actions.",
        }
        logger.info("analysis_complete", extra={"metrics": details["metrics"]})
        return PipelineResult(result, details)
