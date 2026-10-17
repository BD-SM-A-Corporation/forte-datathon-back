import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import AnalysisError, BudgetExceeded, VideoError
from app.schemas.analysis import Result, Violation
from app.services.events import consolidate
from app.services.evidence import extract_evidence
from app.services.inference import Observer
from app.services.motion import MotionDetector
from app.services.rules import RuleEngine
from app.services.verification import (
    FACT_QUESTIONS,
    RuleVerifier,
    context_timestamps,
    rank_candidates,
    select_spread_intervals,
    verified_observation,
)
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
        verifier: RuleVerifier | None = None,
    ):
        self.settings, self.observer, self.rules = settings, observer, rules
        self.providers = providers
        self.verifier = verifier

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
        rule_checks = []
        checked_matches = []
        refinement_calls = 0
        verification_calls = 0
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

            def analyze(interval: Interval, timestamps: list[float], stage_deadline=None) -> None:
                nonlocal calls
                check()
                selected = [t for t in timestamps if interval.start_sec <= t <= interval.end_sec]
                if not selected:
                    return
                if len(selected) > self.settings.max_frames_per_window:
                    raise VideoError("Sampling exceeds configured frame cap")
                frames = [Frame(t, np.load(cache[t], allow_pickle=False)) for t in selected]
                events.extend(self.observer.observe(frames, stage_deadline or deadline))
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
            refinements = select_spread_intervals(refinements, self.settings.max_refinement_windows)
            detailed: list[float] = []
            if refinements:
                for frame in sample_frames(
                    path, metadata, self.settings.detailed_sample_fps, check=check
                ):
                    if any(i.start_sec <= frame.timestamp_sec <= i.end_sec for i in refinements):
                        save(frame)
                        detailed.append(frame.timestamp_sec)
                for interval in refinements:
                    refinement_deadline = (
                        deadline - self.settings.verification_reserve_sec
                        if self.verifier else deadline
                    )
                    if self.verifier and time.monotonic() + 40 >= refinement_deadline:
                        break
                    try:
                        analyze(interval, detailed, refinement_deadline)
                        refinement_calls += 1
                    except BudgetExceeded:
                        if not self.verifier:
                            raise
                        break

            events = consolidate(events, metadata.duration_sec, self.settings.event_merge_gap_sec)
            if self.verifier:
                for event, rule in rank_candidates(
                    events, self.rules.catalogue.rules, self.settings.max_rule_checks
                ):
                    record = {
                        "rule_id": rule.id, "rule_name": rule.name,
                        "source": rule.source.model_dump(mode="json"),
                        "event_type": event.event_type,
                        "start_sec": event.start_sec, "end_sec": event.end_sec,
                        "status": "not_checked", "facts": [], "review_required": True,
                    }
                    rule_checks.append(record)
                    if verification_calls >= self.settings.max_rule_checks:
                        record["reason"] = "check_limit"
                        continue
                    if time.monotonic() + 10 >= deadline:
                        record["reason"] = "time_budget"
                        continue
                    selected = context_timestamps(
                        list(cache), event, self.settings.verification_context_sec,
                        self.settings.max_frames_per_window,
                    )
                    record["sampled_timestamps"] = selected
                    if not selected:
                        record["reason"] = "no_frames"
                        continue
                    record["context_start_sec"], record["context_end_sec"] = selected[0], selected[-1]
                    frames = [Frame(t, np.load(cache[t], allow_pickle=False)) for t in selected]
                    verification_calls += 1
                    try:
                        verification = self.verifier.verify(
                            frames, rule, event,
                            min(deadline - 5, time.monotonic()
                                + self.settings.verification_check_timeout_sec),
                        )
                    except AnalysisError as exc:
                        record.update(status="unavailable", reason=(
                            "time_budget" if isinstance(exc, BudgetExceeded) else "invalid_response"
                        ))
                        continue
                    record.update(
                        facts=[{
                            "id": name, "label": FACT_QUESTIONS.get(name, (name, name))[1],
                            **fact.model_dump(),
                        } for name, fact in verification.checks.items()],
                        episode_link=verification.episode_link.model_dump(),
                        confidence=verification.confidence,
                        uncertainties=verification.uncertainties,
                    )
                    witness = verified_observation(event, rule, verification, selected)
                    matched = witness is not None and rule in self.rules.match(witness)
                    record["status"] = "potential_match" if matched else (
                        "contradicted" if any(
                            fact.status == "contradicted" for fact in
                            [*verification.checks.values(), verification.episode_link]
                        ) else "insufficient_evidence"
                    )
                    record["blocking_reasons"] = [
                        name for name, blocked in {
                            "unknown_facts": any(f.status == "unknown"
                                                 for f in verification.checks.values()),
                            "episode_unconfirmed": verification.episode_link.status != "confirmed",
                            "low_confidence": verification.confidence < rule.conditions.min_confidence,
                            "uncertainties": bool(verification.uncertainties),
                        }.items() if blocked
                    ]
                    if matched:
                        checked_matches.append((witness, rule))

        matches = checked_matches if self.verifier else [
            (event, rule) for event in events for rule in self.rules.match(event)
        ]
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
            "rule_checks": rule_checks,
            "verification_summary": {
                "enabled": self.verifier is not None,
                "candidate_count": len(rule_checks),
                "checked_count": sum(r["status"] not in {"not_checked", "unavailable"}
                                     for r in rule_checks),
                "not_checked_count": sum(r["status"] == "not_checked" for r in rule_checks),
                "unavailable_count": sum(r["status"] == "unavailable" for r in rule_checks),
                "insufficient_count": sum(r["status"] == "insufficient_evidence"
                                          for r in rule_checks),
            },
            "motion_intervals": [asdict(i) for i in motion_intervals],
            "metrics": {
                "processing_sec": elapsed,
                "video_duration_sec": metadata.duration_sec,
                "inference_windows": calls,
                "baseline_frames": len(baseline),
                "refinement_windows": refinement_calls,
                "refinement_windows_planned": len(refinements),
                "verification_calls": verification_calls,
            },
            "scope": "No-match verdicts apply only to the configured rules and sampled visible actions.",
        }
        logger.info("analysis_complete", extra={"metrics": details["metrics"]})
        return PipelineResult(result, details)
