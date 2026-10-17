from dataclasses import dataclass
from typing import List, Optional
from app.models import Verdict
from app.pipeline.rules_engine import rules_engine


@dataclass
class DetectedViolation:
    event_type: str
    title: str
    description: str
    timestamp_sec: float
    end_timestamp_sec: float
    confidence: float
    frame_idx: int
    rule_source: Optional[str] = None
    rule_section: Optional[str] = None
    rule_url: Optional[str] = None


class ComplianceClassifier:
    """
    State-machine based compliance classifier evaluating temporal event patterns.
    Uses strict factual templates rather than generative LLMs.
    """

    def __init__(self, fps: int = 5, min_violation_duration_sec: float = 3.0):
        self.fps = fps
        self.min_violation_duration_sec = min_violation_duration_sec

    def evaluate_sequence(self, frame_features_list: List[dict]) -> Tuple_Verdict_Violations:
        violations: List[DetectedViolation] = []

        unattended_cash_streak = 0
        streak_start_time = 0.0

        for item in frame_features_list:
            feat = item["features"]
            t = item["timestamp"]
            frame_idx = item["frame_idx"]

            # Rule 1: Cash left unattended
            if feat.get("cash_count", 0) > 0 and not feat.get("cashier_present", False):
                if unattended_cash_streak == 0:
                    streak_start_time = t
                unattended_cash_streak += 1
            else:
                if unattended_cash_streak >= (self.fps * self.min_violation_duration_sec):
                    rule = rules_engine.get_rule("cash_left_unattended")
                    violations.append(
                        DetectedViolation(
                            event_type="cash_left_unattended",
                            title=rule.title if rule else "Оставление денежных средств без присмотра",
                            description=(
                                f"Денежные средства находились на рабочем месте без присутствия кассира "
                                f"с {streak_start_time:.1f}с по {t:.1f}с (длительность: {t - streak_start_time:.1f}с)."
                            ),
                            timestamp_sec=streak_start_time,
                            end_timestamp_sec=t,
                            confidence=0.88,
                            frame_idx=frame_idx,
                            rule_source=rule.source if rule else None,
                            rule_section=rule.section if rule else None,
                            rule_url=rule.url if rule else None,
                        )
                    )
                unattended_cash_streak = 0

        # Check trailing streak at end of video
        if unattended_cash_streak >= (self.fps * self.min_violation_duration_sec):
            t = frame_features_list[-1]["timestamp"] if frame_features_list else streak_start_time
            rule = rules_engine.get_rule("cash_left_unattended")
            violations.append(
                DetectedViolation(
                    event_type="cash_left_unattended",
                    title=rule.title if rule else "Оставление денежных средств без присмотра",
                    description=(
                        f"Денежные средства находились на рабочем месте без присутствия кассира "
                        f"с {streak_start_time:.1f}с по {t:.1f}с."
                    ),
                    timestamp_sec=streak_start_time,
                    end_timestamp_sec=t,
                    confidence=0.88,
                    frame_idx=frame_features_list[-1]["frame_idx"] if frame_features_list else 0,
                    rule_source=rule.source if rule else None,
                    rule_section=rule.section if rule else None,
                    rule_url=rule.url if rule else None,
                )
            )

        verdict = Verdict.VIOLATION if violations else Verdict.NO_VIOLATION
        return verdict, violations


Tuple_Verdict_Violations = tuple[Verdict, List[DetectedViolation]]
