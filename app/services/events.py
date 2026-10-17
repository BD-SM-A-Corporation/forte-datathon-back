from difflib import SequenceMatcher

from app.schemas.analysis import Observation


def consolidate(events: list[Observation], duration: float, gap: float = 0.2) -> list[Observation]:
    """Conservative duplicate heuristic: same action/objects, similar text, overlapping time.

    Distant events and dissimilar descriptions remain separate. No identity inference.
    The strongest description is retained verbatim, never synthesized.
    """
    merged: list[Observation] = []
    for event in sorted(events, key=lambda e: e.start_sec):
        if not 0 <= event.start_sec <= event.end_sec < duration:
            raise ValueError("Event outside video")
        candidate = next(
            (
                old
                for old in reversed(merged)
                if old.event_type == event.event_type
                and set(old.objects) == set(event.objects)
                # Never manufacture a conjunction by unioning facts from different windows.
                and set(old.facts) == set(event.facts)
                and old.uncertainties == event.uncertainties
                and event.start_sec <= old.end_sec + gap
                and SequenceMatcher(
                    None, old.description.lower(), event.description.lower()
                ).ratio()
                >= 0.8
            ),
            None,
        )
        if candidate is None:
            merged.append(event.model_copy(deep=True))
            continue
        candidate.end_sec = max(candidate.end_sec, event.end_sec)
        if event.confidence > candidate.confidence:
            candidate.description = event.description
            candidate.facts = {name: list(times) for name, times in event.facts.items()}
        candidate.confidence = max(candidate.confidence, event.confidence)
        times = sorted(set(candidate.evidence_timestamps + event.evidence_timestamps))
        candidate.evidence_timestamps = (
            times if len(times) <= 3 else [times[0], times[len(times) // 2], times[-1]]
        )
    return merged
