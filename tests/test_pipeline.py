import pytest

from app.core.errors import BudgetExceeded, ConfigurationError, InferenceError
from app.services.inference import MockObserver, QwenObserver
from app.services.pipeline import Pipeline
from app.services.rules import Catalogue, RuleEngine


def test_full_pipeline(video, settings, rules, event):
    observer = MockObserver([event])
    result = Pipeline(settings, observer, rules).run(video)
    assert result.result.verdict == "violation"
    assert len(result.result.violations) == 1
    assert len(result.result.violations[0].evidence_frames) == 2
    assert result.details["matches"][0]["rule"]["id"] == "TEST_ONLY"
    assert result.details["metrics"]["baseline_frames"] == 4
    assert observer.calls > 1


def test_no_events(video, settings, rules):
    result = Pipeline(settings, MockObserver(), rules).run(video)
    assert result.result.model_dump() == {"verdict": "no_violation", "violations": []}


def test_unconfigured_fails(video, settings):
    with pytest.raises(ConfigurationError):
        Pipeline(settings, MockObserver(), RuleEngine(Catalogue(rules=[]))).run(video)


def test_failed_observation_never_clear(video, settings, rules):
    with pytest.raises(InferenceError):
        Pipeline(settings, QwenObserver(settings, lambda *args: "oops"), rules).run(video)


def test_budget(video, settings, rules):
    settings = settings.model_copy(update={"processing_budget_sec": 0})
    with pytest.raises(BudgetExceeded):
        Pipeline(settings, MockObserver(), rules).run(video)


@pytest.mark.parametrize("duration", [10, 145])
def test_complete_baseline_timeline(tmp_path, settings, rules, duration):
    import cv2
    import numpy as np

    path = tmp_path / "long.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(duration * 10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    class RecordingObserver:
        def __init__(self):
            self.seen = set()

        def observe(self, frames, deadline):
            self.seen.update(f.timestamp_sec for f in frames)
            return []

    observer = RecordingObserver()
    output = Pipeline(settings, observer, rules).run(path)
    assert observer.seen == set(range(duration))
    assert output.details["motion_intervals"] == []


def test_multiple_distinct_events(video, settings, rules, event):
    first = event.model_copy(update={"start_sec": 0, "end_sec": 0, "evidence_timestamps": [0]})
    second = event.model_copy(update={"start_sec": 2, "end_sec": 2, "evidence_timestamps": [2]})
    output = Pipeline(settings, MockObserver([first, second]), rules).run(video)
    assert len(output.result.violations) == 2
    assert [v.timecode_sec for v in output.result.violations] == [0, 2]


def test_unmatched_observations(video, settings, rules, event):
    event = event.model_copy(update={"event_type": "phone_interaction"})
    output = Pipeline(settings, MockObserver([event]), rules).run(video)
    assert output.result.verdict == "no_violation"
    assert len(output.details["observations"]) == 1
