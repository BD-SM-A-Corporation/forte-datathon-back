import json
import time

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.errors import ConfigurationError, InferenceError
from app.services.inference import QwenObserver, parse_observations
from app.services.model_loader import LocalQwen
from app.services.video import Frame


@pytest.fixture
def frames():
    return [Frame(t, np.zeros((56, 56, 3), dtype=np.uint8)) for t in [0, 1, 2, 3]]


def test_valid(event, frames):
    assert parse_observations(json.dumps({"events": [event.model_dump()]}), frames) == [event]


def test_recovery(event, frames, settings):
    outputs = iter(["broken", json.dumps({"events": [event.model_dump()]})])
    observer = QwenObserver(settings, lambda *args: next(outputs))
    assert observer.observe(frames, time.monotonic() + 30) == [event]


def test_retry_exhausted(frames, settings):
    calls = []

    def generate(*args):
        calls.append(1)
        return '{"events": ['

    with pytest.raises(InferenceError):
        QwenObserver(settings, generate).observe(frames, time.monotonic() + 30)
    assert len(calls) == 2


def test_empty(frames, settings):
    assert (
        QwenObserver(settings, lambda *args: '{"events": []}').observe(
            frames, time.monotonic() + 30
        )
        == []
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_type", "Illegal action!"),
        ("start_sec", 20),
        ("confidence", 1.1),
        ("evidence_timestamps", [0.5]),
        ("end_sec", -1),
        ("description", ""),
    ],
)
def test_invalid_fields(event, frames, field, value):
    data = event.model_dump()
    data[field] = value
    with pytest.raises((ValueError, ValidationError)):
        parse_observations(json.dumps({"events": [data]}), frames)


def test_missing_field(event, frames):
    data = event.model_dump()
    del data["objects"]
    with pytest.raises(ValidationError):
        parse_observations(json.dumps({"events": [data]}), frames)


def test_checkpoint_missing(settings, tmp_path):
    with pytest.raises(ConfigurationError, match="checkpoint missing"):
        LocalQwen(settings.model_copy(update={"qwen_model_path": tmp_path / "missing"}))


def test_wrong_model(settings, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "other"}')
    with pytest.raises(ConfigurationError, match="Expected Qwen"):
        LocalQwen(settings.model_copy(update={"qwen_model_path": tmp_path}))


def test_fenced_output(frames):
    assert parse_observations('```json\n{"events": []}\n```', frames) == []
