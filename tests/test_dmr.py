import asyncio
import base64
import json
import time

import cv2
import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import BudgetExceeded, InferenceError
from app.services.dmr import DMRObserver
from app.services.inference import QwenObserver, create_observer
from app.services.pipeline import Pipeline
from app.services.video import Frame


@pytest.fixture
def dmr_settings(settings):
    return Settings.model_validate(
        {
            **settings.model_dump(),
            "inference_backend": "dmr",
            "model_base_url": "http://model-runner.test/engines/v1/",
            "model_name": "test/vision:fixture",
        }
    )


@pytest.fixture
def frames():
    return [
        Frame(t, np.full((56, 56, 3), color, dtype=np.uint8))
        for t, color in [(1.25, (0, 0, 255)), (2.75, (255, 0, 0))]
    ]


def completion(raw='{"events": []}', finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": raw},
                }
            ]
        },
    )


def test_timestamped_images_and_request(dmr_settings, frames):
    def handle(request):
        assert str(request.url) == "http://model-runner.test/engines/v1/chat/completions"
        data = json.loads(request.content)
        assert data["model"] == "test/vision:fixture"
        assert data["temperature"] == 0
        assert data["stream"] is False
        assert data["max_tokens"] == dmr_settings.max_new_tokens
        assert data["response_format"] == {"type": "json_object"}
        assert data["messages"][0]["role"] == "user"
        content = data["messages"][0]["content"]
        assert len(content) == 5
        assert "Supplied timestamps: [1.25, 2.75]" in content[0]["text"]
        for index, frame in enumerate(frames):
            assert content[1 + 2 * index]["text"] == f"Timestamp {frame.timestamp_sec:.9f} seconds"
            part = content[2 + 2 * index]
            assert part["type"] == "image_url"
            prefix, encoded = part["image_url"]["url"].split(",", 1)
            assert prefix == "data:image/jpeg;base64"
            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded.shape == frame.image.shape
            assert np.max(np.abs(decoded.astype(int) - frame.image.astype(int))) <= 2
        assert 0 < request.extensions["timeout"]["read"] <= 30
        return completion()

    observer = DMRObserver(dmr_settings, httpx.MockTransport(handle))
    assert observer.observe(frames, time.monotonic() + 30) == []


@pytest.mark.parametrize("invalid", ["broken", '{"events": [{"event_type": "bad"}]}'])
def test_invalid_output_retried(dmr_settings, frames, invalid):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return completion(invalid if len(requests) == 1 else '{"events": []}')

    assert (
        DMRObserver(dmr_settings, httpx.MockTransport(handle)).observe(
            frames, time.monotonic() + 30
        )
        == []
    )
    assert len(requests) == 2
    assert "Previous response failed validation" in requests[1]["messages"][0]["content"][0]["text"]


def test_ungrounded_output_never_cleared(dmr_settings, frames, event):
    calls = []

    def handle(request):
        calls.append(request)
        return completion(json.dumps({"events": [event.model_dump()]}))

    with pytest.raises(InferenceError, match="failed validation"):
        DMRObserver(dmr_settings, httpx.MockTransport(handle)).observe(
            frames, time.monotonic() + 30
        )
    assert len(calls) == dmr_settings.generation_retries + 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="invalid envelope"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [None]}),
        completion(None),
        completion(""),
        completion(finish="length"),
        completion(finish="content_filter"),
    ],
)
def test_bad_server_response(dmr_settings, frames, response):
    with pytest.raises(InferenceError):
        DMRObserver(dmr_settings, httpx.MockTransport(lambda _: response)).observe(
            frames, time.monotonic() + 30
        )


@pytest.mark.parametrize("status", [302, 400, 404, 429, 500])
def test_http_failure_is_safe_and_not_retried(dmr_settings, frames, status, caplog):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status, text="private server data", headers={"Location": "http://other"}
        )

    with pytest.raises(InferenceError, match="rejected request") as error:
        DMRObserver(dmr_settings, httpx.MockTransport(handle)).observe(
            frames, time.monotonic() + 30
        )
    assert len(calls) == 1
    assert "private server data" not in str(error.value) + caplog.text


def test_connection_failure(dmr_settings, frames):
    def handle(request):
        raise httpx.ConnectError("private host", request=request)

    with pytest.raises(InferenceError, match="server unavailable"):
        DMRObserver(dmr_settings, httpx.MockTransport(handle)).observe(
            frames, time.monotonic() + 30
        )


@pytest.mark.parametrize("job_expires", [False, True])
def test_total_request_timeout(dmr_settings, frames, job_expires):
    cancelled = []

    async def handle(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        return completion()

    settings = dmr_settings.model_copy(
        update={"model_http_timeout_sec": 30 if job_expires else 0.05}
    )
    deadline = time.monotonic() + (0.05 if job_expires else 30)
    with pytest.raises(BudgetExceeded if job_expires else InferenceError):
        DMRObserver(settings, httpx.MockTransport(handle)).observe(frames, deadline)
    assert cancelled == [True]


def test_expired_deadline_makes_no_request(dmr_settings, frames):
    def handle(request):
        pytest.fail("Expired jobs must not call the model server")

    with pytest.raises(BudgetExceeded):
        DMRObserver(dmr_settings, httpx.MockTransport(handle)).observe(frames, time.monotonic() - 1)


def test_frame_checks_before_request(dmr_settings, frames):
    def handle(request):
        pytest.fail("Invalid frames must not call the model server")

    observer = DMRObserver(dmr_settings, httpx.MockTransport(handle))
    assert observer.observe([], time.monotonic() + 30) == []
    with pytest.raises(InferenceError, match="chronological"):
        observer.observe(frames[::-1], time.monotonic() + 30)
    with pytest.raises(InferenceError, match="frame limit"):
        observer.observe(frames * dmr_settings.max_frames_per_window, time.monotonic() + 30)


def test_factory(settings, dmr_settings, monkeypatch):
    monkeypatch.setattr(
        "app.services.inference.get_model", lambda _: pytest.fail("Local model loaded")
    )
    assert isinstance(create_observer(settings), QwenObserver)
    assert isinstance(create_observer(dmr_settings), DMRObserver)


@pytest.mark.parametrize(
    "updates",
    [
        {"model_base_url": ""},
        {"model_base_url": "file:///models"},
        {"model_base_url": "http://host:bad"},
        {"model_base_url": "http://user:password@host/v1"},
        {"model_base_url": "http://host/v1?key=secret"},
        {"model_name": "  "},
        {"model_http_timeout_sec": 0},
        {"model_http_timeout_sec": float("inf")},
    ],
)
def test_invalid_dmr_configuration(dmr_settings, updates):
    with pytest.raises(ValidationError):
        Settings.model_validate({**dmr_settings.model_dump(), **updates})


def test_dmr_pipeline(video, dmr_settings, rules, event):
    def handle(request):
        content = json.loads(request.content)["messages"][0]["content"]
        times = {
            float(part["text"].split()[1])
            for part in content
            if part["type"] == "text" and part["text"].startswith("Timestamp ")
        }
        events = [event.model_dump()] if {1, 2} <= times else []
        return completion(json.dumps({"events": events}))

    result = Pipeline(
        dmr_settings, DMRObserver(dmr_settings, httpx.MockTransport(handle)), rules
    ).run(video)
    assert result.result.verdict == "violation"
    assert result.result.violations[0].evidence_frames


def test_worker_selects_dmr(dmr_settings, rule_file, monkeypatch):
    from app.workers import analysis

    monkeypatch.setattr(analysis, "_pipeline", None)
    monkeypatch.setattr(
        analysis, "Settings", lambda: dmr_settings.model_copy(update={"rules_path": rule_file})
    )
    assert isinstance(analysis.get_pipeline().observer, DMRObserver)
