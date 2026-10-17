from uuid import uuid4

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker

from app.main import create_app
from app.services.inference import MockObserver
from app.services.jobs import JobStore
from app.services.pipeline import Pipeline
from app.workers import analysis


@pytest.fixture
def api(settings, rule_file, rules, event, monkeypatch):
    settings = settings.model_copy(update={"rules_path": rule_file, "max_upload_mb": 1})
    store = JobStore(settings)
    store.connection = fakeredis.FakeRedis()
    store.queue = Queue(settings.queue_name, connection=store.connection)
    monkeypatch.setattr(analysis, "_pipeline", Pipeline(settings, MockObserver([event]), rules))
    with TestClient(create_app(settings, store)) as client:
        yield client, store, settings


def work(store):
    SimpleWorker([store.queue], connection=store.connection).work(burst=True)


def test_upload_and_result(api, video):
    client, store, settings = api
    response = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={"file": ("camera.mov", video.read_bytes(), "video/quicktime")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    url = f"/api/v1/jobs/{job_id}"
    assert client.get(url, headers={"X-API-Key": "test-key"}).json()["status"] == "processing"
    work(store)
    data = client.get(url, headers={"X-API-Key": "test-key"}).json()
    assert data["status"] == "done", data
    assert data["verdict"] == "violation"
    assert set(data["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert data["violations"][0]["evidence_frames"][0]["image_base64"].startswith("/9j/")
    details = client.get(url + "/details", headers={"X-API-Key": "test-key"}).json()
    assert details["details"]["matches"][0]["rule"]["id"] == "TEST_ONLY"
    assert not list(settings.upload_dir.glob("*.mov"))


@pytest.mark.parametrize("key", [None, "wrong"])
def test_auth(api, key):
    client, _, _ = api
    headers = {} if key is None else {"X-API-Key": key}
    assert (
        client.post(
            "/api/v1/analyze", headers=headers, files={"file": ("a.mov", b"data")}
        ).status_code
        == 401
    )
    assert client.get(f"/api/v1/jobs/{uuid4()}", headers=headers).status_code == 401


def test_missing_file(api):
    assert api[0].post("/api/v1/analyze", headers={"X-API-Key": "test-key"}).status_code == 422


def test_invalid_extension(api):
    assert (
        api[0]
        .post(
            "/api/v1/analyze",
            headers={"X-API-Key": "test-key"},
            files={"file": ("bad.mp4", b"bad")},
        )
        .status_code
        == 415
    )


def test_corruption(api):
    client, store, _ = api
    response = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={"file": ("bad.mov", b"corrupt")},
    )
    work(store)
    data = client.get(
        "/api/v1/jobs/" + response.json()["job_id"], headers={"X-API-Key": "test-key"}
    ).json()
    assert data["status"] == "failed"
    assert "verdict" not in data
    assert "Traceback" not in data["error"]


def test_empty(api):
    assert (
        api[0]
        .post(
            "/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("empty.mov", b"")}
        )
        .status_code
        == 422
    )


def test_limit(api):
    assert (
        api[0]
        .post(
            "/api/v1/analyze",
            headers={"X-API-Key": "test-key"},
            files={"file": ("huge.mov", b"x" * (2 * 1024 * 1024))},
        )
        .status_code
        == 413
    )


def test_unknown_job(api):
    assert (
        api[0].get(f"/api/v1/jobs/{uuid4()}", headers={"X-API-Key": "test-key"}).status_code == 404
    )


def test_failed_job(api):
    client, store, _ = api
    job = store.queue.enqueue("builtins.int", "bad", job_id=str(uuid4()))
    work(store)
    data = client.get(f"/api/v1/jobs/{job.id}", headers={"X-API-Key": "test-key"}).json()
    assert data["status"] == "failed"
    assert "bad" not in data["error"]


def test_health(api):
    assert api[0].get("/health").json() == {"status": "ok", "regulations_configured": True}


def test_queue_failure_cleanup(api, monkeypatch):
    client, store, settings = api

    def broken(*args):
        raise RuntimeError("secret")

    monkeypatch.setattr(store, "enqueue", broken)
    response = client.post(
        "/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("bad.mov", b"video")}
    )
    assert response.status_code == 503
    assert "secret" not in response.text
    assert not list(settings.upload_dir.glob("*.mov"))
