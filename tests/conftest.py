import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.schemas.analysis import Observation
from app.services.rules import Catalogue, RuleEngine


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, api_key="test-key", upload_dir=tmp_path / "uploads")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for i in range(40):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        if i >= 10:
            cv2.rectangle(image, (20 + i * 2, 30), (50 + i * 2, 90), (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    return path


@pytest.fixture
def vfr_video(tmp_path: Path) -> Path:
    frames = []
    for i, duration in enumerate([0.12, 0.48, 0.24, 0.72]):
        file = tmp_path / f"{i}.png"
        cv2.imwrite(str(file), np.full((96, 128, 3), i * 60, dtype=np.uint8))
        frames += [f"file '{file}'", f"duration {duration}"]
    frames.append(f"file '{file}'")
    manifest = tmp_path / "concat.txt"
    manifest.write_text("\n".join(frames))
    output = tmp_path / "vfr.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-fps_mode",
            "vfr",
            "-c:v",
            "mpeg4",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.fixture
def event() -> Observation:
    return Observation(
        event_type="object_transfer",
        start_sec=1,
        end_sec=2,
        description="A person appears to transfer an object across the counter.",
        objects=["unknown_object"],
        confidence=0.8,
        evidence_timestamps=[1, 2],
    )


@pytest.fixture
def catalogue_data() -> dict:
    # Deliberately fictional test fixture, NEVER a production regulation or legal assertion.
    return {
        "rules": [
            {
                "id": "TEST_ONLY",
                "name": "Synthetic test condition",
                "source": {
                    "document": "TEST FIXTURE ONLY",
                    "section": "TEST",
                    "url": "https://adilet.zan.kz/test-fixture-not-a-regulation",
                    "jurisdiction": "KZ",
                    "verified_on": "2025-01-01",
                    "verified_by": "Unit test fixture",
                    "excerpt": "Synthetic fixture only",
                },
                "observable_event_types": ["object_transfer"],
                "conditions": {"objects_all": ["unknown_object"], "min_confidence": 0.7},
                "applicability": "Synthetic unit test only",
                "explanation_template": "Synthetic test condition matched.",
            }
        ]
    }


@pytest.fixture
def rules(catalogue_data: dict) -> RuleEngine:
    return RuleEngine(Catalogue.model_validate(catalogue_data))


@pytest.fixture
def rule_file(tmp_path: Path, catalogue_data: dict) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(catalogue_data))
    return path
