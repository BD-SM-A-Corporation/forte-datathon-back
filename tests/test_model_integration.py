"""Opt-in smoke test; never downloads weights and never makes legal assertions."""

import os
import time

import numpy as np
import pytest

from app.core.config import Settings
from app.services.inference import QwenObserver
from app.services.video import Frame


@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_MODEL_TEST") != "1", reason="Local checkpoint opt-in required"
)
def test_real_local_qwen():
    settings = Settings()
    frames = [Frame(t, np.zeros((112, 112, 3), dtype=np.uint8)) for t in [0, 1]]
    output = QwenObserver(settings).observe(frames, time.monotonic() + 300)
    assert output == []
