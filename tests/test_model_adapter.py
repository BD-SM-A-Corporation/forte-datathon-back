"""Adapter contract tests with fake tensors; these do not validate actual Qwen inference."""

import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.services.inference import QwenObserver
from app.services.model_loader import get_model
from app.services.video import Frame


def test_timestamped_processor_contract(monkeypatch, settings):
    from app.services import inference

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.zeros((1, 4)))
            self.input_ids = self["input_ids"]

        def to(self, device):
            assert device == "cpu"
            return self

    processor = Mock()
    processor.apply_chat_template.return_value = "formatted prompt"
    processor.return_value = Batch()
    processor.batch_decode.return_value = ['{"events": []}']
    model = SimpleNamespace(device="cpu", generate=Mock(return_value=np.zeros((1, 7))))
    monkeypatch.setattr(
        inference, "get_model", lambda _: SimpleNamespace(processor=processor, model=model)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list),
    )
    frames = [
        Frame(1.25, np.zeros((56, 56, 3), np.uint8)),
        Frame(2.75, np.zeros((56, 56, 3), np.uint8)),
    ]
    assert QwenObserver(settings).observe(frames, time.monotonic() + 10) == []
    content = processor.apply_chat_template.call_args.args[0][0]["content"]
    assert "1.250000000" in content[1]["text"]
    assert "2.750000000" in content[3]["text"]
    assert len(processor.call_args.kwargs["images"]) == 2
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert processor.batch_decode.call_args.args[0].shape == (1, 3)


def test_model_cached_per_process(monkeypatch, settings):
    import app.services.model_loader as loader

    factory = Mock(return_value=object())
    monkeypatch.setattr(loader, "LocalQwen", factory)
    get_model.cache_clear()
    try:
        assert get_model(settings.model_dump_json()) is get_model(settings.model_dump_json())
        assert factory.call_count == 1
    finally:
        get_model.cache_clear()
