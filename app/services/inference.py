import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

import cv2
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import BudgetExceeded, InferenceError
from app.schemas.analysis import Observation, Observations
from app.services.model_loader import get_model
from app.services.prompts import SYSTEM_PROMPT, observation_prompt, observation_schema
from app.services.video import Frame

logger = logging.getLogger(__name__)


class Observer(Protocol):
    def observe(self, frames: list[Frame], deadline: float) -> list[Observation]: ...


class TimestampGroundingError(ValueError):
    """An otherwise valid event references a frame that was not supplied."""


def validation_diagnostics(exc: Exception) -> dict:
    """Report structure errors without copying model text or arbitrary JSON keys."""
    if isinstance(exc, json.JSONDecodeError):
        return {"reason": "invalid_json", "position": exc.pos}
    if isinstance(exc, ValidationError):
        fields = {"events", *Observation.model_fields}
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        return {
            "reason": "schema_mismatch",
            "errors": [
                {
                    "type": error["type"],
                    "loc": [
                        item if isinstance(item, int) or item in fields else "<field>"
                        for item in error["loc"]
                    ],
                }
                for error in errors[:8]
            ],
        }
    if isinstance(exc, TimestampGroundingError):
        return {"reason": "ungrounded_timestamp"}
    return {"reason": "invalid_output"}


def parse_observations(raw: str, frames: list[Frame]) -> list[Observation]:
    raw = raw.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    elif raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()
    parsed = Observations.model_validate(json.loads(raw))
    timestamps = [f.timestamp_sec for f in frames]
    for event in parsed.events:
        for timestamp in [
            event.start_sec,
            event.end_sec,
            *event.evidence_timestamps,
            *(t for times in event.facts.values() for t in times),
        ]:
            if not any(abs(timestamp - t) <= 0.001 for t in timestamps):
                raise TimestampGroundingError("Timestamp not grounded in supplied frames")
        # Normalize sub-millisecond model rounding to the actual decoded PTS.
        event.start_sec = min(timestamps, key=lambda t: abs(t - event.start_sec))
        event.end_sec = min(timestamps, key=lambda t: abs(t - event.end_sec))
        event.evidence_timestamps = sorted(
            {min(timestamps, key=lambda t: abs(t - x)) for x in event.evidence_timestamps}
        )
        event.facts = {
            name: sorted({min(timestamps, key=lambda t: abs(t - x)) for x in times})
            for name, times in event.facts.items()
        }
    return parsed.events


class ValidatedObserver(ABC):
    """Shared frame, deadline, and output contract for inference backends."""

    def __init__(
        self, settings: Settings, generate: Callable[[list[Frame], bool, float], str] | None = None
    ):
        self.settings = settings
        self.generate = generate or self._generate
        self.rule_guidance = ""
        self.generation_metrics: dict = {}

    def observe(self, frames: list[Frame], deadline: float) -> list[Observation]:
        if not frames:
            return []
        if len(frames) > self.settings.max_frames_per_window:
            raise InferenceError("Inference frame limit exceeded")
        if any(b.timestamp_sec <= a.timestamp_sec for a, b in itertools.pairwise(frames)):
            raise InferenceError("Inference frames must be chronological")
        for attempt in range(self.settings.generation_retries + 1):
            if time.monotonic() >= deadline:
                raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")
            self.generation_metrics = {}
            raw = self.generate(frames, attempt > 0, deadline)
            if time.monotonic() >= deadline:
                raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")
            try:
                return parse_observations(raw, frames)
            except (ValueError, ValidationError, TypeError) as exc:
                diagnostic = validation_diagnostics(exc)
                if (
                    diagnostic["reason"] == "invalid_json"
                    and self.generation_metrics.get("finish_reason") == "length"
                ):
                    diagnostic["reason"] = "output_token_limit"
                logger.warning(
                    "invalid_model_output",
                    extra={"metrics": {
                        "attempt": attempt + 1,
                        "frame_count": len(frames),
                        "window_start_sec": frames[0].timestamp_sec,
                        "window_end_sec": frames[-1].timestamp_sec,
                        "output_chars": len(raw),
                        **self.generation_metrics,
                        **diagnostic,
                    }},
                )
                if attempt == self.settings.generation_retries:
                    raise InferenceError(
                        f"Model output failed validation ({diagnostic['reason']}); "
                        "regulatory verdict unavailable"
                    ) from exc
        raise AssertionError("unreachable")

    @abstractmethod
    def _generate(self, frames: list[Frame], retry: bool, deadline: float) -> str:
        raise NotImplementedError


def build_prefix_constraint(tokenizer_data, timestamps: list[float]):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer_data, JsonSchemaParser(observation_schema(timestamps))
    )


class QwenObserver(ValidatedObserver):
    def _generate(self, frames: list[Frame], retry: bool, deadline: float) -> str:
        if time.monotonic() >= deadline:
            raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")
        service = get_model(self.settings.model_dump_json())
        if time.monotonic() >= deadline:
            raise BudgetExceeded("Processing time budget exceeded during model initialization")
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        images = [Image.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)) for f in frames]
        content = [
            {
                "type": "text",
                "text": observation_prompt(
                    [f.timestamp_sec for f in frames], retry, include_system=False,
                    rule_guidance=self.rule_guidance,
                ),
            }
        ]
        for frame, image in zip(frames, images):
            content += [
                {"type": "text", "text": f"Timestamp {frame.timestamp_sec:.9f} seconds"},
                {"type": "image", "image": image},
            ]
        # Native Qwen multi-image chat pathway; avoids assuming uniform video FPS for VFR inputs.
        text = service.processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = service.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        ).to(service.model.device)
        with torch.inference_mode():
            # A retry may need more room to finish multiple observed actions.
            # The same job deadline still bounds both attempts.
            token_limit = self.settings.max_new_tokens * (2 if retry else 1)
            generated = service.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=token_limit,
                prefix_allowed_tokens_fn=build_prefix_constraint(
                    service.tokenizer_data, [f.timestamp_sec for f in frames]
                ),
                stopping_criteria=StoppingCriteriaList([Deadline()]),
            )
        output_tokens = generated.shape[1] - inputs.input_ids.shape[1]
        eos_ids = getattr(getattr(service.model, "generation_config", None), "eos_token_id", [])
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        if time.monotonic() >= deadline:
            finish_reason = "deadline"
        elif output_tokens and generated[0, -1].item() in eos_ids:
            finish_reason = "eos"
        elif output_tokens >= token_limit:
            finish_reason = "length"
        else:
            finish_reason = "stopped"
        self.generation_metrics = {
            "output_tokens": output_tokens,
            "token_limit": token_limit,
            "finish_reason": finish_reason,
        }
        logger.info("model_generation_finished", extra={"metrics": {
            **self.generation_metrics,
            "retry": retry,
            "window_start_sec": frames[0].timestamp_sec,
            "window_end_sec": frames[-1].timestamp_sec,
        }})
        return service.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def create_observer(settings: Settings, *, rule_guidance: str = "") -> Observer:
    if settings.inference_backend == "dmr":
        from app.services.dmr import DMRObserver

        observer = DMRObserver(settings)
    else:
        observer = QwenObserver(settings)
    observer.rule_guidance = rule_guidance
    return observer


class MockObserver:
    """Explicit dependency injection for tests/benchmarks only; never a model fallback."""

    def __init__(self, events: list[Observation] | None = None):
        self.events = events or []
        self.calls = 0

    def observe(self, frames: list[Frame], deadline: float) -> list[Observation]:
        self.calls += 1
        times = {f.timestamp_sec for f in frames}
        return [
            e.model_copy(deep=True)
            for e in self.events
            if {
                e.start_sec,
                e.end_sec,
                *e.evidence_timestamps,
                *(t for values in e.facts.values() for t in values),
            }
            <= times
        ]
