"""Docker Model Runner adapter for the synchronous RQ worker."""

import asyncio
import base64
import logging
import time

import cv2
import httpx

from app.core.config import Settings
from app.core.errors import BudgetExceeded, InferenceError
from app.services.inference import ValidatedObserver
from app.services.prompts import observation_prompt
from app.services.video import Frame

logger = logging.getLogger(__name__)


class DMRObserver(ValidatedObserver):
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(settings)
        self.transport = transport

    def _generate(self, frames: list[Frame], retry: bool, deadline: float) -> str:
        content = [
            {"type": "text", "text": observation_prompt(
                [f.timestamp_sec for f in frames], retry, rule_guidance=self.rule_guidance
            )}
        ]
        for frame in frames:
            if time.monotonic() >= deadline:
                raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")
            try:
                ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except cv2.error as exc:
                raise InferenceError("Cannot encode inference frame") from exc
            if not ok:
                raise InferenceError("Cannot encode inference frame")
            content.extend(
                [
                    {"type": "text", "text": f"Timestamp {frame.timestamp_sec:.9f} seconds"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(encoded.tobytes()).decode("ascii")
                        },
                    },
                ]
            )
        payload = {
            "model": self.settings.model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": self.settings.max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        response = asyncio.run(self._request(payload, deadline))
        try:
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise InferenceError("Model response incomplete; regulatory verdict unavailable")
            raw = choice["message"]["content"]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Missing text response")
            return raw
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise InferenceError("Invalid model server response; verdict unavailable") from exc

    async def _request(self, payload: dict, deadline: float) -> httpx.Response:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Processing time budget exceeded; verdict unavailable")
        timeout = min(self.settings.model_http_timeout_sec, remaining)
        try:
            # Enforce a total request deadline as well as HTTPX's per-operation timeouts.
            # Closing this request does not guarantee server-side generation cancellation.
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(
                    timeout=timeout,
                    transport=self.transport,
                    trust_env=False,
                    follow_redirects=False,
                ) as client:
                    response = await client.post(
                        self.settings.model_base_url + "/chat/completions", json=payload
                    )
                    response.raise_for_status()
                    return response
        except (TimeoutError, httpx.TimeoutException) as exc:
            if time.monotonic() >= deadline:
                raise BudgetExceeded(
                    "Processing time budget exceeded; verdict unavailable"
                ) from exc
            raise InferenceError("Model server request timed out; verdict unavailable") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "model_server_http_error",
                extra={"metrics": {"status_code": exc.response.status_code}},
            )
            raise InferenceError(
                "Model server rejected request; check model, endpoint, and server logs"
            ) from exc
        except httpx.RequestError as exc:
            raise InferenceError("Model server unavailable; check endpoint and server") from exc
