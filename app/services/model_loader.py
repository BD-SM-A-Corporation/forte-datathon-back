import json
import logging
import os
import time
from functools import lru_cache

from app.core.config import Settings
from app.core.errors import ConfigurationError

logger = logging.getLogger(__name__)


class LocalQwen:
    def __init__(self, settings: Settings):
        path = settings.qwen_model_path
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ConfigurationError(
                "Local Qwen checkpoint missing; download it before starting worker"
            )
        try:
            config = json.loads((path / "config.json").read_text())
            if config.get("model_type") != "qwen2_5_vl" or config.get("hidden_size") != 2048:
                raise ValueError("Wrong model")
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                "Expected Qwen2.5-VL-3B-Instruct checkpoint configuration"
            ) from exc
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        started = time.monotonic()
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )

            if settings.model_device == "cuda" and not torch.cuda.is_available():
                raise ConfigurationError("CUDA requested but no CUDA device is available")
            if settings.model_quantization != "none" and settings.model_device != "cuda":
                raise ConfigurationError("Quantization requires CUDA in this implementation")
            dtype = (
                torch.float32
                if settings.model_device == "cpu"
                else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            )
            kwargs = {}
            if settings.model_quantization != "none":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=settings.model_quantization == "4bit",
                    load_in_8bit=settings.model_quantization == "8bit",
                    bnb_4bit_compute_dtype=dtype,
                )
            self.processor = AutoProcessor.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                min_pixels=4 * 28 * 28,
                max_pixels=settings.inference_max_side**2,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                device_map={"": settings.model_device},
                attn_implementation="sdpa",
                **kwargs,
            ).eval()
        except ConfigurationError:
            raise
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise ConfigurationError(
                "Cannot load local Qwen checkpoint; check model files and runtime dependencies"
            ) from exc
        logger.info(
            "model_initialized",
            extra={
                "metrics": {
                    "seconds": time.monotonic() - started,
                    "device": settings.model_device,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated()
                    if torch.cuda.is_available()
                    else 0,
                }
            },
        )


@lru_cache(maxsize=1)
def get_model(settings_json: str) -> LocalQwen:
    """One model per long-lived worker process; settings are fixed at startup."""
    return LocalQwen(Settings.model_validate_json(settings_json))
