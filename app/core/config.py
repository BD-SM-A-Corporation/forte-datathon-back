from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "analysis"
    upload_dir: Path = Path("storage/uploads")
    rules_path: Path = Path("rules/regulations.json")
    model_rules_path: Path | None = Path("rules/regulations.en.json")
    max_upload_mb: int = Field(512, ge=1)
    max_duration_sec: float = Field(150, gt=0)
    max_video_pixels: int = Field(3840 * 2160, gt=0)
    max_decoded_frames: int = Field(18000, gt=0)
    base_sample_fps: float = Field(1, gt=0, le=30)
    detailed_sample_fps: float = Field(3, gt=0, le=30)
    window_duration_sec: float = Field(5, gt=0)
    window_overlap_sec: float = Field(1, ge=0)
    motion_threshold: float = Field(0.04, gt=0, le=1)
    motion_pixel_threshold: int = Field(25, ge=1, le=255)
    motion_merge_gap_sec: float = Field(1, ge=0)
    inference_max_side: int = Field(448, ge=56)
    roi: tuple[float, float, float, float] | None = None
    inference_backend: Literal["local", "dmr"] = "local"
    model_base_url: str = ""
    model_name: str = ""
    model_http_timeout_sec: float = Field(120, gt=0, allow_inf_nan=False)
    qwen_model_path: Path = Path("weights/Qwen2.5-VL-3B-Instruct")
    model_device: Literal["cpu", "cuda"] = "cpu"
    model_quantization: Literal["none", "4bit", "8bit"] = "none"
    max_new_tokens: int = Field(512, ge=32)
    max_frames_per_window: int = Field(12, ge=2)
    generation_retries: int = Field(1, ge=0, le=2)
    max_refinement_windows: int = Field(8, ge=0)
    rule_verification_enabled: bool = True
    max_rule_checks: int = Field(4, ge=1, le=16)
    verification_context_sec: float = Field(6, ge=0, le=30)
    verification_reserve_sec: float = Field(140, gt=0)
    verification_check_timeout_sec: float = Field(40, gt=0)
    verification_max_new_tokens: int = Field(768, ge=128, le=2048)
    processing_budget_sec: int = Field(300, ge=1)
    result_ttl_sec: int = Field(86400, ge=60)
    queue_ttl_sec: int = Field(3600, ge=60)
    event_merge_gap_sec: float = Field(0.2, ge=0)
    cors_origins: list[str] = ["http://localhost:3000"]

    @model_validator(mode="after")
    def consistent(self) -> "Settings":
        self.model_base_url = self.model_base_url.strip().rstrip("/")
        self.model_name = self.model_name.strip()
        if self.inference_backend == "dmr":
            url = urlsplit(self.model_base_url)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.query
                or url.fragment
            ):
                raise ValueError("DMR requires MODEL_BASE_URL as an HTTP(S) API base URL")
            # Accessing port also validates malformed and out-of-range port values.
            _ = url.port
            if not self.model_name:
                raise ValueError("DMR requires MODEL_NAME identifying a prepared vision model")
        if self.window_overlap_sec >= self.window_duration_sec:
            raise ValueError("Window overlap must be shorter than window duration")
        if self.window_duration_sec * self.base_sample_fps + 1 > self.max_frames_per_window:
            raise ValueError("Frame cap must fit every baseline sample in a window")
        if self.detailed_sample_fps < self.base_sample_fps:
            raise ValueError("Detailed sampling must be at least the baseline rate")
        if self.roi:
            x1, y1, x2, y2 = self.roi
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("ROI must be normalized [left, top, right, bottom]")
        return self
