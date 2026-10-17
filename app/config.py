from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Security
    API_KEY: str = "forte_hackathon_jury_secret_key"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            import json
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

    # Storage paths (relative to backend dir or absolute)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATABASE_URL: str = "sqlite+aiosqlite:///storage/db.sqlite3"
    UPLOAD_DIR: Path = Path("storage/uploads")
    FRAMES_DIR: Path = Path("storage/frames")
    RULES_PATH: Path = Path("rules/rules.yaml")
    WEIGHTS_DIR: Path = Path("weights")

    # Pipeline config
    FPS_SAMPLE_RATE: int = 5
    CONFIDENCE_THRESHOLD: float = 0.50
    DEVICE: str = "cpu"

    # Redis Queue
    USE_REDIS_QUEUE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"


settings = Settings()
