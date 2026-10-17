"""Network-enabled setup only. Never imported by the inference worker."""

import fcntl
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from app.core.logging import configure_logging

REPO_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MARKER = ".forte-checkpoint.json"
REQUIRED = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
)
logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_checkpoint(path: Path) -> list[Path]:
    """Structural completeness, not a replacement for an actual model smoke test."""
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl" or config.get("hidden_size") != 2048:
        raise ValueError("Expected Qwen2.5-VL-3B-Instruct configuration")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards or any(
        Path(name).name != name or not name.endswith(".safetensors") for name in shards
    ):
        raise ValueError("Invalid checkpoint shard index")
    tokenizer = json.loads((path / "tokenizer_config.json").read_text())
    template_files = ["chat_template.json"] if (path / "chat_template.json").exists() else []
    if not template_files and not tokenizer.get("chat_template"):
        raise ValueError("Checkpoint chat template missing")
    files = [path / name for name in sorted(set(REQUIRED) | shards | set(template_files))]
    for file in files:
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Checkpoint file missing or empty: {file.name}")
        if file.suffix == ".json":
            json.loads(file.read_text())
    return files


def initialize(
    path: Path,
    revision: str = DEFAULT_REVISION,
    *,
    offline: bool = False,
    download: Callable | None = None,
) -> bool:
    """Return True after downloading, False after verifying an already prepared checkpoint.

    Interrupted downloads have no completion marker and resume through Hub metadata.
    Existing completed checkpoints are immutable: corruption or revision changes fail
    clearly instead of rewriting files that a running worker may be reading.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("QWEN_MODEL_REVISION must be a full 40-character commit SHA")
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".init.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = path / MARKER
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if manifest.get("repo_id") != REPO_ID or manifest.get("revision") != revision:
                raise ValueError("Checkpoint revision differs; use a fresh MODEL_WEIGHTS_DIR")
            files = validate_checkpoint(path)
            expected = manifest.get("sha256", {})
            if set(expected) != {file.name for file in files} or any(
                expected[file.name] != digest(file) for file in files
            ):
                raise ValueError("Checkpoint integrity check failed; use a fresh MODEL_WEIGHTS_DIR")
            logger.info("model_init_already_complete", extra={"metrics": {"revision": revision}})
            return False
        if offline:
            raise ValueError(
                "Offline setup requires a checkpoint prepared by model-init, including its completion marker"
            )
        if download is None:
            # Set these before importing Hub; offline constants are read at import time.
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            from huggingface_hub import snapshot_download

            download = snapshot_download
        logger.info(
            "model_init_downloading", extra={"metrics": {"repo_id": REPO_ID, "revision": revision}}
        )
        download(
            repo_id=REPO_ID,
            revision=revision,
            local_dir=str(path),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE"],
            max_workers=2,
            local_files_only=False,
        )
        files = validate_checkpoint(path)
        manifest = {
            "repo_id": REPO_ID,
            "revision": revision,
            "sha256": {file.name: digest(file) for file in files},
        }
        pending = path / (MARKER + ".tmp")
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(marker)
        logger.info(
            "model_init_complete", extra={"metrics": {"revision": revision, "files": len(files)}}
        )
        return True


def main() -> None:
    configure_logging()
    try:
        initialize(
            Path(os.environ.get("QWEN_MODEL_PATH", "weights/Qwen2.5-VL-3B-Instruct")),
            os.environ.get("QWEN_MODEL_REVISION", DEFAULT_REVISION),
            offline=os.environ.get("MODEL_INIT_OFFLINE", "0").lower() in {"1", "true"},
        )
    except Exception as exc:  # noqa: BLE001 -- setup must exit nonzero without exposing token-bearing errors
        logger.error(
            "model_init_failed: check network, disk space, permissions, revision and checkpoint integrity",
            extra={"metrics": {"exception_type": type(exc).__name__}},
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
