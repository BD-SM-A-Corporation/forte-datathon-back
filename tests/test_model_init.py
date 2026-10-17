import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.workers.model_init import DEFAULT_REVISION, MARKER, REQUIRED, initialize


def checkpoint(**kwargs):
    path = Path(kwargs["local_dir"])
    for name in REQUIRED:
        (path / name).write_text("{}" if name.endswith(".json") else "merges")
    (path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl", "hidden_size": 2048}))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model.safetensors"}})
    )
    (path / "model.safetensors").write_bytes(b"fixture weights")


def test_download_and_offline_reuse(tmp_path):
    download = Mock(side_effect=checkpoint)
    assert initialize(tmp_path, download=download)
    assert download.call_args.kwargs["revision"] == DEFAULT_REVISION
    assert download.call_args.kwargs["repo_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert download.call_args.kwargs["local_files_only"] is False
    assert (tmp_path / MARKER).is_file()
    assert not initialize(tmp_path, offline=True, download=download)
    assert download.call_count == 1


def test_interrupted_download_can_retry(tmp_path):
    with pytest.raises(ConnectionError):
        initialize(tmp_path, download=Mock(side_effect=ConnectionError))
    assert not (tmp_path / MARKER).exists()
    assert initialize(tmp_path, download=checkpoint)


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "wrong_model", "no_template", "unsafe_shard"]
)
def test_incomplete_checkpoint_never_marked_ready(tmp_path, damage):
    def broken(**kwargs):
        checkpoint(**kwargs)
        if damage == "missing":
            (tmp_path / "model.safetensors").unlink()
        elif damage == "empty":
            (tmp_path / "model.safetensors").write_bytes(b"")
        elif damage == "wrong_model":
            (tmp_path / "config.json").write_text("{}")
        elif damage == "no_template":
            (tmp_path / "tokenizer_config.json").write_text("{}")
        else:
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer": "../model.safetensors"}})
            )

    with pytest.raises(ValueError):
        initialize(tmp_path, download=broken)
    assert not (tmp_path / MARKER).exists()


def test_completed_checkpoint_corruption_is_detected(tmp_path):
    initialize(tmp_path, download=checkpoint)
    (tmp_path / "model.safetensors").write_bytes(b"changed weights")
    download = Mock()
    with pytest.raises(ValueError, match="integrity"):
        initialize(tmp_path, download=download)
    download.assert_not_called()


def test_revision_change_requires_fresh_directory(tmp_path):
    initialize(tmp_path, download=checkpoint)
    with pytest.raises(ValueError, match="revision differs"):
        initialize(tmp_path, revision="a" * 40, download=Mock())


def test_branch_revision_rejected(tmp_path):
    with pytest.raises(ValueError, match="commit SHA"):
        initialize(tmp_path, revision="main", download=Mock())


def test_offline_without_preparation_fails(tmp_path):
    download = Mock()
    with pytest.raises(ValueError, match="Offline setup requires"):
        initialize(tmp_path, offline=True, download=download)
    download.assert_not_called()
