"""Regression coverage for profile-owned Kokoro backend blend state."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager
from tldw_chatbook.TTS.backends import kokoro
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)


@pytest.mark.asyncio
async def test_manager_backends_keep_default_blends_in_their_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Actual manager sessions give each default backend its own profile path."""
    first_config = tmp_path / "first" / "config.toml"
    second_config = tmp_path / "second" / "config.toml"
    first_config.parent.mkdir(mode=0o700)
    second_config.parent.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def initialize_without_model(self: KokoroTTSBackend) -> None:
        """Avoid loading model weights while preserving manager construction."""

    monkeypatch.setattr(KokoroTTSBackend, "initialize", initialize_without_model)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first_config))
    first_manager = TTSBackendManager({})
    first_backend = await first_manager.get_backend("local_kokoro_onnx")

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second_config))
    second_manager = TTSBackendManager({})
    second_backend = await second_manager.get_backend("local_kokoro_onnx")

    assert isinstance(first_backend, KokoroTTSBackend)
    assert isinstance(second_backend, KokoroTTSBackend)
    assert first_backend is not second_backend
    assert first_backend.voice_blends_dir == first_config.parent / "kokoro_voice_blends"
    assert second_backend.voice_blends_dir == second_config.parent / "kokoro_voice_blends"


def test_explicit_backend_blend_directory_is_preserved(tmp_path: Path) -> None:
    """A configured backend blend directory remains unchanged."""
    explicit = tmp_path / "configured-blends"

    backend = KokoroTTSBackend(config={"KOKORO_VOICE_BLENDS_DIR": explicit})

    assert backend.voice_blends_dir == explicit


def test_save_voice_blend_restores_memory_when_private_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed atomic save preserves both existing disk data and memory state."""
    blend_directory = tmp_path / "blends"
    blend_directory.mkdir()
    blend_file = blend_directory / "voice_blends.json"
    original_text = (
        '{\n'
        '  "existing": {\n'
        '    "voices": [["af_sarah", 1.0]],\n'
        '    "description": "Existing blend",\n'
        '    "created_at": "2026-01-01T00:00:00",\n'
        '    "metadata": {}\n'
        '  }\n'
        '}\n'
    )
    blend_file.write_text(original_text, encoding="utf-8")
    backend = KokoroTTSBackend({"KOKORO_VOICE_BLENDS_DIR": blend_directory})
    backend.saved_blends = {
        "existing": {
            "voices": [("af_sarah", 1.0)],
            "description": "Existing blend",
            "created_at": "2026-01-01T00:00:00",
            "metadata": {},
        }
    }

    def fail_private_write(*args: object, **kwargs: object) -> object:
        raise PrivatePathError(
            PrivatePathResult(blend_file, PrivatePathStatus.OPERATION_FAILED)
        )

    monkeypatch.setattr(kokoro, "write_private_json", fail_private_write)

    assert backend.save_voice_blend("new", [("af_bella", 1.0)]) is False
    assert "new" not in backend.saved_blends
    assert blend_file.read_text(encoding="utf-8") == original_text


def test_delete_voice_blend_restores_memory_when_private_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed atomic deletion restores the removed in-memory blend."""
    blend_directory = tmp_path / "blends"
    blend_directory.mkdir()
    blend_file = blend_directory / "voice_blends.json"
    original_text = (
        '{\n'
        '  "saved": {\n'
        '    "voices": [["af_bella", 1.0]],\n'
        '    "description": "Saved blend",\n'
        '    "created_at": "2026-01-01T00:00:00",\n'
        '    "metadata": {}\n'
        '  }\n'
        '}\n'
    )
    blend_file.write_text(original_text, encoding="utf-8")
    backend = KokoroTTSBackend({"KOKORO_VOICE_BLENDS_DIR": blend_directory})
    original_blend = {
        "voices": [("af_bella", 1.0)],
        "description": "Saved blend",
        "created_at": "2026-01-01T00:00:00",
        "metadata": {},
    }
    backend.saved_blends = {"saved": original_blend}

    def fail_private_write(*args: object, **kwargs: object) -> object:
        raise PrivatePathError(
            PrivatePathResult(blend_file, PrivatePathStatus.OPERATION_FAILED)
        )

    monkeypatch.setattr(kokoro, "write_private_json", fail_private_write)

    assert backend.delete_voice_blend("saved") is False
    assert backend.saved_blends["saved"] == original_blend
    assert blend_file.read_text(encoding="utf-8") == original_text
