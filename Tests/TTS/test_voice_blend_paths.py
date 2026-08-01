"""Regression coverage for profile-owned UI Kokoro blend state."""

import json
import os
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest

from tldw_chatbook.TTS import voice_blend_paths
from tldw_chatbook.TTS.voice_blend_paths import (
    kokoro_ui_blend_file,
    write_kokoro_ui_blends,
)
from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget
from tldw_chatbook.UI.Speech.speech_catalog_mixin import SpeechCatalogMixin


def test_ui_blend_path_retargets_after_module_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The active profile selects the blend path at each helper call."""
    first = tmp_path / "first" / "config.toml"
    second = tmp_path / "second" / "config.toml"

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    assert kokoro_ui_blend_file() == first.parent / "kokoro_voice_blends.json"

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    assert kokoro_ui_blend_file() == second.parent / "kokoro_voice_blends.json"


def test_ui_blend_write_is_private_and_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Saving UI state uses the private atomic writer rather than a plain file write."""
    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    result = write_kokoro_ui_blends({"warm": {"voices": [["af_bella", 1.0]]}})

    assert result.lexical_path == kokoro_ui_blend_file()
    assert json.loads(result.lexical_path.read_text()) == {
        "warm": {"voices": [["af_bella", 1.0]]}
    }
    if os.name == "posix":
        assert stat.S_IMODE(result.lexical_path.stat().st_mode) == 0o600


def test_serialization_failure_preserves_existing_ui_blends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-serializable blend cannot truncate the currently saved UI state."""
    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    target = kokoro_ui_blend_file()
    target.write_text('{"existing": true}\n', encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setattr(voice_blend_paths.json, "dumps", Mock(side_effect=TypeError))

    with pytest.raises(TypeError):
        write_kokoro_ui_blends({"broken": object()})

    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_production_blend_choice_readers_retarget_between_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both production catalog readers expose only their active profile's blends."""
    first = tmp_path / "first" / "config.toml"
    second = tmp_path / "second" / "config.toml"
    first.parent.mkdir(mode=0o700)
    second.parent.mkdir(mode=0o700)
    (first.parent / "kokoro_voice_blends.json").write_text(
        '{"first": {"voices": [["af_bella", 1.0]]}}\n', encoding="utf-8"
    )
    (second.parent / "kokoro_voice_blends.json").write_text(
        '{"second": {"voices": [["af_sarah", 1.0]]}}\n', encoding="utf-8"
    )

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    assert TTSPlaygroundWidget._kokoro_blend_choices() == [
        ("Voice blend: first", "blend:first")
    ]
    assert SpeechCatalogMixin._kokoro_blend_choices() == [
        ("Voice blend: first", "blend:first")
    ]

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    assert TTSPlaygroundWidget._kokoro_blend_choices() == [
        ("Voice blend: second", "blend:second")
    ]
    assert SpeechCatalogMixin._kokoro_blend_choices() == [
        ("Voice blend: second", "blend:second")
    ]


def test_production_blend_choice_readers_ignore_legacy_home_decoy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy home state is neither read nor changed for a selected profile."""
    home = tmp_path / "home"
    decoy = home / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
    decoy.parent.mkdir(parents=True, mode=0o700)
    decoy.write_text(
        '{"legacy": {"voices": [["af_bella", 1.0]]}}\n', encoding="utf-8"
    )
    before = decoy.read_bytes()
    profile_config = tmp_path / "profile" / "config.toml"
    profile_config.parent.mkdir(mode=0o700)
    (profile_config.parent / "kokoro_voice_blends.json").write_text(
        '{"active": {"voices": [["af_sarah", 1.0]]}}\n', encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_config))
    original_read_text = Path.read_text

    def reject_legacy_decoy_read(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == decoy:
            raise AssertionError("Production reader accessed the legacy blend decoy")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_legacy_decoy_read)

    assert kokoro_ui_blend_file() == profile_config.parent / "kokoro_voice_blends.json"
    assert TTSPlaygroundWidget._kokoro_blend_choices() == [
        ("Voice blend: active", "blend:active")
    ]
    assert SpeechCatalogMixin._kokoro_blend_choices() == [
        ("Voice blend: active", "blend:active")
    ]
    assert decoy.read_bytes() == before
    assert decoy.exists()
