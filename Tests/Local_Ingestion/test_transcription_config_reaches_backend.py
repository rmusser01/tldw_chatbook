"""TASK-1754: the live transcription backend must read `[transcription]`.

`_LegacyTranscriptionBackend.__init__` reads its configuration with the
dotted 2-arg `get_cli_setting("transcription.default_provider", fallback)`
form. Before the config.py fix, that form dropped the caller's default
whenever the default was a string (see `Tests/Utils/test_config_nested_settings.py
::TestDottedFormStringDefaultRegression` for the accessor-level RED/GREEN
proof) -- so provider, model, language, source/target language and device
all silently fell back to whatever the constructor's Python-level fallback
computed, never the user's `[transcription]` config.

These tests exercise the real consumer end of that fix: the backend itself
(AC #1/#3), and the actual `transcribe()` dispatch seam media ingest calls
into (AC #4) -- both with the real `get_cli_setting` and a real, isolated
config file (TLDW_CONFIG_PATH + force_reload), never a mock of the accessor
itself (the `test_config_nested_settings.py` C1 lesson: accessor mocks hid
this exact class of bug through five review gates before).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.Local_Ingestion.transcription_service import (
    _LegacyTranscriptionBackend,
)


@contextmanager
def _real_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml_text: str):
    """Point the real loader at a scratch TOML; restore + reload afterwards.

    Mirrors `Tests/Utils/test_config_nested_settings.py::_real_config`
    exactly (kept local rather than imported to avoid coupling two
    otherwise-independent test modules to one shared private helper).
    """
    config_path = tmp_path / "scratch-transcription-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield
    finally:
        if original_env is not None:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        else:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        config_mod.load_cli_config_and_ensure_existence(force_reload=True)


# Every value here is deliberately something none of `_LegacyTranscriptionBackend
# .__init__`'s own Python-level fallbacks would ever compute (not
# "faster-whisper"/"parakeet-mlx"/"base"/"en"/"cpu"/etc), so a passing
# assertion cannot be explained by coincidentally matching a hardcoded
# fallback instead of actually reading the file.
CONFIGURED_TRANSCRIPTION_TOML = """
[transcription]
default_provider = "remote-whisper"
default_model = "custom-configured-model"
default_language = "fr"
default_source_language = "de"
default_target_language = "es"
device = "mps"
"""


class TestConfiguredProviderReachesBackend:
    """AC #1 + AC #3: the backend's `self.config` must reflect a configured,
    non-default provider/model/language/device -- not the constructor's own
    platform-fallback guesses."""

    def test_backend_config_reflects_transcription_section(
        self, tmp_path, monkeypatch
    ):
        with _real_config(tmp_path, monkeypatch, CONFIGURED_TRANSCRIPTION_TOML):
            backend = _LegacyTranscriptionBackend()

        assert backend.config["default_provider"] == "remote-whisper"
        assert backend.config["default_model"] == "custom-configured-model"
        assert backend.config["default_language"] == "fr"
        assert backend.config["default_source_language"] == "de"
        assert backend.config["default_target_language"] == "es"
        assert backend.config["device"] == "mps"

    def test_backend_config_still_falls_back_when_unset(self, tmp_path, monkeypatch):
        """Sanity counterpart: an empty `[transcription]` table must still
        produce the constructor's platform-preference fallback, not crash
        or silently produce `None` -- proving the fix didn't just make
        every read return the configured value unconditionally."""
        with _real_config(tmp_path, monkeypatch, "[transcription]\n"):
            backend = _LegacyTranscriptionBackend()

        # The template itself bakes a platform-preference default_provider
        # (task-867), so it is never literally unset -- but it must be a
        # real provider string, never None, and device/model/language must
        # match the constructor's own documented fallbacks.
        assert isinstance(backend.config["default_provider"], str)
        assert backend.config["default_provider"]
        assert backend.config["default_language"] == "en"
        assert backend.config["device"] == "cpu"


class TestMediaIngestEndToEnd:
    """AC #4: media ingest transcription (`transcribe()`) must honour a
    user-configured provider end to end.

    `transcribe()` resolves `provider = provider or self.config
    ["default_provider"]` and, for an unrecognised provider, raises
    `ValueError(f"Unknown or unavailable transcription provider: {provider}.
    ...")` before touching any model/network code. Configuring a
    provider name no real backend implements turns that raise into a
    cheap, deterministic probe: if the configured name appears in the
    error, the value travelled all the way from `[transcription]` through
    `__init__` into the real `transcribe()` dispatch -- the exact path a
    real ingest call takes -- with no audio decoding, model download, or
    network access involved.
    """

    def test_transcribe_dispatches_on_the_configured_provider(
        self, tmp_path, monkeypatch
    ):
        configured_provider = "totally-bespoke-nonexistent-provider"
        with _real_config(
            tmp_path,
            monkeypatch,
            f'[transcription]\ndefault_provider = "{configured_provider}"\n',
        ):
            backend = _LegacyTranscriptionBackend()

        # `.wav` suffix short-circuits `_ensure_wav_format`'s ffmpeg
        # conversion path (it only converts non-.wav inputs), so an empty
        # placeholder file is sufficient -- no real audio content, no
        # ffmpeg, no model.
        fake_audio = tmp_path / "sample.wav"
        fake_audio.write_bytes(b"")

        with pytest.raises(ValueError) as excinfo:
            backend.transcribe(str(fake_audio), provider=None)

        assert configured_provider in str(excinfo.value)
