"""Meetings resolve exact user selection before any runtime/device effects."""

import builtins

import pytest

from Tests.Audio.test_meeting_owner import _owner
from tldw_chatbook.Audio import meeting_owner as meetings
from tldw_chatbook.Chat import console_voice_input as voice


@pytest.fixture
def selected(monkeypatch):
    values = {"provider": "faster-whisper"}

    def setting(section, key=None, default=None):
        if (section, key) == ("transcription", "default_provider"):
            return values["provider"]
        if (section, key) == ("transcription", "default_language"):
            return "fr"
        return default

    monkeypatch.setattr(voice, "get_cli_setting", setting)
    monkeypatch.setattr(voice, "installed_local_providers", lambda: ("faster-whisper",))
    monkeypatch.setattr(meetings, "diarization_requirements", lambda: ())
    monkeypatch.setattr(meetings, "resolve_engine", lambda settings: (None, ()))
    monkeypatch.setattr(meetings, "scan_recoverable", lambda directory: ())
    return values


@pytest.mark.parametrize(
    "provider", ["auto", "lightning-whisper-mlx", "lightning-whisper"]
)
def test_meeting_prepare_refuses_deferred_selection_before_facade_or_devices(
    tmp_path, monkeypatch, selected, provider
):
    selected["provider"] = "lightning-whisper"
    owner, _, _ = _owner(tmp_path, provider=provider, voice_match=False)
    owner._facade_factory = lambda: pytest.fail("facade created")
    owner._tap_probe = lambda **kwargs: pytest.fail("device probed")
    monkeypatch.setattr(
        voice,
        "installed_local_providers",
        lambda: pytest.fail("provider packages probed"),
    )
    with pytest.raises(RuntimeError, match="Your saved settings have not been changed"):
        owner.prepare()
    assert owner.prepared is None
    assert not owner.settings.recordings_dir.exists()
    assert selected["provider"] == "lightning-whisper"


def test_meeting_start_rechecks_saved_deferral_before_capture_module_import(
    tmp_path, monkeypatch, selected
):
    owner, _, _ = _owner(tmp_path, voice_match=False)
    owner.prepare()
    selected["provider"] = "lightning-whisper-mlx"
    original_import = builtins.__import__

    def no_capture_import(name, *args, **kwargs):
        if name.endswith("meeting_capture"):
            pytest.fail("capture module imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_capture_import)
    with pytest.raises(RuntimeError, match="Lightning Whisper MLX is deferred"):
        owner.start()
    assert owner.session is None
    assert not owner.settings.recordings_dir.exists()


def test_explicit_installed_meeting_override_preserves_model_language_with_deferred_global(
    tmp_path, selected
):
    selected["provider"] = "lightning-whisper"
    owner, _, _ = _owner(
        tmp_path, provider="faster-whisper", model="manual.en", voice_match=False
    )
    prepared = owner.prepare()
    assert prepared.provider == "faster-whisper"
    assert prepared.model == "manual.en"
    assert owner._cfg.language == "fr"
    assert selected["provider"] == "lightning-whisper"


def test_missing_exact_meeting_override_does_not_fall_back(
    tmp_path, monkeypatch, selected
):
    selected["provider"] = "parakeet-mlx"
    monkeypatch.setattr(voice, "installed_local_providers", lambda: ("parakeet-mlx",))
    owner, _, _ = _owner(tmp_path, provider="faster-whisper", voice_match=False)
    owner._facade_factory = lambda: pytest.fail("absent override reached facade")
    with pytest.raises(
        RuntimeError, match="selected transcription provider is unavailable"
    ):
        owner.prepare()
    assert owner.prepared is None
    assert not owner.settings.recordings_dir.exists()
