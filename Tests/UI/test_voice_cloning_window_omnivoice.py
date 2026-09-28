"""Voice Cloning window offers the omnivoice backend with transcript cloning."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from tldw_chatbook.TTS.omnivoice_voice_manager import OmniVoiceVoiceManager
from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow
from tldw_chatbook.Widgets.voice_profile_dialog import VoiceProfileDialog


class _DialogApp(App):
    def __init__(self, request_reference_text: bool) -> None:
        super().__init__()
        self.request_reference_text = request_reference_text

    def compose(self) -> ComposeResult:
        yield VoiceProfileDialog(
            "/tmp/ref.wav",
            on_submit=None,
            request_reference_text=self.request_reference_text,
        )


class _WindowApp(App):
    def compose(self) -> ComposeResult:
        yield VoiceCloningWindow()


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 24000)


@pytest.mark.asyncio
async def test_backend_select_offers_omnivoice_and_defaults_to_higgs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_gate(
        monkeypatch, tmp_path, _local_availability(higgs=True, omnivoice=True)
    )
    app = _WindowApp()
    async with app.run_test(size=(200, 60)):
        backend_select = app.query_one("#backend-select", Select)
        option_values = {value for _label, value in backend_select._options}
        assert "omnivoice" in option_values
        assert "higgs" in option_values
        assert backend_select.value == "higgs"


@pytest.mark.asyncio
async def test_initialize_backends_includes_omnivoice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Hermetic config: the real get_cli_setting is storage-gated (ADR-126),
    # so a lone run would log "Error initializing backends" and build nothing.
    voices = tmp_path / "omni_voices"

    def fake_setting(section, key, default=None):
        if (section, key) == ("OmniVoiceSettings", "voice_samples_dir"):
            return str(voices)
        return default

    monkeypatch.setattr(
        "tldw_chatbook.UI.Voice_Cloning_Window.get_cli_setting", fake_setting
    )
    app = _WindowApp()
    async with app.run_test(size=(200, 60)):
        window = app.query_one(VoiceCloningWindow)
        await window._initialize_backends()

        assert "omnivoice" in window.backend_managers
        manager = window.backend_managers["omnivoice"]
        assert isinstance(manager, OmniVoiceVoiceManager)
        assert manager.voice_samples_dir == voices
        assert "higgs" in window.backend_managers


@pytest.mark.asyncio
async def test_profile_dialog_collects_reference_text_when_requested() -> None:
    app = _DialogApp(request_reference_text=True)
    async with app.run_test(size=(100, 40)):
        assert app.query_one("#reference-text-input")


@pytest.mark.asyncio
async def test_profile_dialog_omits_reference_text_by_default() -> None:
    app = _DialogApp(request_reference_text=False)
    async with app.run_test(size=(100, 40)):
        assert not app.query("#reference-text-input")


# --- the dependency gate asks about cloning backends, not Kokoro ----------------


def _local_availability(**available: bool):
    from tldw_chatbook.UI.Speech.speech_runtime_status import (
        SpeechLocalDependencyAvailability,
    )

    flags = {"stt": False, "kokoro": False, "chatterbox": False, "higgs": False}
    flags.update(available)
    return SpeechLocalDependencyAvailability(
        **{key: value for key, value in flags.items() if key != "omnivoice"},
        omnivoice=available.get("omnivoice", False),
    )


def _patch_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, availability) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.UI.Lab_Modules.lab_speech_status."
        "speech_local_dependency_availability",
        lambda **_kwargs: availability,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Voice_Cloning_Window.get_cli_setting",
        lambda section, key, default=None: str(tmp_path / section)
        if key == "voice_samples_dir"
        else default,
    )


@pytest.mark.asyncio
async def test_no_alert_when_a_cloning_backend_is_installed_without_kokoro(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """UAT: the window demanded kokoro-onnx + pyaudio (Kokoro's local TTS),
    which no cloning backend uses, even with OmniVoice fully installed."""
    from tldw_chatbook.Utils.widget_helpers import FeatureNotAvailableDialog

    _patch_gate(monkeypatch, tmp_path, _local_availability(omnivoice=True))
    app = _WindowApp()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause(0.3)
        assert not app.query(FeatureNotAvailableDialog)


@pytest.mark.asyncio
async def test_alert_names_cloning_backends_when_none_is_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from textual.widgets import Static

    from tldw_chatbook.Utils.widget_helpers import FeatureNotAvailableDialog

    _patch_gate(monkeypatch, tmp_path, _local_availability(kokoro=True))
    app = _WindowApp()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause(0.3)
        dialogs = app.query(FeatureNotAvailableDialog)
        assert len(dialogs) == 1
        text = " ".join(str(item.renderable) for item in dialogs.first().query(Static))
        assert "Voice Cloning" in text
        assert "omnivoice_tts" in text
        assert "kokoro" not in text.lower()


@pytest.mark.asyncio
async def test_defaults_to_an_installed_backend_when_higgs_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OmniVoice-only install must not open on the uninstalled Higgs."""
    _patch_gate(monkeypatch, tmp_path, _local_availability(omnivoice=True))
    app = _WindowApp()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause(0.3)
        window = app.query_one(VoiceCloningWindow)
        assert app.query_one("#backend-select", Select).value == "omnivoice"
        assert window.current_backend == "omnivoice"


# --- pure policy units (no Textual app) -----------------------------------------


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ({}, None),
        ({"kokoro": True}, None),  # Kokoro cannot clone
        ({"omnivoice": True}, "omnivoice"),
        ({"chatterbox": True}, "chatterbox"),
        ({"omnivoice": True, "chatterbox": True}, "omnivoice"),
        ({"higgs": True, "omnivoice": True}, "higgs"),  # Higgs stays default
    ],
)
def test_default_cloning_backend_policy(installed: dict, expected) -> None:
    from tldw_chatbook.UI.Voice_Cloning_Window import default_cloning_backend

    assert default_cloning_backend(_local_availability(**installed)) == expected


@pytest.mark.parametrize(
    ("absent", "expected"),
    [
        (set(), []),
        ({"tokenizers"}, ["tokenizers"]),
        ({"onnxruntime", "tokenizers"}, ["onnxruntime", "tokenizers"]),
    ],
)
def test_alert_lists_only_the_missing_omnivoice_modules(
    monkeypatch: pytest.MonkeyPatch, absent: set, expected: list
) -> None:
    """Qodo (PR #2833): the alert named both modules even when one was present."""
    import importlib.util

    from tldw_chatbook.Utils import widget_helpers

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name in absent else real_find_spec("json"),
    )
    assert widget_helpers.missing_omnivoice_modules() == expected
