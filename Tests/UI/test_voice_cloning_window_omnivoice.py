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
async def test_backend_select_offers_omnivoice_and_defaults_to_higgs() -> None:
    app = _WindowApp()
    async with app.run_test(size=(200, 60)):
        backend_select = app.query_one("#backend-select", Select)
        option_values = {value for _label, value in backend_select._options}
        assert "omnivoice" in option_values
        assert "higgs" in option_values
        assert backend_select.value == "higgs"


@pytest.mark.asyncio
async def test_initialize_backends_includes_omnivoice() -> None:
    app = _WindowApp()
    async with app.run_test(size=(200, 60)):
        window = app.query_one(VoiceCloningWindow)
        await window._initialize_backends()

        assert "omnivoice" in window.backend_managers
        assert isinstance(window.backend_managers["omnivoice"], OmniVoiceVoiceManager)
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
