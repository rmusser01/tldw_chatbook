import pytest

from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.STTS_Window import STTSWindow
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


@pytest.mark.asyncio
async def test_stts_window_explains_missing_local_speech_dependencies(monkeypatch):
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "tts_processing", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "stt_processing", False)
    monkeypatch.setattr("tldw_chatbook.UI.STTS_Window.check_tts_deps", lambda: False)
    monkeypatch.setattr("tldw_chatbook.UI.STTS_Window.check_stt_deps", lambda: False)

    class STTSCapabilityApp(App):
        def compose(self) -> ComposeResult:
            yield STTSWindow(self)

    app = STTSCapabilityApp()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        status = app.query_one("#speech-capability-status", Static)
        rendered_status = str(status.render())

        assert rendered_status == "Local speech missing: TTS, STT"
        assert (
            'pip install "tldw_chatbook[local_tts,transcription_faster_whisper,speech_recording]"'
            in status.tooltip
        )


@pytest.mark.asyncio
async def test_stts_window_refreshes_local_speech_dependency_flags(monkeypatch):
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "tts_processing", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "stt_processing", False)

    def mark_tts_available() -> bool:
        DEPENDENCIES_AVAILABLE["tts_processing"] = True
        return True

    def mark_stt_available() -> bool:
        DEPENDENCIES_AVAILABLE["stt_processing"] = True
        return True

    monkeypatch.setattr("tldw_chatbook.UI.STTS_Window.check_tts_deps", mark_tts_available)
    monkeypatch.setattr("tldw_chatbook.UI.STTS_Window.check_stt_deps", mark_stt_available)

    class STTSCapabilityApp(App):
        def compose(self) -> ComposeResult:
            yield STTSWindow(self)

    app = STTSCapabilityApp()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        status = app.query_one("#speech-capability-status", Static)

        assert str(status.render()) == "Local speech: ready"
