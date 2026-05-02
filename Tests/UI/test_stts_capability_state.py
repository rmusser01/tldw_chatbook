import pytest

from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.STTS_Window import STTSWindow
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


@pytest.mark.asyncio
async def test_stts_window_explains_missing_local_speech_dependencies():
    original_tts = DEPENDENCIES_AVAILABLE.get("tts_processing")
    original_stt = DEPENDENCIES_AVAILABLE.get("stt_processing")
    DEPENDENCIES_AVAILABLE["tts_processing"] = False
    DEPENDENCIES_AVAILABLE["stt_processing"] = False

    class STTSCapabilityApp(App):
        def compose(self) -> ComposeResult:
            yield STTSWindow(self)

    app = STTSCapabilityApp()

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            status = app.query_one("#speech-capability-status", Static)
            rendered_status = str(status.render())

            assert "Local speech capabilities need optional dependencies" in rendered_status
            assert "Text-to-Speech unavailable" in rendered_status
            assert "Speech Recognition unavailable" in rendered_status
            recovery_command = (
                'pip install -e ".[local_tts,transcription_faster_whisper,speech_recording]"'
            )
            assert recovery_command in rendered_status
    finally:
        if original_tts is None:
            DEPENDENCIES_AVAILABLE.pop("tts_processing", None)
        else:
            DEPENDENCIES_AVAILABLE["tts_processing"] = original_tts

        if original_stt is None:
            DEPENDENCIES_AVAILABLE.pop("stt_processing", None)
        else:
            DEPENDENCIES_AVAILABLE["stt_processing"] = original_stt
