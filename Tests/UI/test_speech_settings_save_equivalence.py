"""The transition Lab save must publish only its Studio-owned controls.

Global-owned controls remain mounted as effective readouts, but Settings is
their sole write owner. Until TASK-1697 moves the remaining controls onto the
separate Studio store, this compatibility event is deliberately limited to
the five controls TASK-1692 proved request-scoped.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)

BASELINE = (
    pathlib.Path(__file__).parent / "fixtures" / "tts_settings_save_baseline.json"
)


class _CapturingHost(App[None]):
    """Records the save event instead of letting it reach persistence."""

    def __init__(self, widget_factory):
        super().__init__()
        self._factory = widget_factory
        self.saved: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield self._factory()

    def post_message(self, message):
        if isinstance(message, STTSSettingsSaveEvent):
            self.saved.append(message)
            return True
        return super().post_message(message)

    def notify(self, *args, **kwargs):
        pass


async def _event_posted_by(widget_factory) -> STTSSettingsSaveEvent:
    app = _CapturingHost(widget_factory)
    async with app.run_test(size=(200, 80)) as pilot:
        await pilot.pause()
        await pilot.pause()
        pane = app.query_one(widget_factory.target)
        pane._save_settings()
        await pilot.pause()
    assert app.saved, "Save posted no event at all"
    return app.saved[-1]


class _RebuiltFactory:
    from tldw_chatbook.UI.Speech.speech_settings_pane import (
        SpeechSettingsPane as target,
    )

    def __call__(self):
        return self.target(id="speech-settings-pane")


@pytest.mark.asyncio
async def test_lab_save_posts_only_studio_owned_compatibility_keys():
    baseline = json.loads(BASELINE.read_text())
    event = await _event_posted_by(_RebuiltFactory())
    posted = {key: repr(value) for key, value in event.settings.items()}
    expected_keys = {
        "ELEVENLABS_DEFAULT_MODEL",
        "CHATTERBOX_EXAGGERATION",
        "CHATTERBOX_CFG_WEIGHT",
        "ALLTALK_TTS_VOICE_DEFAULT",
        "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT",
    }

    assert set(posted) == expected_keys
    assert posted == {key: baseline[key] for key in expected_keys}
    assert event.preferences is None


@pytest.mark.asyncio
async def test_lab_save_never_posts_global_defaults_credentials_or_connections():
    event = await _event_posted_by(_RebuiltFactory())
    forbidden = {
        "audio_cpp",
        "openai_api_key",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "elevenlabs_api_key",
        "ALLTALK_TTS_URL_DEFAULT",
    }

    assert forbidden.isdisjoint(event.settings)
    assert event.preferences is None
