"""Studio Save has one persistence owner and no compatibility event."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Select

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
    StudioTTSWriteResult,
    StudioTTSWriteStatus,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane


class _Store:
    def __init__(self) -> None:
        self.saved: list[StudioTTSPreferencesSnapshot] = []

    def save(self, snapshot: StudioTTSPreferencesSnapshot) -> StudioTTSWriteResult:
        self.saved.append(snapshot)
        return StudioTTSWriteResult(
            StudioTTSWriteStatus.SAVED,
            replace(snapshot, revision=snapshot.revision + 1),
        )


class _Host(App[None]):
    def __init__(self, store: _Store) -> None:
        super().__init__()
        self.store = store
        self.global_events: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            store=self.store,
            global_preferences=TTSPreferencesSnapshot.from_settings({}),
            load_result=StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.LOADED,
            ),
            id="speech-settings-pane",
        )

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSSettingsSaveEvent):
            self.global_events.append(message)
            return True
        return super().post_message(message)


@pytest.mark.asyncio
async def test_save_publishes_only_supported_studio_request_values() -> None:
    store = _Store()
    app = _Host(store)
    async with app.run_test(size=(140, 56)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        pane._apply_provider("chatterbox")
        app.query_one("#studio-tts-model-mode", Select).value = "first_available"
        app.query_one("#studio-tts-voice-mode", Select).value = "server_default"
        app.query_one("#chatterbox-exaggeration-input", Input).value = "0.65"
        app.query_one("#chatterbox-cfg-weight-input", Input).value = "0.35"
        assert await pane.save_preferences()

    assert not app.global_events
    assert len(store.saved) == 1
    assert store.saved[0].selection.to_mapping() == {
        "provider_id": "chatterbox",
        "model_mode": "first_available",
        "voice_mode": "server_default",
    }
    assert dict(store.saved[0].provider_options["chatterbox"]) == {
        "exaggeration": 0.65,
        "cfg_weight": 0.35,
    }


@pytest.mark.asyncio
async def test_save_never_emits_the_legacy_global_compatibility_event() -> None:
    store = _Store()
    app = _Host(store)
    async with app.run_test(size=(140, 56)) as pilot:
        await pilot.pause()
        assert await app.query_one(SpeechSettingsPane).save_preferences()

    assert len(store.saved) == 1
    assert not app.global_events
