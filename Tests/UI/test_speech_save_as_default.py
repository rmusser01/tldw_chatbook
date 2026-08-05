"""The Playground links to Studio preferences without mutating global defaults."""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import (
    OpenStudioPreferencesRequested,
    SpeechPlaygroundPane,
)


class _CapturingHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.open_requests: list[OpenStudioPreferencesRequested] = []
        self.global_saves: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(id="speech-playground-pane")

    def post_message(self, message: Any) -> bool:
        if isinstance(message, OpenStudioPreferencesRequested):
            self.open_requests.append(message)
            return True
        if isinstance(message, STTSSettingsSaveEvent):
            self.global_saves.append(message)
            return True
        return super().post_message(message)


@pytest.mark.asyncio
async def test_studio_preferences_action_is_visible_and_reachable() -> None:
    app = _CapturingHost()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        button = app.query_one("#tts-open-studio-preferences-btn", Button)
        assert button.label.plain == "Studio preferences"
        assert button.allow_focus()
        pane = app.query_one("#speech-playground-pane")
        assert pane.region.contains_region(button.region)


@pytest.mark.asyncio
async def test_playground_opens_studio_editor_without_writing_global_defaults() -> None:
    app = _CapturingHost()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        await pilot.click("#tts-open-studio-preferences-btn")
        await pilot.pause()

    assert len(app.open_requests) == 1
    assert not app.global_saves
