"""Presentation and event contracts for the Console header speech switches."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Switch

from tldw_chatbook.Widgets.Console.console_speech_controls import (
    ConsoleAutoSpeakChanged,
    ConsoleHandsFreeToggleRequested,
    ConsoleSpeechControls,
)


class SpeechControlsHarness(App[None]):
    """Mount speech controls while recording only their public messages."""

    def __init__(self, *, sync_before_mount: bool = False) -> None:
        super().__init__()
        self.controls = ConsoleSpeechControls(id="console-speech-controls")
        self.auto_speak_requests: list[bool] = []
        self.hands_free_requests: list[bool] = []
        if sync_before_mount:
            self.controls.sync_auto_speak(enabled=True, paused=True)
            self.controls.sync_hands_free_state(True)

    def compose(self) -> ComposeResult:
        yield self.controls

    @on(ConsoleAutoSpeakChanged)
    def record_auto_speak_request(self, event: ConsoleAutoSpeakChanged) -> None:
        self.auto_speak_requests.append(event.enabled)

    @on(ConsoleHandsFreeToggleRequested)
    def record_hands_free_request(
        self,
        event: ConsoleHandsFreeToggleRequested,
    ) -> None:
        self.hands_free_requests.append(event.enabled)


@pytest.mark.asyncio
async def test_programmatic_sync_before_and_after_mount_is_silent() -> None:
    app = SpeechControlsHarness(sync_before_mount=True)

    async with app.run_test(size=(50, 5)) as pilot:
        await pilot.pause()
        auto_speak = app.query_one("#console-auto-speak", Switch)
        hands_free = app.query_one("#console-hands-free-switch", Switch)

        assert auto_speak.value is True
        assert hands_free.value is True
        assert "paused" in str(auto_speak.tooltip).lower()
        assert app.auto_speak_requests == []
        assert app.hands_free_requests == []

        app.controls.sync_auto_speak(enabled=False, paused=False)
        app.controls.sync_hands_free_state(False)
        await pilot.pause()

        assert auto_speak.value is False
        assert hands_free.value is False
        assert app.auto_speak_requests == []
        assert app.hands_free_requests == []


@pytest.mark.asyncio
async def test_each_user_gesture_posts_one_request() -> None:
    app = SpeechControlsHarness()

    async with app.run_test(size=(50, 5)) as pilot:
        auto_speak = app.query_one("#console-auto-speak", Switch)
        hands_free = app.query_one("#console-hands-free-switch", Switch)

        auto_speak.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.auto_speak_requests == [True]
        assert auto_speak.value is False

        app.controls.sync_auto_speak(enabled=True, paused=False)
        await pilot.pause()
        assert auto_speak.value is True
        assert app.auto_speak_requests == [True]

        hands_free.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert hands_free.value is True
        assert app.hands_free_requests == [True]

        app.controls.sync_hands_free_state(False)
        await pilot.pause()
        assert hands_free.value is False
        assert app.hands_free_requests == [True]
