"""Every Speech rail entry must actually reach its view.

This shipped broken and 295 tests did not notice. While the rebuilt pane was
returned as the Lab body directly, `STTSScreen.stts_window` stayed None, and
the rail's press handler bails on exactly that:

    if self.stts_window is None:
        logger.warning("Speech rail pressed before the body mounted; ignored.")
        return

So TTS Settings, AudioBook and Speech Recognition did nothing at all -- no
error, just a log line nobody reads. Only Voice Cloning worked, because it
pushes its own screen before reaching that guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane

SWITCHABLE = ("playground", "settings", "audiobook", "dictation")


class _RailHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.screen_under_test = STTSScreen(self)

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


@pytest.mark.asyncio
async def test_the_body_owns_view_switching():
    """The rail drives `STTSWindow.current_view`, so the window must exist.

    Asserting the seam rather than a symptom: with no window there is
    nothing for any rail entry to switch.
    """
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        assert screen.stts_window is not None, (
            "no window mounted; every rail entry will silently do nothing"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("view_key", SWITCHABLE)
async def test_each_rail_entry_switches_the_view(view_key):
    """Press the entry and assert the view actually changed."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        row = next(
            button
            for button in screen.query(Button)
            if getattr(button, "lab_view_key", None) == view_key
        )
        row.press()
        await pilot.pause()
        await pilot.pause()

        assert screen.stts_window is not None
        assert screen.stts_window.current_view == view_key, (
            f"pressing {view_key} left the view on {screen.stts_window.current_view}"
        )


@pytest.mark.asyncio
async def test_the_playground_view_mounts_a_playground():
    """The playground view must mount the rebuilt `SpeechPlaygroundPane`.

    dev shipped a profile library into the legacy playground widget while
    this rebuild was in flight; the pane took the view over once its axis
    row and dev's profile presets were reconciled -- the ownership ruling
    at `Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md`
    (`SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of
    record, every axis writer keeps the row's markers in step, and
    defaults are seeded from persisted preferences at construction).
    """
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#speech-playground-pane")


@pytest.mark.asyncio
async def test_rail_navigation_honors_cancel_then_discard_for_dirty_studio() -> None:
    app = _RailHost()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = app.screen_under_test
        await pilot.pause()
        await pilot.pause()

        settings_row = next(
            button
            for button in screen.query(Button)
            if getattr(button, "lab_view_key", None) == "settings"
        )
        playground_row = next(
            button
            for button in screen.query(Button)
            if getattr(button, "lab_view_key", None) == "playground"
        )
        settings_row.press()
        for _ in range(40):
            if (
                screen.query(SpeechSettingsPane)
                and screen.query_one(SpeechSettingsPane).query(
                    "#studio-tts-model-mode #label"
                )
                and "Loading"
                not in str(screen.query_one("#studio-tts-status").render())
            ):
                break
            await pilot.pause(0.02)
        pane = screen.query_one(SpeechSettingsPane)
        pane.query_one("#studio-tts-model-mode", Select).value = "exact"
        await pilot.pause()
        model = pane.query_one("#studio-tts-model-id", Input)
        model.value = "keep-this-draft"
        model.focus()
        await pilot.pause()
        assert screen.stts_window is not None
        assert screen.stts_window.current_view == "settings"
        assert pane.is_dirty

        cancel_choice = AsyncMock(return_value="cancel")
        pane._ask_leave_choice = cancel_choice
        playground_row.press()
        for _ in range(40):
            if cancel_choice.await_count == 1:
                break
            await pilot.pause(0.02)
        assert screen.stts_window is not None
        assert screen.stts_window.current_view == "settings"
        assert model.value == "keep-this-draft"
        assert app.focused is model

        discard_choice = AsyncMock(return_value="discard")
        pane._ask_leave_choice = discard_choice
        playground_row.press()
        for _ in range(40):
            if screen.stts_window.current_view == "playground":
                playgrounds = list(screen.query(SpeechPlaygroundPane))
                if playgrounds and list(
                    playgrounds[0].query("#tts-provider-select SelectOverlay")
                ):
                    break
            await pilot.pause(0.02)
        assert screen.stts_window.current_view == "playground"
        assert screen.query_one(SpeechPlaygroundPane).query_one(
            "#tts-provider-select SelectOverlay"
        )
