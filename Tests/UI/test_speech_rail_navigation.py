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

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import (
    SPEECH_NON_VIEW_KEYS,
    STTSScreen,
)

SWITCHABLE = ("playground", "settings", "audiobook", "dictation")


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
            f"pressing {view_key} left the view on "
            f"{screen.stts_window.current_view}"
        )


@pytest.mark.asyncio
async def test_the_playground_view_mounts_a_playground():
    """The playground view must mount the rebuilt `SpeechPlaygroundPane`.

    dev shipped a profile library into the legacy `TTSPlaygroundWidget`
    while this rebuild was in flight; the pane took the view over once its
    axis row and dev's profile presets were reconciled -- the ownership
    ruling at
    `Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md`
    (`SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of
    record, every axis writer keeps the row's markers in step, and
    defaults are seeded from persisted preferences at construction). Retiring
    the legacy widget's own code is a separate task and not part of this
    ruling.
    """
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#speech-playground-pane")
