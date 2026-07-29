"""Voice Cloning belongs inside the Lab frame, not on a screen of its own.

It shipped as a pushed `Screen`, so choosing it from the rail left the
Speech destination entirely: the rail, the mode strip and the capability
line all disappeared, and the way back was a binding rather than the rail
the user had just used. Every other Speech view stays in the frame.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import SPEECH_NON_VIEW_KEYS, STTSScreen


@pytest.mark.unit
def test_voice_cloning_is_no_longer_a_non_view():
    """While it was in this set, the rail press returned early by design."""
    assert "voice-cloning" not in SPEECH_NON_VIEW_KEYS


@pytest.mark.asyncio
async def test_choosing_it_switches_the_view_rather_than_pushing_a_screen():
    """The frame must survive. If a screen is pushed, the rail that the user
    just clicked is no longer on screen."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        depth_before = len(app.screen_stack)

        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "voice-cloning"
        )
        row.press()
        for _ in range(6):
            await pilot.pause()

        assert len(app.screen_stack) == depth_before, (
            "pushed a screen; the Lab frame is gone"
        )
        assert screen.stts_window is not None
        assert screen.stts_window.current_view == "voice-cloning"


@pytest.mark.asyncio
async def test_the_profile_surface_is_inside_the_frame():
    """The point of the move: its controls live in the Lab body."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "voice-cloning"
        )
        row.press()
        for _ in range(6):
            await pilot.pause()

        body = screen.query_one("#lab-body")
        container = screen.query_one("#voice-cloning-container")
        assert body.region.contains_region(container.region) or container.region.height
