"""Audio Effects: a placeholder that says something, not a dead row.

The rail entry shipped `disabled=True`, so it could not be opened and
explained nothing -- the user sees a greyed line called "Audio Effects" and
has to guess whether it is broken, unavailable to them, or unbuilt.

The spec asks for one line stating what it will be and that it is not built,
naming the studio view it belongs to. That is strictly more informative than
a control that cannot be pressed.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import SPEECH_NON_VIEW_KEYS, STTSScreen


@pytest.mark.asyncio
async def test_the_rail_entry_is_reachable():
    """A disabled row cannot explain itself. This one must open."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "effects"
        )
        assert not row.disabled, "Audio Effects cannot be opened"


@pytest.mark.asyncio
async def test_opening_it_states_that_it_is_not_built_and_where_it_is_going():
    """Both halves matter. "Coming soon" alone does not say what it is, and
    naming the feature without saying it is unbuilt implies it works."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = STTSScreen(app)
        await app.push_screen(screen)
        await pilot.pause()
        row = next(
            b for b in screen.query(Button)
            if getattr(b, "lab_view_key", None) == "effects"
        )
        row.press()
        for _ in range(6):
            await pilot.pause()

        text = " ".join(
            str(w.renderable) for w in screen.query(Static).results()
        ).lower()

    assert "not built" in text or "not yet" in text, "does not say it is unbuilt"
    assert "studio" in text, "does not name the studio view it belongs to"


@pytest.mark.unit
def test_effects_is_no_longer_a_non_view():
    """It has a view now, so leaving it in the non-view set would route the
    press into the branch that deliberately does nothing."""
    assert "effects" not in SPEECH_NON_VIEW_KEYS
