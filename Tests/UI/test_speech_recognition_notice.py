"""The privacy notice is a line, not a coloured block.

Measured on the running Speech Recognition view: the notice rendered as a
padded, bordered, full-background box FOUR rows tall -- and there are two of
them, so 8 of a 26-row viewport went on notices.

What this does NOT fix is the fold: 30 of 34 controls were below it before
and 29 after, because the notices sit in the sidebar rather than the main
column. That is the layout as a whole, and belongs to the Speech Recognition
rebuild rather than here.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


async def _open_recognition(app, pilot):
    screen = STTSScreen(app)
    await app.push_screen(screen)
    await pilot.pause()
    row = next(
        b for b in screen.query(Button)
        if getattr(b, "lab_view_key", None) == "dictation"
    )
    row.press()
    for _ in range(8):
        await pilot.pause()
    return screen


@pytest.mark.asyncio
async def test_the_privacy_notice_costs_one_row():
    """Four rows each, twice over, is a third of the viewport spent on
    telling the user something that fits on one line."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open_recognition(app, pilot)
        notices = [
            w for w in screen.query(".privacy-notice").results() if w.region.height
        ]
        assert notices, "no privacy notice rendered"
        tall = [w.region.height for w in notices if w.region.height > 1]

    assert not tall, f"privacy notice still costs {tall} rows"


@pytest.mark.asyncio
async def test_the_notice_still_says_something():
    """Shrinking it must not empty it -- the privacy state is the point."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open_recognition(app, pilot)
        text = " ".join(
            str(w.render()) for w in screen.query(".privacy-notice").results()
        )

    assert text.strip(), "the notice renders nothing at all"
