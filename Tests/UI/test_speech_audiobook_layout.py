"""AudioBook: the primary action must be reachable.

Measured on the running view: `generate-audiobook-btn` sat at y=40 in a
26-row viewport, after eight collapsible groups, with 29 of 35 controls
below the fold. The same defect Settings had, and the same fix -- the
actions go in a strip above the scroll region, so they stay on screen
whatever is expanded below.

The spec asks for this view's collapsible structure to be KEPT; it is
already the closest to the Console grammar. Only the actions move.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen


async def _open_audiobook(app, pilot):
    screen = STTSScreen(app)
    await app.push_screen(screen)
    await pilot.pause()
    row = next(
        b for b in screen.query(Button)
        if getattr(b, "lab_view_key", None) == "audiobook"
    )
    row.press()
    for _ in range(8):
        await pilot.pause()
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 60), (80, 24)])
async def test_generate_is_reachable_without_scrolling(size):
    """It is the reason the view exists; it measured at y=40 of 26."""
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _open_audiobook(app, pilot)
        body = screen.query_one("#lab-body")
        generate = screen.query_one("#generate-audiobook-btn", Button)
        assert body.region.contains_region(generate.region), (
            f"Generate below the fold at {size}: y={generate.region.y}"
        )


@pytest.mark.asyncio
async def test_export_stays_beside_generate():
    """Export acts on what Generate produced; separating them puts the
    second half of one task several screens from the first."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open_audiobook(app, pilot)
        body = screen.query_one("#lab-body")
        export = screen.query_one("#audiobook-export-btn", Button)
        assert body.region.contains_region(export.region)


@pytest.mark.asyncio
async def test_the_collapsible_structure_survives():
    """The spec keeps this view's grouping. Only the actions move."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open_audiobook(app, pilot)
        from textual.widgets import Collapsible

        assert len(list(screen.query(Collapsible))) >= 5
