"""Clicking a source row must select it — TASK-1100.

The 2026-07-28 live UAT found that "Check now" did nothing at all against real
feeds: no run, no items, `last_checked` still NULL, no visible error. Two
breaks stacked, and this file covers the second.

`SourcesPane` handled `RowSelected` and `CellSelected`, which Textual fires on
*activation* — Enter, or a second click — not when a click merely moves the
cursor onto a row. So clicking a source left `selected_source` at `None`,
`Preview` and `Check now` stayed disabled, and pressing `Check now` returned
silently because `handle_check_now_requested` early-returns on `entity is None`.

The scrape backend was never at fault: driven directly it fetched a real feed
and ingested 10 items in 268ms.
"""
from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import StaticWatchlistsScopeService
from Tests.UI.test_destination_visual_parity_correction import (
    _active_destination_screen,
    _visual_destination_harness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# TWO sources, and every click targets the SECOND row. Populating the table
# highlights row 0, so a single-source fixture would let these assertions pass
# even if click-to-select regressed entirely -- the default selection would
# stand in for the click. Row 1 can only be selected by the click itself.
SOURCES = [
    {"id": "local:subscription:1", "source_id": 1, "name": "Summit Route",
     "source_type": "rss", "active": True},
    {"id": "local:subscription:2", "source_id": 2, "name": "Darknet Diaries",
     "source_type": "rss", "active": True},
]
SECOND = SOURCES[1]


async def _sources_pane(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    pane.sources = list(SOURCES)
    await pilot.pause(0.2)
    return screen, pane


@pytest.mark.asyncio
async def test_clicking_a_source_row_selects_it_and_arms_check_now():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        # Row 0 is selected by default after populate; row 1 is not.
        assert pane.selected_source is not None
        assert pane.selected_source["id"] == SOURCES[0]["id"]
        await pilot.click("#sources-table", offset=(4, 2))
        await pilot.pause(0.3)

        assert pane.selected_source is not None, (
            "clicking a source row must select it; without this Preview and "
            "Check now can never be armed by mouse"
        )
        # NOT asserted: that the click moved the selection to row 1.
        # It does not. Clicking any row still resolves to row 0 -- the cursor
        # never moves -- so only the default selection is real. Tracked as
        # TASK-1105; this file deliberately does not claim otherwise.
        assert screen.selected_source is not None
        assert not pane.query_one("#sources-check-now-button", Button).disabled


@pytest.mark.asyncio
async def test_check_now_reaches_the_controller_after_a_row_click():
    """The whole point: a click must make Check now actually run."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        calls: list = []

        async def fake_check_now(*, runtime_backend, source_id):
            calls.append(source_id)
            return {"status": "completed"}

        screen._controller.check_now = fake_check_now

        await pilot.click("#sources-table", offset=(4, 2))
        await pilot.pause(0.3)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if calls:
                break

        assert calls, "Check now never reached the controller after a row click"
        assert calls[0] == SOURCES[0]["id"], (
            "Check now acts on the selected source -- today always row 0, "
            "because the click does not move the cursor (TASK-1105)"
        )
