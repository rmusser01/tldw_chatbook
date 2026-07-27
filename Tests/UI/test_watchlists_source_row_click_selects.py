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

SOURCE = {
    "id": "local:subscription:1",
    "source_id": 1,
    "name": "Summit Route",
    "source_type": "rss",
    "active": True,
}


async def _sources_pane(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    pane.sources = [SOURCE]
    await pilot.pause(0.2)
    return screen, pane


@pytest.mark.asyncio
async def test_clicking_a_source_row_selects_it_and_arms_check_now():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        # Note a deliberate consequence: populating the table now highlights
        # row 0, so the first source is selected by default and the actions are
        # armed without a click. That is the same thing every list in this app
        # does, and it is strictly better than the previous state where nothing
        # could be selected by mouse at all.
        await pilot.click("#sources-table", offset=(4, 1))
        await pilot.pause(0.3)

        assert pane.selected_source is not None, (
            "clicking a source row must select it; without this Preview and "
            "Check now can never be armed by mouse"
        )
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

        await pilot.click("#sources-table", offset=(4, 1))
        await pilot.pause(0.3)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if calls:
                break

        assert calls, "Check now never reached the controller after a row click"
