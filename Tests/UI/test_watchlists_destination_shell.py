"""Tests for the new watchlists screen shell structure."""

import pytest
from textual.widgets import Button, Select

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen


@pytest.mark.asyncio
async def test_watchlists_shell_has_navigator_and_panes():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#watchlists-navigator")
        assert screen.query_one("#watchlists-list-pane")
        assert screen.query_one("#watchlists-detail-pane")
        assert screen.query_one("#watchlists-inspector-pane")
        assert screen.query_one("#watchlists-backend-select", Select)


@pytest.mark.asyncio
async def test_watchlists_navigator_updates_active_section():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "overview"
        screen.query_one("#nav-sources", Button).press()
        await pilot.pause()
        assert screen.active_section == "sources"
