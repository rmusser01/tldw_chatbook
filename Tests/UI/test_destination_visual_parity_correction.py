"""Visual parity geometry tests for destination correction pass."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button
from textual.widgets import Static

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.Widgets.destination_workbench import DestinationWorkbench, WorkbenchPane


def _region(widget):
    region = widget.region
    return region.x, region.y, region.width, region.height


def _assert_no_horizontal_overlap(left, right, *, context: str) -> None:
    lx, ly, lw, lh = _region(left)
    rx, ry, rw, rh = _region(right)
    if ly + lh <= ry or ry + rh <= ly:
        return
    assert lx + lw <= rx or rx + rw <= lx, context


@pytest.mark.asyncio
async def test_main_navigation_overflow_hint_does_not_overlap_settings_at_default_size():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        nav = home.query_one(MainNavigationBar)
        settings = nav.query_one("#nav-settings", Button)
        more = nav.query_one("#nav-overflow-hint")
        _assert_no_horizontal_overlap(settings, more, context="More hint overlaps Settings nav item")


@pytest.mark.asyncio
async def test_destination_content_starts_immediately_below_nav():
    app = _build_test_app()
    host = HomeHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-dashboard")
        content = home.query_one("#screen-content")
        dashboard = home.query_one("#home-dashboard")
        assert content.region.y == 3
        assert dashboard.region.y <= 4


class WorkbenchHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield DestinationWorkbench(
            WorkbenchPane("List", Static("left"), id="test-list-pane"),
            WorkbenchPane("Detail", Static("center"), id="test-detail-pane"),
            WorkbenchPane("Inspector", Static("right"), id="test-inspector-pane"),
            id="test-workbench",
        )


@pytest.mark.asyncio
async def test_destination_workbench_renders_three_horizontal_panes():
    app = WorkbenchHarness()
    async with app.run_test(size=(100, 20)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#test-workbench")
        left = app.query_one("#test-list-pane")
        center = app.query_one("#test-detail-pane")
        right = app.query_one("#test-inspector-pane")
        assert left.region.x < center.region.x < right.region.x
        assert left.region.y == center.region.y == right.region.y
