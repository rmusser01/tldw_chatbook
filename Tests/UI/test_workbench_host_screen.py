"""Tests for `WorkbenchHostScreen` -- the pushed, fresh-instance widget
host (redesign PR-4, Task 1).

Generic push/pop mechanics only. The conflicts-badge integration (a real
`ConflictsTab` pushed through this host) is covered in
`test_schedules_workbench.py`, alongside the badge's own behavior.
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.scheduling.workbench_host_screen import (
    WorkbenchHostScreen,
)


class _Marker(Static):
    """A trivially paintable, identity-checkable hosted widget."""


class _BaseScreen(Screen):
    """Stand-in for the pane a host overlay is pushed on top of."""

    def compose(self) -> ComposeResult:
        yield Static("base pane", id="base-marker")


class _HostTestApp(ConsolidatedCSSApp):
    pass


@pytest.mark.asyncio
async def test_push_pop_round_trip_yields_a_fresh_instance_each_time():
    """Two pushes of the same factory must build two distinct widgets --
    the spec's no-reparenting rule (survey §2)."""
    app = _HostTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(_BaseScreen())
        await pilot.pause()

        built: list[_Marker] = []

        def factory() -> _Marker:
            widget = _Marker()
            built.append(widget)
            return widget

        await pilot.app.push_screen(WorkbenchHostScreen(factory, title="Overlay"))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        await pilot.app.push_screen(WorkbenchHostScreen(factory, title="Overlay"))
        await pilot.pause()

        assert len(built) == 2
        assert built[0] is not built[1]


@pytest.mark.asyncio
async def test_escape_pops_the_host_screen():
    app = _HostTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(_BaseScreen())
        await pilot.pause()

        host = WorkbenchHostScreen(lambda: _Marker(), title="Overlay")
        await pilot.app.push_screen(host)
        await pilot.pause()
        assert pilot.app.screen is host

        await pilot.press("escape")
        await pilot.pause()

        assert pilot.app.screen is not host
        assert isinstance(pilot.app.screen, _BaseScreen)


@pytest.mark.asyncio
async def test_pane_behind_survives_the_round_trip_untouched():
    """Pushing/popping an overlay must not recompose or replace the
    screen underneath -- same widget identity before and after."""
    app = _HostTestApp()
    async with app.run_test() as pilot:
        base = _BaseScreen()
        await pilot.app.push_screen(base)
        await pilot.pause()
        marker_before = base.query_one("#base-marker", Static)

        await pilot.app.push_screen(
            WorkbenchHostScreen(lambda: _Marker(), title="Overlay")
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        marker_after = base.query_one("#base-marker", Static)
        assert marker_after is marker_before


@pytest.mark.asyncio
async def test_dismissed_callback_fires_once_on_pop():
    app = _HostTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(_BaseScreen())
        await pilot.pause()

        calls: list[None] = []
        host = WorkbenchHostScreen(
            lambda: _Marker(),
            title="Overlay",
            dismissed=lambda: calls.append(None),
        )
        await pilot.app.push_screen(host)
        await pilot.pause()
        assert calls == []

        await pilot.press("escape")
        await pilot.pause()

        assert calls == [None]


@pytest.mark.asyncio
async def test_hosted_widget_paints_within_the_screen():
    """Real geometry, not internal state: the hosted widget must occupy
    actual screen space and show its content."""
    app = _HostTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(_BaseScreen())
        await pilot.pause()

        await pilot.app.push_screen(
            WorkbenchHostScreen(lambda: _Marker("hosted content"), title="Overlay")
        )
        await pilot.pause()

        marker = pilot.app.screen.query_one(_Marker)
        assert marker.region.width > 0
        assert marker.region.height > 0
        assert str(marker.renderable) == "hosted content"
