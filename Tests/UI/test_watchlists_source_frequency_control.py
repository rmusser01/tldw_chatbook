"""TASK-1210: the check-cadence control, driven the way a user drives it.

The cadence a source is checked on is the whole point of the scheduling fix --
``WatchlistProjection`` computes ``next_run_at`` from ``check_frequency``, so a
source that cannot be given one is never queued. A control that renders but
cannot be operated would satisfy "the field exists" while leaving every source
stuck on the database default.

These run against the production stylesheet in the full shell, for the reason
``test_watchlists_source_create_form.py`` gives: this form's failures are
geometry ones, and a bare ``App`` with no CSS cannot reproduce them. The bare
harness reports the Select at ``width=1``; under the real stylesheet it is 16.
"""

from __future__ import annotations

import time

import pytest
from textual.widgets import Button, Select

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# The small end the Watchlists parity suite covers, and the size the UAT ran at.
SIZES = [(160, 42), (235, 52)]


def _watchlists_host():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    return _visual_destination_harness(app, "watchlists_collections")


async def _open_form(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await pilot.pause(0.2)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    screen.query_one("#sources-new-button", Button).press()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        controls = pane.query("#sources-create-frequency")
        if controls and controls.first().region.width > 1:
            break
        await pilot.pause(0.01)
    return screen, pane


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_frequency_control_is_on_screen_and_sized(size):
    """The control has to be inside the pane, not pushed past its edge.

    The `Active` switch on the row above shipped at x=198 on a 235-column
    terminal once, for exactly this reason.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        _screen, pane = await _open_form(pilot, host)
        select = pane.query_one("#sources-create-frequency", Select)

        assert select.region.width > 1, (
            f"the cadence Select collapsed to {select.region.width} columns"
        )
        assert select.region.right <= size[0], (
            f"the cadence Select ends at x={select.region.right} on a "
            f"{size[0]}-column terminal"
        )
        assert select.region.bottom <= size[1]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_frequency_options_are_reachable_when_expanded(size):
    """Every cadence has to be visible in the overlay, not clipped away.

    A four-option overlay that renders one row is indistinguishable from a
    working one in a screenshot, which is how this nearly shipped.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        _screen, pane = await _open_form(pilot, host)
        select = pane.query_one("#sources-create-frequency", Select)
        select.expanded = True
        await pilot.pause()
        await pilot.pause()

        overlay = select.query_one("SelectOverlay")
        assert overlay.visible, "the cadence overlay is not visible when expanded"
        assert overlay.option_count == len(SourcesPane._FREQUENCY_OPTIONS)
        # border (1 row each side) + one row per option
        assert overlay.region.height >= overlay.option_count, (
            f"the overlay is {overlay.region.height} rows tall for "
            f"{overlay.option_count} options -- options are clipped away"
        )
        assert overlay.region.bottom <= size[1], (
            f"the overlay runs to y={overlay.region.bottom} on a "
            f"{size[1]}-row terminal, so its lower options are off-screen"
        )
