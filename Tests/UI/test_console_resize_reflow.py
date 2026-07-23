"""TASK-361: live-resize reflow convergence + stale-overlay dismissal.

The review saw a live browser-viewport resize (900x620 -> 700x480) leave the
rail full-width with the transcript/inspector gone and a nav tooltip stuck over
the header, whereas a cold start at the same size was fine. On a native resize
the pane reflow converges to the cold-start layout (locked here); the resize now
also dismisses any hover tooltip so a mounted overlay can't survive the repaint.
"""

import pytest

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app

_PANES = (
    "#console-left-rail",
    "#console-transcript-surface",
    "#console-native-composer",
)


def _pane_layout(console) -> dict:
    layout: dict = {}
    for selector in _PANES:
        try:
            layout[selector] = bool(console.query_one(selector).display)
        except Exception:  # noqa: BLE001
            layout[selector] = None
    layout["compact"] = console.query_one("#console-shell").has_class(
        "-console-compact"
    )
    return layout


@pytest.mark.asyncio
async def test_console_live_resize_converges_to_cold_start_layout():
    """TASK-361 AC#1: reflow after a live resize converges to the cold-start
    layout at that size (panes present, header compacted), not the review's
    rail-full-width / panes-gone divergence."""
    cold_host = ConsoleHarness(_build_test_app())
    async with cold_host.run_test(size=(90, 30)) as pilot:
        cold_console = cold_host.screen_stack[-1]
        await pilot.pause()
        await pilot.pause()
        cold = _pane_layout(cold_console)

    live_host = ConsoleHarness(_build_test_app())
    async with live_host.run_test(size=(160, 48)) as pilot:
        live_console = live_host.screen_stack[-1]
        await pilot.pause()
        await pilot.resize_terminal(90, 30)
        await pilot.pause()
        await pilot.pause()
        live = _pane_layout(live_console)

    assert live == cold
    # The panes are actually present (not the "panes gone" state) and the header
    # is compacted at 30 rows.
    assert cold["#console-transcript-surface"] is True
    assert cold["#console-native-composer"] is True
    assert cold["compact"] is True


@pytest.mark.asyncio
async def test_console_resize_dismisses_stale_tooltip():
    """TASK-361 AC#2: a live resize dismisses a hover tooltip so a mounted
    overlay cannot stick over the header across reflows."""
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await pilot.pause()

        cleared: list[bool] = []
        console._clear_tooltip = lambda: cleared.append(True)

        await pilot.resize_terminal(120, 40)
        await pilot.pause()

        assert cleared
