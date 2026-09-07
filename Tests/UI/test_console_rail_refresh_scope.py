"""Full-screen relayout scope for Console rail interactions (TASK-25888).

Every rail interaction funnels through ``_sync_console_rail_visibility``,
which used to end with ``self.refresh(layout=True)`` on the whole ChatScreen.
For a section toggle -- where no pane-level geometry moves -- that cost ~61ms
of ``_refresh_layout`` plus ~39ms of ``render_full_update`` on the main
thread and 54-62KB of terminal output per click (bare Textual baseline: 0.8
full updates, ~9KB), which is what made every Console button press feel
sticky (2026-08-31 latency investigation).

These tests pin the compositor-visible contract, not the implementation:

* a section toggle must produce ZERO full-screen compositor updates;
* collapsing a rail -- where the main column genuinely resizes -- must still
  relayout the screen.

The instrument counts ``Compositor.render_full_update`` calls because that is
the exact branch whose output a terminal must absorb; asserting on internal
flags would pass even while the screen still repainted wholesale.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from textual._compositor import Compositor

from Tests.UI.test_console_left_rail import make_console_pilot


@contextmanager
def count_compositor_updates(counts: dict[str, int]) -> Iterator[None]:
    """Count full and partial compositor updates while the context is live."""
    original_full = Compositor.render_full_update
    original_partial = Compositor.render_partial_update

    def counting_full(self, *args, **kwargs):
        counts["full"] += 1
        return original_full(self, *args, **kwargs)

    def counting_partial(self, *args, **kwargs):
        counts["partial"] += 1
        return original_partial(self, *args, **kwargs)

    Compositor.render_full_update = counting_full
    Compositor.render_partial_update = counting_partial
    try:
        yield
    finally:
        Compositor.render_full_update = original_full
        Compositor.render_partial_update = original_partial


@pytest.mark.asyncio
async def test_section_toggle_never_full_screen_updates() -> None:
    """Opening/closing a rail section must not redraw the whole screen.

    The first click on the toggle is excluded from the count: it MOVES FOCUS,
    and Textual's ``Screen.focused`` reactive full-repaints the screen on any
    focus change (a framework behaviour this stylesheet depends on -- seven
    ``:focus-within`` rules go stale without it). That cost is Textual's and
    is tracked separately; this test pins the app's contribution, which is
    the rail-visibility sync.
    """
    async with make_console_pilot(size=(160, 45), production_styles=True) as pilot:
        await pilot.pause(0.2)
        # Warm the focus so later clicks are pure section toggles.
        assert await pilot.click("#console-rail-section-toggle-model")
        await pilot.pause(0.1)
        counts = {"full": 0, "partial": 0}
        with count_compositor_updates(counts):
            for _ in range(4):
                assert await pilot.click("#console-rail-section-toggle-model")
                await pilot.pause(0.1)
        assert counts["full"] == 0, (
            f"{counts['full']} full-screen compositor updates for 4 section "
            f"toggles (partial={counts['partial']}); a section toggle moves "
            "geometry only inside the rail and must repaint only the rail"
        )
        # The toggle must still actually work: partial updates repainted it.
        assert counts["partial"] > 0
        body = pilot.app.screen.query_one("#console-rail-section-body-model")
        assert body.display is True, (
            "5 toggles from closed must end open -- the scoped refresh path "
            "dropped the toggle itself"
        )


@pytest.mark.asyncio
async def test_rail_collapse_still_relayouts_the_screen() -> None:
    """Hiding a rail resizes the main column -- the screen must relayout."""
    async with make_console_pilot(size=(160, 45), production_styles=True) as pilot:
        await pilot.pause(0.2)
        screen = pilot.app.screen
        rail = screen.query_one("#console-left-rail")
        assert rail.display is True, "test needs the rail open to collapse it"
        counts = {"full": 0, "partial": 0}
        with count_compositor_updates(counts):
            assert await pilot.click("#console-context-rail-collapse")
            await pilot.pause(0.2)
        assert rail.display is False, "collapse control did not hide the rail"
        handle = screen.query_one("#console-context-rail-handle")
        assert handle.display is True, "hidden rail must expose its handle"
        assert counts["full"] >= 1, (
            "collapsing the rail changed pane geometry but produced no "
            "full-screen update; the scoped-refresh path over-reached"
        )
