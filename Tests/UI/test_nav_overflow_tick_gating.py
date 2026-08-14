"""Change-gated nav overflow tick (task-15473).

`MainNavigationBar._update_overflow_hints` is the callback for the nav
bar's periodic 0.5s `set_interval` (and the first post-mount pass, and
`on_resize`'s deferred call). Before this task it ran its full pipeline --
hint toggle, `_refresh_overflow_hint_visibility`'s reclaimable-space math,
and scheduling `_recenter_strip` (which itself chains into
`_ghost_clipped_buttons`) -- unconditionally on every one of those
triggers, forever, on every screen, even when nothing about the strip had
moved since the previous pass.

This file pins two things:
  1. A no-op tick (nothing scrolled, resized, or changed) does none of that
     work -- the counting seam this task's fix is required to satisfy
     (born red against the pre-fix implementation; see the test's
     docstring for the mutation-test result).
  2. The gate does not falsely skip when the strip's CONTENT changes (a
     destination added to the strip) even though nothing here scrolled or
     resized -- the scenario the task explicitly calls out as needing its
     own coverage, distinct from the resize/scroll cases the existing
     `test_master_shell_navigation.py`/`test_chrome_ux_fixes.py` suites
     already pin.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import Horizontal

from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigationButton,
)


class _NavHarness(ConsolidatedCSSApp):
    def __init__(self, active: str = "logs"):
        super().__init__()
        self.active = active

    def compose(self) -> ComposeResult:
        yield MainNavigationBar(active=self.active)


@pytest.mark.asyncio
async def test_a_settled_no_op_tick_does_no_measurement_or_toggle_work() -> None:
    """Once the strip has settled, a later tick with nothing changed must
    not call `_refresh_overflow_hint_visibility` (and, by extension, must
    not toggle a hint or schedule a recenter/ghost pass -- that method is
    the one thing every branch of the gated pipeline funnels through).

    Mutation-tested: reverting `_update_overflow_hints` to call the
    pipeline unconditionally (removing the signature gate) turns this red
    -- the interval fires at least once more inside the second 0.6s
    window and the counter goes non-zero.
    """
    # 80 cols with "logs" active reliably overflows (see
    # test_chrome_ux_fixes.py's identical premise) -- the settled state
    # this test cares about is the STEADY state after that overflow is
    # already resolved, not the transient first pass.
    app = _NavHarness(active="logs")
    async with app.run_test(size=(80, 24)) as pilot:
        # Let the mount-time settle passes (call_after_refresh, the 0.05s
        # and 0.25s one-shot hint timers, and several interval ticks) run
        # and stabilize the signature. Directly measured: at 80 cols the
        # strip's own region width still moves for 2-3 ticks after mount
        # (showing the overflow hint reclaims/costs width, which itself
        # only settles a pass later) -- a shorter wait here would install
        # the spy while the strip was still genuinely converging and see
        # a real, expected change, not a regression.
        for _ in range(6):
            await pilot.pause(0.3)

        nav = app.query_one(MainNavigationBar)
        strip = app.query_one("#nav-destination-strip", Horizontal)
        assert strip.max_scroll_x >= 0  # test premise: strip is measurable
        assert nav._overflow_signature is not None, (
            "test premise: the strip has produced at least one settled "
            "signature by now"
        )

        spy = MagicMock(wraps=nav._refresh_overflow_hint_visibility)
        nav._refresh_overflow_hint_visibility = spy

        # At least one more full interval period, with nothing scrolled,
        # resized, or mounted in the meantime.
        await pilot.pause(0.6)

        assert spy.call_count == 0, (
            f"a no-op tick should skip all downstream work, but "
            f"_refresh_overflow_hint_visibility was called {spy.call_count} "
            "time(s)"
        )


@pytest.mark.asyncio
async def test_a_tick_after_a_destination_is_added_still_does_the_work() -> None:
    """The gate must not falsely skip when the strip's CONTENT changes --
    a new destination button mounted into the strip -- even though nothing
    scrolled or resized. Proves the signature is keyed on the button set,
    not only on scroll/width.
    """
    app = _NavHarness(active="home")
    async with app.run_test(size=(200, 24)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        strip = app.query_one("#nav-destination-strip", Horizontal)

        spy = MagicMock(wraps=nav._refresh_overflow_hint_visibility)
        nav._refresh_overflow_hint_visibility = spy

        # Simulate a new destination joining the strip at runtime.
        extra = NavigationButton(
            "Extra",
            id="nav-overflow-tick-gating-test-extra",
            classes="nav-button",
            target_route="chat",
        )
        await strip.mount(extra)

        await pilot.pause(0.6)

        assert spy.call_count >= 1, (
            "a tick after the button set changed should still run the "
            "overflow-hint pipeline, not skip it as a no-op"
        )
