"""TASK-21244: the left rail's fold copy repaints without a layout pass.

The left rail's outer fold hint was the last Console-rail instance of the
``Static.update()`` trap TASK-21117 documented in
``backlog/docs/lessons-textual.md``: ``update(text)`` defaults to
``layout=True``, so a copy-only repaint of the pinned one-row hint slot
scheduled a whole-screen layout pass on every rail transition that changed
the fold copy. The fix mirrors the right rail's ``_paint_outer_hint``:
read the live widget's content, write with ``layout=False`` (the slot's
height is pinned by INLINE styles, which beat every stylesheet rule), and
no shadow copy on the rail.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_console_rail_reconciliation import (
    _RailHarness,
    _settle,
)
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail


def _arm_screen_layout_counter(monkeypatch, screen) -> dict[str, int]:
    """Count ``Screen._refresh_layout`` invocations."""
    counts = {"screen_layout": 0}
    original = screen._refresh_layout

    def counting_layout(*args, **kwargs):
        counts["screen_layout"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(screen, "_refresh_layout", counting_layout)
    return counts


@pytest.mark.asyncio
async def test_copy_only_outer_hint_update_schedules_no_screen_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy-only hint write must not schedule a screen layout pass.

    Calls the painter directly over the mounted rail so nothing but the
    copy write runs: at the pre-fix behaviour (``update(text)`` with its
    ``layout=True`` default) each write scheduled one ``Screen.
    _refresh_layout``; ``layout=False`` must schedule none, and the slot's
    pinned geometry must hold across the copy change.
    """
    app = _RailHarness()
    async with app.run_test(size=(60, 18)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        outer = rail.query_one("#console-left-rail-body")
        assert outer.max_scroll_y > 0, "precondition: the outer body scrolls"
        hint = rail.query_one("#console-left-rail-outer-hint", Static)
        assert hint.display is True

        # Land a real copy first so the geometry pin reflects the shown slot.
        outer.scroll_home(animate=False)
        await _settle(pilot, passes=3)
        rail._update_outer_hint()
        await _settle(pilot, passes=3)
        assert str(hint.content), "precondition: the fold copy is showing"

        hint_region = hint.region
        counts = _arm_screen_layout_counter(monkeypatch, app.screen)

        ConsoleLeftRail._paint_outer_hint(hint, "▼ probe copy")
        await _settle(pilot, passes=4)
        ConsoleLeftRail._paint_outer_hint(hint, "▼ probe copy two")
        await _settle(pilot, passes=4)
        ConsoleLeftRail._paint_outer_hint(hint, "▼ probe copy")
        await _settle(pilot, passes=4)

        assert counts["screen_layout"] == 0, (
            "a copy-only left-rail hint update scheduled a screen layout "
            "pass -- the Static.update layout=True trap is back"
        )
        assert str(hint.content) == "▼ probe copy", "the copy did not land"
        assert hint.region == hint_region, (
            "the hint slot's pinned one-row geometry moved with only the "
            "copy changing -- layout=False is unsound here"
        )


@pytest.mark.asyncio
async def test_scroll_end_blanks_the_copy_without_a_layout_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The integration shape: blanking the fold copy adds no layout pass.

    Scrolling schedules Textual's own scroll-timer layout pass
    (``Screen._on_timer_update -> _refresh_layout(scroll=True)``) whether
    or not the copy changes, so the honest pin is a within-test A/B: one
    scroll that leaves the copy UNCHANGED (one row down from home) buys
    the baseline; the copy-blanking scroll to the end must cost the SAME
    count. On the pre-fix painter the blanking write added exactly one
    ``layout=True`` pass on top of that baseline.
    """
    app = _RailHarness()
    async with app.run_test(size=(60, 18)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        outer = rail.query_one("#console-left-rail-body")
        hint = rail.query_one("#console-left-rail-outer-hint", Static)
        outer.scroll_home(animate=False)
        await _settle(pilot, passes=3)
        rail._update_outer_hint()
        await _settle(pilot, passes=3)
        assert str(hint.content), "precondition: the fold copy is showing"
        copy_at_top = str(hint.content)

        # Baseline: a small scroll that cannot change the copy.
        baseline_counts = _arm_screen_layout_counter(monkeypatch, app.screen)
        outer.scroll_to(y=1, animate=False, immediate=True)
        await _settle(pilot, passes=6)
        assert str(hint.content) == copy_at_top, (
            "precondition: the baseline scroll must leave the copy unchanged"
        )
        baseline = baseline_counts["screen_layout"]

        # Gesture under test: the copy blanks at the end.
        outer.scroll_home(animate=False)
        await _settle(pilot, passes=3)
        counts = _arm_screen_layout_counter(monkeypatch, app.screen)
        outer.scroll_to(y=outer.max_scroll_y, animate=False, immediate=True)
        await _settle(pilot, passes=6)

        assert str(hint.content) == "", "the copy did not blank at the end"
        assert counts["screen_layout"] == baseline, (
            "blanking the fold copy at scroll end scheduled a screen layout "
            f"pass beyond the scroll baseline ({counts['screen_layout']} vs "
            f"{baseline})"
        )


@pytest.mark.asyncio
async def test_paint_outer_hint_skips_unchanged_copy() -> None:
    """The unchanged check reads the live widget, with no shadow on the rail.

    Also pins the shadow-field removal: the rail class must not carry
    ``_outer_hint_text`` (TASK-21117's AC3 shape) after ``__init__``.
    """
    app = _RailHarness()
    async with app.run_test(size=(60, 18)) as pilot:
        await _settle(pilot, passes=8)
        rail = app.query_one(ConsoleLeftRail)
        hint = rail.query_one("#console-left-rail-outer-hint", Static)
        ConsoleLeftRail._paint_outer_hint(hint, "▼ probe copy")
        updates: list[object] = []
        original = hint.update

        def recording_update(content, *args, **kwargs):
            updates.append(content)
            return original(content, *args, **kwargs)

        hint.update = recording_update  # type: ignore[method-assign]
        ConsoleLeftRail._paint_outer_hint(hint, "▼ probe copy")
        assert updates == [], "an unchanged copy was repainted"

        assert not hasattr(rail, "_outer_hint_text"), (
            "the left rail still carries the _outer_hint_text shadow the "
            "right rail's TASK-21117 fix removed"
        )
