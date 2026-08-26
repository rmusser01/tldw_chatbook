"""The rail's allocation reconcile must only disturb hover when rows moved.

TASK-22221 (holistic perf review of dev ``a71e62e4b``, finding 22221).
``ConsoleLeftRail._run_allocation_reconcile`` ends every pass in
``_refresh_workspace_tree_after_reflow``, which cleared the workspace tree's
hover row and recomputed its tooltip UNCONDITIONALLY. The reconcile runs at
roughly 5 Hz while a run streams, so a user resting the pointer on a tree row
lost the hover highlight and its tooltip several times a second -- two repaints
and a tooltip write per tick, for a tree that never moved.

Clearing hover IS correct when the reflow actually shifted the rows under the
pointer, which is why the leg exists. These gates hold both halves: an
unchanged tree keeps its hover, a moved tree loses it.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_workspace_tree_cursor_layout import (
    APP_SIZE,
    _console_with_probe_tree,
    _settle,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree

#: Reconcile passes per probe -- more than a second of streaming at ~5 Hz.
PASSES = 8


def _count_tooltip_recomputes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every tooltip recompute the tree is asked to perform."""

    calls: list[str] = []
    original = ConsoleWorkspaceTree._update_tooltip

    def counting(tree):
        calls.append("update_tooltip")
        return original(tree)

    monkeypatch.setattr(ConsoleWorkspaceTree, "_update_tooltip", counting)
    return calls


@pytest.mark.asyncio
async def test_reconcile_passes_that_move_nothing_keep_the_hover_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 5 Hz tick must not steal a stationary pointer's hover."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        hovered = tree.workspace_nodes["ws-alpha"]
        tree.hover_line = int(hovered.line)
        await _settle(pilot, passes=4)
        assert tree.hover_line == int(hovered.line)

        cleared = 0
        tooltips = _count_tooltip_recomputes(monkeypatch)
        for _ in range(PASSES):
            rail.request_allocation_reconcile()
            await _settle(pilot, passes=3)
            if tree.hover_line < 0:
                cleared += 1
                tree.hover_line = int(hovered.line)
                await _settle(pilot, passes=2)

        print(
            f"\n[t22221 hover] {PASSES} no-op reconcile passes: "
            f"hover_cleared={cleared}, tooltip_recomputes={len(tooltips)}"
        )
        assert cleared == 0, f"hover cleared on {cleared}/{PASSES} passes that moved nothing"
        # A tree that did not move costs the leg nothing at all.
        assert tooltips == [], tooltips


@pytest.mark.asyncio
async def test_a_reflow_that_moves_the_tree_still_clears_hover() -> None:
    """The leg must keep doing its job when the rows really do shift."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        hovered = tree.workspace_nodes["ws-alpha"]
        tree.hover_line = int(hovered.line)
        await _settle(pilot, passes=4)
        assert tree.hover_line == int(hovered.line)
        before = tree.region

        # A real rail geometry change: the tree's own region moves with it.
        rail.styles.width = 60
        rail.request_allocation_reconcile()
        await _settle(pilot, passes=8)

        assert tree.region != before, "the probe must actually move the tree"
        assert tree.hover_line == -1


@pytest.mark.asyncio
async def test_hover_survives_a_reconcile_burst_then_clears_on_a_real_move() -> None:
    """Both halves in one session: quiet ticks, then a genuine reflow."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        hovered = tree.workspace_nodes["ws-alpha"]
        tree.hover_line = int(hovered.line)
        await _settle(pilot, passes=4)

        for _ in range(PASSES):
            rail.request_allocation_reconcile()
            await _settle(pilot, passes=2)
        assert tree.hover_line == int(hovered.line)

        rail.styles.width = 60
        rail.request_allocation_reconcile()
        await _settle(pilot, passes=8)
        assert tree.hover_line == -1


@pytest.mark.asyncio
async def test_reflow_leg_is_safe_when_the_tree_is_gone() -> None:
    """The deferred check must tolerate a tree removed between pass and settle."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        rail.request_allocation_reconcile()
        await tree.remove()
        await _settle(pilot, passes=6)

        # Directly, too: the leg is reachable from the reveal path as well.
        rail._refresh_workspace_tree_after_reflow()
        await _settle(pilot, passes=3)
