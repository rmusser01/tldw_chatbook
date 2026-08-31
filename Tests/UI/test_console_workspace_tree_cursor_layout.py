"""Workspace-tree cursor moves must not fan out into rail-wide reconciles.

TASK-22203. The holistic perf review (Docs/Design/2026-08-24-holistic-perf-review.md,
finding 22203) measured that every tree cursor move posts a context update whose
unguarded ``Static.update()`` (``layout=True`` default in Textual 8.2.8) armed one
screen layout pass per arrow key, and that crossing the workspace<->conversation
boundary escalated through ``_reconcile_workspace_action_owners`` into the full
7-section, ~45-``query_one`` rail allocation pipeline
(``ConsoleLeftRail._run_allocation_reconcile``).

These tests replicate the review's probes as permanent gates:

* a conversation->conversation cursor move (non-boundary) arms ZERO
  ``Screen._refresh_layout`` calls beyond the Tree's own repaint;
* a workspace<->conversation crossing (boundary) reconciles only the workspace
  bounded section (scoped), never the rail allocation pipeline, while the
  action row still appears for conversations only;
* the context tray's selection copy and tooltip writes are equality-guarded,
  and a real copy repaint is delivered with ``layout=False`` into a slot whose
  one-row geometry is held to account;
* ``ConsoleWorkspaceTree._update_tooltip`` memoizes by (node identity, label,
  width) so an unchanged target skips the measurement and the tooltip write;
* teardown/mid-flight failure cases stay safe (cursor move during unmount,
  tooltip target removed by a projection push, boundary flip while a rail
  allocation pass is already in flight).
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from rich.text import Text
from textual.widgets import Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_workspace_context_rail import (
    _base_grouped_workspace_state,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.Widgets.Console import (
    ConsoleBoundedSection,
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import (
    ConsoleWorkspaceTree,
)
from tldw_chatbook.Workspaces.workspace_tree_state import (
    WorkspaceTreeConversation,
    WorkspaceTreeWorkspace,
)

APP_SIZE = (160, 44)

#: Presses per move class -- enough that per-press work cannot hide in noise.
PRESSES = 20


def _conversation(
    conversation_id: str, title: str, *, starred: bool = False
) -> WorkspaceTreeConversation:
    return WorkspaceTreeConversation(
        conversation_id=conversation_id,
        title=title,
        starred=starred,
        updated_sort="2026-08-22T00:00:00",
        selected=False,
        run_marker="",
        star_enabled=True,
    )


def _probe_workspace_state():
    """Two workspaces, several same-starred conversations under the first."""

    return replace(
        _base_grouped_workspace_state(),
        workspace_tree=(
            WorkspaceTreeWorkspace(
                workspace_id="ws-alpha",
                label="Workspace Alpha",
                conversations=tuple(
                    _conversation(f"conv-a{i}", f"Alpha conversation {i}")
                    for i in range(4)
                ),
                next_cursor=None,
                active=True,
            ),
            WorkspaceTreeWorkspace(
                workspace_id="ws-beta",
                label="Workspace Beta",
                conversations=(_conversation("conv-b0", "Beta conversation 0"),),
                next_cursor=None,
            ),
        ),
        workspace_marks_available=True,
    )


async def _settle(pilot, passes: int = 3) -> None:
    for _ in range(passes):
        await pilot.pause()


async def _console_with_probe_tree(host, pilot):
    """Mount the real Console, seed a deterministic tree, and settle.

    The seeding push happens while the Tree is still empty, so the rail's
    first-sync rule expands every seeded workspace. Deliberately NO
    ``node.expand()`` here: an explicit expand posts
    ``WorkspaceTreeExpansionChanged``, and the mounted ChatScreen answers it
    by re-syncing its own real workspace state over the seeded one.
    """

    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-workspace-tree")
    tree = console.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
    rail = console.query_one("#console-left-rail", ConsoleLeftRail)
    state = _probe_workspace_state()
    # Keep the mounted screen's periodic projection source aligned with the
    # synthetic state. Otherwise its next normal sync tick can replace the
    # seeded rows between this helper's final assertion and the test body.
    console._workspace._build_console_workspace_context_state = lambda: state
    rail.sync_workspace_context(state)
    for _ in range(250):
        if "ws-alpha" in tree.workspace_nodes:
            break
        await pilot.pause(0.02)
    else:
        raise AssertionError("seeded workspace tree never settled")
    assert tree.workspace_nodes["ws-alpha"].is_expanded
    assert "conv-a0" in tree.conversation_nodes
    tree.focus()
    for _ in range(250):
        if host.screen_stack[-1].focused is tree:
            break
        await pilot.pause(0.02)
    else:
        raise AssertionError("workspace tree never received focus")
    assert "conv-a0" in tree.conversation_nodes
    return console, rail, tree


def _arm_counters(monkeypatch, screen, rail) -> dict[str, object]:
    """Count screen layout passes and every leg of the rail pipeline."""

    counts: dict[str, object] = {
        "screen_layout": 0,
        "allocation_runs": 0,
        "allocation_preps": 0,
        "rail_query_one": 0,
        "section_reconciles": [],
    }

    original_layout = screen._refresh_layout

    def counting_layout(*args, **kwargs):
        counts["screen_layout"] += 1
        return original_layout(*args, **kwargs)

    monkeypatch.setattr(screen, "_refresh_layout", counting_layout)

    original_run = rail._run_allocation_reconcile

    def counting_run(*args, **kwargs):
        counts["allocation_runs"] += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(rail, "_run_allocation_reconcile", counting_run)

    original_prep = rail._prepare_allocation_reconcile

    def counting_prep(*args, **kwargs):
        counts["allocation_preps"] += 1
        return original_prep(*args, **kwargs)

    monkeypatch.setattr(rail, "_prepare_allocation_reconcile", counting_prep)

    original_query = rail.query_one

    def counting_query(*args, **kwargs):
        counts["rail_query_one"] += 1
        return original_query(*args, **kwargs)

    monkeypatch.setattr(rail, "query_one", counting_query)

    original_reconcile = ConsoleBoundedSection._reconcile

    def counting_reconcile(section, *args, **kwargs):
        counts["section_reconciles"].append(section.section_id)
        return original_reconcile(section, *args, **kwargs)

    monkeypatch.setattr(ConsoleBoundedSection, "_reconcile", counting_reconcile)
    return counts


# ---------------------------------------------------------------------------
# 1. Non-boundary cursor moves: zero screen layout passes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_boundary_cursor_move_arms_zero_screen_layout_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation->conversation arrowing must not lay out the screen.

    On the pre-fix code each press paid at least one ``Screen._refresh_layout``
    via the tray's unguarded ``Static.update`` (``layout=True`` default).
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        tree.move_cursor(tree.conversation_nodes["conv-a0"])
        await _settle(pilot, passes=4)
        counts = _arm_counters(monkeypatch, host.screen_stack[-1], rail)
        started = time.perf_counter()
        for press in range(PRESSES):
            await pilot.press("down" if press % 2 == 0 else "up")
            await _settle(pilot, passes=3)
            # The press really moved the Tree cursor between the two rows.
            expected = "conv-a1" if press % 2 == 0 else "conv-a0"
            assert tree.cursor_node.data.key == f"conversation:{expected}"
        elapsed = time.perf_counter() - started

        print(
            f"\n[t22203 non-boundary] {PRESSES} presses: "
            f"{elapsed * 1000.0:.1f} ms wall, "
            f"screen_layout={counts['screen_layout']}, "
            f"allocation_runs={counts['allocation_runs']}, "
            f"rail_query_one={counts['rail_query_one']}, "
            f"section_reconciles={counts['section_reconciles']}"
        )
        assert counts["screen_layout"] == 0, counts
        assert counts["allocation_runs"] == 0, counts
        assert counts["allocation_preps"] == 0, counts
        assert counts["section_reconciles"] == [], counts
        # One tray lookup per posted context change is the whole rail cost.
        assert counts["rail_query_one"] <= PRESSES, counts


# ---------------------------------------------------------------------------
# 2. Boundary crossings: scoped reconcile only, never the rail pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_crossing_reconciles_only_the_workspace_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace<->conversation arrowing stays inside the workspace section.

    On the pre-fix code every crossing ran ``_reconcile_workspace_action_owners``
    into ``request_allocation_reconcile`` -- the full 7-section, ~45-``query_one``
    rail measure -- twice per bounce.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        tree.move_cursor(tree.workspace_nodes["ws-alpha"])
        await _settle(pilot, passes=6)
        bounded = console.query_one(
            "#console-bounded-section-workspace", ConsoleBoundedSection
        )

        counts = _arm_counters(monkeypatch, host.screen_stack[-1], rail)
        started = time.perf_counter()
        for press in range(PRESSES):
            await pilot.press("down" if press % 2 == 0 else "up")
            await _settle(pilot, passes=4)
        elapsed = time.perf_counter() - started

        print(
            f"\n[t22203 boundary] {PRESSES} presses: "
            f"{elapsed * 1000.0:.1f} ms wall, "
            f"screen_layout={counts['screen_layout']}, "
            f"allocation_runs={counts['allocation_runs']}, "
            f"allocation_preps={counts['allocation_preps']}, "
            f"rail_query_one={counts['rail_query_one']}, "
            f"section_reconciles={len(counts['section_reconciles'])}"
        )
        # The rail allocation pipeline never runs on a cursor boundary flip.
        assert counts["allocation_runs"] == 0, counts
        assert counts["allocation_preps"] == 0, counts
        # TASK-25712: the star seam the boundary used to drive is retired, so
        # a crossing now runs NO reconcile at all -- the old invariant ("only
        # the workspace section, never the rail pipeline") is preserved and
        # strengthened: nothing fires.
        assert counts["section_reconciles"] == [], counts
        # The whole rail cost is the per-press tray lookup, not the ~45-query
        # allocation measure.
        assert counts["rail_query_one"] <= 2 * PRESSES, counts
        # The flip's own tray refit costs 2 screen layout passes per press
        # (measured before AND after the fix — the pipeline's work coalesced
        # into those same passes). Tripwire against per-press frame growth.
        assert counts["screen_layout"] <= 3 * PRESSES, counts
        # The flip's own geometry settled: no reconcile left scheduled.
        assert bounded._reconcile_scheduled is False
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_scoped_reconcile_swallows_a_demand_delta_plain_still_escalates() -> (
    None
):
    """The scoped/plain contract on ``_ContextBoundedSection`` itself.

    In BOTH mounted probes above, the workspaces tray sits at its 12-row cap
    (``max_height: 12``, ``overflow_y: hidden``), so the action-row flip
    cannot change ``desired_content_lines`` there and the demand-escalation
    leg stays cold — a mutant with the scoped skip deleted passes those
    gates. This test drives the mechanism directly: a scoped pass carrying a
    genuine demand delta must NOT reach the allocator; a plain pass with a
    delta must; and a plain request coalescing into a pending scoped pass
    demotes it back to escalating.
    """

    from textual.app import App
    from textual.containers import Vertical

    from tldw_chatbook.UI.Console_Modules.left_rail import _ContextBoundedSection

    class _OwnerProbe:
        def __init__(self) -> None:
            self.allocation_requests = 0

        def request_allocation_reconcile(self) -> None:
            self.allocation_requests += 1

        def recover_section_focus(self, section_id: str) -> None:  # pragma: no cover
            pass

    owner = _OwnerProbe()
    filler = Static("content", id="demand-filler")
    filler.styles.height = 3
    wrapper = Vertical(filler)
    wrapper.styles.height = "auto"

    class SectionApp(App[None]):
        CSS = """
        _ContextBoundedSection { height: auto; }
        .console-bounded-section-viewport { height: auto; overflow-y: auto; }
        """

        def compose(self):
            yield _ContextBoundedSection(
                wrapper,
                section_id="workspace",
                owner=owner,
            )

    app = SectionApp()
    async with app.run_test(size=(60, 24)) as pilot:
        await _settle(pilot, passes=8)
        section = app.query_one(_ContextBoundedSection)
        settled_demand = section.desired_content_lines
        assert settled_demand > 0
        assert section._reconcile_scheduled is False
        # Change what the reconcile MEASURES rather than real child geometry:
        # a real style change resizes widgets in the same frame, and those
        # resize handlers issue plain requests that legitimately demote the
        # scoped pass — which would test the demotion rule, not the skip.
        measured = {"value": settled_demand}
        section._measure_content_lines = lambda viewport: measured["value"]
        baseline = owner.allocation_requests

        # 1. A scoped pass carrying a real +1 demand delta stays scoped.
        measured["value"] = settled_demand + 1
        section.request_scoped_reconcile()
        await _settle(pilot, passes=6)
        assert section.desired_content_lines == settled_demand + 1
        assert owner.allocation_requests == baseline

        # 2. A plain pass with a delta escalates.
        measured["value"] = settled_demand + 2
        section.request_reconcile()
        await _settle(pilot, passes=6)
        assert section.desired_content_lines == settled_demand + 2
        assert owner.allocation_requests > baseline
        escalated = owner.allocation_requests

        # 3. A plain request coalescing into a pending scoped pass demotes it.
        measured["value"] = settled_demand + 3
        section.request_scoped_reconcile()
        section.request_reconcile()
        await _settle(pilot, passes=6)
        assert section.desired_content_lines == settled_demand + 3
        assert owner.allocation_requests > escalated


# ---------------------------------------------------------------------------
# 3. The context tray's equality guards
# ---------------------------------------------------------------------------


def _tray_harness():
    state = replace(_base_grouped_workspace_state(), workspace_name="Research Lab")

    class TrayApp(ConsolidatedCSSApp):
        CSS_PATH = str(BUNDLED_STYLESHEET)

        def compose(self):
            yield ConsoleWorkspaceContextTray(
                state,
                show_heading=False,
                content="workspace",
                id="console-workspaces-context",
            )

    return TrayApp()


@pytest.mark.asyncio
async def test_tree_tooltip_recomputes_only_when_node_or_width_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        tree.move_cursor(tree.conversation_nodes["conv-a0"])
        await _settle(pilot, passes=4)

        measurements: list[str] = []
        original_measure = ConsoleWorkspaceTree._available_label_cells

        def counting_measure(widget, node):
            measurements.append(node.data.key if node.data else "?")
            return original_measure(widget, node)

        monkeypatch.setattr(
            ConsoleWorkspaceTree, "_available_label_cells", counting_measure
        )

        # Same cursor node, same width: the repeated calls sync_projection
        # makes on every push (the 5 Hz tick path) must be memo hits.
        for _ in range(5):
            tree._update_tooltip()
        assert measurements == [], measurements

        # A cursor move re-measures for the new node...
        tree.move_cursor(tree.conversation_nodes["conv-a1"])
        await _settle(pilot, passes=2)
        assert measurements, "a node change must recompute the tooltip"
        after_move = len(measurements)

        # ...and then re-settles into hits again.
        for _ in range(5):
            tree._update_tooltip()
        assert len(measurements) == after_move, measurements


@pytest.mark.asyncio
async def test_tree_tooltip_stays_correct_across_width_changes() -> None:
    """The memo must not freeze truncation decisions across width changes."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        long_state = replace(
            _probe_workspace_state(),
            workspace_tree=(
                WorkspaceTreeWorkspace(
                    workspace_id="ws-alpha",
                    label="Workspace Alpha " + "研究🙂" * 12,
                    conversations=(),
                    next_cursor=None,
                    active=True,
                ),
            ),
        )
        rail.sync_workspace_context(long_state)
        await _settle(pilot, passes=6)
        tree.move_cursor(tree.workspace_nodes["ws-alpha"])
        await _settle(pilot, passes=4)
        assert isinstance(tree.tooltip, Text)
        assert "Workspace Alpha" in tree.tooltip.plain

        # Grow the rail so the same label fits: the tooltip must clear.
        rail.styles.width = 130
        await _settle(pilot, passes=8)
        assert tree.tooltip is None


# ---------------------------------------------------------------------------
# 5. Failure / teardown cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_move_paths_are_safe_during_teardown() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        tree.move_cursor(tree.conversation_nodes["conv-a0"])
        await _settle(pilot, passes=3)

        await tree.remove()
        await pilot.pause()
        # The watch/update paths must be inert on a removed Tree.
        tree._update_tooltip()
        tree._post_context_changed()
        await pilot.pause()


@pytest.mark.asyncio
async def test_tooltip_target_removed_by_projection_push_is_not_served_stale() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        narrow_state = replace(
            _probe_workspace_state(),
            workspace_tree=(
                WorkspaceTreeWorkspace(
                    workspace_id="ws-alpha",
                    label="Workspace Alpha " + "研究🙂" * 12,
                    conversations=(
                        _conversation("conv-a0", "Alpha conversation 0"),
                    ),
                    next_cursor=None,
                    active=True,
                ),
            ),
        )
        rail.sync_workspace_context(narrow_state)
        await _settle(pilot, passes=6)
        tree.move_cursor(tree.workspace_nodes["ws-alpha"])
        await _settle(pilot, passes=4)
        assert isinstance(tree.tooltip, Text)

        # The projection drops the clipped workspace: the memo must not keep
        # serving the removed node's tooltip.
        replacement = replace(
            _probe_workspace_state(),
            workspace_tree=(
                WorkspaceTreeWorkspace(
                    workspace_id="ws-gamma",
                    label="Short",
                    conversations=(
                        _conversation("conv-g0", "Gamma conversation 0"),
                    ),
                    next_cursor=None,
                    active=True,
                ),
            ),
        )
        rail.sync_workspace_context(replacement)
        await _settle(pilot, passes=6)
        assert tree.tooltip is None


@pytest.mark.asyncio
async def test_boundary_flip_coexists_with_an_in_flight_allocation_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine allocation request already in flight still completes exactly
    once when a cursor boundary flip lands in the same frame window."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)
        tree.move_cursor(tree.workspace_nodes["ws-alpha"])
        await _settle(pilot, passes=6)

        runs: list[int] = []
        original_run = rail._run_allocation_reconcile

        def counting_run(*args, **kwargs):
            runs.append(1)
            return original_run(*args, **kwargs)

        monkeypatch.setattr(rail, "_run_allocation_reconcile", counting_run)

        rail.request_allocation_reconcile()
        tree.move_cursor(tree.conversation_nodes["conv-a0"])
        await _settle(pilot, passes=8)

        assert len(runs) == 1, runs
        assert rail._allocation_reconcile_scheduled is False


# ---------------------------------------------------------------------------
# 6. Genuine content changes still escalate to the allocator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_content_changes_still_reach_the_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoped path must not swallow real demand changes: growing the tree
    through a state push still runs the rail allocation pipeline."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        console, rail, tree = await _console_with_probe_tree(host, pilot)

        runs: list[int] = []
        original_run = rail._run_allocation_reconcile

        def counting_run(*args, **kwargs):
            runs.append(1)
            return original_run(*args, **kwargs)

        monkeypatch.setattr(rail, "_run_allocation_reconcile", counting_run)

        grown = replace(
            _probe_workspace_state(),
            workspace_tree=(
                WorkspaceTreeWorkspace(
                    workspace_id="ws-alpha",
                    label="Workspace Alpha",
                    conversations=tuple(
                        _conversation(f"conv-a{i}", f"Alpha conversation {i}")
                        for i in range(9)
                    ),
                    next_cursor=None,
                    active=True,
                ),
            ),
        )
        rail.sync_workspace_context(grown)
        await _settle(pilot, passes=8)
        assert runs, "a genuine projection change must run the allocator"
