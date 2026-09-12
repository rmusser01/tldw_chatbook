"""Supervisor fleet PR 2b, Task 4: the Agents rail section's three states.

Spec `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
section 7: state 1 (collapsed -- glyph cluster + "N working, M done",
right-aligned), state 2 (expanded -- one two-line row per child), state 3
(drill-in, now reached by clicking a SPECIFIC row rather than cycling
through every run one click at a time -- see `Tests/UI/test_console_agent_
rail.py`'s `test_clicking_a_specific_subagent_row_drills_into_that_run_
directly` for the controller-level half of this same replacement).

Rendered-geometry assertions throughout, not DOM presence -- per the
Library-UAT lesson this repo cites repeatedly: an unbounded-width Static
can be "present" in a headless `query_one` while rendering invisible on a
real terminal. `_assert_widget_and_ancestors_displayed`/`_assert_painted_
at_own_region` (imported from `test_console_parallel_runs.py`, the
precedent this whole PR cites) use the real display-chain walk and the
compositor's own hit-test rather than a raw `region.y` bound.

Harness: `ConsoleHarness`/`_build_test_app`/`_wait_for_selector`, mounted
at `(180, 48)` -- matches `test_console_agent_controller.py`'s own
`_AGENT_SECTION_SIZE` (wide enough that the Agent rail section is expanded
rather than collapsed by the responsive shell).
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot, AgentLiveTurnUsage
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
)


@pytest.fixture(autouse=True)
def _real_fleet_recovery_database(monkeypatch, tmp_path, request):
    """Mount with the real recovery owner; a DB-less mount correctly pauses."""
    from Tests.conftest import _close_database_instance
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs

    build = request.module._build_test_app
    owned_resources = []

    def build_with_db(*args, **kwargs):
        app = build(*args, **kwargs)
        _attach_real_dbs(app, tmp_path)
        _configure_native_ready_console(app)
        owned_resources.append(
            (
                app.chachanotes_db,
                app.local_library_collections_db,
                app.evaluation_orchestrator.db,
                app.local_workspace_db,
                app.subscriptions_db,
                app._instance_lock_status.handle,
            )
        )
        return app

    monkeypatch.setattr(request.module, "_build_test_app", build_with_db)
    yield
    for chacha, collections, evals, workspaces, subscriptions, lock in owned_resources:
        for database in (chacha, collections, evals, workspaces, subscriptions):
            _close_database_instance(database)
        if lock is not None:
            lock.close()


_AGENT_SECTION_SIZE = (180, 48)

#: The fleet mini-section's own `section_id` (`CONSOLE_AGENT_FLEET_SECTION_
#: ID` in `agent.py`) -- duplicated here as a literal rather than imported
#: so this test file pins the DOM id contract independently of that
#: constant ever changing silently.
_SECTION_ID = "agent-fleet"


def _ready_test_app():
    app = _build_test_app()
    _configure_native_ready_console(app)
    return app


def _static_text(console, widget_id: str) -> str:
    return str(console.query_one(widget_id, Static).renderable)


class _FleetBridge:
    """Fake Console agent bridge exposing exactly the live-fleet + drill-in
    surface the Agents section reads -- a stand-in for
    `AgentService.fleet_snapshot()`'s real return via `ConsoleAgentBridge.
    fleet_snapshot` (PR2b Task 1)."""

    def __init__(
        self,
        handles: tuple[FleetHandle, ...],
        conversation_id: str = "conv-A",
        live_by_run_id: dict[str, AgentLiveSnapshot] | None = None,
    ) -> None:
        self._handles = list(handles)
        self._by_run_id = {h.run_id: h for h in handles if h.run_id}
        self._conversation_id = conversation_id
        self._live_by_run_id = live_by_run_id or {}
        #: PR2b Task 5: every `cancel_subagent(conversation_id, handle_id)`
        #: call this fake received, in order -- the seam the per-row-cancel
        #: wiring tests assert against.
        self.cancel_calls: list[tuple[str, str]] = []

    def fleet_snapshot(self, conversation_id: str) -> list[FleetHandle]:
        if conversation_id != self._conversation_id:
            return []
        return list(self._handles)


    def cancel_subagent(self, conversation_id: str, handle_id: str) -> bool:
        self.cancel_calls.append((conversation_id, handle_id))
        return conversation_id == self._conversation_id and any(
            h.handle_id == handle_id for h in self._handles
        )

    def live_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        if conversation_id != self._conversation_id:
            return AgentLiveSnapshot()
        return AgentLiveSnapshot(status="running", step=1)

    def live_run_snapshot(
        self, conversation_id: str, run_id: str
    ) -> AgentLiveSnapshot | None:
        if conversation_id != self._conversation_id:
            return None
        return self._live_by_run_id.get(run_id)

    def subagent_counts(self, conversation_ids: list[str]) -> dict[str, int]:
        if not self._handles or self._conversation_id not in conversation_ids:
            return {}
        return {self._conversation_id: len(self._handles)}

    def subagent_run(self, run_id: str):
        handle = self._by_run_id.get(run_id)
        if handle is None:
            return None
        return {
            "id": run_id,
            "conversation_id": self._conversation_id,
            "status": handle.status,
            "task": handle.task,
            "steps": [],
        }


async def _setup_console(pilot, host, bridge, *, conversation_id: str = "conv-A"):
    """Mount the Console screen, wire the fake fleet bridge in, and force
    the Agent rail section open (bypassing the persisted-preference/auto-
    open machinery entirely -- matches `test_console_parallel_runs.py`'s own
    `_setup_tall_steps_and_parked_fleet` precedent) so the fleet mini-
    section's ancestor chain is genuinely displayed for the geometry
    assertions below. A REAL persisted-preference write (not a bare
    `SimpleNamespace` stand-in for `_current_console_rail_state`) --
    `_drill_into_console_agent_subagent` dispatches the FULL periodic sync
    (`_sync_native_console_chat_ui`), which reads far more of
    `ConsoleRailState` than just `agent_open` (e.g. `_sync_console_rail_
    visibility` reads `left_label`/`left_badge`); a minimal fake namespace
    crashes there with an `AttributeError` the moment a test drills in.

    task-10: the fleet mini-section itself now lives in the right
    (Inspector) rail, closed by default (`CONSOLE_RAIL_RIGHT_DEFAULT_OPEN
    = False`) -- unlike the left (Context) rail this file used to rely on,
    which stays open by default. `right_open=True` is passed alongside the
    existing `agent` section toggle (still real: the Agent status/steps
    Statics and the drilldown controls stay in the left rail) so the fleet
    section's own ancestor chain is genuinely displayed too.
    """
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
    console._console_agent_bridge = bridge
    console._console_agent_drilldown_run_id = None
    console._character._current_console_rail_conversation_id = lambda: conversation_id
    console._agent._console_agent_drilldown_conversation_id = conversation_id
    console._set_console_rail_preference(
        right_open=True,
        section_updates={"agent": True},
        notify_on_failure=False,
    )
    console._sync_console_agent_section()
    await pilot.pause()
    await _scroll_into_view(pilot, console, "#console-agent-section-subagents")
    return console


async def _scroll_into_view(pilot, console, selector: str) -> None:
    """Scroll a widget inside its rail's `VerticalScroll` into view before
    a geometry assertion or a real click.

    task-10: the fleet mini-section now lives in `#console-inspector-rail-
    body` (the right/Inspector rail), not `#console-left-rail-body` as
    before -- `Widget.scroll_visible()` walks whichever scrollable
    ancestor actually contains it, so this helper needed no code change,
    only this note. It can sit well past a short terminal's fold, and
    `Widget.region` is reported UNCLIPPED (a below-the-fold widget still
    has a non-empty region) -- `pilot.click` and the compositor hit-test
    both need the widget's OWN screen offset to be genuinely on-screen.
    """
    widget = console.query_one(selector)
    widget.scroll_visible(animate=False)
    await pilot.pause(0.2)


# -- State 1: collapsed summary line (spec §7) --------------------------


@pytest.mark.asyncio
async def test_state_1_summary_line_shows_glyph_cluster_and_working_done_counts():
    """Two running + one done -> "2 working, 1 done", with a glyph per
    child (in handle order) prefixed -- the exact grammar spec §7 state 1
    describes. The counts are read straight off the same handles a real
    `FleetCoordinator.snapshot()` publish would return -- this IS "the
    coordinator snapshot" the brief's AC asks the summary to match."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="find pricing",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="researcher",
            task="find comps",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h3",
            run_id="run-3",
            agent="writer",
            task="draft summary",
            status="done",
            result="drafted",
            started_at=1000.0,
            finished_at=1005.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)

        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        # The section MOUNTS collapsed, but task-13 (addition A) opens it
        # once on first fleet activity: after the move into the Inspect rail
        # a collapsed header with no body surfaced nothing at all. The
        # summary line this test is really about is painted either way --
        # the header stays visible in both states.
        assert fleet_section.open is True
        assert fleet_section.summary == "●●✓ 2 working, 1 done"

        # Not just the widget's own state -- prove it actually RENDERS
        # (Task 3's own mutation-testing lesson: a structural assertion
        # alone can pass vacuously against a broken `.update()` call).
        assert (
            _static_text(console, f"#console-inspector-section-{_SECTION_ID}-summary")
            == "●●✓ 2 working, 1 done"
        )
        summary_static = console.query_one(
            f"#console-inspector-section-{_SECTION_ID}-summary", Static
        )
        _assert_widget_and_ancestors_displayed(summary_static)
        _assert_painted_at_own_region(host, summary_static)


@pytest.mark.asyncio
async def test_state_1_summary_counts_every_terminal_status_as_done_not_just_literal_done():
    """ "Working" is `status == "running"`; every other status this
    codebase's fleet vocabulary uses (`done`/`error`/`stuck`/`cancelled` --
    see `SubAgentSummary.status`'s own docstring, and `TERMINAL_RUN_
    STATUSES`) is terminal, i.e. counted as "done" in the summary's second
    bucket -- not just a literal `status == "done"` check."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="a",
            task="t1",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="a",
            task="t2",
            status="done",
            started_at=1000.0,
            finished_at=1001.0,
        ),
        FleetHandle(
            handle_id="h3",
            run_id="run-3",
            agent="a",
            task="t3",
            status="error",
            error="boom",
            started_at=1000.0,
            finished_at=1001.0,
        ),
        FleetHandle(
            handle_id="h4",
            run_id="run-4",
            agent="a",
            task="t4",
            status="cancelled",
            started_at=1000.0,
            finished_at=1001.0,
        ),
        FleetHandle(
            handle_id="h5",
            run_id="run-5",
            agent="a",
            task="t5",
            status="stuck",
            started_at=1000.0,
            finished_at=1001.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        # Large fleets keep full counts without an unbounded glyph cluster.
        assert fleet_section.summary == "1 working, 4 done"
        assert len(fleet_section.rows) == 5


# -- State 2: expanded rows, two lines each (spec §7) --------------------


@pytest.mark.asyncio
async def test_state_2_expanded_rows_render_two_painted_lines_per_child():
    """Line 1: glyph + agent name + elapsed. Line 2: dimmed last-step
    summary (task while running, result/error once terminal). Both lines
    are asserted painted at their OWN region, not just "present" -- a row
    positioned past the fold would still satisfy a bare `query_one`."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="find pricing",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="writer",
            task="draft summary",
            status="done",
            result="drafted the summary",
            started_at=1000.0,
            finished_at=1005.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        # The user expands it (spec §7 state 2) -- the section's own
        # chevron, not a rail-wide toggle.
        fleet_section.set_open(True)
        await pilot.pause()

        for index, handle in enumerate(handles):
            primary = console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-{index}-primary",
                Static,
            )
            secondary = console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-{index}-secondary",
                Static,
            )
            await _scroll_into_view(
                pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-{index}"
            )
            _assert_widget_and_ancestors_displayed(primary)
            _assert_painted_at_own_region(host, primary)
            _assert_widget_and_ancestors_displayed(secondary)
            _assert_painted_at_own_region(host, secondary)
            assert handle.agent in str(primary.renderable)

        # Line 1: the DONE child's elapsed segment is deterministic
        # (started_at/finished_at are both fixed floats, independent of
        # wall-clock timing).
        assert "5s" in str(
            console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-1-primary", Static
            ).renderable
        )
        # Line 2: the RUNNING child's secondary is its task (nothing else
        # to summarize yet); the DONE child's is its result.
        assert "find pricing" in str(
            console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-0-secondary", Static
            ).renderable
        )
        assert "drafted the summary" in str(
            console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-1-secondary", Static
            ).renderable
        )


# -- State 3: drill-in via a specific row click (spec §7) -----------------


@pytest.mark.asyncio
async def test_state_3_drilling_into_a_row_hides_the_fleet_section_and_shows_the_transcript():
    """Once drilled in, the fleet mini-section hides entirely (its rows
    would be redundant next to the drilled-in child's own status/steps,
    which the pre-existing Statics already carry -- state 3 is unchanged
    plumbing, only how you REACH it changed) -- and the Back button
    becomes the only way out."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="find pricing",
            status="done",
            result="42",
            started_at=1000.0,
            finished_at=1002.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)

        console._agent._drill_into_console_agent_subagent("run-1")
        await pilot.pause()
        console._sync_console_agent_section()

        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        assert not fleet_section.display

        assert _static_text(console, "#console-agent-section-status").startswith(
            "Sub-agent · done"
        )
        back_button = console.query_one("#console-agent-drilldown-back")
        assert back_button.styles.display != "none"


@pytest.mark.asyncio
async def test_clicking_the_last_row_drills_into_that_child_directly_not_via_a_cycle():
    """A REAL click (`pilot.click`, not calling the controller method by
    hand) on the LAST row lands on THAT child immediately -- a cycling
    mechanism would have needed three clicks (or landed on whichever run
    happened to be "next" from wherever the cursor was) to reach it."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="t1",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="writer",
            task="t2",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h3",
            run_id="run-3",
            agent="reviewer",
            task="t3",
            status="running",
            started_at=1000.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        fleet_section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-2"
        )

        await pilot.click(f"#console-inspector-section-{_SECTION_ID}-row-2")
        await pilot.pause()

        assert console._console_agent_drilldown_run_id == "run-3"


@pytest.mark.asyncio
async def test_clicking_the_first_row_drills_into_that_child_directly():
    """The row-order mirror of the test above -- proves row 0 is reachable
    on its own too, i.e. each row is independently addressable rather than
    only the position a cycling cursor would land on next."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="t1",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="writer",
            task="t2",
            status="running",
            started_at=1000.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        fleet_section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-0"
        )

        await pilot.click(f"#console-inspector-section-{_SECTION_ID}-row-0")
        await pilot.pause()

        assert console._console_agent_drilldown_run_id == "run-1"


# -- PR2b Task 5: per-child token spend + per-row cancel -------------------


@pytest.mark.asyncio
async def test_live_usage_is_attributed_per_child_without_replacing_task_or_order():
    """Looking up one shared snapshot would put the same token count on every child."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="compare provider pricing across regions",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h2",
            run_id="run-2",
            agent="writer",
            task="draft the decision memo with citations",
            status="running",
            started_at=1000.0,
        ),
        FleetHandle(
            handle_id="h3",
            run_id="run-3",
            agent="reviewer",
            task="check the recommendation",
            status="running",
            started_at=1000.0,
        ),
    )
    bridge = _FleetBridge(
        handles,
        live_by_run_id={
            "run-1": AgentLiveSnapshot(
                status="running",
                turn_usage=AgentLiveTurnUsage(12, "provider", 1001.0, 1),
            ),
            "run-2": AgentLiveSnapshot(
                status="running",
                turn_usage=AgentLiveTurnUsage(7, "local", 1002.0, 1),
            ),
        },
    )
    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        section.set_open(True)
        await pilot.pause()

        assert [row.row_id for row in section.rows] == ["h1", "h2", "h3"]
        assert section.rows[0].secondary_text == (
            "compare provider pricing across regions · 12 provider output tok"
        )
        assert section.rows[1].secondary_text == (
            "draft the decision memo with citations · ~7 local output tok"
        )
        assert section.rows[2].secondary_text == "check the recommendation"


@pytest.mark.asyncio
async def test_terminal_budget_tokens_do_not_retain_live_usage():
    """A settled child must keep final budget accounting distinct from live usage."""
    handle = FleetHandle(
        handle_id="h1",
        run_id="run-1",
        agent="writer",
        task="draft summary",
        status="done",
        result="drafted",
        started_at=1000.0,
        finished_at=1002.0,
        total_tokens=1234,
    )
    bridge = _FleetBridge(
        (handle,),
        live_by_run_id={
            "run-1": AgentLiveSnapshot(
                status="done",
                turn_usage=AgentLiveTurnUsage(99, "provider", 1001.0, 1),
            )
        },
    )
    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        assert section.rows[0].secondary_text == "drafted · 1.2k budget tok"
        assert "output tok" not in section.rows[0].secondary_text


@pytest.mark.asyncio
async def test_live_usage_task_and_count_are_painted_wide_and_narrow():
    """Responsive rail paint must leave realistic task context and usage visible."""
    import time as _time
    from xml.etree import ElementTree

    def painted_cells(svg: str) -> list[tuple[str, str]]:
        root = ElementTree.fromstring(svg)
        return [
            (
                element.attrib.get("x", ""),
                "".join(element.itertext()).replace("\xa0", " "),
            )
            for element in root.iter()
            if element.tag.endswith("}text") or element.tag == "text"
        ]

    def assert_usage_cells_are_painted(svg: str) -> None:
        cells = painted_cells(svg)
        task_x = next(x for x, text in cells if "compare regional provider" in text)
        assert any(x == task_x and "128" in text for x, text in cells)
        assert any(x == task_x and "provider output tok" in text for x, text in cells)

    started_at = _time.monotonic() - 12.0
    handle = FleetHandle(
        handle_id="h1",
        run_id="run-1",
        agent="researcher",
        task="compare regional provider pricing and summarize material differences",
        status="running",
        started_at=started_at,
    )
    bridge = _FleetBridge(
        (handle,),
        live_by_run_id={
            "run-1": AgentLiveSnapshot(
                status="running",
                turn_usage=AgentLiveTurnUsage(128, "provider", started_at, 1),
            )
        },
    )
    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-row-0"
        )
        host.save_screenshot(
            filename="live-usage-wide-180x48.svg",
            path=".superpowers/sdd/2026-09-12-live-per-run-usage/task-2-evidence",
        )
        assert_usage_cells_are_painted(host.export_screenshot(simplify=True))

        await pilot.resize_terminal(100, 32)
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-row-0"
        )
        host.save_screenshot(
            filename="live-usage-narrow-100x32.svg",
            path=".superpowers/sdd/2026-09-12-live-per-run-usage/task-2-evidence",
        )
        assert_usage_cells_are_painted(host.export_screenshot(simplify=True))
        secondary = console.query_one(
            "#console-inspector-section-agent-fleet-row-0-secondary", Static
        )
        _assert_widget_and_ancestors_displayed(secondary)
        _assert_painted_at_own_region(host, secondary)
        assert "regional provider pricing" in str(secondary.renderable)
        assert "128 provider output tok" in str(secondary.renderable)


@pytest.mark.asyncio
async def test_state_2_secondary_line_shows_token_spend_for_a_finished_child():
    """A finished child's measured `total_tokens` (`FleetHandle.
    total_tokens`, PR2b Task 5) appends a ` · N tok` segment to its row's
    secondary line -- a still-running child (0, not yet final) shows none,
    covered by every OTHER state-2 test above never setting it."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="writer",
            task="draft summary",
            status="done",
            result="drafted the summary",
            started_at=1000.0,
            finished_at=1002.0,
            total_tokens=1234,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        fleet_section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-0"
        )

        secondary = console.query_one(
            f"#console-inspector-section-{_SECTION_ID}-row-0-secondary", Static
        )
        _assert_widget_and_ancestors_displayed(secondary)
        _assert_painted_at_own_region(host, secondary)
        text = str(secondary.renderable)
        assert "drafted the summary" in text
        assert "1.2k budget tok" in text


@pytest.mark.asyncio
async def test_pressing_delete_on_a_running_row_cancels_the_child_through_the_bridge():
    """PR2b Task 5 (per-row cancel), end to end through the REAL screen
    handler: a real Delete keypress on a running (cancellable) row reaches
    `ConsoleAgentBridge.cancel_subagent` with the row's own handle id --
    the row-initiated cancel's actual production wiring, not just the
    controller method in isolation."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="find pricing",
            status="running",
            started_at=1000.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        fleet_section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-0"
        )

        row_widget = console.query_one(
            f"#console-inspector-section-{_SECTION_ID}-row-0",
            ConsoleInspectorSectionRow,
        )
        assert row_widget.cancellable is True
        row_widget.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()

        assert bridge.cancel_calls == [("conv-A", "h1")]


@pytest.mark.asyncio
async def test_pressing_delete_on_a_finished_row_does_nothing():
    """A terminal child is not cancellable (`_fleet_row_from_handle`'s
    `status not in TERMINAL_RUN_STATUSES` gate) -- Delete on that row must
    never reach the bridge at all."""
    handles = (
        FleetHandle(
            handle_id="h1",
            run_id="run-1",
            agent="researcher",
            task="find pricing",
            status="done",
            result="42",
            started_at=1000.0,
            finished_at=1001.0,
        ),
    )
    bridge = _FleetBridge(handles)

    app = _ready_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        fleet_section.set_open(True)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, f"#console-inspector-section-{_SECTION_ID}-row-0"
        )

        row_widget = console.query_one(
            f"#console-inspector-section-{_SECTION_ID}-row-0",
            ConsoleInspectorSectionRow,
        )
        assert row_widget.cancellable is False
        row_widget.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()

        assert bridge.cancel_calls == []
