"""PR 3a-2 Task 4: the survivor tick (task-15664), the unseen badge, and
the deep-link mount claim.

The defect (15664, found live in PR 3a-1 verification): the 0.2s Console
transcript poll self-stops the moment no run occupies a slot -- and a
cross-turn SURVIVOR occupies no slot -- so while only survivors run,
nothing repaints the Sub-agents rows' elapsed segment, the tab glyphs, or
(on settle) anything at all. A child working for a minute still shows
``· 1s``.

The fix under test: a 1s survivor tick armed at the transcript poll's own
self-stop edge (the exact spot the UI used to go blind), driven by
``ConsoleChatController.fleet_has_unsettled_children`` (the drain-paired
counter), stopping itself -- after one final settle paint -- when nothing
is live (15664 AC#2).

Rendered-geometry assertions throughout (`_assert_painted_at_own_region`
et al.), per this repo's Library-UAT lesson. The clock advance in the
headline test moves the handle's ``started_at`` back rather than patching
``time.monotonic`` -- same arithmetic (elapsed = now - started_at), and
patching the global clock would destabilize the event loop's own timers.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_console_fleet_panel import (
    _AGENT_SECTION_SIZE,
    _scroll_into_view,
    _setup_console,
)
from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot
from tldw_chatbook.Chat.console_chat_models import ConsoleFleetCompletionTarget
from tldw_chatbook.Chat.console_fleet_attention import bump_fleet_unseen_revision
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)


class _SurvivorBridge:
    """A live-fleet bridge double whose one child's clock the test controls.

    ``fleet_snapshot`` builds the handle fresh per call from mutable
    fields, so 'a minute passes' is one attribute write; ``has_unsettled_
    children`` is the drain-paired flag the survivor tick's drive reads.
    """

    def __init__(self, conversation_id: str = "conv-A") -> None:
        self.conversation_id = conversation_id
        self.unsettled = True
        self.started_at = 0.0
        self.status = "running"
        self.finished_at: float | None = None

    def fleet_snapshot(self, conversation_id: str) -> list[FleetHandle]:
        if conversation_id != self.conversation_id:
            return []
        return [
            FleetHandle(
                handle_id="h1",
                run_id="run-1",
                agent="researcher",
                task="long job",
                status=self.status,
                started_at=self.started_at,
                finished_at=self.finished_at,
            )
        ]

    def live_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        if conversation_id != self.conversation_id:
            return AgentLiveSnapshot()
        return AgentLiveSnapshot(status="running", step=1)

    def has_unsettled_children(self, conversation_id: str) -> bool:
        return self.unsettled

    def subagent_run(self, run_id: str):
        return None


def _fleet_row_text(console) -> str:
    return str(
        console.query_one(
            "#console-inspector-section-agent-fleet-row-0-primary", Static
        ).renderable
    )


async def _wire_survivor(pilot, host, bridge):
    """Mount Console, open the rail, expand the Sub-agents section (the
    elapsed segment lives on the expanded rows), and wire the bridge into
    BOTH slots: the agent module's (rail rows) and the controller's (the
    tick's drive)."""
    console = await _setup_console(pilot, host, bridge)
    controller = console._ensure_console_chat_controller()
    controller._agent_bridge = bridge
    section = console.query_one(
        "#console-agent-section-subagents", ConsoleInspectorSection
    )
    section.set_open(True)
    await pilot.pause()
    return console, controller


@pytest.mark.asyncio
async def test_survivor_elapsed_advances_with_no_other_interaction():
    """15664 AC#1 + AC#3, red against unmodified production: only a
    survivor runs (the transcript poll self-stops -- its production stop
    edge is driven for real); the clock advances a minute; the row's
    rendered elapsed segment must advance ON ITS OWN. Frozen paint =
    failure."""
    import time as _time

    bridge = _SurvivorBridge()
    # Idle at mount, so the 0.3s mount hedge provably no-ops and the ONLY
    # thing that can arm the tick below is the poll's stop edge -- the
    # production seam under test (and the seam whose absence is the bug).
    bridge.unsettled = False
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller = await _wire_survivor(pilot, host, bridge)
        await pilot.pause(0.5)  # let the mount hedge fire against idle
        assert console._console_fleet_survivor_timer is None
        bridge.unsettled = True
        bridge.started_at = _time.monotonic() - 1.0
        console._sync_console_agent_section()
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-row-0"
        )
        first = _fleet_row_text(console)
        assert "· 1s" in first or "· 2s" in first, f"precondition paint: {first!r}"

        # Drive the production edge that used to go blind: the 0.2s poll
        # starts (as every send does), sees no in-flight run, and
        # self-stops -- ONLY THEN does the clock advance, so the poll's
        # own last paint cannot mask a frozen rail.
        console._start_console_transcript_sync_timer()
        await pilot.pause(0.5)
        assert console._console_transcript_sync_timer is None, (
            "precondition: the transcript poll must have self-stopped -- "
            "this is the exact state 15664 describes"
        )
        # A minute passes while ONLY the survivor runs.
        bridge.started_at -= 60.0
        await pilot.pause(2.5)  # >= 2 survivor-tick beats

        after = _fleet_row_text(console)
        assert "1m " in after, (
            "the elapsed segment froze with no other interaction: "
            f"{after!r} (was {first!r}) -- task-15664's defect"
        )
        # Geometry, not DOM presence: the advanced row is actually painted.
        primary = console.query_one(
            "#console-inspector-section-agent-fleet-row-0-primary", Static
        )
        _assert_widget_and_ancestors_displayed(primary)
        _assert_painted_at_own_region(host, primary)


@pytest.mark.asyncio
async def test_the_tick_stops_itself_with_one_final_settle_paint():
    """15664 AC#2: the refresh must not keep repainting on a timer when no
    sub-agent is live. When the last child settles, the tick paints ONCE
    more (that pass flips the row to its terminal glyph with no user
    interaction) and stops -- no survivor timer remains."""
    import time as _time

    bridge = _SurvivorBridge()
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller = await _wire_survivor(pilot, host, bridge)
        bridge.started_at = _time.monotonic() - 5.0
        console._sync_console_agent_section()
        await pilot.pause()
        console._start_console_transcript_sync_timer()
        await pilot.pause(0.5)
        assert console._console_fleet_survivor_timer is not None, (
            "precondition: the survivor tick armed at the poll's stop edge"
        )

        # The child settles: drain owed -> none; handle goes terminal.
        bridge.unsettled = False
        bridge.status = "done"
        bridge.finished_at = bridge.started_at + 5.0
        await pilot.pause(2.5)

        assert console._console_fleet_survivor_timer is None, (
            "15664 AC#2: the survivor tick must stop itself when nothing is live"
        )
        after = _fleet_row_text(console)
        assert "✓" in after, (
            "the tick's final pass must paint the terminal glyph without "
            f"any user interaction: {after!r}"
        )


@pytest.mark.asyncio
async def test_an_idle_console_never_gains_a_survivor_timer():
    """15664 AC#2's other face: with nothing live, neither the mount hedge
    nor the poll's stop edge may create the timer at all."""
    bridge = _SurvivorBridge()
    bridge.unsettled = False
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller = await _wire_survivor(pilot, host, bridge)
        console._maybe_start_console_fleet_survivor_tick()
        console._start_console_transcript_sync_timer()
        await pilot.pause(0.6)
        assert console._console_transcript_sync_timer is None
        assert console._console_fleet_survivor_timer is None


# ---------------------------------------------------------------------------
# The unseen badge (durable-mark-backed) on session tabs.
# ---------------------------------------------------------------------------


def _attach_real_marks_service(app, tmp_path):
    """The factory app fakes ``chachanotes_db`` to ``None`` (no marks
    service); give it the REAL service over a real on-disk DB so the
    mark -> badge chain under test stays production-shaped."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="ui-test")
    app.conversation_local_marks_service = ConversationLocalMarksService(db)
    return app.conversation_local_marks_service


@pytest.mark.asyncio
async def test_unseen_mark_paints_the_tab_badge_and_viewing_clears_it(tmp_path):
    """The durable ``fleet_unseen`` mark drives a ◈ glyph on the marked
    session's TAB -- rendered, not just derived -- distinct from the
    turn-outcome glyphs; switching to that session (viewing it) clears the
    mark through the named seam and the glyph disappears on the next
    sync."""
    app = _build_test_app()
    marks = _attach_real_marks_service(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        controller = console._ensure_console_chat_controller()
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        second = store.create_session(title="Background research")
        store.switch_session(first.id)
        # The production write pair: the consumer writes the mark on the
        # child thread and bumps the badge-cache revision on the app loop.
        marks.set_mark(second.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        await console._sync_console_native_session_tabs()
        await pilot.pause()
        tab = console.query_one(f"#console-session-tab-{second.id}")
        assert "◈" in str(tab.label), (
            f"the marked session's tab must carry the unseen glyph: {tab.label!r}"
        )
        _assert_widget_and_ancestors_displayed(tab)
        _assert_painted_at_own_region(host, tab)
        active_tab = console.query_one(f"#console-session-tab-{first.id}")
        assert "◈" not in str(active_tab.label), "unmarked tabs stay clean"

        # Viewing IS the clear: switch to the marked session and resync.
        controller.switch_session(second.id)
        await console._sync_console_native_session_tabs()
        await pilot.pause()
        assert not marks.has_mark(
            second.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "viewing the conversation must clear the durable mark"
        tab = console.query_one(f"#console-session-tab-{second.id}")
        assert "◈" not in str(tab.label), (
            f"the badge must not outlive the mark: {tab.label!r}"
        )


@pytest.mark.asyncio
async def test_turn_outcome_glyphs_outrank_the_unseen_badge(tmp_path):
    """A session with BOTH an unvisited turn outcome and the durable mark
    shows the turn glyph -- the unseen badge is the lowest-precedence
    marker, never a mask over fresher turn news."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker

    app = _build_test_app()
    marks = _attach_real_marks_service(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        controller = console._ensure_console_chat_controller()
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        second = store.create_session(title="Both markers")
        store.switch_session(first.id)
        marks.set_mark(second.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)
        controller._unvisited_outcomes[second.id] = ConsoleRunMarker.FINISHED_FAILED

        await console._sync_console_native_session_tabs()
        await pilot.pause()
        label = str(console.query_one(f"#console-session-tab-{second.id}").label)
        assert "✗" in label and "◈" not in label, (
            f"the turn-outcome glyph must outrank the unseen badge: {label!r}"
        )


@pytest.mark.asyncio
async def test_unseen_mark_reaches_the_sidebar_browser_rows(tmp_path):
    """The sidebar conversation-browser pipeline carries the ◈ glyph for a
    marked, non-active session's row -- the same durable-mark backing as
    the tab badge, threaded as a resolved glyph string per the PA-T8
    no-model-import discipline."""
    app = _build_test_app()
    marks = _attach_real_marks_service(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        console._ensure_console_chat_controller()
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        second = store.create_session(title="Marked in browser")
        store.switch_session(first.id)
        marks.set_mark(second.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        rows = console._workspace._native_console_browser_rows()
        by_session = {row.native_session_id: row for row in rows}
        assert by_session[second.id].run_marker == "◈", (
            f"the marked row must carry the unseen glyph: {rows}"
        )
        assert by_session[first.id].run_marker == "", "unmarked rows stay clean"


# ---------------------------------------------------------------------------
# The deep-link mount claim.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_claim_switches_to_the_settled_conversations_session():
    """A staged ``CONSOLE_FLEET_COMPLETION`` target is claimed and Console
    switches to that session; a target whose session is gone is
    acknowledged and dropped (the durable mark stays the pointer)."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        second = store.create_session(title="Settled elsewhere")
        store.switch_session(first.id)

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_FLEET_COMPLETION,
            ConsoleFleetCompletionTarget(
                conversation_id=second.id, session_id=second.id
            ),
        )
        assert console.consume_pending_console_fleet_completion() is True
        assert store.active_session_id == second.id, (
            "the claim must land the user on the settled conversation"
        )
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_FLEET_COMPLETION
        )

        # A closed session's target: acknowledged, dropped, no switch.
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_FLEET_COMPLETION,
            ConsoleFleetCompletionTarget(conversation_id="conv-gone"),
        )
        assert console.consume_pending_console_fleet_completion() is False
        assert store.active_session_id == second.id
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_FLEET_COMPLETION
        ), "an unmatchable target must be acknowledged, not retried forever"
        assert (
            app.pending_handoffs.claim(HandoffChannel.CONSOLE_FLEET_COMPLETION) is None
        )
