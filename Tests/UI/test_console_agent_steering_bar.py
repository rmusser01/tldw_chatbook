"""Fleet PR 3b Task 3: the panel steering input + honest queued state.

Spec `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
section 7 ("phase 3 adds the steering input + mailbox 'queued' state") and
section 1's owner pin: the panel watches/steers, never launches — so the
input exists ONLY while drilled into a LIVE child, never for a finished/
historical one and never on the overview.

Painted-frame assertions throughout (`_assert_painted_at_own_region`, the
compositor's own hit-test) — this programme has three times had "rendered"
assertions that only proved data arrived.

The bridge fake here delegates its ``steer_subagent`` to the REAL
``ConsoleAgentBridge.steer_subagent`` (unbound, over a REAL
``FleetCoordinator`` in ``_fleet_coordinators``) — deliberately, so the
end-to-end submit test measures the PRODUCTION posting path and its exact
``("user", text)`` label, not a fake's re-implementation of it. That works
because ``steer_subagent``'s whole state surface is ``_fleet_coordinators``
(the no-service-hop rule, pinned by
``Tests/Chat/test_console_agent_bridge_steering.py``).
"""

from __future__ import annotations

import time

import pytest
from textual.widgets import Input, Static

from Tests.UI.test_console_fleet_panel import (
    _SECTION_ID,
    _scroll_into_view,
    _setup_console,
    _static_text,
)
from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.agent_models import MAX_STEERING_CHARS
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot,
    ConsoleAgentBridge,
)
from tldw_chatbook.Widgets.Console.console_agent_steering_bar import (
    ConsoleAgentSteeringBar,
    ConsoleAgentSteeringState,
)

_SIZE = (180, 48)
CONV = "conv-A"

_BAR = "#console-agent-steering-bar"
_INPUT = "#console-agent-steering-input"
_QUEUED = "#console-agent-steering-queued"
_NOTE = "#console-agent-steering-note"


class _SteeringFleetBridge:
    """Fake bridge exposing the fleet surface the Agent rail reads, whose
    steering path IS the real ``ConsoleAgentBridge.steer_subagent``."""

    def __init__(self, coordinator: FleetCoordinator, conversation_id: str = CONV):
        self.coordinator = coordinator
        #: The real method's whole state surface.
        self._fleet_coordinators = {conversation_id: coordinator}
        self._conversation_id = conversation_id
        #: Every steer call the screen wiring routed here, in order.
        self.steer_calls: list[tuple[str, str, str]] = []

    def fleet_snapshot(self, conversation_id: str):
        if conversation_id != self._conversation_id:
            return []
        return self.coordinator.snapshot()

    def live_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        if conversation_id != self._conversation_id:
            return AgentLiveSnapshot()
        return AgentLiveSnapshot(status="running", step=1)

    def subagent_run(self, run_id: str):
        for handle in self.coordinator.snapshot():
            if handle.run_id == run_id:
                return {
                    "id": run_id,
                    "conversation_id": self._conversation_id,
                    "status": handle.status,
                    "task": handle.task,
                    "steps": [],
                }
        return None

    def steer_subagent(self, conversation_id: str, row_id: str, text: str) -> bool:
        self.steer_calls.append((conversation_id, row_id, text))
        return ConsoleAgentBridge.steer_subagent(self, conversation_id, row_id, text)


def _fleet_with_live_child(run_id: str = "run-1", task: str = "find pricing"):
    coordinator = FleetCoordinator(max_live=4, clock=time.monotonic)
    handle = coordinator.reserve(task, "researcher")
    coordinator.attach_run(handle.handle_id, run_id)
    return coordinator, handle.handle_id


def _fleet_with_finished_child(run_id: str = "run-1"):
    coordinator, handle_id = _fleet_with_live_child(run_id=run_id)
    coordinator.finish(handle_id, "done", result="42")
    return coordinator, handle_id


async def _drill_in(pilot, console, run_id: str) -> None:
    console._agent._drill_into_console_agent_subagent(run_id)
    await pilot.pause()
    console._sync_console_agent_section()
    await pilot.pause()


# -- visibility: live drill-in ONLY -------------------------------------


@pytest.mark.asyncio
async def test_steering_input_paints_in_a_live_childs_drill_in():
    """Drilled into a RUNNING child, the input is genuinely painted at its
    own region — not merely present in the DOM."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        steer_input = console.query_one(_INPUT, Input)
        _assert_widget_and_ancestors_displayed(steer_input)
        _assert_painted_at_own_region(host, steer_input)


@pytest.mark.asyncio
async def test_steering_input_does_not_paint_for_a_finished_childs_drill_in():
    """A finished child takes no more model turns — drilling into it must
    never offer a steering input (spec section 1: the panel watches/steers,
    never launches; continuation is supervisor-only)."""
    coordinator, _handle_id = _fleet_with_finished_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")

        # The drill-in itself is real (state-3 header shows the child)...
        assert _static_text(console, "#console-agent-section-status").startswith(
            "Sub-agent · done"
        )
        # ...but the steering bar is not on the live surface.
        bar = console.query_one(_BAR, ConsoleAgentSteeringBar)
        assert not bar.display


@pytest.mark.asyncio
async def test_steering_input_does_not_paint_in_the_overview():
    """Not drilled in at all: even with a LIVE child on the fleet list,
    the overview offers no steering input (the input belongs to one
    specific child's drill-in, never the aggregate)."""
    coordinator, _handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)

        bar = console.query_one(_BAR, ConsoleAgentSteeringBar)
        assert not bar.display


# -- submit: the production path, end to end -----------------------------


@pytest.mark.asyncio
async def test_submitting_steering_lands_an_exact_user_labeled_mailbox_entry():
    """A REAL Enter keypress in the input reaches the REAL
    ``steer_subagent`` and the REAL coordinator mailbox — which must hold
    exactly ``("user", text)``. A mislabeling to the supervisor source
    fails here, at the entry itself."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        steer_input = console.query_one(_INPUT, Input)
        steer_input.focus()
        await pilot.pause()
        steer_input.value = "focus on pricing first"
        await pilot.press("enter")
        await pilot.pause()

        assert bridge.steer_calls == [(CONV, handle_id, "focus on pricing first")]
        assert coordinator.drain_steering(handle_id) == [
            ("user", "focus on pricing first")
        ]
        # The input cleared for the next message.
        assert console.query_one(_INPUT, Input).value == ""


# -- the honest queued state ---------------------------------------------


@pytest.mark.asyncio
async def test_queued_count_paints_on_the_fleet_row_and_clears_after_a_drain():
    """Spec section 6 latency honesty on the OVERVIEW: a row with queued
    steering says so (`· steering queued (N)`), painted; after the child
    drains (simulated), the next sync clears the segment."""
    coordinator, handle_id = _fleet_with_live_child()
    assert coordinator.post_steering(handle_id, "user", "first") is True
    assert coordinator.post_steering(handle_id, "user", "second") is True
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        fleet_section = console.query_one("#console-agent-section-subagents")
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
        assert "steering queued (2)" in str(secondary.renderable)

        # The child consumes the mailbox at its next drain boundary
        # (simulated) — the next sync must drop the segment.
        coordinator.drain_steering(handle_id)
        console._sync_console_agent_section()
        await pilot.pause()
        assert "steering queued" not in str(
            console.query_one(
                f"#console-inspector-section-{_SECTION_ID}-row-0-secondary", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_queued_count_line_paints_in_the_drill_in_and_clears_after_a_drain():
    """The bar's own queued line: painted with the honest count while
    entries wait, gone once the child drained them."""
    coordinator, handle_id = _fleet_with_live_child()
    assert coordinator.post_steering(handle_id, "user", "first") is True
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        queued = console.query_one(_QUEUED, Static)
        _assert_widget_and_ancestors_displayed(queued)
        _assert_painted_at_own_region(host, queued)
        assert str(queued.renderable) == "steering queued (1)"

        coordinator.drain_steering(handle_id)
        console._sync_console_agent_section()
        await pilot.pause()
        assert not console.query_one(_QUEUED, Static).display


# -- refusals at the input's own boundary --------------------------------


@pytest.mark.asyncio
async def test_empty_input_submit_is_inert():
    """Whitespace-only submit: nothing reaches the bridge, nothing is
    queued, and no refusal copy appears (in particular never an unknown-id
    refusal naming `''` — Task 2's report bound this task to that)."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        steer_input = console.query_one(_INPUT, Input)
        steer_input.focus()
        await pilot.pause()
        steer_input.value = "   "
        await pilot.press("enter")
        await pilot.pause()

        assert bridge.steer_calls == []
        assert coordinator.drain_steering(handle_id) == []
        note = console.query_one(_NOTE, Static)
        assert str(note.renderable) == ""
        assert not note.display


@pytest.mark.asyncio
async def test_submit_with_an_empty_target_never_posts_a_message_at_all():
    """The WIDGET's empty-target guard itself: a bar whose state carries
    no target (a race the sync can produce) swallows the submit — no
    ``SteeringSubmitted`` posts at all, so nothing downstream can ever
    draw a refusal naming ``''``.

    Asserted at the message layer deliberately (mutation finding, first
    round): "nothing reached the bridge" alone is VACUOUS for this guard,
    because the controller's own ``not target_id`` arm also refuses — the
    guard-dropped mutant survived an outcome-only assertion. Spying on
    ``post_message`` pins the layer the plan names ("disable submit on an
    empty target"), not just the shared outcome."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        bar = console.query_one(_BAR, ConsoleAgentSteeringBar)
        bar.sync_state(
            ConsoleAgentSteeringState(visible=True, target_id="", queued=0)
        )
        await pilot.pause()

        posted: list[object] = []
        original_post = bar.post_message

        def _spy(message, *args, **kwargs):
            posted.append(message)
            return original_post(message, *args, **kwargs)

        bar.post_message = _spy

        steer_input = console.query_one(_INPUT, Input)
        steer_input.focus()
        await pilot.pause()
        steer_input.value = "hello there"
        await pilot.press("enter")
        await pilot.pause()

        assert [
            m
            for m in posted
            if isinstance(m, ConsoleAgentSteeringBar.SteeringSubmitted)
        ] == []
        assert bridge.steer_calls == []
        assert coordinator.drain_steering(handle_id) == []


@pytest.mark.asyncio
async def test_oversize_submit_refused_with_own_painted_copy():
    """One char over ``MAX_STEERING_CHARS``: refused at the input with the
    bar's OWN copy (painted), nothing posted anywhere, and the draft kept
    in the input so the user can shorten it."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        steer_input = console.query_one(_INPUT, Input)
        steer_input.focus()
        await pilot.pause()
        steer_input.value = "x" * (MAX_STEERING_CHARS + 1)
        await pilot.press("enter")
        await pilot.pause()

        assert bridge.steer_calls == []
        assert coordinator.drain_steering(handle_id) == []
        await _scroll_into_view(pilot, console, _NOTE)
        note = console.query_one(_NOTE, Static)
        _assert_widget_and_ancestors_displayed(note)
        _assert_painted_at_own_region(host, note)
        text = str(note.renderable)
        assert "too long" in text and str(MAX_STEERING_CHARS) in text
        # The draft is kept for shortening, not discarded.
        assert console.query_one(_INPUT, Input).value == "x" * (
            MAX_STEERING_CHARS + 1
        )


@pytest.mark.asyncio
async def test_at_cap_submit_is_accepted():
    """Exactly at the cap is legal — the refusal boundary is `>`type, not
    `>=` (mirrors the bridge-level boundary test)."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _SteeringFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        steer_input = console.query_one(_INPUT, Input)
        steer_input.focus()
        await pilot.pause()
        steer_input.value = "x" * MAX_STEERING_CHARS
        await pilot.press("enter")
        await pilot.pause()

        assert len(bridge.steer_calls) == 1
        entries = coordinator.drain_steering(handle_id)
        assert entries == [("user", "x" * MAX_STEERING_CHARS)]
