"""Fleet PR 3b Task 5: the panel's "Cancel all agents" affordance.

With Stop decoupled from the children (a stopped turn's survivors keep
working -- `Tests/Agents/test_fleet_stop_semantics.py`), the panel owes
the user a whole-fleet kill switch. The affordance lives in the Agent
rail section, is offered ONLY while live rows exist (a fleet of finished
children has nothing to cancel -- a dead button would be a lie), and
routes through `ConsoleAgentBridge.cancel_all_subagents` (whose walk and
per-handle-path reuse are pinned in
`Tests/Chat/test_console_agent_bridge_cancel_all.py`).

Painted-frame assertions throughout (`_assert_painted_at_own_region`,
the compositor's own hit-test), including the does-NOT-paint case --
this programme has three times had "rendered" assertions that only
proved data arrived.
"""

from __future__ import annotations

import time

import pytest
from textual.widgets import Button

from Tests.UI.test_console_agent_steering_bar import (
    _SteeringFleetBridge,
    _drill_in,
    _fleet_with_finished_child,
    _fleet_with_live_child,
)
from Tests.UI.test_console_fleet_panel import (
    _scroll_into_view,
    _setup_console,
)
from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_agent_steering_bar import (
    ConsoleAgentSteeringBar,
)

_SIZE = (180, 48)
CONV = "conv-A"

_BUTTON = "#console-agent-cancel-all"
_BAR = "#console-agent-steering-bar"


class _CancelAllFleetBridge(_SteeringFleetBridge):
    """The steering suite's fake bridge plus the Cancel-all seam.

    `cancel_all_subagents` records the call and answers with the number
    of live handles -- the wiring under test here is the SCREEN's (does
    the affordance paint honestly, does pressing it reach the bridge
    seam once); the bridge method's own walk is pinned by execution in
    the Chat-level suite.
    """

    def __init__(self, coordinator, conversation_id: str = CONV):
        super().__init__(coordinator, conversation_id)
        self.cancel_all_calls: list[str] = []

    def cancel_all_subagents(self, conversation_id: str) -> int:
        self.cancel_all_calls.append(conversation_id)
        live = [
            h
            for h in self.coordinator.snapshot()
            if h.status not in ("done", "error", "cancelled", "superseded")
        ]
        for handle in live:
            self.coordinator.finish(handle.handle_id, "cancelled")
        return len(live)


# -- visibility: live rows only ------------------------------------------


@pytest.mark.asyncio
async def test_cancel_all_paints_while_a_live_child_exists():
    """With a live child on the fleet, the affordance is genuinely painted
    at its own region -- not merely present in the DOM."""
    coordinator, _handle_id = _fleet_with_live_child()
    bridge = _CancelAllFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _scroll_into_view(pilot, console, _BUTTON)

        button = console.query_one(_BUTTON, Button)
        _assert_widget_and_ancestors_displayed(button)
        _assert_painted_at_own_region(host, button)


@pytest.mark.asyncio
async def test_cancel_all_does_not_paint_with_no_live_rows():
    """A fleet of FINISHED children has nothing to cancel: the affordance
    must not paint even though the section still shows their rows --
    offering a kill switch for work that already ended would be a lie."""
    coordinator, _handle_id = _fleet_with_finished_child()
    bridge = _CancelAllFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)

        button = console.query_one(_BUTTON, Button)
        assert not button.display


@pytest.mark.asyncio
async def test_cancel_all_does_not_paint_with_no_fleet_at_all():
    """No coordinator handles at all (never spawned): hidden likewise."""
    coordinator = None
    # An empty coordinator: reserve nothing.
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator

    coordinator = FleetCoordinator(max_live=4, clock=time.monotonic)
    bridge = _CancelAllFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)

        button = console.query_one(_BUTTON, Button)
        assert not button.display


# -- press: one call to the bridge seam ----------------------------------


@pytest.mark.asyncio
async def test_pressing_cancel_all_reaches_the_bridge_once_and_clears_the_affordance():
    """A REAL click routes to `cancel_all_subagents` exactly once, and the
    next sync -- reading the now-terminal handles -- hides the button
    again (nothing left to cancel)."""
    coordinator, _handle_id = _fleet_with_live_child()
    bridge = _CancelAllFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _scroll_into_view(pilot, console, _BUTTON)

        await pilot.click(_BUTTON)
        await pilot.pause()

        assert bridge.cancel_all_calls == [CONV]
        # The fake finished every live handle; the resync the handler
        # requested must hide the affordance again.
        console._sync_console_agent_section()
        await pilot.pause()
        assert not console.query_one(_BUTTON, Button).display


# -- Task 3's cross-pin: the steering bar and Cancel-all move together ----


@pytest.mark.asyncio
async def test_cancel_all_and_the_steering_bar_hide_together_after_cancel_all():
    """Drilled into a LIVE survivor, both the steering bar and Cancel-all
    are painted; after Cancel-all finishes the fleet, the SAME sync hides
    both -- the two affordances must never disagree about whether live
    work exists (Task 3's landing report asked Task 5 to pin exactly
    this)."""
    coordinator, handle_id = _fleet_with_live_child()
    bridge = _CancelAllFleetBridge(coordinator)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _setup_console(pilot, host, bridge)
        await _drill_in(pilot, console, "run-1")
        await _scroll_into_view(pilot, console, _BAR)

        bar = console.query_one(_BAR, ConsoleAgentSteeringBar)
        assert bar.display
        button = console.query_one(_BUTTON, Button)
        _assert_widget_and_ancestors_displayed(button)
        _assert_painted_at_own_region(host, button)

        # The user cancels the whole fleet (bridge-level; the press path
        # is covered above) -- the child goes terminal...
        assert bridge.cancel_all_subagents(CONV) == 1
        console._sync_console_agent_section()
        await pilot.pause()

        # ... and BOTH affordances left the live surface on the same sync.
        assert not console.query_one(_BUTTON, Button).display
        assert not console.query_one(_BAR, ConsoleAgentSteeringBar).display
