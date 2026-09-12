"""Fleet PR 3b Task 5: `ConsoleAgentBridge.cancel_all_subagents`.

The panel's "Cancel all agents" seam. Contract (plan Task 5):

* The walk is the EXISTING one -- the current published service first,
  then the retained survivor owners -- executed once per live handle
  through the EXISTING per-handle `cancel_subagent` path, so approval-
  card revocation and the honest ownership refusals ride along for free.
  No second cancellation mechanism.
* Every LIVE handle of the conversation is cancelled, whatever turn's
  service owns it; terminal handles are not counted.
* The count of children actually cancelled is returned; 0 -- never a
  raise -- for an unknown conversation or an idle fleet.
* A cancel-all'd child is NOT retained (cancelled is never a retained
  status), pinned here at the coordinator seam; the refusal copy a later
  `send_to_agent` draws for it is pinned at the service seam in
  `Tests/Agents/test_fleet_stop_semantics.py` (the exact seam each layer
  owns -- Task 3's layered-guard lesson).

Fixtures are the real heavyweight bridge harness from
`test_console_agent_bridge.py` -- real `run_reply` turns, real gated
children on real threads -- because "cancel all" is precisely a claim
about which OWNERS the walk reaches, and only real turns produce the
published-service/retained-owner split.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from Tests.Agents.conftest import pin_agent_settings
from Tests.Chat.test_console_agent_bridge import (
    _bridge_with_gateway,
    _CancelDuringParentTurnGateway,
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
    _second_turn_message,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import RUN_DONE
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _pin_outlive_on(monkeypatch):
    """Deterministic config for the walk under test: fleet on, outlive on.

    The shipped defaults already say both; pinning keeps these tests
    independent of the environment's `[agents]` table.
    """
    pin_agent_settings(
        monkeypatch,
        **{
            agent_service.MAX_LIVE_SUBAGENTS_KEY: 3,
            agent_service.SUBAGENTS_OUTLIVE_TURN_KEY: True,
        },
    )


def test_cancel_all_returns_zero_for_an_unknown_conversation(tmp_path):
    """Nothing has ever run here -- 0, never a raise."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.cancel_all_subagents("never-seen-conversation") == 0


@pytest.mark.parametrize("create_first", [False, True])
def test_fence_fleet_blocks_a_late_child_reservation(
    tmp_path,
    monkeypatch,
    create_first,
):
    """A close fence wins whether the coordinator exists before or after it."""

    _pin_outlive_on(monkeypatch)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    existing = (
        bridge._conversation_fleet_coordinator("closing") if create_first else None
    )

    bridge.fence_fleet("closing", generation=1)
    fleet = bridge._conversation_fleet_coordinator("closing")

    assert fleet is not None
    if existing is not None:
        assert fleet is existing
    assert fleet.reserve("too late", None) is None


def test_fleet_reservation_publishes_lifecycle_activity_at_admission(
    tmp_path,
    monkeypatch,
):
    """Destructive consent is invalidated before the child thread starts."""

    _pin_outlive_on(monkeypatch)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    activity: list[str] = []
    bridge.on_fleet_activity("test", activity.append)

    fleet = bridge._conversation_fleet_coordinator("conversation")
    assert fleet is not None
    assert fleet.reserve("new child", None) is not None

    assert activity == ["conversation"]


def test_gracefully_drained_close_releases_fence_for_saved_conversation_reopen(
    tmp_path,
    monkeypatch,
):
    """A provisional close fence must not poison a later saved-chat resume."""

    _pin_outlive_on(monkeypatch)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    original = bridge._conversation_fleet_coordinator("saved")
    assert original is not None

    bridge.fence_fleet("saved", generation=1)
    assert original.reserve("blocked during close", None) is None

    assert bridge.release_fleet_fence("saved", generation=1) is True
    reopened = bridge._conversation_fleet_coordinator("saved")

    assert reopened is not None
    assert reopened is not original
    assert reopened.reserve("new incarnation", None) is not None


def test_latched_fleet_fence_cannot_be_replaced_by_a_later_close_generation(
    tmp_path,
    monkeypatch,
):
    """A timed-out incarnation's fence stays authoritative process-wide."""

    _pin_outlive_on(monkeypatch)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    bridge.fence_fleet("saved", generation=1)
    bridge.fence_fleet("saved", generation=2)

    assert bridge.release_fleet_fence("saved", generation=2) is False
    assert bridge._conversation_fleet_coordinator("saved").reserve("late", None) is None


def test_terminal_waiters_publish_only_after_stale_drain_consumers_are_fenced(
    tmp_path,
    monkeypatch,
):
    """A graceful close may release its fence only after drain fan-out returns."""

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    order: list[str] = []
    bridge.on_fleet_drained("order", lambda _event: order.append("fanout"))
    monkeypatch.setattr(
        bridge,
        "_notify_fleet_activity",
        lambda _conversation_id: order.append("terminal-waiter"),
    )
    bridge._unsettled_child_counts["saved"] = 1

    bridge._on_fleet_child_settled(
        "saved",
        "old-session",
        "old-assistant",
        None,
        RUN_DONE,
    )

    assert order == ["fanout", "terminal-waiter"]


@pytest.mark.asyncio
async def test_await_fleet_terminal_returns_immediately_for_idle_conversation(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    assert await bridge.await_fleet_terminal("idle") is True


@pytest.mark.asyncio
async def test_await_fleet_terminal_registers_before_snapshot_race(tmp_path, monkeypatch):
    """A drain fired by the first snapshot cannot strand the waiter."""

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snapshots = 0

    def _racing_snapshot(conversation_id: str):
        nonlocal snapshots
        snapshots += 1
        if snapshots == 1:
            bridge._notify_fleet_activity(conversation_id)
        return []

    monkeypatch.setattr(bridge, "fleet_snapshot", _racing_snapshot)

    assert await asyncio.wait_for(bridge.await_fleet_terminal("conv"), timeout=1)
    assert snapshots == 2, (
        "the activity callback must see the registered waiter before the "
        "await method completes its own post-registration snapshot"
    )


@pytest.mark.asyncio
async def test_await_fleet_terminal_resolves_after_real_coordinator_finish(tmp_path):
    """The child-settled hook observes the already-terminal fleet handle."""

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    fleet = bridge._conversation_fleet_coordinator("conv")
    assert fleet is not None
    handle = fleet.reserve("child", None)
    assert handle is not None

    class _PublishedService:
        def fleet_snapshot(self):
            return fleet.snapshot()

    bridge._fleet_services["conv"] = _PublishedService()
    waiter = asyncio.create_task(bridge.await_fleet_terminal("conv"))
    await asyncio.sleep(0)
    assert waiter.done() is False

    fleet.finish(handle.handle_id, RUN_DONE)
    bridge._on_fleet_child_settled(
        "conv",
        "session",
        "assistant",
        None,
        RUN_DONE,
    )

    assert await asyncio.wait_for(waiter, timeout=1) is True


def test_cancel_all_takes_the_published_services_child_and_a_retained_survivor(
    tmp_path, monkeypatch
):
    """The full walk, in one press: current service PLUS retained owner.

    Turn 1 leaves a survivor (its owner service is retained); turn 2 is
    in flight with its own live child (its service is the published one)
    when Cancel-all fires. One call must stop BOTH -- the count says 2 --
    which is exactly what a single-tier lookup cannot do: the published
    service can SEE the survivor in the shared coordinator but holds none
    of its cancel Events, and `AgentService.cancel_subagent`'s ownership
    refusal forces the walk through to the retained owner.
    """
    _pin_outlive_on(monkeypatch)
    gate = threading.Event()
    counts: list[int] = []
    gateway = _CancelDuringParentTurnGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "job two"})],
            ["turn 2 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
        # Fires as turn 2's FINAL parent call starts: turn 2's own child
        # is reserved and registered (reserve/Event wiring is synchronous
        # inside the spawn dispatch), turn 1's survivor is still gated --
        # the one moment both ownership tiers hold a live child each.
        on_parent_turn=4,
        callback=lambda: counts.append(bridge.cancel_all_subagents("conv-1")),
    )
    bridge, db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)
    try:
        outcome_1 = _run(bridge, store, session, assistant_id)
        assert outcome_1.status == "done"
        assert gateway.entered_event.wait(5), "turn 1's child never started"
        live_before = bridge.fleet_snapshot("conv-1")
        assert [h.status for h in live_before] == ["running"]

        second = _second_turn_message(store, session)
        outcome_2 = _run(bridge, store, session, second)
        assert outcome_2.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    assert counts == [2], (
        f"Cancel-all reported {counts}; the walk must take the published "
        f"service's child AND the retained owner's survivor"
    )
    children = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(children) == 2
    assert {row["status"] for row in children} == {"cancelled"}, children
    # Nothing left live anywhere the panel could see.
    assert bridge.fleet_snapshot("conv-1") == []


def test_cancel_all_reaches_a_survivor_with_no_run_in_flight(tmp_path, monkeypatch):
    """The retained-owner-only tier: no published service exists at all.

    The turn is over, so `_fleet_services` has no entry -- the ONLY way
    to this survivor's cancel Event is the retained-owner walk. This is
    the owner test for the mandated mutation "cancel-all skips retained
    owners": with that walk gone the count reads 0 and the child keeps
    running.

    Also pins the retention interaction at the coordinator seam: the
    cancel-all'd child is NOT retained (cancelled is never a retained
    status), so a later continuation attempt has nothing to resume --
    and a second Cancel-all press honestly reports 0.
    """
    _pin_outlive_on(monkeypatch)
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn 1 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    bridge, db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)
    try:
        outcome = _run(bridge, store, session, assistant_id)
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
        live = bridge.fleet_snapshot("conv-1")
        assert [h.status for h in live] == ["running"]
        handle_id = live[0].handle_id

        assert bridge.cancel_all_subagents("conv-1") == 1
    finally:
        gate.set()
    _join_fleet_threads()

    child = next(
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    )
    assert child["status"] == "cancelled", child["status"]
    assert bridge.fleet_snapshot("conv-1") == []
    # Retention interaction: cancelled is never retained, so the child
    # cannot be resumed (the refusal copy itself is pinned at the
    # service seam in test_fleet_stop_semantics).
    coordinator = bridge._fleet_coordinators.get("conv-1")
    assert coordinator is not None
    assert coordinator.get_retained(handle_id) is None
    # Idempotent: nothing live is left to count.
    assert bridge.cancel_all_subagents("conv-1") == 0


def test_cancel_all_reuses_the_per_handle_cancel_path_one_call_per_live_handle(
    tmp_path, monkeypatch
):
    """No second mechanism -- the layered-guard lesson applied.

    The revocation guarantee (cancel -> approval-card revoke) lives in
    the per-handle path (`cancel_subagent` -> `_cancel_fleet_handles` ->
    `_revoke_handle_approvals`), so Cancel-all must go THROUGH that
    method, once per live handle -- asserted at the method seam itself
    with a delegating spy, not via the shared outcome (which a parallel
    second mechanism could also produce).
    """
    _pin_outlive_on(monkeypatch)
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "job one"})],  # primary turn 1
            [_fence("spawn_subagent", {"task": "job two"})],  # primary turn 2
            ["turn 1 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=2,
    )
    bridge, db, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)
    per_handle_calls: list[str] = []
    real_cancel = bridge.cancel_subagent

    def spying_cancel(conversation_id: str, handle_id: str) -> bool:
        per_handle_calls.append(handle_id)
        return real_cancel(conversation_id, handle_id)

    try:
        outcome = _run(bridge, store, session, assistant_id)
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the children never started"
        live_ids = {h.handle_id for h in bridge.fleet_snapshot("conv-1")}
        assert len(live_ids) == 2

        monkeypatch.setattr(bridge, "cancel_subagent", spying_cancel)
        assert bridge.cancel_all_subagents("conv-1") == 2
    finally:
        gate.set()
    _join_fleet_threads()

    assert sorted(per_handle_calls) == sorted(live_ids), (
        "Cancel-all must delegate exactly once per live handle to the "
        "per-handle revocation path"
    )
