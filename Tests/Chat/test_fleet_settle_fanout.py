"""PR 3a-2 Task 2: the last-child-settled fan-out (`FleetDrainFanout`).

Task 1 pinned the EXISTING hook's ordering (`_child_run_scope` exit -->
`_close_post_turn_change_window`): it fires before `fleet.finish`, and on
the raise path before the child's `agent_runs` row is terminal. A wake or
usage consumer needs the opposite guarantee -- "the last child is DONE and
its row is terminal" -- and cannot get it from that hook even with a
bounded DB wait, because on the raise path the terminal write happens
LATER ON THE SAME THREAD (`run_child`'s finally): a consumer blocking at
scope exit would wait on a write its own thread performs after it returns.

So Task 2 added a second, later hook (`AgentService(on_child_settled=...)`,
last act of `run_child`'s finally, after `fleet.finish` and the terminal-
status fallback) feeding one bridge-lifetime fan-out. These tests pin:

1. the drain fires exactly once, on the last child's thread, with every
   settled row terminal AND the coordinator finished at fire time;
2. the row is terminal at fire time ON THE RAISE PATH TOO -- the property
   the old hook cannot offer;
3. the change window closes strictly BEFORE the drain fires (consumers may
   read what it wrote);
4. consumers run in registration order and one raising does not starve
   the rest;
5. a raising settle hook never reaches the child thread's excepthook
   (the `AgentService` call-site wrap -- the second isolation layer);
6. re-registering a name replaces in place (no duplicate accumulation);
7. registration is bridge-lifetime: a consumer registered once fires once
   per drain across turns, each event carrying its OWN turn's identity.

Each test is mutation-tested (Task 2 report): every listed mutation makes
the named test fail on its own assertion.
"""

from __future__ import annotations

import threading

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    FleetDrained,
    FleetDrainFanout,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _install_drain_recorder(bridge, db, name="test-recorder"):
    """Register a real fan-out consumer that records fire-time state.

    Unlike Task 1's recorder (installed after `run_reply` returned, by
    instance-attribute replacement), this registers through the public
    seam BEFORE the turn runs -- the honest wiring a Task 3-5 consumer
    will use.
    """
    fires: list[dict] = []

    def recorder(event: FleetDrained) -> None:
        subagent_rows = [
            r
            for r in db.list_runs(event.conversation_id)
            if r["agent_kind"] == "subagent"
        ]
        fires.append(
            {
                "thread": threading.current_thread().name,
                "event": event,
                "db_statuses": sorted(r["status"] for r in subagent_rows),
                "db_run_ids": sorted(r["id"] for r in subagent_rows),
                "coordinator_statuses": sorted(
                    h.status for h in bridge.fleet_snapshot(event.conversation_id)
                ),
            }
        )

    bridge.on_fleet_drained(name, recorder)
    return fires


def test_drain_fires_once_with_terminal_rows_and_a_finished_coordinator(
    tmp_path,
):
    """Two survivors settle; the drain fires ONCE, on a child's own
    thread, and at fire time every row is terminal and no coordinator
    handle still says "running" -- the exact pair the earlier hook cannot
    offer (Task 1 pinned it firing with the coordinator still running).
    The event carries both children with their run ids and this turn's
    session + originating-assistant identity."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "job A"})],
            [_fence("spawn_subagent", {"task": "job B"})],
            ["parent final"],
        ],
        needed=2,
    )
    fires = _install_drain_recorder(bridge, db)
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-drain")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the children never started"
        assert fires == [], "no drain may fire while children are live"
    finally:
        gate.set()
    _join_fleet_threads()

    assert len(fires) == 1, (
        f"the drain must fire exactly once per drain, not per child: {fires}"
    )
    fire = fires[0]
    assert fire["thread"].startswith("fleet-"), fire
    assert fire["db_statuses"] == ["done", "done"], (
        f"every settled row must be terminal at fire time on the happy path: {fire}"
    )
    assert all(s != "running" for s in fire["coordinator_statuses"]), (
        "the drain fires strictly after fleet.finish -- unlike the "
        "scope-exit hook, no handle may still be running at fire time: "
        f"{fire}"
    )
    event = fire["event"]
    assert event.conversation_id == "conv-drain"
    assert len(event.children) == 2, event
    assert sorted(c.run_id for c in event.children) == fire["db_run_ids"], (
        "the event must carry the settled children's real run row ids -- "
        f"the wake reads results from those rows: {event}"
    )
    assert all(c.status == "done" for c in event.children), event
    assert all(c.session_id == session.id for c in event.children), event
    assert all(c.assistant_message_id == aid for c in event.children), event
    # Bookkeeping fully unwound: nothing leaks for the next drain.
    assert bridge._unsettled_child_counts == {}
    assert bridge._settling_children == {}


def test_drain_row_is_terminal_at_fire_time_on_the_raise_path_too(
    tmp_path, monkeypatch
):
    """The property the old hook cannot offer: with `_run_one` unwinding
    past `_persist` (the setup-phase-exception class), the scope-exit
    hook fires with the row still `running` (Task 1's pin) -- but the
    DRAIN still fires with the row already terminal, because it runs
    after `run_child`'s finally `set_status` fallback."""
    original_persist = AgentService._persist

    def raising_persist(self, run_id, outcome):
        if threading.current_thread().name.startswith("fleet-"):
            raise RuntimeError("induced: _run_one unwinds past _persist")
        return original_persist(self, run_id, outcome)

    monkeypatch.setattr(AgentService, "_persist", raising_persist)

    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "doomed job"})],
            ["parent final"],
        ],
        needed=1,
    )
    fires = _install_drain_recorder(bridge, db)
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-rp2")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
    finally:
        gate.set()
    _join_fleet_threads()

    assert len(fires) == 1, fires
    fire = fires[0]
    assert fire["db_statuses"] == ["error"], (
        "the drain must not fire until the raise-path row has settled "
        f"terminal via run_child's finally: {fire}"
    )
    event = fire["event"]
    assert len(event.children) == 1, event
    assert event.children[0].status == "error", event
    assert event.children[0].run_id == fire["db_run_ids"][0], event


def test_change_window_close_completes_before_the_drain_fires(tmp_path):
    """Deterministic cross-hook order: the change window (the earlier
    hook's consumer, which existed first) closes strictly before the
    drain fires, so drain consumers may read what it wrote. Structural on
    the last child's own thread: scope exit -> close -> finally -> settle
    -> drain."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "ordered job"})],
            ["parent final"],
        ],
        needed=1,
    )
    order: list[str] = []
    original_close = bridge._close_post_turn_change_window

    def recording_close(conv_id):
        original_close(conv_id)
        order.append("change-window-closed")

    bridge._close_post_turn_change_window = recording_close
    bridge.on_fleet_drained("order-probe", lambda event: order.append("drain"))
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-order")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
    finally:
        gate.set()
    _join_fleet_threads()

    assert order == ["change-window-closed", "drain"], (
        f"the change window must have fully closed before the drain fires: {order}"
    )


def test_a_raising_consumer_neither_starves_later_consumers_nor_reorders(
    tmp_path,
):
    """Isolation + order in one scenario: three consumers registered
    a-then-b-then-c; a raises; b and c still run, in registration order."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "isolated job"})],
            ["parent final"],
        ],
        needed=1,
    )
    calls: list[str] = []

    def raising_a(event):
        calls.append("a")
        raise RuntimeError("induced: consumer a is broken")

    bridge.on_fleet_drained("a", raising_a)
    bridge.on_fleet_drained("b", lambda event: calls.append("b"))
    bridge.on_fleet_drained("c", lambda event: calls.append("c"))
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-iso")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
    finally:
        gate.set()
    _join_fleet_threads()

    assert calls == ["a", "b", "c"], (
        "a raising consumer must not starve those registered after it, "
        f"and delivery order is registration order: {calls}"
    )


def test_a_raising_settle_hook_never_reaches_the_child_threads_excepthook(
    tmp_path, monkeypatch
):
    """The second isolation layer, targeted directly: even if the
    bridge's whole settle hook raises (a seam bug, not a consumer bug --
    the per-consumer catch never runs), nothing propagates out of
    `run_child`'s finally into the thread's default excepthook, and the
    child's terminal row is untouched (it was written BEFORE the hook)."""
    seen: list[str] = []

    def recording_excepthook(args):
        seen.append(f"{args.thread.name}: {args.exc_type.__name__}")

    monkeypatch.setattr(threading, "excepthook", recording_excepthook)

    def raising_hook(
        self, conversation_id, session_id, assistant_message_id, run_id, status
    ):
        raise RuntimeError("induced: the settle hook itself is broken")

    monkeypatch.setattr(ConsoleAgentBridge, "_on_fleet_child_settled", raising_hook)

    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "hooked job"})],
            ["parent final"],
        ],
        needed=1,
    )
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-hook")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
    finally:
        gate.set()
    _join_fleet_threads()

    fleet_deaths = [entry for entry in seen if entry.startswith("fleet-")]
    assert fleet_deaths == [], (
        "a raising settle hook must be contained inside run_child's "
        f"finally, never killing the child's thread: {fleet_deaths}"
    )
    assert not [t for t in threading.enumerate() if t.name.startswith("fleet-")], (
        "the child thread must have finished cleanly"
    )
    row = next(r for r in db.list_runs("conv-hook") if r["agent_kind"] == "subagent")
    assert row["status"] == "done", row


def test_reregistering_a_name_replaces_in_place_keeping_its_order_slot():
    """Belt to the bridge-lifetime braces: even a misplaced repeated
    registration cannot accumulate -- same name replaces, same slot."""
    fanout = FleetDrainFanout()
    calls: list[str] = []
    fanout.register("w", lambda event: calls.append("w-old"))
    fanout.register("x", lambda event: calls.append("x"))
    fanout.register("w", lambda event: calls.append("w-new"))
    fanout.fire(FleetDrained(conversation_id="c", children=()))
    assert calls == ["w-new", "x"], (
        "re-registering a name must replace in place (old consumer gone, "
        f"order slot kept): {calls}"
    )


def test_a_consumer_registered_once_fires_once_per_drain_across_turns(
    tmp_path,
):
    """Registration is bridge-lifetime and exactly-once: one consumer,
    registered once before turn 1, sees exactly one event per drain
    across two turns on the SAME bridge -- and each event carries its own
    turn's identity (turn 2's originating assistant message, not turn
    1's), because identity is bound per turn into the settle hook while
    the registry is never touched per turn."""
    gate = threading.Event()
    gate.set()  # children need not outlive their turns for a drain
    gateway = _FleetTwoChildGateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "turn-1 job"})],
            ["turn 1 final"],
            [_fence("spawn_subagent", {"task": "turn-2 job"})],
            ["turn 2 final"],
        ],
        child_result=["child answer"],
        gate=gate,
        needed=1,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    aid_1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    events: list[FleetDrained] = []
    bridge.on_fleet_drained("once", events.append)

    outcome = _run(bridge, store, session, aid_1, conversation_id="conv-turns")
    assert outcome.status == "done"
    _join_fleet_threads()
    assert len(events) == 1, events

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="again")
    aid_2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
    outcome = _run(bridge, store, session, aid_2, conversation_id="conv-turns")
    assert outcome.status == "done"
    _join_fleet_threads()

    assert len(events) == 2, (
        "one registration, two drains, exactly two fires -- a per-turn "
        "re-registration (or a per-turn registry) breaks this: "
        f"{len(events)} events"
    )
    first, second = events
    assert [len(first.children), len(second.children)] == [1, 1]
    assert first.children[0].run_id != second.children[0].run_id
    assert first.children[0].assistant_message_id == aid_1, first
    assert second.children[0].assistant_message_id == aid_2, (
        "each drain event must carry its OWN turn's originating "
        f"assistant message: {second}"
    )
