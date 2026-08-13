"""PR 3a-2 Task 1: pin `_child_run_scope`'s exit ordering (survey hazard 8).

Nothing else in Tests/ names `_child_run_scope` or `_live_child_counts`,
yet PR 3a-2's whole wake chain hangs off three facts of that seam:

1. The LAST child's scope exit fires the consumer exactly once -- not once
   per child -- on the child's own thread.
2. On the happy path the child's `agent_runs` row is already terminal when
   the scope exits (`_persist` runs inside the scope, at the end of
   `_run_one`), while the fleet coordinator STILL reports the child as
   running at that instant (`fleet.finish` lands only after the scope) --
   so a consumer of this signal must read the DB, never the coordinator.
3. On the raise path (`_run_one` unwinds past `_persist`) the row is NOT
   yet terminal at scope exit; it settles to a terminal status via
   `run_child`'s finally (`db.set_status`, first-writer-wins) strictly
   AFTER the signal. A wake consumer must tolerate a not-yet-terminal row
   at signal time on this path.

The tests instrument the consumer seam itself
(`ConsoleAgentBridge._close_post_turn_change_window`) by instance-attribute
replacement -- the same late-binding call `_child_run_scope` makes -- with
`change_tracker=None`, so the ONLY caller in these scenarios is the scope
exit (the turn's own finally close is gated on `change_handle is not None`,
console_agent_bridge.py `run_reply`).

Each test was mutation-tested (Task 1 report): firing per-child instead of
last-child fails test 1; removing `_persist` from `_run_one` (final row
state identical, ordering broken) fails test 2 alone; removing
`run_child`'s finally `set_status` fallback fails test 3.
"""
from __future__ import annotations

import threading

from Tests.Chat.test_console_agent_bridge import (
    _FleetTwoChildGateway,
    _fence,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _survivor_bridge(tmp_path, parent_script, needed):
    """A bridge + gated fleet gateway whose children outlive their turn."""
    gate = threading.Event()
    gateway = _FleetTwoChildGateway(
        parent_script=parent_script,
        child_result=["child answer"],
        gate=gate,
        needed=needed,
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )
    return gate, gateway, db, store, session, assistant.id, bridge


def _install_consumer_recorder(bridge, db, conversation_id):
    """Record every consumer fire: thread, live count, DB + coordinator state.

    Installed AFTER `run_reply` returns, so every recorded call is one the
    scope exit made for a surviving child.
    """
    calls: list[dict] = []
    original = bridge._close_post_turn_change_window

    def recording_close(conv_id):
        subagent_rows = [
            r for r in db.list_runs(conv_id) if r["agent_kind"] == "subagent"
        ]
        calls.append(
            {
                "thread": threading.current_thread().name,
                "live_count": bridge._live_child_count(conv_id),
                "db_statuses": sorted(r["status"] for r in subagent_rows),
                "coordinator_statuses": sorted(
                    h.status for h in bridge.fleet_snapshot(conv_id)
                ),
            }
        )
        original(conv_id)

    bridge._close_post_turn_change_window = recording_close
    return calls


def test_last_child_scope_exit_fires_the_consumer_exactly_once_on_the_childs_thread(
    tmp_path,
):
    """Two survivors settle; the consumer fires ONCE, from the last child's
    own `fleet-*` thread, with the live count already at zero."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "job A"})],
            [_fence("spawn_subagent", {"task": "job B"})],
            ["parent final"],
        ],
        needed=2,
    )
    try:
        outcome = _run(
            bridge, store, session, aid, conversation_id="conv-ordering"
        )
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the children never started"
        assert bridge._live_child_count("conv-ordering") == 2, (
            "precondition: two live survivors"
        )
        calls = _install_consumer_recorder(bridge, db, "conv-ordering")
    finally:
        gate.set()
    _join_fleet_threads()

    assert len(calls) == 1, (
        "the last-child-done consumer must fire exactly once for the "
        f"conversation, not per child: {calls}"
    )
    assert calls[0]["thread"].startswith("fleet-"), (
        "the signal fires on the child's own thread -- every consumer "
        f"inherits that thread context: {calls[0]}"
    )
    assert calls[0]["live_count"] == 0, calls[0]
    # Bookkeeping fully unwound: no leaked per-conversation count entry.
    assert bridge._live_child_counts == {}


def test_child_db_row_is_terminal_before_scope_exit_on_the_happy_path(tmp_path):
    """At the instant the consumer fires, the child's `agent_runs` row is
    already terminal (`_persist` sits inside the scope, at the end of
    `_run_one`) -- while the coordinator still says "running", because
    `fleet.finish` lands only after the scope exits. A wake consumer must
    therefore read the DB, never the coordinator, at signal time."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["parent final"],
        ],
        needed=1,
    )
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-hp")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
        calls = _install_consumer_recorder(bridge, db, "conv-hp")
    finally:
        gate.set()
    _join_fleet_threads()

    assert len(calls) == 1, calls
    assert calls[0]["db_statuses"] == ["done"], (
        "the DB row must already be terminal when the signal fires on the "
        f"happy path -- _persist runs inside the scope: {calls[0]}"
    )
    assert calls[0]["coordinator_statuses"] == ["running"], (
        "documented ordering hazard: the coordinator has NOT caught up at "
        "signal time (fleet.finish lands after the scope exit) -- a wake "
        f"consumer reading it here would see a live child: {calls[0]}"
    )


def test_raise_path_row_not_terminal_at_scope_exit_settles_via_run_child_finally(
    tmp_path, monkeypatch
):
    """`_run_one` unwinding past `_persist` (the setup-phase-exception class
    named in `run_child`'s finally) leaves the row `running` at signal
    time; `run_child`'s finally then settles it terminal. This IS the
    contract a wake consumer must tolerate: at signal time the row may not
    be terminal yet on this path."""
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
    try:
        outcome = _run(bridge, store, session, aid, conversation_id="conv-rp")
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
        calls = _install_consumer_recorder(bridge, db, "conv-rp")
    finally:
        gate.set()
    _join_fleet_threads()

    # The signal still fired exactly once, on the child's thread, DURING
    # the unwind -- before any terminal write existed.
    assert len(calls) == 1, calls
    assert calls[0]["thread"].startswith("fleet-"), calls[0]
    assert calls[0]["db_statuses"] == ["running"], (
        "on the raise path the row must NOT yet be terminal at signal "
        f"time -- that is the contract consumers tolerate: {calls[0]}"
    )

    # ... and `run_child`'s finally settles it, strictly after the signal.
    row = next(
        r for r in db.list_runs("conv-rp") if r["agent_kind"] == "subagent"
    )
    assert row["status"] == "error", (
        "the raise path's row must still settle terminal via run_child's "
        f"finally set_status fallback: {row['status']!r}"
    )
