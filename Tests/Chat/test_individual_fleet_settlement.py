"""ADR-135: each persisted child settles before a slow sibling drains.

These use the real bridge, AgentService, fleet threads and SQLite. Only the
external provider is scripted; its independent gates make the drain barrier
observable without sleeps or manually invoking settlement callbacks.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Chat import console_agent_bridge as bridge_module
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class _GatedSiblingsGateway(_FleetTwoChildGateway):
    def __init__(self, *, fail_fast=False):
        self.gates = {task: threading.Event() for task in ("fast", "slow")}
        self.entered = {task: threading.Event() for task in self.gates}
        self.fail_fast = fail_fast
        super().__init__(
            parent_script=[
                [_fence("spawn_subagent", {"task": "fast"})],
                [_fence("spawn_subagent", {"task": "slow"})],
                ["parent final"],
            ],
            child_result=[],
            gate=threading.Event(),
        )

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if not system.startswith(SUBAGENT_PROMPT_PREFIX):
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk
            return
        task = messages[-1]["content"]
        assert task in self.gates, task
        self.entered[task].set()
        await asyncio.get_running_loop().run_in_executor(None, self.gates[task].wait)
        if task == "fast" and self.fail_fast:
            raise RuntimeError("scripted provider failure")
        yield f"{task} answer"

    def release_all(self):
        for gate in self.gates.values():
            gate.set()


def _bridge(tmp_path, *, fail_fast=False, db_class=AgentRunsDB):
    gateway = _GatedSiblingsGateway(fail_fast=fail_fast)
    db = db_class(tmp_path / "runs.db", client_id="individual-settlement-test")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant_id = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    return gateway, db, store, session, assistant_id, bridge


@pytest.mark.parametrize("failure", [None, "provider", "result_write"])
def test_individual_event_has_terminal_row_before_slow_sibling_drains(
    tmp_path, failure
):
    """Moving delivery to scope exit or last-child drain breaks this contract."""

    class FailedResultWriteDB(AgentRunsDB):
        def set_status(self, run_id, status, result=None, **kwargs):
            if result == "fast answer":
                raise sqlite3.OperationalError("scripted result write failure")
            return super().set_status(run_id, status, result, **kwargs)

    gateway, db, store, session, assistant_id, bridge = _bridge(
        tmp_path,
        fail_fast=failure == "provider",
        db_class=FailedResultWriteDB if failure == "result_write" else AgentRunsDB,
    )
    events = []
    drains = []
    notified = threading.Event()

    def record(event):
        # A nonblocking acquisition makes lock-order regressions fail cleanly.
        outside_lock = bridge._change_window_lock.acquire(blocking=False)
        if outside_lock:
            bridge._change_window_lock.release()
        row = db.get_run(event.child.run_id)
        events.append((event, row, outside_lock, threading.current_thread().name))
        notified.set()

    bridge.on_fleet_child_settled("recorder", record)
    bridge.on_fleet_drained("drain", drains.append)
    try:
        assert (
            _run(
                bridge, store, session, assistant_id, conversation_id="conversation"
            ).status
            == "done"
        )
        assert all(entered.wait(5) for entered in gateway.entered.values())
        gateway.gates["fast"].set()
        assert notified.wait(5), "fast child must notify while slow remains gated"
        assert len(events) == 1
        event, row, outside_lock, thread = events[0]
        assert isinstance(event, bridge_module.FleetChildSettled)
        assert event.conversation_id == "conversation"
        assert event.child.session_id == session.id
        assert event.child.assistant_message_id == assistant_id
        assert event.child.settled_after_turn is True
        assert event.child.run_id == row["id"]
        assert event.child.status == row["status"] == ("error" if failure else "done")
        if not failure:
            assert row["result"] == "fast answer"
        assert outside_lock, "consumers must not run under the bridge lock"
        assert thread.startswith("fleet-")
        assert bridge.has_unsettled_children("conversation")
        assert drains == [], "individual completion must not reconcile final usage"
        assert any(
            r["task"] == "slow" and r["status"] == "running"
            for r in db.list_runs("conversation")
        )
    finally:
        gateway.release_all()
        _join_fleet_threads()

    assert len(events) == 2
    assert len(drains) == 1
    assert drains[0].children == tuple(item[0].child for item in events)
    assert len({item[0].child.run_id for item in events}) == 2
    assert not bridge.has_unsettled_children("conversation")


def test_replacing_and_raising_consumers_preserve_siblings_and_final_drain(tmp_path):
    """Duplicate registration or uncontained failures must not multiply/drop events."""
    gateway, _db, store, session, assistant_id, bridge = _bridge(tmp_path)
    calls = []
    drains = []

    def raising(event):
        calls.append((event.child.run_id, "raising"))
        raise RuntimeError("scripted notification failure")

    bridge.on_fleet_child_settled(
        "first", lambda e: calls.append((e.child.run_id, "old"))
    )
    bridge.on_fleet_child_settled("raising", raising)
    bridge.on_fleet_child_settled(
        "last", lambda e: calls.append((e.child.run_id, "last"))
    )
    bridge.on_fleet_child_settled(
        "first", lambda e: calls.append((e.child.run_id, "new"))
    )
    bridge.on_fleet_drained("drain", drains.append)
    try:
        assert (
            _run(
                bridge, store, session, assistant_id, conversation_id="conversation"
            ).status
            == "done"
        )
        assert all(entered.wait(5) for entered in gateway.entered.values())
    finally:
        gateway.release_all()
        _join_fleet_threads()

    assert len(drains) == 1
    assert len(drains[0].children) == 2
    for child in drains[0].children:
        assert [name for run_id, name in calls if run_id == child.run_id] == [
            "new",
            "raising",
            "last",
        ]
    assert len(calls) == 6


@pytest.mark.parametrize("failure", ["create", "terminal_write", "terminal_read"])
def test_unavailable_durable_row_suppresses_individual_event_but_preserves_drain(
    tmp_path, failure
):
    """A real child must never authorize wake intake from a failed DB boundary."""

    class UnavailableChildRowDB(AgentRunsDB):
        # Faults live at the storage boundary. Run creation, execution, settlement
        # and every successful SQLite operation remain production behavior.
        def create_run(self, **kwargs):
            if failure == "create" and kwargs.get("agent_kind") == "subagent":
                raise sqlite3.OperationalError("scripted create failure")
            return super().create_run(**kwargs)

        def set_status(self, run_id, status, result=None, **kwargs):
            row = super().get_run(run_id)
            if failure == "terminal_write" and row["agent_kind"] == "subagent":
                raise sqlite3.OperationalError("scripted terminal write failure")
            return super().set_status(run_id, status, result, **kwargs)

        def get_run_fresh(self, run_id):
            row = super().get_run_fresh(run_id)
            if failure == "terminal_read" and row["agent_kind"] == "subagent":
                raise sqlite3.OperationalError("scripted terminal read failure")
            return row

    gateway, db, store, session, assistant_id, bridge = _bridge(
        tmp_path, db_class=UnavailableChildRowDB
    )
    events = []
    drains = []
    bridge.on_fleet_child_settled("recorder", events.append)
    bridge.on_fleet_drained("drain", drains.append)
    try:
        assert (
            _run(
                bridge, store, session, assistant_id, conversation_id="conversation"
            ).status
            == "done"
        )
    finally:
        gateway.release_all()
        _join_fleet_threads()

    assert events == [], "no durable terminal evidence means no individual wake"
    assert sum(len(event.children) for event in drains) == 2
    rows = [
        row for row in db.list_runs("conversation") if row["agent_kind"] == "subagent"
    ]
    if failure == "create":
        assert rows == []
        assert all(child.run_id is None for event in drains for child in event.children)
    elif failure == "terminal_write":
        assert [row["status"] for row in rows] == ["running", "running"]
    else:
        assert [row["status"] for row in rows] == ["done", "done"]


def test_registration_survives_turns_and_keeps_within_turn_classification(tmp_path):
    """Rebuilding the consumer list or reclassifying at drain loses turn identity."""
    gateway, _db, store, session, assistant_id, bridge = _bridge(tmp_path)
    events = []
    bridge.on_fleet_child_settled("once", events.append)
    gateway.release_all()
    assistant_ids = [assistant_id]
    for turn in range(2):
        if turn:
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="again"
            )
            assistant_ids.append(
                store.append_message(
                    session.id, role=ConsoleMessageRole.ASSISTANT, content=""
                ).id
            )
        gateway._parent = [
            [_fence("spawn_subagent", {"task": "fast"})],
            [_fence("wait_agents", {})],
            ["parent final"],
        ]
        try:
            assert (
                _run(
                    bridge,
                    store,
                    session,
                    assistant_ids[-1],
                    conversation_id="conversation",
                ).status
                == "done"
            )
        finally:
            _join_fleet_threads()

    assert len(events) == 2
    assert [event.child.assistant_message_id for event in events] == assistant_ids
    assert all(not event.child.settled_after_turn for event in events)
    assert events[0].child.run_id != events[1].child.run_id


def test_individual_status_respects_earlier_durable_cancellation(tmp_path):
    """A late successful provider reply must not undo first-writer cancellation."""
    gateway, db, store, session, assistant_id, bridge = _bridge(tmp_path)
    events = []
    bridge.on_fleet_child_settled("recorder", events.append)
    try:
        assert (
            _run(
                bridge, store, session, assistant_id, conversation_id="conversation"
            ).status
            == "done"
        )
        assert all(entered.wait(5) for entered in gateway.entered.values())
        fast_row = next(
            row for row in db.list_runs("conversation") if row["task"] == "fast"
        )
        # This is the real DB transition used when an abandoned child has
        # exceeded its join grace; its still-pending provider can return later.
        assert db.set_status(fast_row["id"], "cancelled")
    finally:
        gateway.release_all()
        _join_fleet_threads()

    fast_event = next(event for event in events if event.child.run_id == fast_row["id"])
    assert db.get_run(fast_row["id"])["status"] == "cancelled"
    assert fast_event.child.status == "cancelled"
