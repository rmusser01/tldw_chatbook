"""Native automatic model-loop cancellation retires its physical SQLite owner."""

import asyncio
import sqlite3
import threading

import pytest

from Tests.Chat.test_automatic_provider_budget import context_for, response
from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run
from Tests.Chat.test_console_fleet_wake import _settle
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_console_stop_silent_provider import _SilentGateway
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Agents.agent_models import WorkOrigin

stores = _stores_fixture

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


def _model_loop_handles(database):
    return [
        (connection, lease.resource_thread)
        for connection, lease in tuple(
            database._maintenance_participant.connections.items()
        )
        if lease.resource_thread.name == "console-agent-loop"
    ]


async def test_accepted_wake_kill_checks_and_retires_actual_model_loop_handle(
    tmp_path, monkeypatch
):
    gateway = _SilentGateway()
    bridge, db, store, session, assistant = _bridge_with_gateway(tmp_path, gateway)
    context = context_for(db)
    existing_runs = {row["id"] for row in db.list_runs("conversation")}

    def run():
        with context.scope():
            return _run(
                bridge,
                store,
                session,
                assistant,
                conversation_id="conversation",
                work_origin=WorkOrigin.AUTOMATIC,
                work_chain_id=context.chain_id,
            )

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        assert await _settle(gateway.entered.is_set), task
        assert await _settle(lambda: bool(_model_loop_handles(db)))
        handles = _model_loop_handles(db)
        assert len(handles) == 1
        connection, thread = handles[0]
        assert thread is not threading.current_thread() and thread.is_alive()
        assert not sqlite3.Connection.in_transaction.__get__(connection)
        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
        assert await _settle(gateway.closed.is_set)
        outcome = await asyncio.wait_for(task, 15)
        assert outcome.status == "cancelled"
        assert not gateway.released.is_set()
        assert not _model_loop_handles(db)
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(connection)
        assert (
            db.automatic_work.snapshot(context.chain_id).pause_reason
            == "autowake_disabled"
        )
        new_runs = [
            row
            for row in db.list_runs("conversation")
            if row["id"] not in existing_runs
        ]
        assert len(new_runs) == 1 and new_runs[0]["status"] == "cancelled"
    finally:
        gateway.released.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


async def test_accepted_goal_kill_retires_model_loop_runs_and_nested_goal_handles(
    stores, monkeypatch
):
    entered, release = threading.Event(), threading.Event()

    def provider(**_kwargs):
        entered.set()
        assert release.wait(15), "test provider was not physically released"
        return response()

    goal, _store, _session, _controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    db = stores[0]
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await _settle(entered.is_set)
        assert await _settle(lambda: bool(_model_loop_handles(db)))
        handles = _model_loop_handles(db)
        assert len(handles) == 1
        connection, thread = handles[0]
        assert thread is not threading.current_thread() and thread.is_alive()
        assert not sqlite3.Connection.in_transaction.__get__(connection)
        monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "false")
        assert await _settle(lambda: not _model_loop_handles(db))
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(connection)
        # The actual native cancellation and DB retirement happened while the
        # provider worker was still held; release it only for physical drain.
        assert not release.is_set()
        release.set()
        result = await asyncio.wait_for(task, 15)
        assert result.native_run_id and result.outcome.status == "cancelled", result
        assert len(calls) == 1
        snapshot = db.automatic_work.snapshot(goal.chain_id)
        assert snapshot.status == "review_required"
        assert snapshot.pause_reason == "usage_unknown"
        for database in (db, stores[1].db, stores[2].db):
            assert not _model_loop_handles(database)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


async def test_automatic_survivor_kill_retires_its_exact_child_loop_handle(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_agent_bridge import (
        SUBAGENT_PROMPT_PREFIX,
        _fence,
        _FleetTwoChildGateway,
        _join_fleet_threads,
    )

    release, closed = threading.Event(), threading.Event()

    class Gateway(_FleetTwoChildGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                self.child_calls += 1
                self.entered_event.set()
                try:
                    while not release.is_set():
                        await asyncio.sleep(0.01)
                    yield "must not require release for cancellation"
                finally:
                    closed.set()
                return
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk

    gateway = Gateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "automatic survivor"})],
            ["parent done"],
        ],
        child_result=[],
        gate=release,
        needed=1,
    )
    bridge, db, store, session, assistant = _bridge_with_gateway(tmp_path, gateway)
    context = context_for(db)

    def run():
        with context.scope():
            return _run(
                bridge,
                store,
                session,
                assistant,
                conversation_id="conversation",
                work_origin=WorkOrigin.AUTOMATIC,
                work_chain_id=context.chain_id,
            )

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        outcome = await asyncio.wait_for(task, 15)
        assert outcome.status == "done" and outcome.final_text == "parent done"
        assert await _settle(gateway.entered_event.is_set)
        live = bridge.fleet_snapshot("conversation")
        assert len(live) == 1 and live[0].status == "running"
        child = live[0]
        child_thread = next(
            thread
            for thread in threading.enumerate()
            if thread.name == "fleet-" + child.handle_id[:8]
        )
        loop_name = "child-loop-" + child.handle_id[:8]

        def child_handles():
            return [
                (connection, lease.resource_thread)
                for connection, lease in tuple(
                    db._maintenance_participant.connections.items()
                )
                if lease.resource_thread.name == loop_name
            ]

        assert await _settle(lambda: bool(child_handles()))
        handles = child_handles()
        assert len(handles) == 1
        connection, loop_thread = handles[0]
        assert loop_thread.is_alive() and child_thread.is_alive()
        assert not sqlite3.Connection.in_transaction.__get__(connection)
        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
        assert await _settle(closed.is_set)
        await asyncio.to_thread(child_thread.join, 5)
        assert not child_thread.is_alive() and not loop_thread.is_alive()
        assert not release.is_set() and gateway.child_calls == 1
        assert db.get_run(child.run_id)["status"] == "cancelled"
        assert db.get_run(child.run_id)["work_chain_id"] == context.chain_id
        assert not child_handles()
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(connection)
        assert (
            db.automatic_work.snapshot(context.chain_id).pause_reason
            == "autowake_disabled"
        )
    finally:
        bridge.cancel_all_subagents("conversation")
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(_join_fleet_threads)
        db.close()


@pytest.mark.parametrize("inline", [False, True])
async def test_native_coupled_or_inline_child_keeps_primary_stop(
    tmp_path, monkeypatch, inline
):
    from Tests.Chat.test_console_agent_bridge import (
        SUBAGENT_PROMPT_PREFIX,
        _fence,
        _FleetTwoChildGateway,
        _join_fleet_threads,
    )

    monkeypatch.setenv("TLDW_AGENTS_SUBAGENTS_OUTLIVE_TURN", "false")
    if inline:
        monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "1")
    cancel, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Gateway(_FleetTwoChildGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                self.child_calls += 1
                self.entered_event.set()
                try:
                    while not release.is_set():
                        await asyncio.sleep(0.01)
                    yield "must not need provider release"
                finally:
                    closed.set()
                return
            if len(self._parent) == 1:
                while not cancel.is_set():
                    await asyncio.sleep(0.01)
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk

    gateway = Gateway(
        parent_script=[
            [_fence("spawn_subagent", {"task": "coupled child"})],
            ["parent"],
        ],
        child_result=[],
        gate=release,
        needed=1,
    )
    bridge, db, store, session, assistant = _bridge_with_gateway(tmp_path, gateway)
    task = asyncio.create_task(
        asyncio.to_thread(
            _run,
            bridge,
            store,
            session,
            assistant,
            should_cancel=cancel.is_set,
        )
    )
    try:
        assert await _settle(gateway.entered_event.is_set)
        cancel.set()
        assert await _settle(closed.is_set)
        outcome = await asyncio.wait_for(task, 15)
        assert outcome.status == "cancelled"
        assert not release.is_set() and gateway.child_calls == 1
        children = db.list_runs("conv-1", agent_kind="subagent")
        assert len(children) == 1 and children[0]["status"] == "cancelled"
    finally:
        cancel.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(_join_fleet_threads)
        db.close()
