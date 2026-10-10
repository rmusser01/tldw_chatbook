"""Original subagent-count worker must retain its actual database callback."""

import asyncio
import inspect
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen
from textual.worker import WorkerState

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_original_subagent_count_worker_retires_before_host_drain(
    request, tmp_path
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
    from tldw_chatbook.DB import base_db

    database = AgentRunsDB(tmp_path / "runs.db")
    conversation_id = "original-count-conversation"
    row_ids = frozenset({conversation_id})
    host, screen = App(), Screen()
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    query, definition, query_key, stock_query = (
        ConsoleAgentController._subagent_count_query(bridge, database)
    )
    assert stock_query and query.__self__ is database
    authority = object()
    live_token = ConsoleAgentController._subagent_count_live_token(
        bridge, row_ids, stock_query=stock_query
    )
    state = {
        "key": (bridge, row_ids, live_token, authority, query_key, database),
        "query": query,
        "query_definition": definition,
        "stock_query": stock_query,
        "database": authority,
        "pending": True,
        "values": {},
        "at": 0.0,
    }
    published = []
    controller = SimpleNamespace(
        _console_subagent_counts_read={row_ids: state},
        _console_agent_bridge=bridge,
        app_instance=SimpleNamespace(chachanotes_db=authority),
        _subagent_count_query=ConsoleAgentController._subagent_count_query,
        _subagent_count_live_token=ConsoleAgentController._subagent_count_live_token,
        run_worker=lambda *args, **kwargs: published.append((args, kwargs)),
    )
    original = ConsoleAgentController._load_subagent_counts
    reader = AgentRunsDB.count_subagents_by_conversation
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (ConsoleAgentController, "_load_subagent_counts"),
        (ConsoleAgentController, "_console_subagent_counts_for_rows"),
        (ConsoleAgentController, "_subagent_count_query"),
        (ConsoleAgentController, "_subagent_count_live_token"),
        (AgentRunsDB, "count_subagents_by_conversation"),
        (base_db, "run_owned_db_call"),
        (_WorkItem, "run"),
    ):
        callback = getattr(owner, name)
        monitor._pin(callback)
        monitor.slots.append((owner, name, inspect.getattr_static(owner, name)))
    invoke_code = next(
        code
        for code in base_db.run_owned_db_call.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "invoke"
    )
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    worker = drain = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold(code, _offset, _value):
        frame = ancestor = invocation = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not database or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            ancestor = frame.f_back
            invocation = None
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                if ancestor.f_code is invoke_code:
                    invocation = ancestor
                ancestor = ancestor.f_back
            assert invocation is not None
            assert invocation.f_globals is base_db.run_owned_db_call.__globals__
            assert invocation.f_locals["database"] is database
            assert invocation.f_locals["operation"] is query
            assert invocation.f_locals["args"] == (list(row_ids),)
            assert invocation.f_locals["kwargs"] == {}
            invocation = None
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._thread_local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases and not closed(connection)
                assert lease.resource_thread is threading.current_thread()
            held.update(
                future=item.future,
                connection=connection,
                participant=participant,
                lease=lease,
            )
            entered.set()
            assert release.wait(10), "Original subagent-count query was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor, invocation

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "subagent-count-host-native")
        except ValueError:
            continue
        tool = candidate
        break
    else:
        raise AssertionError("No local monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, hold)
        is None
    )
    sys.monitoring.set_local_events(
        tool, reader.__code__, sys.monitoring.events.PY_RETURN
    )
    with host._context():
        try:
            issued = original(controller, bridge, database, state)
            worker = screen.run_worker(
                issued,
                group="console-subagent-counts",
                exit_on_error=False,
            )
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and not errors, errors
            assert worker._node is screen and worker in host.workers
            task = worker._task
            assert worker._work is issued and type(task) is asyncio.Task
            assert issued.cr_code is original.__code__ and not task.done()
            worker.cancel()
            captured = capture_console_view_workers(host)
            selected = any(
                row[0] is worker and row[1] is screen for row in captured[-1]
            )
            drain = asyncio.create_task(drain_console_view_workers(captured))
            await asyncio.wait({drain}, timeout=0.05)
            retained = not drain.done()
            for _ in range(2):
                drain.cancel()
                await asyncio.sleep(0.01)
                retained = retained and not drain.done()
            assert not held["future"].done() and not closed(held["connection"])
        finally:
            release.set()
            try:
                if held:
                    outcome = await asyncio.gather(
                        asyncio.wrap_future(held["future"]), return_exceptions=True
                    )
                await asyncio.gather(
                    *(
                        task
                        for task in (drain, worker._task if worker else None)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
            finally:
                sys.monitoring.set_local_events(tool, reader.__code__, 0)
                assert (
                    sys.monitoring.register_callback(
                        tool, sys.monitoring.events.PY_RETURN, None
                    )
                    is hold
                )
                sys.monitoring.free_tool_id(tool)
                try:
                    receipt = monitor.close()
                finally:
                    database.close()
    assert receipt["original_source_current"] and not errors, (receipt, errors)
    assert outcome == [{}], outcome
    assert not state["pending"]
    assert worker._task is task and worker._work is issued and task.done()
    assert worker.state is WorkerState.CANCELLED and not published
    assert state["at"] == 0.0 and state["values"] == {}
    assert held["future"].done() and closed(held["connection"])
    with storage._lock:
        assert held["lease"] not in storage._live_leases
        assert held["connection"] not in held["participant"].connections
    assert (
        selected and retained
    ), "Host drain released original native-live subagent-count query"
