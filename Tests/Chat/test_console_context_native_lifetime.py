"""PYTEST_DONT_REWRITE: observe original Context callback source and lifetime."""

import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
import inspect
from pathlib import Path
import sqlite3
import sys
import threading
from types import CodeType

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@private_profile_test
async def test_original_context_read_retains_native_callback(
    request, tmp_path, borrowed, cancel
):
    await _verify_original_context_read(tmp_path, borrowed, cancel)


async def _verify_original_context_read(tmp_path, borrowed, cancel):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.base_db import run_owned_db_call

    database, store, controller, _ = _controller(tmp_path)
    store.append_message(
        "session-1",
        role=ConsoleMessageRole.USER,
        content="context anchor",
        persist=True,
    )
    original = ConsoleChatController.context_control_presentation_inputs
    read_code = next(
        code
        for code in original.__code__.co_consts
        if type(code) is CodeType and code.co_name == "read"
    )
    tree = ast.parse(Path(inspect.getsourcefile(ConsoleChatController)).read_bytes())
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ConsoleChatController"
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == original.__name__
    )
    line = next(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "snapshots"
    )
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for function in (original, run_owned_db_call, _WorkItem.run):
        pin._pin(function)
    pin.slots.extend(
        (
            (ConsoleChatController, original.__name__, original),
            (
                sys.modules[run_owned_db_call.__module__],
                "run_owned_db_call",
                run_owned_db_call,
            ),
            (_WorkItem, "run", _WorkItem.run),
        )
    )
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="context-lifetime")
    loop.set_default_executor(executor)
    previous = (
        await loop.run_in_executor(executor, database.get_connection)
        if borrowed
        else None
    )
    entered, release = asyncio.Event(), threading.Event()
    held, errors = {}, []
    tool = outer = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return "closed" in str(error).lower()
        return False

    def hold(code, current_line):
        if code is not read_code or current_line != line or held:
            return
        frame = sys._getframe(1)
        ancestor = None
        try:
            if frame.f_locals.get("self") is not controller:
                return
            assert frame.f_locals["snapshots"]
            connection = database._local.conn
            assert not closed(connection)
            assert threading.current_thread() is not threading.main_thread()
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                operation = storage._operation_local.operation
                assert operation in storage._operations
                assert operation.participant is participant
                assert lease in storage._live_leases
                assert lease.resource_thread is threading.current_thread()
            held.update(
                connection=connection,
                operation=operation,
                lease=lease,
                future=item.future,
                participant=participant,
                thread=threading.current_thread(),
            )
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "original Context read was never released"
        except BaseException as error:
            errors.append(type(error).__name__)
            loop.call_soon_threadsafe(entered.set)
        finally:
            frame = ancestor = None

    try:
        pin._pin(hold)
        pin._pin(closed)
        for candidate in range(5, 0, -1):
            if candidate == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(candidate, "context-native-lifetime")
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None
        assert sys.monitoring.get_events(tool) == 0
        assert (
            sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, hold)
            is None
        )
        assert sys.monitoring.get_local_events(tool, read_code) == 0
        sys.monitoring.set_local_events(tool, read_code, sys.monitoring.events.LINE)
        outer = asyncio.Task(
            original(controller, "session-1", _presentation_global_overrides=None)
        )
        await asyncio.wait_for(entered.wait(), 10)
        assert held and not errors
        if cancel:
            outer.cancel("first Context cancellation")
            await asyncio.sleep(0)
            outer.cancel("second Context cancellation")
        await asyncio.sleep(0.1)
        assert (
            not outer.done()
        ), "Context waiter detached while original native read was held"
        assert not held["future"].done()
        with storage._lock:
            assert held["operation"] in storage._operations
            assert held["lease"] in storage._live_leases
        release.set()
        result = (await asyncio.gather(outer, return_exceptions=True))[0]
        if cancel:
            assert isinstance(result, asyncio.CancelledError)
        else:
            assert isinstance(result, tuple) and len(result) == 3
            assert result[1] is None
        assert held["future"].done()

        def verify_retirement():
            assert threading.current_thread() is held["thread"]
            with storage._lock:
                assert held["operation"] not in storage._operations
                if borrowed:
                    assert held["connection"] is previous and not closed(previous)
                    assert database._local.conn is previous
                    assert held["participant"].connections[previous] is held["lease"]
                    assert held["lease"] in storage._live_leases
                else:
                    assert closed(held["connection"])
                    assert held["connection"] not in held["participant"].connections
                    assert held["lease"] not in storage._live_leases
                    assert getattr(database._local, "conn", None) is None

        await loop.run_in_executor(executor, verify_retirement)
    finally:
        release.set()
        if held:
            await loop.run_in_executor(executor, held["future"].result, 10)
        if outer is not None:
            await asyncio.gather(outer, return_exceptions=True)
        if tool is not None:
            sys.monitoring.set_local_events(tool, read_code, 0)
            actual = sys.monitoring.register_callback(
                tool, sys.monitoring.events.LINE, None
            )
            if actual is not hold:
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.LINE, actual
                )
                raise RuntimeError("context_monitor_owner_changed")
            assert sys.monitoring.get_events(tool) == 0
            sys.monitoring.free_tool_id(tool)
        await loop.run_in_executor(executor, database.close)
        database.close()
        receipt = pin.close()
        assert receipt["original_source_current"] and not receipt["invalid"]
        assert not errors
