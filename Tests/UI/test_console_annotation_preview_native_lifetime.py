"""PYTEST_DONT_REWRITE: verify actual observer Code against its source bytes."""

import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
import inspect
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@private_profile_test
async def test_original_annotation_query_keeps_native_custody_until_waiter_return(
    request, tmp_path, borrowed, cancel
):
    await _verify_original_annotation_native_custody(tmp_path, borrowed, cancel)


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_original_host_shutdown_joins_issued_annotation_worker(
    request, tmp_path, borrowed
):
    await _verify_original_annotation_native_custody(
        tmp_path, borrowed, True, host_shutdown=True
    )


async def _verify_original_annotation_native_custody(
    tmp_path, borrowed, cancel, *, host_shutdown=False
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.base_db import run_owned_db_call
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.review_selection import (
        ConsoleReviewSelectionController,
    )

    database = CharactersRAGDB(
        tmp_path / "annotations.sqlite", client_id="annotation-lifetime"
    )
    participant = database._maintenance_participant
    controller = object.__new__(ConsoleReviewSelectionController)
    controller.annotation_loaded_conversation = "annotation-lifetime"
    controller.annotation_previews = {}
    controller._native_messages_accessor = lambda: []
    original = ConsoleReviewSelectionController._load_console_annotation_previews
    query = CharactersRAGDB.get_transcript_annotations
    query_tree = ast.parse(Path(inspect.getsourcefile(CharactersRAGDB)).read_bytes())
    query_body = next(
        node
        for cls in query_tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "CharactersRAGDB"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_transcript_annotations"
    )
    query_line = next(
        node.lineno
        for node in ast.walk(query_body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "fetchall"
    )
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (ConsoleReviewSelectionController, "_load_console_annotation_previews"),
        (CharactersRAGDB, "get_transcript_annotations"),
        (_WorkItem, "run"),
    ):
        function = inspect.getattr_static(owner, name)
        pin._pin(function)
        pin.slots.append((owner, name, function))
    pin._pin(run_owned_db_call)
    pin.slots.append(
        (
            sys.modules[run_owned_db_call.__module__],
            "run_owned_db_call",
            run_owned_db_call,
        )
    )
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="annotation-custody"
    )
    loop.set_default_executor(executor)
    previous = (
        await loop.run_in_executor(executor, database.get_connection)
        if borrowed
        else None
    )
    entered, release = asyncio.Event(), threading.Event()
    held, invalid = {}, []
    monitoring = sys.monitoring
    tool = None
    outer = None
    host = shutdown_waiter = host_retirement = worker = None

    def native_closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError as error:
            return "closed" in str(error).lower()
        return False

    def hold(code, line):
        frame = ancestor = item = None
        try:
            if code is not query.__code__ or line != query_line or held:
                return
            frame = sys._getframe(1)
            if (
                frame.f_code is not code
                or frame.f_globals is not query.__globals__
                or frame.f_locals.get("self") is not database
            ):
                return
            assert threading.current_thread() is not threading.main_thread()
            cursor = frame.f_locals["conn"]
            assert isinstance(cursor, sqlite3.Cursor)
            connection = sqlite3.Cursor.connection.__get__(cursor)
            assert isinstance(connection, sqlite3.Connection) and not native_closed(
                connection
            )
            ancestor = frame.f_back
            for _ in range(16):
                if ancestor is None or ancestor.f_code is _WorkItem.run.__code__:
                    break
                ancestor = ancestor.f_back
            assert (
                ancestor is not None and ancestor.f_globals is _WorkItem.run.__globals__
            )
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            with storage._lock:
                lease = participant.connections[connection]
                operations = tuple(
                    operation
                    for operation in storage._operations
                    if operation.participant is participant
                )
                assert (
                    lease in storage._live_leases
                    and lease.resource_thread is threading.current_thread()
                )
                assert (
                    len(operations) == 1
                    and operations[0].thread is threading.current_thread()
                )
            held.update(
                connection=connection,
                lease=lease,
                operation=operations[0],
                thread=threading.current_thread(),
                future=item.future,
            )
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "Original annotation query was never released"
        except BaseException as error:
            invalid.append(type(error).__name__)
            loop.call_soon_threadsafe(entered.set)
        finally:
            frame = ancestor = item = None

    try:
        pin._pin(hold)
        pin._pin(native_closed)
        for candidate in range(5, 0, -1):
            if candidate == monitoring.DEBUGGER_ID:
                continue
            try:
                monitoring.use_tool_id(candidate, "original-annotation-custody")
            except ValueError:
                continue
            tool = candidate
            break
        assert tool is not None
        assert monitoring.get_events(tool) == 0
        assert monitoring.register_callback(tool, monitoring.events.LINE, hold) is None
        assert monitoring.get_local_events(tool, query.__code__) == 0
        monitoring.set_local_events(tool, query.__code__, monitoring.events.LINE)
        if host_shutdown:
            from textual.app import App
            from textual.worker import Worker, WorkerState
            from tldw_chatbook.app_lifecycle import LifecycleMixin
            from tldw_chatbook.UI.Console_Modules.view_workers import (
                capture_console_view_workers,
                drain_console_view_workers,
            )

            for owner, name in (
                (App, "run_worker"),
                (Worker, "_run"),
                (Worker, "wait"),
                (Worker, "cancel"),
                (LifecycleMixin, "_shutdown_console_runtime"),
            ):
                function = inspect.getattr_static(owner, name)
                pin._pin(function)
                pin.slots.append((owner, name, function))
            for function in (capture_console_view_workers, drain_console_view_workers):
                pin._pin(function)
                pin.slots.append(
                    (sys.modules[function.__module__], function.__name__, function)
                )
            host = App()
            host._console_runtime_shutdown_task = None
            host._plugin_service = None
            host.console_runtime = None
            with host._context():
                worker = host.run_worker(
                    original(controller, database, None, "annotation-lifetime"),
                    group="console-annotation-previews",
                    exclusive=True,
                    exit_on_error=False,
                )
            outer = worker._task
        else:
            outer = asyncio.create_task(
                original(controller, database, None, "annotation-lifetime")
            )
        await asyncio.wait_for(entered.wait(), 10)
        assert held and not invalid
        if host_shutdown:
            selected = capture_console_view_workers(host)
            assert selected[-1] == ((worker, host, outer, worker._work),)
            shutdown_waiter = asyncio.create_task(
                LifecycleMixin._shutdown_console_runtime(host)
            )
            await asyncio.sleep(0)
            host_retirement = host._console_runtime_shutdown_task
            assert type(host_retirement) is asyncio.Task
            await asyncio.sleep(0)
            shutdown_waiter.cancel()
            await asyncio.sleep(0)
            shutdown_waiter.cancel()
        elif cancel:
            outer.cancel()
            await asyncio.sleep(0)
            outer.cancel()
        progress = []
        loop.call_soon(progress.append, True)
        await asyncio.sleep(0.1)
        assert progress == [True]
        assert not held["future"].done()
        with storage._lock:
            assert held["operation"] in storage._operations
            assert held["lease"] in storage._live_leases
        assert (
            not outer.done()
        ), "Annotation waiter returned while its original native query remained held"
        if host_shutdown:
            assert not host_retirement.done()
            assert worker.is_cancelled
        release.set()
        result = await asyncio.gather(outer, return_exceptions=True)
        if host_shutdown:
            assert result == [None]
            assert worker.state is WorkerState.CANCELLED
            cancelled_wait = await asyncio.gather(
                shutdown_waiter, return_exceptions=True
            )
            assert isinstance(cancelled_wait[0], asyncio.CancelledError)
            await LifecycleMixin._shutdown_console_runtime(host)
            assert host._console_runtime_shutdown_task is host_retirement
            assert host_retirement.done() and host_retirement.result() is None
            assert worker not in host.workers
        else:
            assert (
                isinstance(result[0], asyncio.CancelledError)
                if cancel
                else result == [None]
            )
        assert held["future"].done()
        assert controller.annotation_previews == {}

        def inspect_worker():
            assert threading.current_thread() is held["thread"]
            with storage._lock:
                assert held["operation"] not in storage._operations
                if borrowed:
                    assert held["connection"] is previous and not native_closed(
                        previous
                    )
                    assert participant.connections[previous] is held["lease"]
                    assert held["lease"] in storage._live_leases
                    assert database._local.conn is previous
                else:
                    assert native_closed(held["connection"])
                    assert held["connection"] not in participant.connections
                    assert held["lease"] not in storage._live_leases
                    assert getattr(database._local, "conn", None) is None

        await loop.run_in_executor(executor, inspect_worker)
    finally:
        release.set()
        if held:
            await loop.run_in_executor(executor, held["future"].result, 10)
        if outer is not None:
            await asyncio.gather(outer, return_exceptions=True)
        if host_retirement is not None:
            await asyncio.gather(host_retirement, return_exceptions=True)
        if shutdown_waiter is not None:
            await asyncio.gather(shutdown_waiter, return_exceptions=True)
        if tool is not None:
            monitoring.set_local_events(tool, query.__code__, 0)
            actual = monitoring.register_callback(tool, monitoring.events.LINE, None)
            if actual is not hold:
                monitoring.register_callback(tool, monitoring.events.LINE, actual)
                raise RuntimeError("annotation_monitor_callback_owner_changed")
            assert monitoring.get_events(tool) == 0
            monitoring.free_tool_id(tool)
        await loop.run_in_executor(executor, database.close)
        database.close()
        receipt = pin.close()
        assert receipt["original_source_current"] and not receipt["invalid"]
        assert not invalid


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["memory", "custom"])
@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_annotation_loader_preserves_memory_and_custom_cache_lifetimes(
    request, tmp_path, owner, borrowed
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.review_selection import (
        ConsoleReviewSelectionController,
    )

    class CustomDatabase(CharactersRAGDB):
        pass

    database = (
        CharactersRAGDB(":memory:", client_id="annotation-memory")
        if owner == "memory"
        else CustomDatabase(tmp_path / "custom.sqlite", client_id="annotation-custom")
    )
    main_connection = database.get_connection()
    controller = object.__new__(ConsoleReviewSelectionController)
    controller.annotation_loaded_conversation = "conversation"
    controller.annotation_previews = {}
    controller._native_messages_accessor = lambda: []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotation-route")
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)
    previous = (
        await loop.run_in_executor(executor, database.get_connection)
        if borrowed
        else None
    )
    try:
        await controller._load_console_annotation_previews(
            database, None, "conversation"
        )

        def inspect_worker():
            connection = database._local.conn
            assert connection is not main_connection
            if borrowed:
                assert connection is previous
            assert database._connection_quiescence.is_registered(connection)
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            assert not sqlite3.Connection.in_transaction.__get__(connection)

        await loop.run_in_executor(executor, inspect_worker)
        assert database.get_connection() is main_connection
        assert main_connection.execute("SELECT 1").fetchone()[0] == 1
        assert database.registered_connection_count() == 2
        assert controller.annotation_previews == {}
    finally:
        await loop.run_in_executor(executor, database.close)
        database.close()
        assert database.registered_connection_count() == 0
