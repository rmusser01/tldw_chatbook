"""Actual original browser worker must outlive native callback retirement."""

import asyncio
import inspect
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("cancellation", ["shutdown", "repeated", "all_created"])
@private_profile_test
async def test_current_host_shutdown_drains_original_browser_callback(
    request, tmp_path, borrowed, cancellation
):
    from textual.app import App
    from textual.worker import Worker

    from Tests.UI.test_console_workspace_controller import _workspace_controller
    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "browser.sqlite", client_id="browser-custody")
    service = ChatConversationService(database)
    service.create_conversation(title="Global Native", scope_type="global")
    service.create_conversation(
        title="Default Native", scope_type="workspace", workspace_id="workspace-default"
    )
    database.close_connection()
    participant = database._maintenance_participant
    host = App()
    host._console_runtime_shutdown_task = None
    host._plugin_service = None
    host.console_runtime = None
    controller = _workspace_controller(
        screen=host,
        app_instance=SimpleNamespace(local_chat_conversation_service=service),
    )
    loop = asyncio.get_running_loop()
    entered, release = threading.Event(), threading.Event()
    captured = {}
    reader = ChatConversationService.list_conversations
    code = reader.__code__
    shutdown = worker = borrower_scope = None

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def returned(observed_code, _offset, result):
        frame = sys._getframe(1)
        if (
            observed_code is not code
            or frame.f_locals.get("self") is not service
            or captured
        ):
            return
        assert frame.f_locals["scope_type"] == "global"
        assert result["items"] and result["pagination"]["total"] == 1
        connection = database._local.conn
        assert not closed(connection)
        ancestor = frame
        while ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__:
            ancestor = ancestor.f_back
        assert ancestor is not None
        future = ancestor.f_locals["self"].future
        assert future.running() and not future.done()
        with storage._lock:
            lease = participant.connections[connection]
            assert (
                lease in storage._live_leases
                and lease.resource_thread is threading.current_thread()
            )
        captured.update(
            connection=connection,
            lease=lease,
            future=future,
            thread=threading.current_thread(),
        )
        entered.set()
        assert release.wait(10), "original browser callback was not released"

    async def wait_entered():
        while not entered.is_set():
            if worker._task.done():
                worker._task.result()
                raise AssertionError("original browser read was not reached")
            await asyncio.sleep(0.01)

    def enter_borrower():
        nonlocal borrower_scope
        borrower_scope = database.transaction()
        borrower_scope.__enter__()
        return database._local.conn

    def inspect_and_close():
        assert threading.current_thread() is captured["thread"]
        connection = captured["connection"]
        with storage._lock:
            result = (
                closed(connection),
                captured["lease"] in storage._live_leases,
                connection in participant.connections,
            )
        if borrowed:
            assert connection is previous and sqlite3.Connection.in_transaction.__get__(
                connection
            )
            borrower_scope.__exit__(
                RuntimeError, RuntimeError("fixture rollback"), None
            )
        database.close_connection()
        assert closed(connection)
        return result

    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "current-browser-host-native")
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="current-browser"
    ) as executor:
        loop.set_default_executor(executor)
        previous = (
            await loop.run_in_executor(executor, enter_borrower) if borrowed else None
        )
        try:
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
            monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            tasks_before = asyncio.all_tasks()
            with host._context():
                controller._sync_persisted_console_browser_rows()
            candidates = [
                row
                for row in host.workers
                if type(row) is Worker
                and row.group == "console-persisted-browser-cache"
            ]
            assert len(candidates) == 1
            worker = candidates[0]
            assert worker._node is host
            await asyncio.wait_for(wait_entered(), 10)
            shutdown = asyncio.create_task(
                LifecycleMixin._shutdown_console_runtime(host)
            )
            progress = []
            loop.call_soon(progress.append, True)
            await asyncio.sleep(0.1)
            assert progress == [True]
            assert not captured["future"].done()
            with storage._lock:
                assert captured["lease"] in storage._live_leases
                assert (
                    participant.connections[captured["connection"]] is captured["lease"]
                )
            assert not shutdown.done(), (
                "host shutdown returned before its original native callback retired"
            )
            assert not worker._task.done(), "original Worker lost its physical callback"
            owned_shutdown = host._console_runtime_shutdown_task
            assert owned_shutdown is not None and not owned_shutdown.done()
            if cancellation != "shutdown":
                targets = (
                    asyncio.all_tasks() - tasks_before
                    if cancellation == "all_created"
                    else {owned_shutdown, worker._task}
                )
                assert owned_shutdown in targets and worker._task in targets
                for target in targets:
                    target.cancel()
                await asyncio.sleep(0)
                for target in targets:
                    target.cancel()
                await asyncio.sleep(0.1)
                # The public shielded waiter keeps its existing cancellation ABI;
                # the actual shutdown owner and native worker retain custody.
                assert not owned_shutdown.done()
                assert not worker._task.done()
                assert not captured["future"].done()
                with storage._lock:
                    assert captured["lease"] in storage._live_leases
        finally:
            release.set()
            await asyncio.gather(
                *(
                    task
                    for task in (
                        getattr(worker, "_task", None),
                        shutdown,
                        host._console_runtime_shutdown_task,
                    )
                    if task is not None
                ),
                return_exceptions=True,
            )
            # Wait for actual native completion even when the intended baseline
            # assertion proves the current host returned early.
            if captured:
                from tldw_chatbook.UI.Console_Modules.console_browser_read import (
                    _BrowserSourceChanged,
                )

                native_outcome = await asyncio.gather(
                    asyncio.wrap_future(captured["future"]), return_exceptions=True
                )
                assert len(native_outcome) == 1
                assert isinstance(native_outcome[0], _BrowserSourceChanged)
                boundary = await loop.run_in_executor(executor, inspect_and_close)
                assert boundary == (
                    (False, True, True) if borrowed else (True, False, False)
                )
            elif borrower_scope is not None:

                def close_unobserved_fixture():
                    borrower_scope.__exit__(
                        RuntimeError, RuntimeError("fixture rollback"), None
                    )
                    database.close_connection()

                await loop.run_in_executor(executor, close_unobserved_fixture)
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
            database.close()
    assert (
        reader.__code__ is code
        and inspect.getattr_static(ChatConversationService, "list_conversations")
        is reader
    )
