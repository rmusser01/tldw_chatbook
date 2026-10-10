"""Original receipt hydration owns both of its worker databases until retirement."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import inspect
import sqlite3
import sys
import threading
from types import CodeType, SimpleNamespace

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["normal", "cancel-and-dispose"])
@private_profile_test
async def test_original_receipt_hydration_retires_both_databases(
    request, tmp_path, cancel
):
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_activity_receipts import (
        ConsoleActivityReceiptService,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    notes = CharactersRAGDB(tmp_path / "chat.db", "receipt-retirement")
    conversation_id = notes.add_conversation({"title": "Receipt retirement"})
    marks = ConversationLocalMarksService(notes)
    marks.set_mark(conversation_id, marks.FLEET_UNSEEN)
    borrowed_notes_connection = notes._local.conn
    app = SimpleNamespace(chachanotes_db=notes, conversation_local_marks_service=marks)
    runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
    app.console_runtime = runtime
    service = runtime.ensure_activity_receipt_service()
    delegate = service._get_delegate()
    runs = runtime._agent_runs_db
    assert delegate._db is runs and delegate._marks is marks and marks.db is notes
    reader = ConsoleActivityReceiptService.hydrate_from_storage
    wrapper_code = next(
        code
        for code in ConsoleRuntime.ensure_activity_hydration.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "read_receipts"
    )
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (ConsoleRuntime, "ensure_activity_hydration"),
        (ConsoleRuntime, "dispose"),
        (ConsoleRuntime, "_dispose_owned"),
        (ConsoleActivityReceiptService, "hydrate_from_storage"),
        (ConversationLocalMarksService, "list_marked_conversation_ids"),
        (ConversationLocalMarksService, "clear_mark"),
        (_WorkItem, "run"),
    ):
        callback = getattr(owner, name)
        monitor._pin(callback)
        monitor.slots.append((owner, name, inspect.getattr_static(owner, name)))
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    task = dispose = None
    retained = joined = True
    retired = False

    def closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold(code, _offset, value):
        frame = ancestor = invocation = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not delegate or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            assert value == 0
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                if ancestor.f_code is wrapper_code:
                    invocation = ancestor
                ancestor = ancestor.f_back
            assert invocation is not None and ancestor is not None
            assert invocation.f_locals["service"] is service
            assert invocation.f_locals["runs_db"] is runs
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            rows = []
            for database, local in ((notes, notes._local), (runs, runs._thread_local)):
                connection = local.conn
                participant = database._maintenance_participant
                with storage._lock:
                    lease = participant.connections[connection]
                    assert lease in storage._live_leases and not closed(connection)
                    assert lease.resource_thread is threading.current_thread()
                rows.append((database, connection, participant, lease))
            held.update(future=item.future, rows=rows)
            entered.set()
            assert release.wait(10), "Original hydration callback was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor, invocation

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "receipt-hydration-native")
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
    try:
        task = runtime.ensure_activity_hydration()
        assert task is runtime._activity_hydration_task
        assert await asyncio.to_thread(entered.wait, 10)
        assert held and not errors, errors
        if cancel:
            task.cancel()
            await asyncio.sleep(0.01)
            retained = not task.done()
            dispose = asyncio.create_task(runtime.dispose())
            await asyncio.wait({dispose}, timeout=0.05)
            joined = not dispose.done()
            for _ in range(2):
                task.cancel()
                dispose.cancel()
                await asyncio.sleep(0.01)
                retained = retained and not task.done()
                joined = joined and not dispose.done()
        assert not held["future"].done()
        assert all(not closed(row[1]) for row in held["rows"])
    finally:
        release.set()
        try:
            if held:
                outcome = await asyncio.gather(
                    asyncio.wrap_future(held["future"]), return_exceptions=True
                )
            results = await asyncio.gather(
                *(t for t in (task, dispose) if t is not None), return_exceptions=True
            )
            if held:
                with storage._lock:
                    retired = all(
                        closed(connection)
                        and lease not in storage._live_leases
                        and connection not in participant.connections
                        for _database, connection, participant, lease in held["rows"]
                    )
            reconciled = conversation_id not in marks.list_marked_conversation_ids(
                marks.FLEET_UNSEEN
            )
            borrower_preserved = not closed(borrowed_notes_connection)
            if dispose is None:
                await runtime.dispose()
        finally:
            try:
                sys.monitoring.set_local_events(tool, reader.__code__, 0)
                assert (
                    sys.monitoring.register_callback(
                        tool, sys.monitoring.events.PY_RETURN, None
                    )
                    is hold
                )
                sys.monitoring.free_tool_id(tool)
                receipt = monitor.close()
            finally:
                # Only fixture cleanup follows the captured production-retirement result.
                _close_database_instance(notes)
                runs.close()
    assert receipt["original_source_current"] and not errors, (receipt, errors)
    assert outcome == [0], outcome
    assert task.done()
    if cancel:
        assert isinstance(results[0], asyncio.CancelledError), results
        assert results[1] is None or isinstance(
            results[1], asyncio.CancelledError
        ), results
    else:
        assert results == [0], results
    assert reconciled and borrower_preserved
    assert retired, "Original hydration left a Notes or activity worker connection open"
    assert retained and joined, "Runtime released original native-live hydration"
