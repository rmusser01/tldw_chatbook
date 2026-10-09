"""Original rail-prune worker must retain its actual database callback."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import inspect
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_original_rail_prune_worker_retires_before_host_drain(request, tmp_path):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    database = CharactersRAGDB(tmp_path / "notes.db", "rail-prune")
    host, screen = App(), Screen()
    screen.app_instance = SimpleNamespace(
        chachanotes_db=database, app_config={"console": {"rail_state": {"live": {}}}}
    )
    original = ChatScreen._prune_console_rail_preferences
    reader = inspect.unwrap(CharactersRAGDB.list_all_active_conversations)
    monitor = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for owner, name in (
        (ChatScreen, "_prune_console_rail_preferences"),
        (CharactersRAGDB, "list_all_active_conversations"),
        (_WorkItem, "run"),
    ):
        callback = getattr(owner, name)
        monitor._pin(callback)
        monitor.slots.append((owner, name, callback))
    monitor._pin(reader)
    monitor._pin(inspect.unwrap(original))
    monitor.slots.append((original, "__wrapped__", original.__wrapped__))
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
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not database or held:
                return
            assert code is reader.__code__ and frame.f_globals is reader.__globals__
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._local.conn
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
            assert release.wait(10), "Original rail query was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "rail-prune-native")
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
            worker = original(screen, {"live"})
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and not errors, errors
            assert worker._node is screen and worker in host.workers
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
            sys.monitoring.set_local_events(tool, reader.__code__, 0)
            assert (
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_RETURN, None
                )
                is hold
            )
            sys.monitoring.free_tool_id(tool)
            receipt = monitor.close()
            database.close()
    assert receipt["original_source_current"] and not errors, errors
    assert outcome == [None], outcome
    assert held["future"].done() and closed(held["connection"])
    with storage._lock:
        assert held["lease"] not in storage._live_leases
        assert held["connection"] not in held["participant"].connections
    assert (
        selected and retained
    ), "Host drain released original native-live rail pruning"
