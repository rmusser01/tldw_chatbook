"""Original Collections reconciliation must retire before owner shutdown."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import sqlite3
import sys
import threading

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["interrupt", "offline"])
@private_profile_test
async def test_original_collections_reconciliation_retained_through_shutdown(
    request, tmp_path, route
):
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )
    from tldw_chatbook.Library.collections_offline_store import CollectionsOfflineStore

    database = LibraryCollectionsDB(tmp_path / "collections.db", client_id="reconcile")
    repository = CollectionsCaptureRepository(database, authority_key="local:test")
    offline = CollectionsOfflineStore(
        repository, data_root=tmp_path, authority_fingerprint="a" * 64
    )
    owner = ServiceWiringMixin()
    owner.collections_capture_repository = repository
    owner.collections_offline_store = offline
    callback = (
        repository.interrupt_stale_extractions
        if route == "interrupt"
        else offline.reconcile_batch
    )
    original = callback.__func__
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for function in (
        original,
        ServiceWiringMixin._reconcile_collections_capture_startup,
        ServiceWiringMixin._shutdown_collections_capture_runtime,
        _WorkItem.run,
    ):
        pin._pin(function)
    pin.slots.append((type(callback.__self__), original.__name__, original))
    entered, release = threading.Event(), threading.Event()
    held, errors = {}, []
    loop = asyncio.get_running_loop()
    main = threading.current_thread()
    issued = shutdown = None

    def is_closed(connection):
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def hold(code, _offset, _value):
        frame = ancestor = None
        try:
            frame = sys._getframe(1)
            if frame.f_locals.get("self") is not callback.__self__:
                return
            assert code is original.__code__ and frame.f_globals is original.__globals__
            assert not held and threading.current_thread() is not main
            ancestor = frame.f_back
            while (
                ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__
            ):
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            connection = database._thread_local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases and not is_closed(connection)
                assert lease.resource_thread is threading.current_thread()
            held.update(
                future=item.future,
                connection=connection,
                lease=lease,
                participant=participant,
            )
            entered.set()
            assert release.wait(10), "Original reconciliation callback was not released"
        except BaseException as error:
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor

    for candidate in range(5, 0, -1):
        if candidate == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(candidate, "collections-reconciliation")
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
        tool, original.__code__, sys.monitoring.events.PY_RETURN
    )
    try:
        issued = asyncio.create_task(owner._reconcile_collections_capture_startup())
        deadline = loop.time() + 10
        while not entered.is_set():
            assert loop.time() < deadline, "Original reconciliation did not enter"
            await asyncio.sleep(0.005)
        assert held and not errors, errors
        issued.cancel()
        shutdown = asyncio.create_task(owner._shutdown_collections_capture_runtime())
        await asyncio.sleep(0.05)
        retained = not issued.done() and not shutdown.done()
        for _ in range(2):
            shutdown.cancel()
            await asyncio.sleep(0.01)
            retained = retained and not issued.done() and not shutdown.done()
        assert not held["future"].done() and not is_closed(held["connection"])
    finally:
        release.set()
        if held:
            # Shutdown closes admission before the original callback returns.
            # Its final currentness check may report that terminal cancellation.
            physical_outcome = await asyncio.gather(
                asyncio.wait_for(asyncio.wrap_future(held["future"]), 10),
                return_exceptions=True,
            )
        await asyncio.gather(
            *(task for task in (issued, shutdown) if task is not None),
            return_exceptions=True,
        )
        sys.monitoring.set_local_events(tool, original.__code__, 0)
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, None
            )
            is hold
        )
        sys.monitoring.free_tool_id(tool)
        receipt = pin.close()
        await owner._shutdown_collections_capture_runtime()
        database.close()
    assert receipt["original_source_current"] and not errors, errors
    assert not isinstance(physical_outcome[0], BaseException) or isinstance(
        physical_outcome[0], asyncio.CancelledError
    ), physical_outcome
    assert held["future"].done() and is_closed(held["connection"])
    with storage._lock:
        assert held["lease"] not in storage._live_leases
        assert held["connection"] not in held["participant"].connections
    assert (
        retained
    ), "Shutdown returned while original reconciliation was still native-live"
