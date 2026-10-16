"""Real interrupted capture settlement must retire before owner shutdown."""
import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["single", "repeated", "all_created"])
@private_profile_test
async def test_collections_shutdown_retains_real_interrupted_settlement(request, tmp_path, cancellation):
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )
    from tldw_chatbook.Library.collections_capture_service import (
        LocalCollectionsCaptureService,
        build_local_capture_authority,
    )

    database = LibraryCollectionsDB(tmp_path / "collections.db", client_id="settlement")
    authority = build_local_capture_authority("profile", database.db_path)
    repository = CollectionsCaptureRepository(database, authority_key=authority.key)
    started, extractor_release = threading.Event(), threading.Event()
    entered, release = threading.Event(), threading.Event()

    def extractor(_url, *, url_provenance):
        started.set()
        assert extractor_release.wait(10)
        return {"content": "fixture extraction body"}

    service = LocalCollectionsCaptureService(authority, repository, extractor=extractor)
    outcome = repository.save_capture(CaptureSaveRequest(authority.key, "https://example.invalid/fixture"))
    owner = ServiceWiringMixin()
    owner.local_collections_capture_service = service
    original = CollectionsCaptureRepository.fail_extraction
    pin = OriginalStorageUnitObserver({}, False, lambda _name: None)
    for function in (original, LocalCollectionsCaptureService._call, LocalCollectionsCaptureService._run_extraction, LocalCollectionsCaptureService.cancel_extractions, ServiceWiringMixin._shutdown_collections_capture_runtime, _WorkItem.run):
        pin._pin(function)
    pin.slots.append((CollectionsCaptureRepository, "fail_extraction", original))
    held, errors = {}, []
    main = threading.current_thread()

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
            if frame.f_locals.get("self") is not repository:
                return
            assert code is original.__code__ and frame.f_globals is original.__globals__
            assert frame.f_locals["reason"] == "interrupted"
            assert threading.current_thread() is not main and not held
            ancestor = frame.f_back
            while ancestor is not None and ancestor.f_code is not _WorkItem.run.__code__:
                ancestor = ancestor.f_back
            assert ancestor is not None
            item = ancestor.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running()
            connection = database._thread_local.conn
            participant = database._maintenance_participant
            with storage._lock:
                lease = participant.connections[connection]
                assert lease in storage._live_leases and lease.resource_thread is threading.current_thread()
            assert not closed(connection)
            held.update(future=item.future, connection=connection, participant=participant, lease=lease)
            entered.set()
            assert release.wait(10)
        except BaseException as error:  # noqa: BLE001 - record observer failures without skipping physical test cleanup.
            errors.append(type(error).__name__)
            entered.set()
        finally:
            del frame, ancestor

    for tool in range(5, 0, -1):
        if tool == sys.monitoring.DEBUGGER_ID:
            continue
        try:
            sys.monitoring.use_tool_id(tool, "collections-settlement")
        except ValueError:
            continue
        break
    else:
        raise AssertionError("no monitoring slot")
    assert sys.monitoring.get_events(tool) == 0
    assert sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, hold) is None
    sys.monitoring.set_local_events(tool, original.__code__, sys.monitoring.events.PY_RETURN)
    before = asyncio.all_tasks()
    shutdown = None
    created = set()
    try:
        service._start_extraction(outcome.capture)
        assert await asyncio.to_thread(started.wait, 10)
        shutdown = asyncio.create_task(owner._shutdown_collections_capture_runtime())
        deadline = asyncio.get_running_loop().time() + 10
        while not entered.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(.005)
        assert held and not errors, errors
        created = asyncio.all_tasks() - before - {asyncio.current_task()}
        targets = created if cancellation == "all_created" else {shutdown}
        for target in targets:
            target.cancel()
        await asyncio.sleep(.05)
        retained = not shutdown.done()
        if cancellation == "repeated":
            for _ in range(2):
                shutdown.cancel()
                await asyncio.sleep(.01)
                retained = retained and not shutdown.done()
        assert held["future"].running() and not closed(held["connection"])
    finally:
        release.set()
        extractor_release.set()
        if held:
            await asyncio.gather(asyncio.wait_for(asyncio.wrap_future(held["future"]), 10), return_exceptions=True)
        await asyncio.gather(*created, *tuple(service._extraction_tasks), *( [shutdown] if shutdown else []), return_exceptions=True)
        sys.monitoring.set_local_events(tool, original.__code__, 0)
        assert sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, None) is hold
        sys.monitoring.free_tool_id(tool)
        receipt = pin.close()
        detail = repository.get_detail(outcome.capture.identity)
        database.close()
    assert receipt["original_source_current"] and not errors, errors
    assert held["future"].done() and closed(held["connection"])
    with storage._lock:
        assert held["lease"] not in storage._live_leases
        assert held["connection"] not in held["participant"].connections
    assert detail.processing_state != "processing" and detail.last_fetch_error == "interrupted"
    assert retained, "Collections shutdown returned while original interrupted settlement was native-live"


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
@private_profile_test
async def test_collections_call_preserves_direct_memory_custom_args_and_errors(request, tmp_path, memory):
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )
    from tldw_chatbook.Library.collections_capture_service import (
        LocalCollectionsCaptureService,
        build_local_capture_authority,
    )

    database = LibraryCollectionsDB(":memory:" if memory else tmp_path / "call.db")
    authority = build_local_capture_authority("profile", database.db_path)
    repository = CollectionsCaptureRepository(database, authority_key=authority.key)
    service = LocalCollectionsCaptureService(authority, repository)
    main = threading.current_thread()
    argument, keyword, outcome = object(), object(), object()
    failure = RuntimeError("custom callback outcome")
    calls = []

    def custom(value, *, named, fail=False):
        assert value is argument and named is keyword
        calls.append(threading.current_thread())
        if fail:
            raise failure
        return outcome

    try:
        assert await service._call(custom, argument, named=keyword) is outcome
        with pytest.raises(RuntimeError) as caught:
            await service._call(custom, argument, named=keyword, fail=True)
        assert caught.value is failure
        assert len(calls) == 2 and all((thread is main) == memory for thread in calls)
        # Cancelling existing extraction work does not close ordinary service calls.
        await service.cancel_extractions()
        assert await service._call(custom, argument, named=keyword) is outcome
    finally:
        database.close()


@pytest.mark.asyncio
async def test_collections_shutdown_preserves_primary_cause_and_reports_later_failures():
    from types import SimpleNamespace

    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    primary = RuntimeError("deactivate")
    original_cause = LookupError("original cause")
    primary.__cause__ = original_cause
    initializer_error = ValueError("initializer")
    extraction_error = OSError("extraction")
    calls = []

    def deactivate():
        calls.append("deactivate")
        raise primary

    async def initializer():
        raise initializer_error

    async def cancel_extractions():
        calls.append("extractions")
        raise extraction_error

    task = asyncio.create_task(initializer())
    await asyncio.gather(task, return_exceptions=True)
    owner = ServiceWiringMixin()
    owner._collections_capture_initializer_task = task
    owner.collections_capture_scope_service = SimpleNamespace(deactivate=deactivate)
    owner.local_collections_capture_service = SimpleNamespace(cancel_extractions=cancel_extractions)
    with pytest.raises(RuntimeError) as caught:
        await owner._shutdown_collections_capture_runtime()
    assert caught.value is primary and primary.__cause__ is original_cause
    assert calls == ["deactivate", "extractions"] and task.done()
    assert primary.__notes__ == [
        "Collections initializer retirement also failed: ValueError",
        "Collections extraction shutdown also failed: OSError",
    ]
