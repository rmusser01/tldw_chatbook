"""Finite database callbacks count one real installed repository interval."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import MappingProxyType, SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_console_metadata_batching import (
    observe_worker_connections,
)
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.base_db import operation_owned_connection, run_owned_db_call
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB


pytestmark = pytest.mark.usefixtures("local_root")
OWNER_TYPES = (CharactersRAGDB, AgentRunsDB, WorkspaceDB)


def make_database(owner_type, path):
    if issubclass(owner_type, CharactersRAGDB):
        return owner_type(path, "finite-interval")
    return owner_type(path)


@pytest.fixture(params=OWNER_TYPES, ids=("notes", "runs", "workspaces"))
def database(request, tmp_path):
    owner = make_database(request.param, tmp_path / "finite.sqlite")
    owner.close()
    try:
        yield owner
    finally:
        owner.close()


def connection_accessor(database):
    return (
        "get_connection"
        if isinstance(database, CharactersRAGDB)
        else "_held_connection"
    )


def read_value(database, value):
    manager = (
        database.transaction()
        if isinstance(database, CharactersRAGDB)
        else database.connection()
    )
    with manager as native:
        return native.execute("SELECT ? AS finite_probe", (value,)).fetchone()[0]


def live_operations(database):
    with storage._lock:
        return [
            operation
            for operation in storage._operations
            if operation.participant.repository() is database
        ]


def observe_admissions(monkeypatch):
    """Call the actual admission seam, recording ordinary versus descendant IO."""
    actual = storage.acquire_storage
    records = []

    def observed(path=None, **kwargs):
        operation = getattr(storage._operation_local, "operation", None)
        lease = actual(path, **kwargs)
        records.append((path, operation, lease))
        return lease

    monkeypatch.setattr(storage, "acquire_storage", observed)
    return records


@pytest.mark.asyncio
async def test_finite_reads_share_one_admission_but_execute_every_sql(
    database, monkeypatch
):
    records = observe_admissions(monkeypatch)
    connections, statements = observe_worker_connections(
        monkeypatch, database, connection_accessor(database)
    )

    def reads():
        return tuple(read_value(database, value) for value in (11, 22, 33))

    def separate_intervals():
        with operation_owned_connection(database):
            return reads()

    assert await asyncio.to_thread(separate_intervals) == (11, 22, 33)
    separate_count = sum(
        path == database.db_path and operation is None for path, operation, _ in records
    )
    assert separate_count == 3  # Three independently admitted method intervals.
    assert not worker_leases(database)
    records.clear()
    statements.clear()

    assert await run_owned_db_call(database, reads) == (11, 22, 33)
    ordinary = [
        lease
        for path, operation, lease in records
        if path == database.db_path and operation is None
    ]
    descendants = [
        lease
        for path, operation, lease in records
        if path == database.db_path and operation is not None
    ]
    assert (
        len(ordinary) == 1
    ), "one callback must admit one complete repository interval"
    assert len(descendants) == 1, "the real native connection retains its own lease"
    assert [sql for _, sql in statements if "finite_probe" in sql] == [
        "SELECT 11 AS finite_probe",
        "SELECT 22 AS finite_probe",
        "SELECT 33 AS finite_probe",
    ]
    assert len(connections) == 2, "each finite callback owns and retires its new handle"
    assert not worker_leases(database)
    assert not live_operations(database)
    assert all(lease not in storage._live_leases for _, _, lease in records)


@pytest.mark.asyncio
async def test_pause_preserves_counted_callback_and_rejects_fresh_owners(
    database, tmp_path
):
    other = make_database(type(database), tmp_path / "other.sqlite")
    other.close()
    entered, release = threading.Event(), threading.Event()
    fresh_bodies = []
    refusals = []

    def body():
        entered.set()
        assert release.wait(10)
        # No handle existed before pause: this exact descendant allocation must
        # be allowed only by the original continuously counted operation.
        values = [read_value(database, value) for value in (11, 22)]
        with operation_owned_connection(other):
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                getattr(other, connection_accessor(other))()
        refusals.append(True)
        return values

    pending = asyncio.create_task(run_owned_db_call(database, body))
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pause = storage._begin_local_pause()
        assert live_operations(
            database
        ), "the callback counts before its first SQL read"
        assert not pause.drain(time.monotonic() + 0.02)
        for owner in (database, other):
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                await run_owned_db_call(owner, lambda: fresh_bodies.append(owner))
        assert not fresh_bodies
        release.set()
        assert await pending == [11, 22]
        assert refusals == [True]
        assert not worker_leases(database)
        assert not live_operations(database)
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        if pause is not None:
            pause.resume()
        await asyncio.gather(pending, return_exceptions=True)
        other.close()


@pytest.mark.asyncio
async def test_cancelled_callback_counts_until_later_sql_and_native_retirement(
    database,
):
    entered, release, finished = (threading.Event() for _ in range(3))
    results = []

    def body():
        try:
            first = read_value(database, 11)
            entered.set()
            assert release.wait(10)
            results.append((first, read_value(database, 22)))
        finally:
            finished.set()

    pending = asyncio.create_task(run_owned_db_call(database, body))
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pause = storage._begin_local_pause()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert live_operations(database)
        assert worker_leases(database)
        assert not pause.drain(time.monotonic() + 0.02)
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        for _ in range(200):
            if not live_operations(database) and not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert results == [(11, 22)]
        assert not live_operations(database)
        assert not worker_leases(database)
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        if pause is not None:
            pause.resume()
        await asyncio.gather(pending, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "same_path", [False, True], ids=("different-file", "same-file")
)
async def test_other_repository_getter_requires_independent_ordinary_admission(
    tmp_path, monkeypatch, same_path
):
    first = WorkspaceDB(tmp_path / "first.sqlite")
    second = WorkspaceDB(first.db_path if same_path else tmp_path / "second.sqlite")
    first.close()
    second.close()
    records = observe_admissions(monkeypatch)
    observed_operations = []

    def body():
        original = storage._operation_local.operation
        assert original.participant.repository() is first
        assert read_value(first, 11) == 11
        with operation_owned_connection(second):
            connection = second._held_connection()
            assert connection.execute("SELECT 22 AS finite_probe").fetchone()[0] == 22
            # The getter restores the original receiver's operation after its
            # independent ordinary admission; it cannot adopt the other owner.
            observed_operations.append(storage._operation_local.operation)
        return read_value(first, 33)

    try:
        assert await run_owned_db_call(first, body) == 33
        ordinary = [
            (path, lease) for path, operation, lease in records if operation is None
        ]
        assert [path for path, _ in ordinary] == [first.db_path, second.db_path]
        assert len(observed_operations) == 1
        assert observed_operations[0].participant.repository() is first
        assert not worker_leases(first)
        assert not worker_leases(second)
        assert not live_operations(first)
        assert not live_operations(second)
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_type", OWNER_TYPES, ids=("notes", "runs", "workspaces"))
@pytest.mark.parametrize("shape", ["subclass", "memory"])
async def test_unqualified_real_owners_keep_their_caller_owned_native_handle(
    tmp_path, monkeypatch, owner_type, shape
):
    selected_type = (
        type("DerivedOwner", (owner_type,), {}) if shape == "subclass" else owner_type
    )
    database = make_database(
        selected_type, ":memory:" if shape == "memory" else tmp_path / "derived.sqlite"
    )
    records = observe_admissions(monkeypatch)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def dispatch(callback, *args, **kwargs):
            return await loop.run_in_executor(
                executor, partial(callback, *args, **kwargs)
            )

        monkeypatch.setattr(asyncio, "to_thread", dispatch)
        try:
            assert await run_owned_db_call(database, read_value, database, 42) == 42
            connection = await dispatch(
                getattr(database, connection_accessor(database))
            )
            assert (
                await dispatch(lambda: connection.execute("SELECT 43").fetchone()[0])
                == 43
            )
            assert not live_operations(database)
            assert all(operation is None for _, operation, _ in records)
            if shape == "subclass":
                assert worker_leases(
                    database
                ), "an unqualified caller retains its own handle"
        finally:
            await dispatch(database.close)
            database.close()


@pytest.mark.asyncio
async def test_custom_callback_retains_its_arguments_result_and_original_thread_route():
    def forbidden_close():
        raise AssertionError(
            "custom owner cannot be closed by the finite callback helper"
        )

    owner = SimpleNamespace(is_memory_db=False, close=forbidden_close)
    origin = threading.current_thread()
    result = await run_owned_db_call(
        owner,
        lambda prefix, *, suffix: (threading.current_thread(), prefix + suffix),
        "finite",
        suffix="-result",
    )
    assert result[0] is not origin
    assert result[1] == "finite-result"


def availability_controller(registry, workspace_ids):
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

    controller = object.__new__(ConsoleWorkspaceController)
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)
    controller._screen = SimpleNamespace(app=controller.app_instance)
    controller._preparation_reads = set()
    controller._workspace_files_availability_by_id = MappingProxyType({})
    controller._workspace_files_runtime_bindings_by_id = MappingProxyType({})
    controller._screen_running_accessor = lambda: True
    controller._workspace_files_availability_generation = 1
    controller._workspace_files_availability_requested_ids = workspace_ids
    controller._workspace_files_availability_refresh_in_flight = True
    controller._sync_workspace_context_fn = lambda: None
    return controller


@pytest.mark.asyncio
async def test_workspace_availability_offload_counts_one_interval_and_closes_once(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    database = WorkspaceDB(tmp_path / "availability.sqlite")
    registry = LocalWorkspaceRegistryService(database)
    default = registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="named", name="Named")
    folder = tmp_path / "ready"
    folder.mkdir()
    registry.add_folder_binding("named", folder)
    database.close()
    controller = availability_controller(registry, (default.workspace_id, "named"))
    records = observe_admissions(monkeypatch)
    _, statements = observe_worker_connections(
        monkeypatch, database, "_held_connection"
    )
    closes = []
    actual_close = database.close

    def observed_close():
        closes.append(tuple(live_operations(database)))
        actual_close()

    monkeypatch.setattr(database, "close", observed_close)
    try:
        await controller._refresh_workspace_files_availability_snapshot()
        assert dict(controller._workspace_files_availability_by_id) == {
            default.workspace_id: False,
            "named": True,
        }
        assert (
            sum(
                path == database.db_path and operation is None
                for path, operation, _ in records
            )
            == 1
        )
        assert (
            len(
                [
                    sql
                    for _, sql in statements
                    if "SELECT" in sql and "FROM workspace_runtime_bindings" in sql
                ]
            )
            == 2
        )
        assert closes == [
            ()
        ], "physical close runs once after the callback interval exits"
        assert not worker_leases(database)
    finally:
        actual_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["registry", "database", "generation"])
async def test_workspace_availability_rejects_owner_changes_across_await(
    tmp_path, monkeypatch, member
):
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    first = WorkspaceDB(tmp_path / "first-availability.sqlite")
    second = WorkspaceDB(tmp_path / "second-availability.sqlite")
    original = LocalWorkspaceRegistryService(first)
    replacement = LocalWorkspaceRegistryService(second)
    default = original.ensure_default_workspace()
    replacement.ensure_default_workspace()
    for registry in (original, replacement):
        registry.create_workspace(workspace_id="named", name="Named")
    folder = tmp_path / "old-ready"
    folder.mkdir()
    original.add_folder_binding("named", folder)
    first.close()
    second.close()
    controller = availability_controller(original, (default.workspace_id, "named"))
    previous_availability = controller._workspace_files_availability_by_id
    entered, release = threading.Event(), threading.Event()
    old_read = original.list_runtime_bindings
    blocked = False

    def delayed(workspace_id):
        nonlocal blocked
        result = old_read(workspace_id)
        if not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(original, "list_runtime_bindings", delayed)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def dispatch(callback, *args, **kwargs):
            return await loop.run_in_executor(
                executor, partial(callback, *args, **kwargs)
            )

        monkeypatch.setattr(asyncio, "to_thread", dispatch)
        pending = asyncio.create_task(
            controller._refresh_workspace_files_availability_snapshot()
        )
        try:
            # The one worker is blocked in the actual read; wait for its signal
            # from the event loop without dispatching behind that worker.
            for _ in range(2000):
                if entered.is_set():
                    break
                await asyncio.sleep(0.005)
            assert entered.is_set()
            if member == "registry":
                controller.app_instance.workspace_registry_service = replacement
            elif member == "database":
                original.db = second
            else:
                controller._workspace_files_availability_generation += 1
                controller._workspace_files_availability_requested_ids = (
                    default.workspace_id,
                )
            release.set()
            await pending
            if member == "generation":
                assert dict(controller._workspace_files_availability_by_id) == {
                    default.workspace_id: False
                }
            else:
                assert (
                    controller._workspace_files_availability_by_id
                    is previous_availability
                ), "obsolete read replaced the previous availability cache"
            assert not worker_leases(first)
            assert not worker_leases(second)
            assert not live_operations(first)
            assert not live_operations(second)
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await dispatch(first.close)
            await dispatch(second.close)
            first.close()
            second.close()


@pytest.mark.asyncio
async def test_workspace_availability_cancelled_owner_keeps_interval_until_retired(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    database = WorkspaceDB(tmp_path / "cancel-availability.sqlite")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="named", name="Named")
    database.close()
    controller = availability_controller(registry, ("named",))
    entered, release, finished = (threading.Event() for _ in range(3))
    actual_read = registry.list_runtime_bindings

    def delayed(workspace_id):
        try:
            result = actual_read(workspace_id)
            entered.set()
            assert release.wait(10)
            return result
        finally:
            finished.set()

    monkeypatch.setattr(registry, "list_runtime_bindings", delayed)
    pending = asyncio.create_task(
        controller._refresh_workspace_files_availability_snapshot()
    )
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pause = storage._begin_local_pause()
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        assert live_operations(database)
        assert worker_leases(database)
        assert not pause.drain(time.monotonic() + 0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await asyncio.to_thread(finished.wait, 10)
        for _ in range(200):
            if not live_operations(database) and not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(database)
        assert not live_operations(database)
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        if pause is not None:
            pause.resume()
        await asyncio.gather(pending, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 10)
        database.close()


@pytest.mark.asyncio
async def test_actual_trace_vacuum_completes_inside_owned_callback_interval(tmp_path):
    from Tests.Chat.test_console_trace_compaction import (
        _add_orphan_trace_payload,
        _permissive_policy,
        _shared_fork_fixture,
    )
    from tldw_chatbook.Chat.console_trace_maintenance import (
        PhysicalTraceCompactor,
        TraceGarbageCollector,
    )
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    path = tmp_path / "owned-compaction.sqlite"
    database = CharactersRAGDB(path, "owned-compaction")
    source_id, child_id, child_segment_id, call_id = _shared_fork_fixture(database)
    _add_orphan_trace_payload(database, rows=16)
    gc_result = TraceGarbageCollector(database).collect(
        request_id="gc-owned-compaction"
    )
    assert gc_result.deleted_rows["console_trace_artifacts"] == 16
    assert gc_result.freelist_bytes_after > 0
    database.close()
    before_size = path.stat().st_size
    stages = []
    compactor = PhysicalTraceCompactor(
        database,
        policy=_permissive_policy(),
        progress=lambda event: stages.append(event.stage),
    )
    try:
        outcome = await run_owned_db_call(database, compactor.run_after_gc, gc_result)
        assert outcome.completed, outcome.reason_code
        assert outcome.reason_code == "complete"
        assert outcome.allocated_bytes_after < outcome.allocated_bytes_before
        assert path.stat().st_size < before_size
        assert "vacuum" in stages and stages[-1] == "complete"
        assert not worker_leases(database)
        assert not live_operations(database)
        assert database.get_console_trace_compaction_status()["status"] == "complete"
        repository = ConsoleTraceRepository()
        with database.transaction() as cursor:
            assert cursor.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_owners "
                    "WHERE attached = 1 AND conversation_id IN (?, ?)",
                    (source_id, child_id),
                ).fetchone()[0]
                == 2
            )
            assert (
                repository.get_segment(cursor, child_segment_id).parent_segment_id
                is not None
            )
            assert [
                call.call_id
                for call in repository.read_conversation_call_lineage(cursor, child_id)
            ] == [call_id]
    finally:
        database.close()
