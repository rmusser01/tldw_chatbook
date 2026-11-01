"""Finite binding reads retire native handles on their actual producer thread."""

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Workspaces.test_workspace_registry_service import (
    _insert_stale_default_binding,
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


@pytest.fixture
def binding_registry(tmp_path):
    db = WorkspaceDB(tmp_path / "workspace.db", client_id="retirement")
    registry = LocalWorkspaceRegistryService(db)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="review", name="Review")
    root = tmp_path / "source"
    root.mkdir()
    binding = registry.add_folder_binding("review", root)
    try:
        yield registry, binding.binding_id
    finally:
        db.close()


def _assert_retired(db, connections):
    assert connections
    for connection in connections:
        # Executed by the producer: a foreign-thread ProgrammingError is no proof.
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    with storage._lock:
        assert not any(
            lease.resource_thread is threading.current_thread()
            for lease in db._maintenance_participant.connections.values()
        )


@pytest.mark.parametrize(
    "outcome", ["found", "missing", "default", "read_error", "repair_error"]
)
def test_binding_read_retires_complete_operation(
    binding_registry, monkeypatch, outcome
):
    registry, binding_id = binding_registry
    db = registry.db
    main_connection = db._held_connection()
    if outcome in {"default", "repair_error"}:
        _insert_stale_default_binding(db, "stale-default")
        binding_id = "stale-default"
    elif outcome == "missing":
        binding_id = "absent"
    generation = registry.mutation_generation
    connections = []
    original = db._get_connection

    def acquire():
        connection = original()
        connections.append(connection)
        if outcome in {"read_error", "repair_error"}:
            denied = (
                sqlite3.SQLITE_READ
                if outcome == "read_error"
                else sqlite3.SQLITE_DELETE
            )
            connection.set_authorizer(
                lambda action, table, *_: (
                    sqlite3.SQLITE_DENY
                    if action == denied and table == "workspace_runtime_bindings"
                    else sqlite3.SQLITE_OK
                )
            )
        return connection

    monkeypatch.setattr(db, "_get_connection", acquire)

    def worker():
        try:
            if outcome.endswith("error"):
                with pytest.raises(WorkspaceRegistryServiceError):
                    registry.get_runtime_binding(binding_id)
            else:
                result = registry.get_runtime_binding(binding_id)
                assert (result is not None) is (outcome == "found")
            _assert_retired(db, connections)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=10)
    assert main_connection.execute("SELECT 1").fetchone()[0] == 1
    if outcome in {"default", "repair_error"}:
        row = main_connection.execute(
            "SELECT 1 FROM workspace_runtime_bindings WHERE binding_id = ?",
            (binding_id,),
        ).fetchone()
        assert (row is None) is (outcome == "default")
        assert registry.mutation_generation == generation + (outcome == "default")


@pytest.mark.parametrize("transaction", [False, True])
def test_binding_read_preserves_borrowed_worker_handle(binding_registry, transaction):
    registry, binding_id = binding_registry
    db = registry.db

    def worker():
        borrowed = db._held_connection()
        try:
            if transaction:
                borrowed.execute("BEGIN")
                borrowed.execute(
                    "UPDATE workspace_runtime_bindings SET label = 'uncommitted' WHERE binding_id = ?",
                    (binding_id,),
                )
            result = registry.get_runtime_binding(binding_id)
            assert result is not None
            assert db._held_connection() is borrowed
            assert borrowed.in_transaction is transaction
            if transaction:
                assert result.label == "uncommitted"
                borrowed.rollback()
                assert registry.get_runtime_binding(binding_id).label != "uncommitted"
            assert borrowed.execute("SELECT 1").fetchone()[0] == 1
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=10)


def test_repeated_binding_reads_retire_reopened_stale_cache(
    binding_registry, monkeypatch
):
    registry, binding_id = binding_registry
    db = registry.db
    connections = []
    original = db._get_connection

    def acquire():
        connection = original()
        connections.append(connection)
        return connection

    monkeypatch.setattr(db, "_get_connection", acquire)

    def worker():
        db._held_connection().close()
        try:
            for _ in range(2):
                assert registry.get_runtime_binding(binding_id) is not None
                _assert_retired(db, connections)
            assert len(connections) == 3
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=10)


def test_binding_native_close_failure_retains_exact_lease(
    binding_registry, monkeypatch
):
    registry, binding_id = binding_registry
    db = registry.db
    original = db._get_connection
    connections = []
    attempts = []

    def acquire():
        connection = original()
        connections.append((connection, connection.close))

        def fail_close():
            attempts.append(connection)
            raise sqlite3.OperationalError("injected native close failure")

        connection.close = fail_close
        return connection

    monkeypatch.setattr(db, "_get_connection", acquire)

    def worker():
        try:
            assert registry.get_runtime_binding(binding_id) is not None
            connection, close = connections[0]
            assert attempts == [connection]
            assert db._thread_local.conn is connection
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            with storage._lock:
                assert connection in db._maintenance_participant.connections
            connection.close = close
            db.close()
            _assert_retired(db, [connection])
        finally:
            for connection, close in connections:
                connection.close = close
            db.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=10)


@pytest.mark.asyncio
async def test_cancelled_binding_waiter_preserves_live_producer_and_other_reader(
    binding_registry, monkeypatch
):
    registry, binding_id = binding_registry
    db = registry.db
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    observed = {}
    errors = []
    original = db._get_connection

    def acquire():
        connection = original()
        name = threading.current_thread().name
        observed[name] = connection
        if name.startswith("binding-held"):

            def hold(action, table, *_):
                if (
                    action == sqlite3.SQLITE_READ
                    and table == "workspace_runtime_bindings"
                ):
                    entered.set()
                    assert release.wait(5)
                return sqlite3.SQLITE_OK

            connection.set_authorizer(hold)
        return connection

    monkeypatch.setattr(db, "_get_connection", acquire)

    def read():
        try:
            assert registry.get_runtime_binding(binding_id) is not None
            connection = observed[threading.current_thread().name]
            _assert_retired(db, [connection])
        except BaseException as error:
            errors.append(error)
            raise
        finally:
            db.close()
            if threading.current_thread().name.startswith("binding-held"):
                settled.set()

    loop = asyncio.get_running_loop()
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="binding-held") as held,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="binding-other") as other,
    ):
        pending = loop.run_in_executor(held, read)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await loop.run_in_executor(other, read)
            assert not settled.is_set()
            with storage._lock:
                remaining = tuple(db._maintenance_participant.connections.values())
            assert any(
                lease.resource_thread.name.startswith("binding-held")
                for lease in remaining
            )
            assert not any(
                lease.resource_thread.name.startswith("binding-other")
                for lease in remaining
            )
        finally:
            release.set()
            assert await asyncio.to_thread(settled.wait, 5)
        assert not errors


def test_binding_read_inside_an_outer_operation_defers_to_its_owner(binding_registry):
    """A nested finite read leaves retirement to the enclosing operation owner.

    The project-index provider owns one registry handle across its whole
    operation; an inner binding read must not retire it before the owner's
    own pins are taken and released.
    """
    from tldw_chatbook.DB.base_db import operation_owned_connection

    registry, binding_id = binding_registry
    db = registry.db
    seen = []

    def worker():
        try:
            with operation_owned_connection(db):
                assert registry.get_runtime_binding(binding_id) is not None
                held = db._thread_local.conn
                assert held is not None
                assert held.execute("SELECT 1").fetchone()[0] == 1
                seen.append(held)
            _assert_retired(db, seen)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=10)
