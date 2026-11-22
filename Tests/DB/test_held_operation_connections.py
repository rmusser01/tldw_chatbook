"""TASK-40608 D1: on Windows an owned call keeps its Console handle open.

Opening a private SQLite connection on Windows re-verifies the folder chain and
both sidecars with native handles, so an owned call that opened and closed one
per call paid that every time. The handle now stays in its worker thread's
cache, and a local pause, app quit or the owner's explicit close retires the
idle ones through ``retire_held_connections``. POSIX keeps the per-call close.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import close_settled_core_cache
from tldw_chatbook.Backup_Recovery import held_connections
from tldw_chatbook.Backup_Recovery.held_connections import (
    held_connection_count,
    retire_held_connections,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

pytestmark = pytest.mark.bootstrap_profile

windows_only = pytest.mark.skipif(os.name != "nt", reason="handles are held on Windows only")


def _connection(db):
    return db.get_connection() if type(db) is CharactersRAGDB else db._held_connection()


def _close(db):
    retire_held_connections(db)
    if type(db) is CharactersRAGDB:
        db.close_connection()
    else:
        db.close()


def _closed(connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.fixture(autouse=True)
def no_held_connections_leak():
    """Every test starts and ends with no held handle from another test."""
    assert retire_held_connections()
    yield
    assert retire_held_connections(), "a held handle was still in use at teardown"
    assert held_connection_count() == 0


@windows_only
@pytest.mark.parametrize("database_type", [CharactersRAGDB, AgentRunsDB, WorkspaceDB])
def test_an_owned_call_keeps_its_handle_and_the_next_call_reuses_it(
    tmp_path, database_type
):
    db = database_type(tmp_path / "held.sqlite", "held")
    _close(db)

    def call():
        with operation_owned_connection(db):
            connection = _connection(db)
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            return connection

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(call).result(timeout=20)
            assert not _closed(first)
            assert held_connection_count() == 1
            second = executor.submit(call).result(timeout=20)
            assert second is first, "the second owned call opened a new handle"
            assert held_connection_count() == 1
            assert retire_held_connections(db)
            assert _closed(first)
            third = executor.submit(call).result(timeout=20)
            assert third is not first and not _closed(third)
    finally:
        _close(db)


@pytest.mark.skipif(os.name == "nt", reason="the per-call close is the POSIX contract")
def test_posix_still_closes_the_handle_an_owned_call_opened(tmp_path):
    db = WorkspaceDB(tmp_path / "posix.sqlite", "posix")
    db.close()

    def call():
        with operation_owned_connection(db):
            return db._held_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        connection = executor.submit(call).result(timeout=20)
    assert _closed(connection)
    assert held_connection_count() == 0


@windows_only
def test_other_stores_and_memory_stores_keep_the_per_call_close(tmp_path):
    collections = LibraryCollectionsDB(tmp_path / "collections.sqlite", "per-call")
    collections.close()

    def call():
        with operation_owned_connection(collections):
            return collections._held_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        connection = executor.submit(call).result(timeout=20)
    assert _closed(connection)
    assert held_connection_count() == 0


@windows_only
def test_a_handle_in_use_is_not_retired(tmp_path):
    db = WorkspaceDB(tmp_path / "busy.sqlite", "busy")
    db.close()
    entered, release = threading.Event(), threading.Event()
    seen = {}

    def first_call():
        with operation_owned_connection(db):
            seen["connection"] = db._held_connection()

    def running_call():
        with operation_owned_connection(db):
            entered.set()
            assert release.wait(20)
            return db._held_connection().execute("SELECT 1").fetchone()[0]

    def open_transaction():
        seen["connection"].execute("BEGIN")

    def roll_back():
        seen["connection"].rollback()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(first_call).result(timeout=20)
            future = executor.submit(running_call)
            assert entered.wait(20)
            assert not retire_held_connections(db), "closed under a running owned call"
            assert not _closed(seen["connection"])
            release.set()
            assert future.result(timeout=20) == 1

            executor.submit(open_transaction).result(timeout=20)
            assert not retire_held_connections(db), "closed inside a transaction"
            executor.submit(roll_back).result(timeout=20)
            assert retire_held_connections(db)
            assert _closed(seen["connection"])
    finally:
        release.set()
        _close(db)


@windows_only
def test_a_handle_whose_worker_thread_ended_is_closed_by_the_next_hold(tmp_path):
    db = WorkspaceDB(tmp_path / "ended.sqlite", "ended")
    db.close()
    seen = []

    def call():
        with operation_owned_connection(db):
            seen.append(db._held_connection())

    try:
        worker = threading.Thread(target=call)
        worker.start()
        worker.join(20)
        assert held_connection_count() == 1 and not _closed(seen[0])
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(call).result(timeout=20)
            assert _closed(seen[0]), "the ended worker's handle stayed open"
            assert held_connection_count() == 1
    finally:
        _close(db)


@windows_only
def test_the_cap_falls_back_to_the_per_call_close(tmp_path, monkeypatch):
    monkeypatch.setattr(held_connections, "_HELD_CONNECTIONS_MAX", 0)
    db = WorkspaceDB(tmp_path / "cap.sqlite", "cap")
    db.close()

    def call():
        with operation_owned_connection(db):
            return db._held_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        connection = executor.submit(call).result(timeout=20)
    assert _closed(connection)
    assert held_connection_count() == 0


@windows_only
def test_a_local_pause_closes_idle_held_handles_and_then_drains(tmp_path):
    import asyncio
    import time

    from tldw_chatbook.Backup_Recovery.participants import retire_current_thread_caches

    db = WorkspaceDB(tmp_path / "pause.sqlite", "pause")
    db.close()
    seen = {}

    def call():
        with operation_owned_connection(db):
            seen["connection"] = db._held_connection()

    async def pause_and_drain():
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic()), "a held handle must block the drain"
            retire_current_thread_caches(pause)
            assert _closed(seen["connection"])
            assert pause.drain(time.monotonic() + 5)
        finally:
            pause.resume()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(call).result(timeout=20)
            assert not _closed(seen["connection"])
            asyncio.run(pause_and_drain())
            # After resume the worker opens a fresh handle.
            executor.submit(call).result(timeout=20)
            assert not _closed(seen["connection"])
    finally:
        _close(db)


@windows_only
def test_no_handle_is_held_once_storage_is_paused(tmp_path):
    import time

    db = WorkspaceDB(tmp_path / "paused.sqlite", "paused")
    connection = db._held_connection()
    pause = storage._begin_local_pause()
    try:
        assert not held_connections.hold_operation_connection(db, connection)
        assert held_connection_count() == 0
    finally:
        db.close()
        assert pause.drain(time.monotonic() + 5)
        pause.resume()


@windows_only
def test_the_owner_close_retires_held_handles_first(tmp_path):
    db = WorkspaceDB(tmp_path / "owner-close.sqlite", "owner-close")
    db.close()
    seen = {}

    def call():
        with operation_owned_connection(db):
            seen["connection"] = db._held_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(call).result(timeout=20)
        assert not _closed(seen["connection"])
        assert close_settled_core_cache(db)
        assert _closed(seen["connection"])
        assert held_connection_count() == 0


@windows_only
def test_the_project_index_accepts_a_held_registry_handle(tmp_path):
    """Its per-operation close proof must not refuse a handle held on purpose.

    Before this the first index operation on a worker without a cached handle
    raised ``index_registry_retirement_unproven``, which refused every later
    index operation and failed the app's shutdown.
    """
    from tldw_chatbook.Backup_Recovery.held_connections import is_held_connection
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    db = WorkspaceDB(tmp_path / "index-registry.sqlite", "index-registry")
    db.close()
    seen = {}

    def operation():
        with project_index_registry_connections(db):
            seen["connection"] = db._held_connection()
            seen["connection"].execute("SELECT 1").fetchone()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(operation).result(timeout=20)
            assert is_held_connection(seen["connection"])
            executor.submit(operation).result(timeout=20)  # reuses it, still no refusal
            assert retire_held_connections(db)
            assert not is_held_connection(seen["connection"])
    finally:
        _close(db)


def test_the_project_index_still_proves_a_per_call_close(tmp_path, monkeypatch):
    """With handles not held, the original close proof is unchanged."""
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    monkeypatch.setattr(held_connections, "_HOLD_OPERATION_CONNECTIONS", False)
    db = WorkspaceDB(tmp_path / "index-proof.sqlite", "index-proof")
    db.close()
    seen = {}

    def operation():
        with project_index_registry_connections(db):
            seen["connection"] = db._held_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(operation).result(timeout=20)
    assert _closed(seen["connection"])
    assert held_connection_count() == 0
