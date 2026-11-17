"""Finite callbacks retire replacement handles, not unchanged native borrowers."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
def test_metadata_retires_replacement_after_exact_file_quiescence(tmp_path, borrowed):
    """A stale borrowed flag must not retain a reopened worker connection."""
    db = CharactersRAGDB(tmp_path / "departed-profile.sqlite", "late-metadata")
    db.close_connection()
    entered, release = threading.Event(), threading.Event()

    def read():
        previous = db.get_connection() if borrowed else None
        with operation_owned_connection(db):
            entered.set()
            assert release.wait(5), "metadata callback was not released"
            metadata = ConsoleCharacterContextController._read_database_scope_metadata(
                db
            )
            replacement = db._local.conn
        return previous, replacement, metadata

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read)
        try:
            assert entered.wait(5), "callback did not enter its ownership guard"
            assert db.registered_connection_count() == int(borrowed)
            with db.quiesce_connections(timeout_seconds=5):
                assert db.registered_connection_count() == 0
            release.set()
            previous, replacement, metadata = future.result(timeout=5)
            assert metadata[0] and metadata[1] >= 0
            assert replacement is not previous
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                replacement.execute("SELECT 1")
            assert db.registered_connection_count() == 0
        finally:
            release.set()
            try:
                future.result(timeout=5)
            finally:
                # Settle physical work before retiring this exact test-owned file.
                with db.quiesce_connections(timeout_seconds=5):
                    pass


@pytest.mark.parametrize(
    "database_type", [AgentRunsDB, WorkspaceDB, LibraryCollectionsDB]
)
@pytest.mark.parametrize("replace", [False, True], ids=["unchanged", "replacement"])
@pytest.mark.parametrize("fail_sql", [False, True], ids=["success", "sql-error"])
def test_core_guard_retires_only_replacement_borrowers(
    tmp_path, database_type, replace, fail_sql
):
    """Cleanup follows native identity on success/error, preserving live transactions."""
    db = database_type(tmp_path / "core-owner.sqlite", "finite-core")
    try:
        previous = db._held_connection()
        previous.execute("BEGIN")
        error = (
            pytest.raises(sqlite3.OperationalError, match="no such table")
            if fail_sql
            else nullcontext()
        )
        with error, operation_owned_connection(db):
            if replace:
                db.close()
            current = db._held_connection()
            assert current.execute("SELECT 1").fetchone()[0] == 1
            if fail_sql:
                # Let an actual SQLite error cross the guard's finally boundary.
                current.execute("SELECT * FROM missing_finite_operation_table")
        if replace:
            assert current is not previous
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                current.execute("SELECT 1")
        else:
            assert current is previous
            assert current.in_transaction
            assert current.execute("SELECT 1").fetchone()[0] == 1
            current.rollback()
    finally:
        db.close()


def test_closing_worker_does_not_hold_other_writers_behind_a_live_reader(tmp_path):
    """TASK-41115: a finite worker's close (``run_owned_db_call`` passes no
    deadline) ran ``wal_checkpoint(TRUNCATE)`` with the 15 s busy timeout. A
    pending TRUNCATE checkpoint holds the WAL writer lock while it waits for
    readers, so the app's own writes stalled behind it, and under load
    ``add_note`` failed ``database is locked``."""
    import sys
    import time

    path = tmp_path / "close-checkpoint.sqlite"
    db = CharactersRAGDB(path, "close-checkpoint")
    reader = sqlite3.connect(path)
    executor = ThreadPoolExecutor(max_workers=1)
    worker = {}
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM notes").fetchone()
        db.add_note("Seed", "WAL frames newer than the reader's snapshot.")

        def read():
            worker["ident"] = threading.get_ident()
            with operation_owned_connection(db):
                db.get_connection().execute("SELECT COUNT(*) FROM notes").fetchone()

        future = executor.submit(read)
        # Write while the worker closes its handle (or once it has closed).
        deadline = time.monotonic() + 10
        while not future.done() and time.monotonic() < deadline:
            frame = sys._current_frames().get(worker.get("ident"))
            names = set()
            while frame is not None:
                names.add(frame.f_code.co_name)
                frame = frame.f_back
            if "_close_connection_handle" in names:
                break
            time.sleep(0.005)
        started = time.monotonic()
        db.add_note("Main", "A write while another thread closes.")
        waited = time.monotonic() - started
        future.result(timeout=20)
        assert reader.in_transaction
        assert waited < 3, f"the write waited {waited:.1f}s behind a closing worker"
    finally:
        reader.rollback()
        reader.close()
        executor.shutdown(wait=True)
        db.close_connection()


def test_cleanup_deadline_preserves_borrowed_timeout_and_transaction(tmp_path):
    db = CharactersRAGDB(tmp_path / "borrowed-deadline.sqlite", "borrowed-deadline")
    try:
        connection = db.get_connection()
        connection.execute("PRAGMA busy_timeout = 3141")
        connection.execute("BEGIN")
        with operation_owned_connection(db, operation_deadline=0):
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert connection.in_transaction
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 3141
    finally:
        db.close_connection()


@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_deadline_tolerates_already_retired_native_handle(tmp_path, body_fails):
    db = CharactersRAGDB(tmp_path / "retired-deadline.sqlite", "retired-deadline")
    db.close_connection()
    error = (
        pytest.raises(RuntimeError, match="original operation failure")
        if body_fails
        else nullcontext()
    )
    try:
        with error, operation_owned_connection(db, operation_deadline=0):
            connection = db.get_connection()
            with db.quiesce_connections(timeout_seconds=2):
                pass
            assert db._local.conn is connection
            if body_fails:
                raise RuntimeError("original operation failure")
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
        assert db.registered_connection_count() == 0
    finally:
        db.close_connection()


def test_expiry_after_connection_registration_bounds_internal_checkpoint(
    tmp_path, monkeypatch
):
    """Acquisition's own close must honor the deadline before outer cleanup."""
    import time

    path = tmp_path / "late-registration.sqlite"
    db = CharactersRAGDB(path, "late-registration")
    reader = sqlite3.connect(path)
    executor = ThreadPoolExecutor(max_workers=1)
    opened = []
    original_register = db._connection_quiescence.register

    def register(conn):
        original_register(conn)
        opened.append(conn)
        time.sleep(0.35)  # Cross the acquisition deadline after setup succeeds.

    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM conversations").fetchone()
        db.add_conversation({"title": "WAL beyond the reader snapshot"})
        monkeypatch.setattr(db._connection_quiescence, "register", register)

        def acquire():
            with pytest.raises(TimeoutError, match="budget exhausted"):
                db.get_connection(operation_deadline=time.monotonic() + 0.3)

        future = executor.submit(acquire)
        future.result(timeout=2)
        assert reader.in_transaction
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
        assert db.registered_connection_count() == 1  # Unchanged main borrower.
    finally:
        reader.rollback()
        reader.close()
        executor.shutdown(wait=True)
        db.close_connection()
