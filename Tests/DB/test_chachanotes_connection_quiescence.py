"""All-thread ChaChaNotes connection quiescence for physical maintenance."""

from __future__ import annotations

import threading
import sqlite3
import time
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.base_db import (
    SQLiteConnectionQuiescenceRegistry,
    _QuiescentSQLiteConnection,
)


@pytest.mark.parametrize("abort", [False, True])
def test_backup_holds_quiescence_reservation_and_releases_after_exit(
    abort: bool,
) -> None:
    registry = SQLiteConnectionQuiescenceRegistry()
    source = sqlite3.connect(":memory:", factory=_QuiescentSQLiteConnection)
    destination = sqlite3.connect(":memory:")
    source.attach_quiescence_registry(registry)
    callbacks = []

    def progress(status: int, remaining: int, total: int) -> None:
        callbacks.append((status, remaining, total))
        with pytest.raises(TimeoutError, match="connection_quiescence_timeout"):
            registry.begin_quiescence(timeout_seconds=0.001)
        if abort:
            raise RuntimeError("injected backup cancellation")

    try:
        source.execute("CREATE TABLE payload(value INTEGER)").close()
        source.execute("INSERT INTO payload VALUES (42)").close()
        source.commit()
        if abort:
            with pytest.raises(RuntimeError, match="injected backup cancellation"):
                source.backup(destination, pages=1, progress=progress, sleep=0)
        else:
            source.backup(destination, pages=1, progress=progress, sleep=0)
            assert destination.execute("SELECT value FROM payload").fetchall() == [
                (42,)
            ]
        assert callbacks
        token = registry.begin_quiescence(timeout_seconds=0.01)
        registry.end_quiescence(token)
        assert source.execute("SELECT value FROM payload").fetchall() == [(42,)]
    finally:
        source.close()
        destination.close()


def test_acquisition_started_before_barrier_registers_then_is_closed() -> None:
    registry = SQLiteConnectionQuiescenceRegistry()
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    errors: list[BaseException] = []
    registry.begin_acquisition()

    def maintenance() -> None:
        try:
            token = registry.begin_quiescence(timeout_seconds=0.5)
            try:
                registry.close_registered(token)
            finally:
                registry.end_quiescence(token)
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)

    thread = threading.Thread(target=maintenance, daemon=True)
    thread.start()
    deadline = time.monotonic() + 0.5
    while True:
        try:
            registry.begin_acquisition()
        except RuntimeError as exc:
            assert str(exc) == "database_maintenance_in_progress"
            break
        else:
            registry.finish_acquisition()
        assert time.monotonic() < deadline
        time.sleep(0.001)

    registry.register(connection)
    registry.finish_acquisition()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_quiescence_discards_an_already_closed_registered_handle() -> None:
    registry = SQLiteConnectionQuiescenceRegistry()
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    registry.begin_acquisition()
    registry.register(connection)
    registry.finish_acquisition()
    connection.close()

    token = registry.begin_quiescence(timeout_seconds=0.5)
    try:
        registry.close_registered(token)
    finally:
        registry.end_quiescence(token)

    assert registry.connection_count() == 0


def test_quiescence_rejects_acquisition_closes_handles_and_reopens(
    tmp_path: Path,
) -> None:
    database = CharactersRAGDB(tmp_path / "quiescence.sqlite", "quiescence")
    original = database.get_connection()

    with database.quiesce_connections(timeout_seconds=0.5):
        assert database.registered_connection_count() == 0
        with pytest.raises(CharactersRAGDBError, match="maintenance_in_progress"):
            database.get_connection()

    reopened = database.get_connection()
    assert reopened is not original
    assert reopened.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(Exception, match="closed database"):
        original.execute("SELECT 1")
    database.close_connection()


def test_quiescence_waits_for_a_transaction_on_another_thread(
    tmp_path: Path,
) -> None:
    database = CharactersRAGDB(tmp_path / "threaded.sqlite", "threaded")
    transaction_started = threading.Event()
    release_transaction = threading.Event()
    transaction_finished = threading.Event()
    resume_worker = threading.Event()
    acquisition_rejected = threading.Event()
    maintenance_finished = threading.Event()
    worker_reopened = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with database.transaction(immediate=True) as cursor:
                cursor.execute("SELECT 1")
                transaction_started.set()
                assert release_transaction.wait(2.0)
            transaction_finished.set()
            assert resume_worker.wait(2.0)
            with pytest.raises(CharactersRAGDBError, match="maintenance_in_progress"):
                database.get_connection()
            acquisition_rejected.set()
            assert maintenance_finished.wait(2.0)
            assert database.get_connection().execute("SELECT 1").fetchone()[0] == 1
            worker_reopened.set()
            database.close_connection()
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
            transaction_started.set()
            transaction_finished.set()
            acquisition_rejected.set()
            worker_reopened.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert transaction_started.wait(2.0)

    with pytest.raises(TimeoutError, match="connection_quiescence_timeout"):
        with database.quiesce_connections(timeout_seconds=0.02):
            pass

    # A timed-out admission must restore ordinary access immediately.
    assert database.get_connection().execute("SELECT 1").fetchone()[0] == 1
    release_transaction.set()
    assert transaction_finished.wait(2.0)

    with database.quiesce_connections(timeout_seconds=0.5):
        assert database.registered_connection_count() == 0
        resume_worker.set()
        assert acquisition_rejected.wait(2.0)
        assert not worker_reopened.is_set()

    maintenance_finished.set()
    assert worker_reopened.wait(2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert errors == []
    database.close_connection()


def test_quiescence_refuses_to_begin_inside_the_callers_transaction(
    tmp_path: Path,
) -> None:
    database = CharactersRAGDB(tmp_path / "nested.sqlite", "nested")
    try:
        with database.transaction(immediate=True):
            with pytest.raises(TimeoutError, match="connection_quiescence_timeout"):
                with database.quiesce_connections(timeout_seconds=0.01):
                    pass
    finally:
        database.close_connection()


def test_quiescence_waits_for_a_direct_read_cursor_to_finish(tmp_path: Path) -> None:
    database = CharactersRAGDB(tmp_path / "direct-read.sqlite", "direct-read")
    cursor = database.get_connection().execute(
        "WITH RECURSIVE rows(value) AS ("
        "SELECT 1 UNION ALL SELECT value + 1 FROM rows WHERE value < 100"
        ") SELECT value FROM rows"
    )
    try:
        assert cursor.fetchone()[0] == 1
        with pytest.raises(TimeoutError, match="connection_quiescence_timeout"):
            with database.quiesce_connections(timeout_seconds=0.01):
                pass
        assert cursor.fetchone()[0] == 2
    finally:
        cursor.close()

    with database.quiesce_connections(timeout_seconds=0.5):
        assert database.registered_connection_count() == 0
    database.close_connection()


def test_quiescence_coordinates_separate_instances_for_the_same_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.sqlite"
    first = CharactersRAGDB(path, "shared-first")
    second = CharactersRAGDB(path, "shared-second")
    second_connection = second.get_connection()

    with first.quiesce_connections(timeout_seconds=0.5):
        assert first.registered_connection_count() == 0
        with pytest.raises(CharactersRAGDBError, match="maintenance_in_progress"):
            second.get_connection()

    reopened = second.get_connection()
    assert reopened is not second_connection
    assert reopened.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        second_connection.execute("SELECT 1")
    first.close_connection()
    second.close_connection()
