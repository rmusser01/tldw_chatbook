"""Real indexing SQLite schema, capture and current-thread retirement."""

import sqlite3
import threading
from datetime import UTC, datetime
from threading import Event

import pytest

from Tests.Backup_Recovery.test_core_owners import application_authority
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.models import StorageItem
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB


def test_cached_indexing_borrower_refuses_pause(tmp_path):
    owner = RAGIndexingDB(tmp_path / "index.db")
    pause = storage._begin_local_pause()
    try:
        with (
            pytest.raises(RecoveryRequired, match="storage_locally_paused"),
            owner.connection(),
        ):
            pass
    finally:
        pause.resume()
        owner.close()


def test_indexing_close_preserves_live_transaction_then_retires(tmp_path):
    owner = RAGIndexingDB(tmp_path / "index.db")
    try:
        with owner.transaction() as connection:
            owner.close()
            assert connection.execute("SELECT 42").fetchone()[0] == 42
        owner.close()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 42")
        with owner.connection() as replacement:
            assert (
                replacement.execute("SELECT COUNT(*) FROM indexed_items").fetchone()[0]
                == 0
            )
    finally:
        owner.close()


def test_indexing_snapshot_preserves_schema_rows_and_source(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.rag_indexing import recovery_adapters

    adapter = recovery_adapters()[0]
    path = tmp_path / "index.db"
    owner = RAGIndexingDB(path)
    owner.mark_item_indexed(
        "same-id", "media", datetime.now(UTC), metadata={"content": "retained"}
    )
    with owner.connection() as connection:
        expected = tuple(connection.iterdump())
    owner.close()
    before = path.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, path, monkeypatch)
    item = StorageItem(
        adapter.owner_id, "profile:test:" + adapter.owner_id, path, "included", ()
    )
    destination = stage / "snapshot.db"
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((path,), stage),
    ):
        adapter.capture(item, destination, Event())
        assert adapter.validate(destination) == ()
    assert path.read_bytes() == before
    with sqlite3.connect(destination) as connection:
        assert tuple(connection.iterdump()) == expected


def test_indexing_native_lease_retires_on_owning_worker(tmp_path):
    errors = []
    observed = []

    def work():
        try:
            owner = RAGIndexingDB(tmp_path / "thread.db")
            with owner.connection() as connection:
                with storage._lock:
                    leases = [
                        lease
                        for lease in storage._live_leases
                        if lease.resource_path == owner.db_path
                    ]
                assert leases and all(
                    lease.resource_thread is threading.current_thread()
                    for lease in leases
                )
                observed.extend(leases)
            owner.close()
            with storage._lock:
                assert not any(lease in storage._live_leases for lease in leases)
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
        except Exception as error:  # noqa: BLE001 - forward worker assertion to test
            errors.append(error)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and not errors, errors
    assert observed


def test_indexing_unknown_schema_is_rejected(tmp_path):
    from tldw_chatbook.Backup_Recovery.rag_indexing import recovery_adapters

    path = tmp_path / "index.db"
    owner = RAGIndexingDB(path)
    with owner.connection() as connection:
        connection.execute("CREATE TABLE unknown_extension(value TEXT)")
    owner.close()
    assert recovery_adapters()[0].validate(path) == ("unsupported_schema",)


def test_idle_indexing_cache_retires_through_actual_caller_pause(tmp_path):
    from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches

    owner = RAGIndexingDB(tmp_path / "idle.db")
    with owner.connection() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    finally:
        pause.resume()
        owner.close()
    with owner.connection() as replacement:
        assert (
            replacement.execute("SELECT COUNT(*) FROM indexed_items").fetchone()[0] == 0
        )
    owner.close()
