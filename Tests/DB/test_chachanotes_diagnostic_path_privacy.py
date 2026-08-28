"""Born-red runtime privacy contracts for ChaChaNotes path diagnostics."""

import sqlite3
from pathlib import Path

from loguru import logger

import tldw_chatbook.DB.ChaChaNotes_DB as chachanotes_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Utils.log_sanitizer import content_fingerprint


def test_file_database_caches_one_diagnostic_fingerprint(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-cached.sqlite"
    fingerprint_calls: list[object] = []

    def record_fingerprint(value: object) -> str:
        fingerprint_calls.append(value)
        return "cached-diagnostic-ref"

    monkeypatch.setattr(
        chachanotes_module,
        "content_fingerprint",
        record_fingerprint,
        raising=False,
    )
    database = CharactersRAGDB(database_path, client_id="task-19864")
    try:
        assert database.check_integrity() is True
        assert database._db_diagnostic_ref == "cached-diagnostic-ref"
    finally:
        database.close()
    assert fingerprint_calls == [str(database_path)]


def test_memory_database_uses_fixed_diagnostic_reference(
    monkeypatch,
) -> None:
    fingerprint_calls: list[object] = []

    def record_fingerprint(value: object) -> str:
        fingerprint_calls.append(value)
        return "must-not-be-used"

    monkeypatch.setattr(
        chachanotes_module,
        "content_fingerprint",
        record_fingerprint,
        raising=False,
    )
    database = CharactersRAGDB(":memory:", client_id="task-19864")
    try:
        assert database._db_diagnostic_ref == "memory"
    finally:
        database.close()
    assert fingerprint_calls == []


def test_backup_failure_logs_stable_metadata_without_database_paths(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-primary.sqlite"
    backup_path = tmp_path / "task-19864-private-backup.sqlite"
    raw_exception = f"TASK-19864 backup failed from {database_path} to {backup_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")

    def fail_backup(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError(raw_exception)

    monkeypatch.setattr(chachanotes_module, "backup_connection_to_private", fail_backup)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        assert database.backup_database(str(backup_path)) is False
    finally:
        logger.remove(sink_id)
        database.close()

    rendered = "".join(records)
    assert "Starting database backup" in rendered
    assert "SQLite error during database backup" in rendered
    assert f"db_sha256={content_fingerprint(str(database_path))}" in rendered
    assert f"backup_sha256={content_fingerprint(str(backup_path))}" in rendered
    assert "exception_type=OperationalError" in rendered
    assert str(database_path) not in rendered
    assert str(backup_path) not in rendered
    assert database_path.name not in rendered
    assert backup_path.name not in rendered
    assert raw_exception not in rendered
