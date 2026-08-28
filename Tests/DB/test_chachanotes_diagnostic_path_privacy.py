"""Born-red runtime privacy contracts for ChaChaNotes path diagnostics."""

import sqlite3
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.DB.ChaChaNotes_DB as chachanotes_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
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


def test_integrity_failure_logs_stable_metadata_without_database_path(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-integrity.sqlite"
    raw_exception = f"TASK-19864 integrity failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref

    def fail_integrity_connection() -> sqlite3.Connection:
        raise sqlite3.OperationalError(raw_exception)

    monkeypatch.setattr(database, "get_connection", fail_integrity_connection)
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        assert database.check_integrity() is False
    finally:
        logger.remove(sink_id)
        database.close()

    rendered = "".join(records)
    assert "Failed to check database integrity" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered
    assert expected_ref == content_fingerprint(str(database_path))
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=OperationalError" in rendered


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


def test_vacuum_failure_preserves_caller_error_and_logs_only_safe_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "task-19864-private-vacuum.sqlite"
    raw_exception = f"TASK-19864 vacuum failure repeated path={database_path}"
    database = CharactersRAGDB(database_path, client_id="task-19864")
    expected_ref = database._db_diagnostic_ref
    connection = database.get_connection()

    class VacuumFailureConnection:
        def execute(self, statement: str, *args: object) -> sqlite3.Cursor:
            if statement == "VACUUM":
                raise sqlite3.OperationalError(raw_exception)
            return connection.execute(statement, *args)

    monkeypatch.setattr(
        database,
        "get_connection",
        lambda: VacuumFailureConnection(),
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            database.vacuum()
    finally:
        logger.remove(sink_id)
        database.close()

    error = exc_info.value
    assert str(error) == f"Vacuum failed: {raw_exception}"
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    assert str(error.__cause__) == raw_exception

    rendered = "".join(records)
    assert "Failed to vacuum database" in rendered
    assert str(database_path) not in rendered
    assert database_path.name not in rendered
    assert raw_exception not in rendered
    assert "Traceback (most recent call last)" not in rendered
    assert expected_ref == content_fingerprint(str(database_path))
    assert f"db_sha256={expected_ref}" in rendered
    assert "exception_type=OperationalError" in rendered
