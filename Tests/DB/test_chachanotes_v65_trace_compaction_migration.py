"""ChaChaNotes v64 -> v65 physical trace-maintenance status migration."""

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_v65_adds_singleton_content_free_compaction_status(tmp_path: Path) -> None:
    path = tmp_path / "trace-compaction-v64.sqlite"
    with chachanotes_db_at_version(path, 64):
        pass

    upgraded = CharactersRAGDB(path, client_id="trace-compaction-v65")
    try:
        connection = upgraded.get_connection()
        version = upgraded._get_db_version(connection)
        row = connection.execute(
            "SELECT status, reason_code, retry_count, next_retry_at, "
            "progress_basis_points, allocated_bytes_before, allocated_bytes_after, "
            "freelist_bytes_before, freelist_bytes_after, wal_bytes_before, "
            "wal_bytes_after, logical_live_bytes "
            "FROM console_trace_compaction_state WHERE singleton_id = 1"
        ).fetchone()

        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert tuple(row) == (
            "pending",
            "awaiting_gc",
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        assert upgraded.get_console_trace_compaction_status() == {
            "status": "pending",
            "reason_code": "awaiting_gc",
            "retry_count": 0,
            "retry_pending": False,
            "progress_basis_points": 0,
            "allocated_bytes_before": 0,
            "allocated_bytes_after": 0,
            "freelist_bytes_before": 0,
            "freelist_bytes_after": 0,
            "wal_bytes_before": 0,
            "wal_bytes_after": 0,
            "logical_live_bytes": 0,
        }
    finally:
        upgraded.close_connection()


def test_v65_compaction_status_enforces_bounds_and_singleton_identity(
    tmp_path: Path,
) -> None:
    database = CharactersRAGDB(tmp_path / "trace-compaction-v65.sqlite", "v65")
    try:
        connection = database.get_connection()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE console_trace_compaction_state "
                "SET progress_basis_points = 10001 WHERE singleton_id = 1"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE console_trace_compaction_state "
                "SET singleton_id = 2 WHERE singleton_id = 1"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM console_trace_compaction_state WHERE singleton_id = 1"
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM console_trace_compaction_state"
        ).fetchone()[0] == 1
    finally:
        database.close_connection()
