"""ChaChaNotes v62 -> v63 epoch-safe Console trace GC guards."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_retention_expiry_index_is_selected_without_statistics() -> None:
    """The collector's expiry predicate uses its index without ANALYZE."""

    db = CharactersRAGDB(":memory:", client_id="trace-gc-query-plans")
    try:
        connection = db.get_connection()
        assert connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'sqlite_stat1'"
        ).fetchone() is None
        plan = "\n".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT entity_id FROM console_trace_retention_roots "
                "WHERE julianday(retain_until) <= julianday('now')"
            )
        )
        assert "idx_console_trace_retention_expiry" in plan, plan
    finally:
        db.close_connection()


def test_genuine_v62_reopen_installs_gc_metadata_and_fail_closed_guards(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-gc-v62.sqlite"
    original_target = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 62
    try:
        historical = CharactersRAGDB(path, client_id="trace-gc-v62")
        historical.close_connection()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_target

    migrated = CharactersRAGDB(path, client_id="trace-gc-v63")
    try:
        connection = migrated.get_connection()
        # Unpinned construction migrates to the CURRENT version (v64 added
        # the auxiliary timed_out step after this test's v63 subject).
        assert (
            migrated._get_db_version(connection)
            == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "console_trace_gc_runs",
            "console_trace_gc_marks",
            "console_trace_gc_segment_scopes",
            "console_trace_retention_roots",
        } <= tables
        run_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(console_trace_gc_runs)")
        }
        assert {
            "operation_kind",
            "target_conversation_id",
            "target_owner_id",
            "target_root_segment_id",
        } <= run_columns
        triggers = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        assert {
            "console_trace_calls_open_root_epoch",
            "console_trace_migration_root_epoch",
            "console_trace_retention_roots_insert_epoch",
            "console_trace_retention_roots_delete_epoch",
        } <= triggers
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        with migrated.transaction(immediate=True) as cursor:
            cursor.execute(
                "INSERT INTO console_trace_artifacts("
                "artifact_id, identity_digest, media_type, normalization_version, "
                "sanitized_bytes, byte_length) VALUES "
                "('orphan', ?, 'text/plain', 'test-v1', X'78', 1)",
                ("0" * 64,),
            )
        with pytest.raises(
            sqlite3.DatabaseError, match="trace GC deletion authorization"
        ):
            with migrated.transaction(immediate=True) as cursor:
                cursor.execute(
                    "DELETE FROM console_trace_artifacts WHERE artifact_id = 'orphan'"
                )
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(path, client_id="trace-gc-v63-reopen")
    try:
        # The unpinned reopen now continues one step past this test's own
        # migration (v63 -> v64 auxiliary timed_out); what this asserts is
        # that the genuine v62 DB migrated cleanly to the CURRENT version.
        assert reopened._get_db_version(reopened.get_connection()) == 64
        assert reopened.get_connection().execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close_connection()
