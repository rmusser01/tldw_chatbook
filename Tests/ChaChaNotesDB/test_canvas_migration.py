"""Focused v66 -> v67 persistence tests for durable Canvas revisions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    SCHEMA_NAME,
    chachanotes_db_at_version,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

CANVAS_TABLES = {
    "canvas_conversation_hints",
    "canvas_documents",
    "canvas_revisions",
}
CANVAS_INDEXES = {
    "idx_canvas_documents_conversation",
    "idx_canvas_revisions_canvas_sequence",
    "idx_canvas_revisions_origin_message",
    "idx_canvas_revisions_parent",
    "uq_canvas_documents_id_conversation",
    "uq_canvas_revisions_id_canvas",
}
CANVAS_TRIGGERS = {
    "canvas_documents_ownership_immutable",
    "canvas_origin_message_owner_guard",
    "canvas_revisions_no_delete",
    "canvas_revisions_no_update",
    "canvas_revisions_origin_owner_guard",
    "canvas_revisions_parent_guard",
}


def _version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def _canvas_schema(connection: sqlite3.Connection) -> dict[tuple[str, str], str]:
    rows = connection.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name LIKE 'canvas_%' OR name LIKE 'idx_canvas_%' "
        "OR name LIKE 'uq_canvas_%' ORDER BY type, name"
    ).fetchall()
    return {(str(row[0]), str(row[1])): str(row[2]) for row in rows}


def test_genuine_v66_database_migrates_to_v67_with_complete_canvas_schema(
    tmp_path: Path,
) -> None:
    """A missing v66 dispatch or schema object leaves this real upgrade red."""

    path = tmp_path / "genuine-v66.sqlite"
    with chachanotes_db_at_version(path, 66) as historical:
        assert _canvas_schema(historical.get_connection()) == {}

    migrated = CharactersRAGDB(path, client_id="canvas-v67-migrated")
    try:
        connection = migrated.get_connection()
        assert _version(path) == 67
        objects = _canvas_schema(connection)
        assert {
            name for (object_type, name) in objects if object_type == "table"
        } == CANVAS_TABLES
        assert {
            name for (object_type, name) in objects if object_type == "index"
        } == CANVAS_INDEXES
        assert {
            name for (object_type, name) in objects if object_type == "trigger"
        } == CANVAS_TRIGGERS
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close_connection()


def test_fresh_v67_schema_matches_migrated_v66_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    """Fresh-only DDL or replay-only DDL cannot silently diverge."""

    fresh_path = tmp_path / "fresh.sqlite"
    migrated_path = tmp_path / "migrated.sqlite"
    with chachanotes_db_at_version(migrated_path, 66):
        pass

    fresh = CharactersRAGDB(fresh_path, client_id="canvas-v67-fresh")
    migrated = CharactersRAGDB(migrated_path, client_id="canvas-v67-replay")
    try:
        fresh_schema = _canvas_schema(fresh.get_connection())
        migrated_schema = _canvas_schema(migrated.get_connection())
        assert fresh._CURRENT_SCHEMA_VERSION == 67
        assert fresh_schema == migrated_schema
        assert fresh_schema
    finally:
        fresh.close_connection()
        migrated.close_connection()

    reopened = CharactersRAGDB(fresh_path, client_id="canvas-v67-reopen")
    try:
        assert _version(fresh_path) == 67
        assert _canvas_schema(reopened.get_connection()) == fresh_schema
        assert (
            reopened.get_connection().execute("PRAGMA foreign_key_check").fetchall()
            == []
        )
    finally:
        reopened.close_connection()


def test_v66_migration_rolls_back_all_ddl_and_version_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after the real DDL must leave a genuine v66 database retryable."""

    path = tmp_path / "rollback.sqlite"
    with chachanotes_db_at_version(path, 66):
        pass

    original = CharactersRAGDB._execute_migration_statements

    def fail_after_canvas_ddl(self, cursor, script, label):
        original(self, cursor, script, label)
        if label == "V66→V67":
            raise sqlite3.OperationalError("injected canvas migration failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        fail_after_canvas_ddl,
    )
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, client_id="canvas-v67-fail")

    connection = sqlite3.connect(path)
    try:
        assert _version(path) == 66
        assert _canvas_schema(connection) == {}
    finally:
        connection.close()

    monkeypatch.setattr(
        CharactersRAGDB,
        "_execute_migration_statements",
        original,
    )
    retried = CharactersRAGDB(path, client_id="canvas-v67-retry")
    try:
        assert _version(path) == 67
        assert _canvas_schema(retried.get_connection())
    finally:
        retried.close_connection()
