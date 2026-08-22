"""Contracts for the shared private Notes sync-state schema owner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.Notes import notes_sync_state_schema as schema_module
from tldw_chatbook.Notes.notes_sync_state_schema import (
    NotesSyncStateSchemaError,
    notes_sync_state_transaction,
)


_V1_TABLES = {
    "import_sessions",
    "import_items",
    "import_payload_effects",
    "import_folder_effects",
    "import_membership_effects",
}
_V1_INDEXES = {
    "idx_import_items_outcome",
    "idx_import_payload_state",
    "idx_import_folder_state",
    "idx_import_membership_state",
    "idx_import_payload_target",
    "idx_import_folder_target",
    "idx_import_membership_path",
    "idx_import_folder_parent",
    "idx_import_items_target",
    "idx_import_items_source_session",
}


def test_empty_database_initializes_the_canonical_v1_receipt_schema(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
            if not row[0].startswith("sqlite_autoindex_")
        }

    assert tables == _V1_TABLES
    assert indexes == _V1_INDEXES


def test_v1_index_compatibility_failure_rolls_back_every_repair(
    tmp_path: Path,
) -> None:
    database = tmp_path / "malformed-v1.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE import_items (session_id TEXT, outcome TEXT);
            CREATE TABLE import_payload_effects (session_id TEXT);
            PRAGMA user_version = 1;
            """
        )

    with pytest.raises(NotesSyncStateSchemaError, match="incompatible"):
        with notes_sync_state_transaction(database):
            pass

    with sqlite3.connect(database) as connection:
        repaired_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_import_items_outcome",),
        ).fetchone()

    assert repaired_index is None


@pytest.mark.parametrize(
    "open_error",
    (
        sqlite3.OperationalError("PRIVATE_OPEN_SENTINEL"),
        OSError("PRIVATE_OPEN_SENTINEL"),
    ),
)
def test_connection_open_failures_are_bounded_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_error: Exception,
) -> None:
    def fail_open(*_args, **_kwargs):
        raise open_error

    monkeypatch.setattr(schema_module, "connect_private_sqlite", fail_open)

    with pytest.raises(NotesSyncStateSchemaError) as raised:
        with notes_sync_state_transaction(tmp_path / "private-sentinel.sqlite3"):
            pass

    assert "PRIVATE_OPEN_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_unknown_schema_version_is_rejected_without_mutation(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE future_private_state (opaque_value TEXT);
            INSERT INTO future_private_state VALUES ('opaque-marker');
            PRAGMA user_version = 2;
            """
        )
        before = (
            connection.execute("PRAGMA user_version").fetchone(),
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
            connection.execute("SELECT * FROM future_private_state").fetchall(),
        )

    with pytest.raises(NotesSyncStateSchemaError, match="Unsupported"):
        with notes_sync_state_transaction(database):
            pass

    with sqlite3.connect(database) as connection:
        after = (
            connection.execute("PRAGMA user_version").fetchone(),
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
            connection.execute("SELECT * FROM future_private_state").fetchall(),
        )

    assert after == before


def test_schema_phase_commits_before_a_failing_operation(tmp_path: Path) -> None:
    database = tmp_path / "schema-phase.sqlite3"

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(database) as connection:
            connection.execute("CREATE TABLE operation_only (value TEXT)")
            raise RuntimeError("operation failed")

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert _V1_TABLES <= tables
    assert "operation_only" not in tables


def test_operation_failure_rolls_back_operation_rows(tmp_path: Path) -> None:
    database = tmp_path / "operation-rollback.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        connection.execute("CREATE TABLE operation_rows (value TEXT)")

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(database) as connection:
            connection.execute(
                "INSERT INTO operation_rows VALUES (?)", ("private-marker",)
            )
            raise RuntimeError("operation failed")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM operation_rows").fetchall() == []


def test_operation_failure_closes_the_private_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[sqlite3.Connection] = []
    original_connect = schema_module.connect_private_sqlite

    def capture_connection(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        captured.append(connection)
        return connection

    monkeypatch.setattr(schema_module, "connect_private_sqlite", capture_connection)

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(tmp_path / "closed.sqlite3"):
            raise RuntimeError("operation failed")

    assert len(captured) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured[0].execute("SELECT 1")


def test_healthy_v1_does_not_reserve_the_writer_slot(tmp_path: Path) -> None:
    database = tmp_path / "healthy-v1.sqlite3"
    with notes_sync_state_transaction(database):
        pass

    with sqlite3.connect(database, timeout=0) as reader:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM import_sessions").fetchone() == (0,)
        with notes_sync_state_transaction(database):
            with sqlite3.connect(database, timeout=0) as second_connection:
                second_connection.execute("BEGIN IMMEDIATE")
                second_connection.rollback()
        reader.rollback()
