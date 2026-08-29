"""Schema-v55 local Console memory scope and selection migration coverage."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


SCHEMA_NAME = "rag_char_chat_schema"
SCOPE_TABLE = "console_conversation_memory_scopes"
SELECTION_TABLE = "console_conversation_memory_selections"


def _version(db: CharactersRAGDB) -> int:
    return int(
        db.get_connection()
        .execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        )
        .fetchone()[0]
    )


def _v54_database(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> CharactersRAGDB:
    """Create a real v54 database without running the v55 migration."""
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 54)
    db = CharactersRAGDB(path, client_id="v54-seed")
    assert _version(db) == 54
    return db


def _insert_generated_memory(
    db: CharactersRAGDB,
    *,
    memory_id: str,
    conversation_id: str,
    captured_leaf_message_id: str | None,
    active: int = 1,
) -> None:
    db.get_connection().execute(
        """
        INSERT INTO console_conversation_memories(
            id, conversation_id, captured_leaf_message_id, lineage_json,
            summary_text, selected_units_json, active, source_kind
        ) VALUES (?, ?, ?, '[]', ?, '[]', ?, 'generated')
        """,
        (memory_id, conversation_id, captured_leaf_message_id, memory_id, active),
    )
    db.get_connection().commit()


def _index_columns(conn: sqlite3.Connection, table: str, index: str) -> list[str]:
    """Return an index's columns in declared order."""
    index_names = {row[1] for row in conn.execute(f"PRAGMA index_list({table})")}
    assert index in index_names
    return [row[2] for row in conn.execute(f"PRAGMA index_info({index})")]


def _foreign_key_edges(conn: sqlite3.Connection, table: str) -> set[tuple[str, str, str, str, str]]:
    """Return exact foreign-key table, columns, and actions for a table."""
    return {
        (row[2], row[3], row[4], row[5], row[6])
        for row in conn.execute(f"PRAGMA foreign_key_list({table})")
    }


def _selection_rows(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Return generated selection events in database-assigned sequence order."""
    return [
        tuple(row)
        for row in conn.execute(
            f"SELECT sequence, selection_id, selected_memory_id "
            f"FROM {SELECTION_TABLE} ORDER BY sequence"
        )
    ]


def test_fresh_database_creates_local_scope_and_selection_schema(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh-v55.sqlite", client_id="fresh-v55")
    try:
        conn = db.get_connection()
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(db) == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 55
        assert {SCOPE_TABLE, SELECTION_TABLE} <= tables

        memory_indexes = conn.execute(
            "PRAGMA index_list(console_conversation_memories)"
        ).fetchall()
        assert any(
            [column[2] for column in conn.execute(f"PRAGMA index_info({row[1]})")]
            == ["id", "conversation_id"]
            and row[2]
            for row in memory_indexes
        )
        assert _index_columns(
            conn,
            SCOPE_TABLE,
            "idx_console_memory_scopes_conversation_origin",
        ) == ["conversation_id", "origin_kind", "coverage_kind"]
        assert _index_columns(
            conn,
            SELECTION_TABLE,
            "idx_console_memory_selections_conversation_active_sequence",
        ) == ["conversation_id", "active", "sequence"]
        assert _index_columns(
            conn,
            SELECTION_TABLE,
            "idx_console_memory_selections_activation",
        ) == ["conversation_id", "activation_message_id"]

        scope_columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({SCOPE_TABLE})")
        }
        selection_columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({SELECTION_TABLE})")
        }
        assert {
            "memory_id",
            "conversation_id",
            "coverage_kind",
            "origin_kind",
            "selection_anchor_message_id",
        } <= scope_columns
        assert {
            "sequence",
            "selection_id",
            "conversation_id",
            "activation_message_id",
            "selected_memory_id",
            "event_kind",
            "suppresses_legacy",
            "created_at",
            "revision",
            "active",
        } <= selection_columns
        assert _foreign_key_edges(conn, SCOPE_TABLE) == {
            ("conversations", "conversation_id", "id", "CASCADE", "CASCADE"),
            ("console_conversation_memories", "memory_id", "id", "CASCADE", "CASCADE"),
            (
                "console_conversation_memories",
                "conversation_id",
                "conversation_id",
                "CASCADE",
                "CASCADE",
            ),
            ("messages", "conversation_id", "conversation_id", "CASCADE", "RESTRICT"),
            ("messages", "selection_anchor_message_id", "id", "CASCADE", "RESTRICT"),
        }
        assert _foreign_key_edges(conn, SELECTION_TABLE) == {
            ("conversations", "conversation_id", "id", "CASCADE", "CASCADE"),
            ("console_conversation_memories", "selected_memory_id", "id", "CASCADE", "RESTRICT"),
            (
                "console_conversation_memories",
                "conversation_id",
                "conversation_id",
                "CASCADE",
                "RESTRICT",
            ),
            ("messages", "conversation_id", "conversation_id", "CASCADE", "RESTRICT"),
            ("messages", "activation_message_id", "id", "CASCADE", "RESTRICT"),
        }
        sync_sql = "\n".join(
            str(row[0] or "")
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE '%sync%'"
            )
        )
        assert SCOPE_TABLE not in sync_sql
        assert SELECTION_TABLE not in sync_sql
    finally:
        db.close_connection()


def test_v54_backfill_is_idempotent_and_uses_memory_insertion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "v54-backfill.sqlite"
    db = _v54_database(path, monkeypatch)
    conversation_id = db.add_conversation({"title": "backfill"})
    first_leaf = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "first"}
    )
    db.set_conversation_context_summary(conversation_id, "legacy recap", first_leaf)
    second_leaf = db.add_message(
        {"conversation_id": conversation_id, "sender": "assistant", "content": "second"}
    )
    _insert_generated_memory(
        db,
        memory_id="first-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=first_leaf,
    )
    _insert_generated_memory(
        db,
        memory_id="inactive-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=first_leaf,
        active=0,
    )
    _insert_generated_memory(
        db,
        memory_id="second-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=second_leaf,
    )
    _insert_generated_memory(
        db,
        memory_id="invalid-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=None,
    )
    db.close_connection()
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 55)

    upgraded = CharactersRAGDB(path, client_id="v55-upgrade")
    try:
        conn = upgraded.get_connection()
        scopes = conn.execute(
            f"SELECT memory_id, coverage_kind, origin_kind, selection_anchor_message_id "
            f"FROM {SCOPE_TABLE} ORDER BY memory_id"
        ).fetchall()
        selections = conn.execute(
            f"SELECT selected_memory_id, activation_message_id, event_kind, suppresses_legacy "
            f"FROM {SELECTION_TABLE} ORDER BY sequence"
        ).fetchall()
        assert [(row[0], row[1], row[2], row[3]) for row in scopes] == [
            ("first-memory", "prefix", "automatic", None),
            ("inactive-memory", "prefix", "automatic", None),
            ("invalid-memory", "prefix", "automatic", None),
            ("second-memory", "prefix", "automatic", None),
        ]
        assert [tuple(row) for row in selections] == [
            ("first-memory", first_leaf, "select", 0),
            ("second-memory", second_leaf, "select", 0),
        ]
        assert upgraded.get_conversation_context_summary(conversation_id) == (
            "legacy recap",
            first_leaf,
        )
        assert _version(upgraded) == 55
        conn.execute(
            f"DELETE FROM {SCOPE_TABLE} WHERE memory_id = 'second-memory'"
        )
        conn.execute(
            "UPDATE db_schema_version SET version = 54 WHERE schema_name = ?",
            (SCHEMA_NAME,),
        )
        conn.commit()
    finally:
        upgraded.close_connection()

    # Re-entry after a v54-stamped partial application restores the missing
    # scope without duplicating/reordering the existing select-event backfill.
    reopened = CharactersRAGDB(path, client_id="v55-reentry")
    try:
        conn = reopened.get_connection()
        assert conn.execute(f"SELECT COUNT(*) FROM {SCOPE_TABLE}").fetchone()[0] == 4
        assert conn.execute(f"SELECT COUNT(*) FROM {SELECTION_TABLE}").fetchone()[0] == 2
    finally:
        reopened.close_connection()


@pytest.mark.parametrize("missing_memory_id", ("first-memory", "second-memory"))
def test_v54_partial_selection_backfill_rebuilds_exact_rowid_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_memory_id: str,
) -> None:
    path = tmp_path / f"v54-partial-selection-{missing_memory_id}.sqlite"
    db = _v54_database(path, monkeypatch)
    conversation_id = db.add_conversation({"title": "partial selections"})
    first_leaf = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "first"}
    )
    second_leaf = db.add_message(
        {"conversation_id": conversation_id, "sender": "assistant", "content": "second"}
    )
    _insert_generated_memory(
        db,
        memory_id="first-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=first_leaf,
    )
    _insert_generated_memory(
        db,
        memory_id="second-memory",
        conversation_id=conversation_id,
        captured_leaf_message_id=second_leaf,
    )
    db.close_connection()
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 55)

    upgraded = CharactersRAGDB(path, client_id="v55-initial")
    try:
        conn = upgraded.get_connection()
        conn.execute(
            f"DELETE FROM {SELECTION_TABLE} WHERE selection_id = ?",
            (f"migration:auto-select:{missing_memory_id}",),
        )
        conn.execute(
            "UPDATE db_schema_version SET version = 54 WHERE schema_name = ?",
            (SCHEMA_NAME,),
        )
        conn.commit()
    finally:
        upgraded.close_connection()

    reopened = CharactersRAGDB(path, client_id="v55-repaired")
    try:
        assert _selection_rows(reopened.get_connection()) == [
            (1, "migration:auto-select:first-memory", "first-memory"),
            (2, "migration:auto-select:second-memory", "second-memory"),
        ]
    finally:
        reopened.close_connection()


def test_scope_and_selection_checks_cross_conversation_guards_and_deletion(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "constraints.sqlite", client_id="constraints")
    try:
        conn = db.get_connection()
        first = db.add_conversation({"title": "first"})
        second = db.add_conversation({"title": "second"})
        first_message = db.add_message(
            {"conversation_id": first, "sender": "user", "content": "first message"}
        )
        second_message = db.add_message(
            {"conversation_id": second, "sender": "user", "content": "second message"}
        )
        _insert_generated_memory(
            db,
            memory_id="first-memory",
            conversation_id=first,
            captured_leaf_message_id=first_message,
        )
        _insert_generated_memory(
            db,
            memory_id="second-memory",
            conversation_id=second,
            captured_leaf_message_id=second_message,
        )
        _insert_generated_memory(
            db,
            memory_id="scope-only-memory",
            conversation_id=first,
            captured_leaf_message_id=first_message,
        )
        conn.execute(
            f"INSERT INTO {SCOPE_TABLE}(memory_id, conversation_id, coverage_kind, origin_kind, selection_anchor_message_id) "
            "VALUES ('first-memory', ?, 'prefix', 'automatic', NULL)",
            (first,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SCOPE_TABLE}(memory_id, conversation_id, coverage_kind, origin_kind, selection_anchor_message_id) "
                "VALUES ('second-memory', ?, 'prefix', 'automatic', NULL)",
                (first,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SCOPE_TABLE}(memory_id, conversation_id, coverage_kind, origin_kind, selection_anchor_message_id) "
                "VALUES ('second-memory', ?, 'prefix', 'automatic', ?)",
                (second, second_message),
            )
        conn.execute(
            f"INSERT INTO {SCOPE_TABLE}(memory_id, conversation_id, coverage_kind, origin_kind, selection_anchor_message_id) "
            "VALUES ('second-memory', ?, 'range', 'manual_rewind', ?)",
            (second, second_message),
        )
        conn.execute(
            f"INSERT INTO {SCOPE_TABLE}(memory_id, conversation_id, coverage_kind, origin_kind, selection_anchor_message_id) "
            "VALUES ('scope-only-memory', ?, 'prefix', 'automatic', NULL)",
            (first,),
        )
        conn.execute(
            "DELETE FROM console_conversation_memories WHERE id = 'scope-only-memory'"
        )
        assert conn.execute(
            f"SELECT COUNT(*) FROM {SCOPE_TABLE} WHERE memory_id = 'scope-only-memory'"
        ).fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
                "VALUES ('bad-select', ?, ?, NULL, 'select', 0, CURRENT_TIMESTAMP, 1, 1)",
                (first, first_message),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
                "VALUES ('bad-reset', ?, ?, 'first-memory', 'reset', 1, CURRENT_TIMESTAMP, 1, 1)",
                (first, first_message),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
                "VALUES ('bad-memory-conversation', ?, ?, 'second-memory', 'select', 0, CURRENT_TIMESTAMP, 1, 1)",
                (first, first_message),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
                "VALUES ('bad-activation-conversation', ?, ?, 'first-memory', 'select', 0, CURRENT_TIMESTAMP, 1, 1)",
                (first, second_message),
            )
        conn.execute(
            f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
            "VALUES ('select', ?, ?, 'first-memory', 'select', 0, CURRENT_TIMESTAMP, 1, 1)",
            (first, first_message),
        )
        conn.execute(
            f"INSERT INTO {SELECTION_TABLE}(selection_id, conversation_id, activation_message_id, selected_memory_id, event_kind, suppresses_legacy, created_at, revision, active) "
            "VALUES ('reset', ?, ?, NULL, 'reset', 1, CURRENT_TIMESTAMP, 1, 1)",
            (first, first_message),
        )
        assert [row[0] for row in conn.execute(f"SELECT sequence FROM {SELECTION_TABLE} ORDER BY sequence")] == [1, 2]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM messages WHERE id = ?", (first_message,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "DELETE FROM console_conversation_memories WHERE id = 'first-memory'"
            )
        conn.commit()
        assert db.soft_delete_message(first_message, expected_version=1) is True
        conn.execute("DELETE FROM conversations WHERE id = ?", (first,))
        assert conn.execute(
            f"SELECT COUNT(*) FROM {SCOPE_TABLE} WHERE conversation_id = ?", (first,)
        ).fetchone()[0] == 0
        assert conn.execute(
            f"SELECT COUNT(*) FROM {SELECTION_TABLE} WHERE conversation_id = ?", (first,)
        ).fetchone()[0] == 0
    finally:
        db.close_connection()


def test_v55_foreign_key_audit_rolls_back_before_version_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _v54_database(tmp_path / "fk-audit.sqlite", monkeypatch)
    conn = db.get_connection()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        INSERT INTO console_conversation_memories(
            id, conversation_id, lineage_json, summary_text, selected_units_json, source_kind
        ) VALUES ('orphan-memory', 'missing-conversation', '[]', 'bad', '[]', 'generated')
        """
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 55)
    with pytest.raises(SchemaError, match="(?i)foreign key audit failed"):
        db._migrate_from_v54_to_v55(conn)
    assert _version(db) == 54
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?", (SCOPE_TABLE,)
    ).fetchone()[0] == 0
    db.close_connection()
