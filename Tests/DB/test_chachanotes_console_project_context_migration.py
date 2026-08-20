"""v32 -> v33 local-only Console project-context migration contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)


SCHEMA_NAME = "rag_char_chat_schema"
COLUMN_NAME = "console_project_context_json"


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _conversation_columns(connection) -> dict[str, object]:
    return {
        str(row[1]): row
        for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
    }


def _seed_v32_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create the actual v32 shape even after the canonical schema advances."""
    with monkeypatch.context() as v32_patch:
        v32_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 32)
        db = CharactersRAGDB(path, client_id="migration-seed")
        conversation_id = db.add_conversation({"title": "preserve me"})
        connection = db.get_connection()
        columns = _conversation_columns(connection)
        if COLUMN_NAME in columns:
            connection.execute(f"ALTER TABLE conversations DROP COLUMN {COLUMN_NAME}")
            connection.commit()
        assert _version(connection) == 32
        assert COLUMN_NAME not in _conversation_columns(connection)
        db.close_connection()
    return str(conversation_id)


def test_v32_to_v33_adds_nullable_local_column(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "chachanotes.db"
    conversation_id = _seed_v32_database(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    columns = _conversation_columns(connection)

    assert _version(connection) == 33
    assert columns[COLUMN_NAME][3] == 0
    row = connection.execute(
        "SELECT title, console_project_context_json FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    assert tuple(row) == ("preserve me", None)
    db.close_connection()


def test_v32_to_v33_recovers_column_present_version_still_32(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "chachanotes.db"
    conversation_id = _seed_v32_database(db_path, monkeypatch)

    with monkeypatch.context() as v32_patch:
        v32_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 32)
        db = CharactersRAGDB(db_path, client_id="partial-migration")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE conversations ADD COLUMN console_project_context_json TEXT"
        )
        connection.execute(
            "UPDATE conversations SET console_project_context_json = ? WHERE id = ?",
            ('{"version":1}', conversation_id),
        )
        connection.commit()
        assert _version(connection) == 32
        db.close_connection()

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    assert _version(connection) == 33
    assert (
        connection.execute(
            "SELECT console_project_context_json FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()[0]
        == '{"version":1}'
    )
    db.close_connection()


def test_v32_to_v33_rejects_wrong_start_version(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="version-test")
    connection = db.get_connection()
    starting_version = _version(connection)

    with pytest.raises(SchemaError, match="requires schema version 32"):
        with db.transaction():
            db._migrate_from_v32_to_v33(connection)

    assert _version(connection) == starting_version
    db.close_connection()


@pytest.mark.parametrize("declared_default", ["DEFAULT 'hostile'", "DEFAULT NULL"])
def test_v32_to_v33_rejects_any_declared_column_default(
    declared_default: str,
) -> None:
    db, connection = _minimal_v32_partial_database(f"TEXT {declared_default}")
    before = _conversation_columns(connection)[COLUMN_NAME]
    assert before[4] is not None

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v32_to_v33(connection)
    connection.rollback()

    assert _version(connection) == 32
    assert _conversation_columns(connection)[COLUMN_NAME] == before
    connection.close()


def test_v32_to_v33_rejects_primary_key_column_shape() -> None:
    db, connection = _minimal_v32_partial_database("TEXT PRIMARY KEY")
    before = _conversation_columns(connection)[COLUMN_NAME]
    assert before[5] != 0

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v32_to_v33(connection)
    connection.rollback()

    assert _version(connection) == 32
    assert _conversation_columns(connection)[COLUMN_NAME] == before
    connection.close()


@pytest.mark.parametrize(
    "column_definition",
    [
        "TEXT UNIQUE",
        "TEXT CHECK (console_project_context_json IS NULL)",
        "TEXT REFERENCES local_project_context_parent(id)",
    ],
)
def test_v32_to_v33_rejects_additional_column_constraints(
    column_definition: str,
) -> None:
    db, connection = _minimal_v32_partial_database(column_definition)

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v32_to_v33(connection)
    connection.rollback()

    assert _version(connection) == 32
    connection.close()


@pytest.mark.parametrize(
    "index_expression",
    ["console_project_context_json", "lower(console_project_context_json)"],
)
def test_v32_to_v33_rejects_separate_unique_index_on_local_column(
    index_expression: str,
) -> None:
    db, connection = _minimal_v32_partial_database("TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX hostile_project_context_unique "
        f"ON conversations({index_expression})"
    )
    connection.commit()

    connection.execute("BEGIN")
    with pytest.raises(SchemaError, match="incompatible shape"):
        db._migrate_from_v32_to_v33(connection)
    connection.rollback()

    assert _version(connection) == 32
    assert connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' "
        "AND name = 'hostile_project_context_unique'"
    ).fetchone()
    connection.close()


def test_v32_to_v33_error_rolls_back_and_leaves_version_32(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "broken.db"
    _seed_v32_database(db_path, monkeypatch)

    with monkeypatch.context() as v32_patch:
        v32_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 32)
        db = CharactersRAGDB(db_path, client_id="broken-migration")
        connection = db.get_connection()
        connection.execute(
            """
            CREATE TRIGGER block_v33_version_update
            BEFORE UPDATE OF version ON db_schema_version
            WHEN OLD.schema_name = 'rag_char_chat_schema'
              AND OLD.version = 32
              AND NEW.version = 33
            BEGIN
                SELECT RAISE(ABORT, 'blocked version update');
            END
            """
        )
        connection.commit()
        db.close_connection()

    with pytest.raises(CharactersRAGDBError, match="Migration from V32 to V33 failed"):
        CharactersRAGDB(db_path, client_id="failed-migration")

    with sqlite3.connect(db_path) as connection:
        assert _version(connection) == 32
        assert COLUMN_NAME not in _conversation_columns(connection)


def test_fresh_schema_contains_console_project_context_column(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh-test")
    connection = db.get_connection()
    columns = _conversation_columns(connection)

    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 33
    assert _version(connection) == 33
    assert columns[COLUMN_NAME][2].upper() == "TEXT"
    assert columns[COLUMN_NAME][3] == 0
    db.close_connection()
