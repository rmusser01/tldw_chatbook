"""v29 -> v30: local-only messages.usage_json column (cost ticker PR1).

Local-only means: the column must NOT appear in any messages_sync_* trigger
payload — same precedent as v24/v25/v26 local tables.
"""

from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    open_current_chachanotes_from_legacy,
)

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


# Matches CharactersRAGDB._SCHEMA_NAME, per the sibling migration test
# (Tests/DB/test_chachanotes_character_authority_migration.py).
SCHEMA_NAME = "rag_char_chat_schema"


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _message_columns(connection) -> set[str]:
    return {
        row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()
    }


def _seed_v29_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as v29_patch:
        v29_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 29)
        db = CharactersRAGDB(path, client_id="migration-seed")
        connection = db.get_connection()
        assert _version(connection) == 29
        assert "usage_json" not in _message_columns(connection)
        db.close_connection()


def test_migration_adds_usage_json_and_bumps_version(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v29_database(db_path, monkeypatch)

    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert "usage_json" in _message_columns(connection)
    db.close_connection()


def test_usage_json_excluded_from_sync_triggers(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v29_database(db_path, monkeypatch)
    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )
    connection = db.get_connection()
    triggers = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_sync%'"
    ).fetchall()
    assert triggers, "expected messages sync triggers to exist"
    for (sql,) in triggers:
        assert "usage_json" not in (sql or "")
    db.close_connection()


def test_add_and_update_message_round_trip_usage_json(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="usage-test")
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "hi",
            "usage_json": '{"uncached_input": 10}',
        }
    )
    row = db.get_message_by_id(msg_id)
    assert row["usage_json"] == '{"uncached_input": 10}'

    db.update_message(
        msg_id,
        {"usage_json": '{"uncached_input": 99}'},
        expected_version=row["version"],
    )
    assert db.get_message_by_id(msg_id)["usage_json"] == '{"uncached_input": 99}'


def test_migration_is_idempotent_when_column_already_present(tmp_path, monkeypatch):
    """F8: SQLite has no ``ADD COLUMN IF NOT EXISTS``, so a v29 database that
    already carries ``usage_json`` -- a half-applied migration, or a row added
    by a concurrent build of this branch -- used to abort the whole upgrade
    with "duplicate column name". The runner now checks
    ``PRAGMA table_info(messages)`` and skips only the DDL, never the version
    bump, so such a database still lands at v30.
    """
    db_path = tmp_path / "chachanotes.db"
    _seed_v29_database(db_path, monkeypatch)

    # Hand-apply the column while the schema still says v29.
    with monkeypatch.context() as v29_patch:
        v29_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 29)
        db = CharactersRAGDB(db_path, client_id="pre-applied")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE messages ADD COLUMN usage_json TEXT DEFAULT NULL"
        )
        connection.commit()
        assert _version(connection) == 29
        assert "usage_json" in _message_columns(connection)
        db.close_connection()

    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )  # must not raise
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert "usage_json" in _message_columns(connection)
    # And the column is still usable, not left in some half-migrated state.
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "hi",
            "usage_json": '{"output": 7}',
        }
    )
    assert db.get_message_by_id(msg_id)["usage_json"] == '{"output": 7}'
    db.close_connection()
