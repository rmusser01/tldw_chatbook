"""v30 -> v31: local-only messages.metadata_json column (task-2364).

Local-only means: the column must NOT appear in any messages_sync_* trigger
payload -- same precedent as the v29->v30 usage_json column and the
v24/v25/v26 local-only migrations.
"""

from pathlib import Path

import pytest

from tldw_chatbook.Chat.message_metadata import (
    CharacterEmoteEventMetadata,
    CharacterEmoteMetadata,
    MessageMetadata,
)
from Tests.ChaChaNotesDB.historical_bootstrap import (
    open_current_chachanotes_from_legacy,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


# Matches CharactersRAGDB._SCHEMA_NAME, per the sibling migration test
# (Tests/DB/test_chachanotes_message_usage_migration.py).
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


def _seed_v30_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as v30_patch:
        v30_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 30)
        db = CharactersRAGDB(path, client_id="migration-seed")
        connection = db.get_connection()
        assert _version(connection) == 30
        assert "metadata_json" not in _message_columns(connection)
        db.close_connection()


def test_migration_adds_metadata_json_and_bumps_version(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v30_database(db_path, monkeypatch)

    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert "metadata_json" in _message_columns(connection)
    db.close_connection()


def test_metadata_json_excluded_from_sync_triggers(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v30_database(db_path, monkeypatch)
    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )
    connection = db.get_connection()
    triggers = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_sync%'"
    ).fetchall()
    assert triggers, "expected messages sync triggers to exist"
    for (sql,) in triggers:
        assert "metadata_json" not in (sql or "")
    db.close_connection()


def test_add_and_update_message_round_trip_metadata_json(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="metadata-test")
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "hi",
            "metadata_json": '{"engine": "realtime"}',
        }
    )
    row = db.get_message_by_id(msg_id)
    assert row["metadata_json"] == '{"engine": "realtime"}'

    db.update_message(
        msg_id,
        {"metadata_json": '{"interrupted": true}'},
        expected_version=row["version"],
    )
    assert db.get_message_by_id(msg_id)["metadata_json"] == '{"interrupted": true}'


def test_local_metadata_write_leaves_version_and_sync_log_untouched(tmp_path):
    """The whole reason the column is local-only: a metadata-only write must
    not bump version/last_modified (the messages_sync_update trigger watches
    those) and so must enqueue no sync_log row whose payload could never
    carry metadata_json anyway.
    """
    db = CharactersRAGDB(tmp_path / "local.db", client_id="metadata-test")
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {"conversation_id": conv_id, "sender": "assistant", "content": "hi"}
    )
    before = db.get_message_by_id(msg_id)
    change_id = db.get_latest_sync_log_change_id()

    assert db.update_message_metadata_local(msg_id, '{"interrupted": true}') is True

    after = db.get_message_by_id(msg_id)
    assert after["metadata_json"] == '{"interrupted": true}'
    assert after["version"] == before["version"]
    assert after["last_modified"] == before["last_modified"]
    assert (
        db.get_sync_log_entries(since_change_id=change_id, entity_type="messages") == []
    )
    assert db.update_message_metadata_local("missing-id", "{}") is False


def test_migration_is_idempotent_when_column_already_present(tmp_path, monkeypatch):
    """SQLite has no ``ADD COLUMN IF NOT EXISTS``, so a v30 database that
    already carries ``metadata_json`` -- a half-applied migration, or a row
    added by a concurrent build of this branch -- would abort the whole
    upgrade with "duplicate column name". The runner checks
    ``PRAGMA table_info(messages)`` and skips only the DDL, never the
    version bump, so such a database still lands at v31.
    """
    db_path = tmp_path / "chachanotes.db"
    _seed_v30_database(db_path, monkeypatch)

    # Hand-apply the column while the schema still says v30.
    #
    # Raw connection, not `db.transaction()`, deliberately (Qodo round). This
    # is fabricating a state the application can never produce -- a column
    # present with the version not yet bumped -- by reaching AROUND the
    # migration machinery, which is exactly what every schema-fabricating
    # test in this directory does: the v29->v30 sibling this file mirrors
    # (`test_chachanotes_message_usage_migration.py`, same ALTER + commit),
    # `test_chachanotes_citation_provenance_migration.py` (`executescript` of
    # migration SQL), and both world-book fixtures (`DROP COLUMN` to rewind a
    # schema). Files here use `db.transaction()` for ordinary DATA writes and
    # raw connections for schema fabrication -- see
    # `test_chachanotes_character_authority_migration.py`, which does both.
    # Routing this through the shared helper would read as an ordinary app
    # write and break that distinction for no safety gained.
    with monkeypatch.context() as v30_patch:
        v30_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 30)
        db = CharactersRAGDB(db_path, client_id="pre-applied")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE messages ADD COLUMN metadata_json TEXT DEFAULT NULL"
        )
        connection.commit()
        assert _version(connection) == 30
        assert "metadata_json" in _message_columns(connection)
        db.close_connection()

    db = open_current_chachanotes_from_legacy(
        db_path, client_id="migration-test"
    )  # must not raise
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert "metadata_json" in _message_columns(connection)
    # And the column is still usable, not left in some half-migrated state.
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "hi",
            "metadata_json": '{"engine": "realtime"}',
        }
    )
    assert db.get_message_by_id(msg_id)["metadata_json"] == '{"engine": "realtime"}'
    db.close_connection()


def test_character_emote_metadata_reopens_without_schema_bump(tmp_path):
    db_path = tmp_path / "emote.db"
    metadata = MessageMetadata(
        character_emote=CharacterEmoteMetadata(
            mood_label="sad",
            emote_events=(
                CharacterEmoteEventMetadata("smug", 0),
                CharacterEmoteEventMetadata("sad", 4),
            ),
            sanitized_utf16_length=8,
            actor_kind="character",
            actor_id=3,
            pack_id=5,
            pack_version_id=7,
            expression_key="sad",
            expression_id=11,
            asset_id=13,
        )
    )
    db = CharactersRAGDB(db_path, client_id="emote-write")
    expected_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    conv_id = db.add_conversation({"title": "emote"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "safe text",
            "metadata_json": metadata.to_json(),
        }
    )
    db.close_connection()

    reopened = CharactersRAGDB(db_path, client_id="emote-read")
    row = reopened.get_message_by_id(msg_id)

    assert _version(reopened.get_connection()) == expected_version
    assert MessageMetadata.from_json(row["metadata_json"]) == metadata
    reopened.close_connection()
