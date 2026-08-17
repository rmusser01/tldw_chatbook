"""V36 -> V37 provider-continuation message ownership migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


SCHEMA_NAME = "rag_char_chat_schema"
MESSAGE_SYNC_TRIGGERS = {
    "messages_sync_create",
    "messages_sync_update",
    "messages_sync_delete",
    "messages_sync_undelete",
}


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _schema_objects(connection, object_type: str) -> dict[str, str]:
    return {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = ? AND sql IS NOT NULL",
            (object_type,),
        )
    }


def _seed_v36_database(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    with monkeypatch.context() as v36_patch:
        v36_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 36)
        db = CharactersRAGDB(path, client_id="migration-seed")
        conversation_id = db.add_conversation({"title": "migration conversation"})
        connection = db.get_connection()
        root_id = "message-root"
        variant_id = "message-variant"
        deleted_id = "message-deleted"
        timestamp = "2026-08-12T00:00:00+00:00"
        with db.transaction() as transaction:
            transaction.executemany(
                """
                INSERT INTO messages (
                    id, conversation_id, parent_message_id, sender, content,
                    image_data, image_mime_type, timestamp, ranking,
                    last_modified, deleted, client_id, version, role,
                    variant_of, variant_number, is_selected_variant,
                    total_variants, feedback, usage_json, metadata_json
                ) VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, NULL, ?, 0,
                          'migration-seed', 1, 'assistant', ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    (
                        root_id,
                        conversation_id,
                        None,
                        "migrationterm visible",
                        b"image-bytes",
                        "image/png",
                        timestamp,
                        timestamp,
                        None,
                        1,
                        1,
                        2,
                        '{"output":7}',
                        '{"local":"kept"}',
                    ),
                    (
                        variant_id,
                        conversation_id,
                        None,
                        "migrationterm variant",
                        None,
                        None,
                        timestamp,
                        timestamp,
                        root_id,
                        2,
                        0,
                        2,
                        None,
                        None,
                    ),
                    (
                        deleted_id,
                        conversation_id,
                        root_id,
                        "deleted content",
                        None,
                        None,
                        timestamp,
                        timestamp,
                        None,
                        1,
                        1,
                        1,
                        None,
                        None,
                    ),
                ),
            )
            transaction.execute(
                """
                UPDATE messages
                   SET deleted = 1, version = 2
                 WHERE id = ?
                """,
                (deleted_id,),
            )
        assert _version(connection) == 36
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
        assert "provider_continuation_json" not in columns
        rows = [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, conversation_id, parent_message_id, sender, content,
                       image_data, image_mime_type, timestamp, ranking, last_modified,
                       deleted, client_id, version, role, variant_of, variant_number,
                       is_selected_variant, total_variants, feedback, usage_json,
                       metadata_json
                  FROM messages
                 ORDER BY id
                """
            )
        ]
        triggers = _schema_objects(connection, "trigger")
        indexes = _schema_objects(connection, "index")
        fts_matches = [
            row[0]
            for row in connection.execute(
                """
                SELECT messages.id
                  FROM messages_fts
                  JOIN messages ON messages.rowid = messages_fts.rowid
                 WHERE messages_fts MATCH 'migrationterm'
                 ORDER BY messages.id
                """
            )
        ]
        sync_count = connection.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
        db.close_connection()

    return {
        "conversation_id": conversation_id,
        "root_id": root_id,
        "variant_id": variant_id,
        "deleted_id": deleted_id,
        "rows": rows,
        "triggers": triggers,
        "indexes": indexes,
        "fts_matches": fts_matches,
        "sync_count": sync_count,
    }


def test_v36_to_v37_preserves_messages_schema_objects_and_fts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chachanotes.db"
    before = _seed_v36_database(db_path, monkeypatch)

    # Pin the target so only the V36 -> V37 step runs (schema has moved on).
    with monkeypatch.context() as v37_patch:
        v37_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 37)
        db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()

    assert _version(connection) == 37
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    assert "provider_continuation_json" in columns
    rows = [
        tuple(row)
        for row in connection.execute(
            """
            SELECT id, conversation_id, parent_message_id, sender, content,
                   image_data, image_mime_type, timestamp, ranking, last_modified,
                   deleted, client_id, version, role, variant_of, variant_number,
                   is_selected_variant, total_variants, feedback, usage_json,
                   metadata_json
              FROM messages
             ORDER BY id
            """
        )
    ]
    assert rows == before["rows"]
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM messages WHERE provider_continuation_json IS NOT NULL"
        ).fetchone()[0]
        == 0
    )

    after_triggers = _schema_objects(connection, "trigger")
    before_triggers = before["triggers"]
    assert {
        name: sql
        for name, sql in after_triggers.items()
        if name not in MESSAGE_SYNC_TRIGGERS
    } == {
        name: sql
        for name, sql in before_triggers.items()
        if name not in MESSAGE_SYNC_TRIGGERS
    }
    assert {
        name: sql
        for name, sql in _schema_objects(connection, "index").items()
        if not name.startswith("idx_visual_identity_")
    } == before["indexes"]
    assert [
        row[0]
        for row in connection.execute(
            """
            SELECT messages.id
              FROM messages_fts
              JOIN messages ON messages.rowid = messages_fts.rowid
             WHERE messages_fts MATCH 'migrationterm'
             ORDER BY messages.id
            """
        )
    ] == before["fts_matches"]
    assert (
        connection.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
        == before["sync_count"]
    )
    db.close_connection()


def test_v37_message_sync_triggers_include_continuation_only_where_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chachanotes.db"
    seeded = _seed_v36_database(db_path, monkeypatch)
    # Pin the target so only the V36 -> V37 step runs (schema has moved on).
    with monkeypatch.context() as v37_patch:
        v37_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 37)
        db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()

    trigger_sql = _schema_objects(connection, "trigger")
    for name in (
        "messages_sync_create",
        "messages_sync_update",
        "messages_sync_undelete",
    ):
        assert "provider_continuation_json" in trigger_sql[name]
    assert (
        "OLD.provider_continuation_json IS NOT NEW.provider_continuation_json"
        in trigger_sql["messages_sync_update"]
    )
    assert "provider_continuation_json" not in trigger_sql["messages_sync_delete"]

    connection.execute("DELETE FROM sync_log")
    connection.commit()
    connection.execute(
        "UPDATE messages SET provider_continuation_json = '{}' WHERE id = ?",
        (seeded["root_id"],),
    )
    connection.commit()
    entries = connection.execute(
        "SELECT operation, payload FROM sync_log WHERE entity = 'messages' ORDER BY change_id"
    ).fetchall()
    assert len(entries) == 1
    assert entries[0]["operation"] == "update"
    assert json.loads(entries[0]["payload"])["provider_continuation_json"] == "{}"
    db.close_connection()


def test_v36_to_v37_requires_exact_precondition_and_handles_preadded_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chachanotes.db"
    _seed_v36_database(db_path, monkeypatch)

    with monkeypatch.context() as v36_patch:
        v36_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 36)
        db = CharactersRAGDB(db_path, client_id="pre-applied")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE messages ADD COLUMN provider_continuation_json TEXT"
        )
        connection.commit()
        db.close_connection()

    # Pin the target so only the V36 -> V37 step runs (schema has moved on).
    with monkeypatch.context() as v37_patch:
        v37_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 37)
        db = CharactersRAGDB(db_path, client_id="migration-test")
    assert _version(db.get_connection()) == 37
    with pytest.raises(SchemaError, match="requires schema version 36"):
        db._migrate_from_v36_to_v37(db.get_connection())
    db.close_connection()


def test_v36_to_v37_rejects_incompatible_preadded_column_without_partial_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chachanotes.db"
    _seed_v36_database(db_path, monkeypatch)

    with monkeypatch.context() as v36_patch:
        v36_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 36)
        db = CharactersRAGDB(db_path, client_id="incompatible-pre-applied")
        connection = db.get_connection()
        connection.execute(
            "ALTER TABLE messages ADD COLUMN provider_continuation_json "
            "INTEGER NOT NULL DEFAULT 7"
        )
        connection.commit()
        before_triggers = {
            name: sql
            for name, sql in _schema_objects(connection, "trigger").items()
            if name in MESSAGE_SYNC_TRIGGERS
        }
        with pytest.raises(SchemaError, match="incompatible"):
            db._migrate_from_v36_to_v37(connection)

        assert _version(connection) == 36
        assert {
            name: sql
            for name, sql in _schema_objects(connection, "trigger").items()
            if name in MESSAGE_SYNC_TRIGGERS
        } == before_triggers
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT provider_continuation_json FROM messages"
            )
        } == {7}
        db.close_connection()
