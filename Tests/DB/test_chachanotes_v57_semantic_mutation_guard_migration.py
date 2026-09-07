"""ChaChaNotes v57 fail-closed semantic mutation guard across later schemas."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.DB.fixtures.chachanotes_v54 import genuine_v54_database
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _seed_referenced_message(db: CharactersRAGDB) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "guarded"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "guarded body",
        }
    )
    assert message_id is not None
    db.set_message_attachments(
        message_id,
        [
            {
                "position": 1,
                "data": b"before-attachment",
                "mime_type": "image/png",
                "display_name": "before.png",
            }
        ],
    )
    return conversation_id, message_id


def test_genuine_v54_upgrades_through_v57_without_rewriting_existing_messages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "genuine-v54.sqlite"
    with genuine_v54_database(path) as historical:
        row = (
            historical.get_connection()
            .execute(
                "SELECT id, conversation_id, sender, content FROM messages LIMIT 1"
            )
            .fetchone()
        )
        assert row is not None
        before = tuple(row)
        assert _version(historical.get_connection()) == 54

    migrated = CharactersRAGDB(path, client_id="v57-upgrade")
    try:
        connection = migrated.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 57
        after = tuple(
            connection.execute(
                "SELECT id, conversation_id, sender, content FROM messages LIMIT 1"
            ).fetchone()
        )
        assert after == before
    finally:
        migrated.close_connection()


def test_fresh_current_schema_registers_v57_guard_function_and_triggers(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.sqlite", client_id="fresh-v57")
    try:
        connection = db.get_connection()
        assert (
            connection.execute(
                "SELECT console_semantic_mutation_authorized('x', 'message_update')"
            ).fetchone()[0]
            == 0
        )
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        assert {
            "messages_semantic_update_guard",
            "messages_semantic_delete_guard",
            "message_attachments_semantic_insert_guard",
            "message_attachments_semantic_update_guard",
            "message_attachments_semantic_delete_guard",
            "console_trace_semantic_revisions_retirement_guard",
        } <= trigger_names
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {
            "idx_console_trace_calls_surface_policy",
            "idx_console_trace_surface_nodes_revision",
        } <= index_names
    finally:
        db.close_connection()


def test_semantic_mutation_authorization_is_not_publicly_obtainable(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "private-authorization.sqlite", "private-auth")
    try:
        assert not hasattr(db, "get_semantic_mutation_authorization")
        authorization = db._semantic_mutation_authorization_for_coordinator(
            db.get_connection()
        )
        assert not hasattr(authorization, "authorize")
        assert not hasattr(authorization, "sqlite_authorized")
        assert not hasattr(authorization, "assert_current_transaction")
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "statement,params_factory",
    [
        ("UPDATE messages SET content = 'bypass' WHERE id = ?", lambda c, m: (m,)),
        ("UPDATE messages SET sender = 'assistant' WHERE id = ?", lambda c, m: (m,)),
        ("UPDATE messages SET role = 'system' WHERE id = ?", lambda c, m: (m,)),
        (
            "UPDATE messages SET image_data = X'01', image_mime_type = 'image/png' WHERE id = ?",
            lambda c, m: (m,),
        ),
        (
            "UPDATE messages SET provider_continuation_json = '{}' WHERE id = ?",
            lambda c, m: (m,),
        ),
        (
            "UPDATE messages SET thinking_blocks_json = '[]' WHERE id = ?",
            lambda c, m: (m,),
        ),
        (
            "UPDATE messages SET assistant_generation_state = 'complete' WHERE id = ?",
            lambda c, m: (m,),
        ),
        ("DELETE FROM messages WHERE id = ?", lambda c, m: (m,)),
        ("DELETE FROM conversations WHERE id = ?", lambda c, m: (c,)),
        (
            "INSERT INTO message_attachments(message_id, position, data, mime_type, display_name) VALUES (?, 1, X'01', 'image/png', 'x')",
            lambda c, m: (m,),
        ),
        (
            "UPDATE message_attachments SET data = X'02' WHERE message_id = ? AND position = 1",
            lambda c, m: (m,),
        ),
        (
            "DELETE FROM message_attachments WHERE message_id = ? AND position = 1",
            lambda c, m: (m,),
        ),
        (
            "UPDATE console_trace_semantic_revisions SET live_message_id = NULL, live_locator_retired_at = CURRENT_TIMESTAMP WHERE live_message_id = ?",
            lambda c, m: (m,),
        ),
    ],
)
def test_registered_connection_rejects_every_direct_sql_bypass_category(
    tmp_path: Path,
    statement: str,
    params_factory: object,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"guard-{new_opaque_id()}.sqlite", client_id="guard"
    )
    try:
        conversation_id, message_id = _seed_referenced_message(db)
        params = params_factory(conversation_id, message_id)  # type: ignore[operator]
        with pytest.raises(sqlite3.DatabaseError, match="semantic mutation"):
            db.get_connection().execute(statement, params)
    finally:
        db.close_connection()


def test_raw_connection_without_guard_function_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "raw-fail-closed.sqlite"
    db = CharactersRAGDB(path, client_id="managed")
    conversation_id, message_id = _seed_referenced_message(db)
    db.close_connection()

    raw = sqlite3.connect(path)
    raw.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            raw.execute(
                "UPDATE messages SET content = 'bypass' WHERE id = ?", (message_id,)
            )
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            raw.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    finally:
        raw.close()


def test_presentation_only_update_does_not_require_authorization(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "presentation.sqlite", client_id="presentation")
    try:
        _conversation_id, message_id = _seed_referenced_message(db)
        db.get_connection().execute(
            "UPDATE messages SET ranking = 4, usage_json = '{}' WHERE id = ?",
            (message_id,),
        )
        row = (
            db.get_connection()
            .execute(
                "SELECT ranking, usage_json FROM messages WHERE id = ?", (message_id,)
            )
            .fetchone()
        )
        assert tuple(row) == (4, "{}")
    finally:
        db.close_connection()
