from __future__ import annotations

import pytest

from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="test-client")


def _assert_local_marks_schema(db):
    conn = db.get_connection()
    columns = conn.execute("PRAGMA table_info(conversation_local_marks)").fetchall()
    assert [
        (row["name"], row["type"], row["notnull"], row["pk"])
        for row in columns
    ] == [
        ("conversation_id", "TEXT", 1, 1),
        ("mark_type", "TEXT", 1, 2),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    ]

    indexes = conn.execute("PRAGMA index_list(conversation_local_marks)").fetchall()
    index_names = {row["name"] for row in indexes}
    assert "idx_conversation_local_marks_type" in index_names

    index_columns = conn.execute(
        "PRAGMA index_info(idx_conversation_local_marks_type)"
    ).fetchall()
    assert [row["name"] for row in index_columns] == [
        "mark_type",
        "updated_at",
        "conversation_id",
    ]


def test_local_marks_table_exists_on_fresh_schema_with_expected_shape(tmp_path):
    db = _db(tmp_path)
    conn = db.get_connection()

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("conversation_local_marks",),
    ).fetchone()

    assert row is not None
    _assert_local_marks_schema(db)


def test_local_marks_migrate_from_v16_to_v17_with_expected_schema(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id="test-client")
    conn = db.get_connection()
    conn.execute("DROP INDEX IF EXISTS idx_conversation_local_marks_type")
    conn.execute("DROP TABLE IF EXISTS conversation_local_marks")
    conn.execute(
        """
        UPDATE db_schema_version
           SET version = 16
         WHERE schema_name = ?
        """,
        (db._SCHEMA_NAME,),
    )
    conn.commit()
    db.close_connection()

    migrated = CharactersRAGDB(str(db_path), client_id="test-client")

    version = migrated.get_connection().execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    assert version["version"] == 17
    _assert_local_marks_schema(migrated)


def test_star_unstar_is_idempotent_and_ordered(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("conv-a")
    service.star_conversation("conv-b")
    service.star_conversation("conv-a")

    assert service.is_starred("conv-a") is True
    assert service.is_starred("conv-b") is True
    assert service.list_marked_conversation_ids() == ("conv-a", "conv-b")

    service.unstar_conversation("conv-a")
    service.unstar_conversation("conv-a")

    assert service.is_starred("conv-a") is False
    assert service.list_marked_conversation_ids() == ("conv-b",)


def test_local_marks_tolerate_missing_conversations(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("missing-conversation")

    assert service.is_starred("missing-conversation") is True
    assert service.list_marked_conversation_ids() == ("missing-conversation",)


@pytest.mark.parametrize("mark_type", ["", "   ", "archived"])
def test_local_marks_reject_blank_and_unsupported_mark_types(tmp_path, mark_type):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    with pytest.raises(ValueError, match="Unsupported conversation mark_type"):
        service.set_mark("conv-a", mark_type)


@pytest.mark.parametrize("limit", [0, -1])
def test_list_marked_conversation_ids_rejects_non_positive_limits(tmp_path, limit):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    with pytest.raises(ValueError, match="limit must be positive"):
        service.list_marked_conversation_ids(limit=limit)


def test_local_marks_do_not_create_sync_log_entries(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    conversation_id = conversations.create_conversation(title="Sync Boundary")
    db.get_connection().execute("DELETE FROM sync_log")
    db.get_connection().commit()

    ConversationLocalMarksService(db).star_conversation(conversation_id)

    rows = db.get_connection().execute(
        "SELECT entity, entity_id, operation, payload FROM sync_log"
    ).fetchall()
    assert rows == []


def test_conversation_metadata_does_not_include_local_marks(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    marks = ConversationLocalMarksService(db)
    conversation_id = conversations.create_conversation(title="Plain Metadata")

    marks.star_conversation(conversation_id)
    metadata = conversations.get_conversation_metadata(conversation_id)

    assert metadata is not None
    assert "starred" not in metadata
    assert "marks" not in metadata
    assert "local_marks" not in metadata
