from __future__ import annotations

from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="test-client")


def test_local_marks_table_exists_on_fresh_schema(tmp_path):
    db = _db(tmp_path)

    row = db.get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("conversation_local_marks",),
    ).fetchone()

    assert row is not None


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
