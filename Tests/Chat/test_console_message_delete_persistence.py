"""Persistence seams behind Console Delete -> Undo (TASK-33628.2).

Real ``CharactersRAGDB`` on in-memory SQLite behind the real
``ChatPersistenceService`` -- the same pair the Console store writes through.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError


def _conversation(db: CharactersRAGDB, count: int = 4) -> tuple[str, list[str]]:
    conversation_id = db.add_conversation({"title": "Delete undo"})
    ids: list[str] = []
    parent = None
    for index in range(count):
        parent = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user" if index % 2 == 0 else "assistant",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index}",
                "parent_message_id": parent,
            }
        )
        ids.append(parent)
    return conversation_id, ids


def _deleted(db: CharactersRAGDB, ids: list[str]) -> list[int]:
    with db.transaction() as conn:
        rows = {
            row["id"]: int(row["deleted"])
            for row in conn.execute(
                f"SELECT id, deleted FROM messages WHERE id IN ({','.join('?' for _ in ids)})",  # nosec B608 - placeholders only
                tuple(ids),
            ).fetchall()
        }
    return [rows[message_id] for message_id in ids]


def test_media_free_subtree_delete_leaves_no_cleanup_warning():
    """AC#4: no recovered-media catalog means no reference can be pending."""
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)

    rows = service.delete_message_subtree(message_id=ids[1])

    assert {row["message_id"] for row in rows} == set(ids[1:])
    assert not service.recovered_media_cleanup_pending
    assert service.recovered_media_cleanup_warning is None


def test_restore_message_subtree_undeletes_exact_tombstones_and_cursor():
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    conversation_id, ids = _conversation(db)
    db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=ids[-1], before_message_id=None
    )
    rows = service.delete_message_subtree(message_id=ids[1])
    db.set_conversation_active_leaf(conversation_id, ids[0])
    tombstones = tuple((row["message_id"], row["version"]) for row in rows)

    restored = service.restore_message_subtree(
        tombstones=tombstones,
        conversation_id=conversation_id,
        active_cursor=(ids[-1], None),
    )

    assert _deleted(db, ids) == [0, 0, 0, 0]
    assert {(row["message_id"], row["version"]) for row in restored} == {
        (message_id, version + 1) for message_id, version in tombstones
    }
    assert db.get_conversation_active_cursor(conversation_id) == (ids[-1], None)


def test_restore_message_subtree_refuses_a_changed_tombstone_atomically():
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)
    rows = service.delete_message_subtree(message_id=ids[1])
    stale = tuple(
        (row["message_id"], row["version"] - (1 if index == 0 else 0))
        for index, row in enumerate(rows)
    )

    with pytest.raises(ConflictError):
        service.restore_message_subtree(tombstones=stale)

    assert _deleted(db, ids) == [0, 1, 1, 1]


def test_restored_rows_are_committed_sync_intents():
    """An undelete is an ordinary committed update a sync projection can read."""
    db = CharactersRAGDB(":memory:", "delete-undo")
    service = ChatPersistenceService(db)
    _conversation_id, ids = _conversation(db)
    rows = service.delete_message_subtree(message_id=ids[2])

    restored = service.restore_message_subtree(
        tombstones=tuple((row["message_id"], row["version"]) for row in rows)
    )

    for row in restored:
        live = db.get_message_by_id(row["message_id"])
        assert live is not None and live["version"] == row["version"]
        intent = db.read_committed_chat_sync_intent(
            message_id=row["message_id"],
            message_version=row["version"],
            payload_hash=CharactersRAGDB._chat_sync_payload_hash_from_row(live),
        )
        assert intent is not None


def test_image_rejection_hint_names_a_non_destructive_path():
    """AC#5: the recovery hint never sends users to the subtree Delete."""
    hint = ConsoleChatController._IMAGE_REJECTION_RECOVERY_HINT

    assert "use Delete" not in hint
    assert "remove that message" not in hint
    assert "vision-capable model" in hint
    assert "/rewind" in hint
