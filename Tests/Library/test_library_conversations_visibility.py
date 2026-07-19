"""Library Browse ▸ Conversations must see Console-created workspace chats.

Regression suite for the live-UAT "Conversations (0)" bug (task-179):
Console chats started inside a workspace session persist with
``scope_type='workspace'`` (``console_chat_store.persist_session_if_needed``
-> ``ChatPersistenceService.create_conversation``), but the Library
conversations snapshot used the service's default 'global' scope and so
neither listed nor counted them. The Library now requests
``scope_type='all'``; these tests exercise that seam against a REAL
in-memory ``CharactersRAGDB``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


class _StaticWorkspaceRegistry:
    """Minimal registry double: real DB behavior stays in CharactersRAGDB."""

    def __init__(self) -> None:
        self.links: list[tuple[str, dict[str, Any]]] = []

    def get_workspace(self, workspace_id: str) -> dict[str, str]:
        return {"id": workspace_id}

    def link_membership(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        self.links.append((workspace_id, kwargs))
        return {"workspace_id": workspace_id, **kwargs}


@pytest.fixture()
def db() -> CharactersRAGDB:
    return CharactersRAGDB(":memory:", "library-visibility-client")


def _create_console_workspace_chat(
    db: CharactersRAGDB,
    *,
    title: str = "Console chat",
    workspace_id: str = "ws-chats",
    message: str | None = "hello",
) -> str:
    """Persist a conversation exactly the way Console does for a workspace session."""
    persistence = ChatPersistenceService(db, workspace_registry=_StaticWorkspaceRegistry())
    conversation_id = persistence.create_conversation(
        assistant_kind="generic",
        assistant_id="console",
        conversation_title=title,
        workspace_id=workspace_id,
        scope_type="workspace",
    )
    if message is not None:
        persistence.create_message(
            conversation_id=conversation_id,
            sender="user",
            content=message,
            image_data=None,
            image_mime_type=None,
        )
    return conversation_id


def test_library_scope_all_lists_console_workspace_chats(db: CharactersRAGDB) -> None:
    console_id = _create_console_workspace_chat(db, title="Console chat")
    seeded_id = db.add_conversation({"title": "Seeded conversation"})

    service = ChatConversationService(db)
    result = service.list_conversations(scope_type="all", limit=50, offset=0)

    by_id = {item["id"]: item for item in result["items"]}
    assert set(by_id) == {console_id, seeded_id}
    assert result["pagination"]["total"] == 2
    assert by_id[console_id]["scope_type"] == "workspace"
    assert by_id[console_id]["workspace_id"] == "ws-chats"
    assert by_id[console_id]["message_count"] == 1
    assert by_id[seeded_id]["scope_type"] == "global"


def test_default_global_scope_still_excludes_workspace_chats(db: CharactersRAGDB) -> None:
    """Console's per-scope rail queries rely on the 'global' default staying strict."""
    console_id = _create_console_workspace_chat(db)
    seeded_id = db.add_conversation({"title": "Seeded conversation"})

    service = ChatConversationService(db)
    result = service.list_conversations(limit=50, offset=0)

    ids = [item["id"] for item in result["items"]]
    assert ids == [seeded_id]
    assert console_id not in ids
    assert result["pagination"]["total"] == 1


def test_scope_all_count_is_exact_when_list_page_is_capped(db: CharactersRAGDB) -> None:
    """The Library rail badge is count-based; the page cap must not cap the COUNT."""
    for index in range(3):
        _create_console_workspace_chat(
            db,
            title=f"Console chat {index}",
            workspace_id=f"ws-{index}",
        )
    db.add_conversation({"title": "Seeded conversation"})

    service = ChatConversationService(db)
    result = service.list_conversations(scope_type="all", limit=2, offset=0)

    assert len(result["items"]) == 2
    assert result["pagination"]["total"] == 4
    assert result["pagination"]["has_more"] is True


def test_scope_all_excludes_soft_deleted_conversations(db: CharactersRAGDB) -> None:
    console_id = _create_console_workspace_chat(db)
    kept_id = db.add_conversation({"title": "Kept conversation"})
    row = db.get_conversation_by_id(console_id)
    db.soft_delete_conversation(console_id, expected_version=int(row["version"]))

    service = ChatConversationService(db)
    result = service.list_conversations(scope_type="all", limit=50, offset=0)

    assert [item["id"] for item in result["items"]] == [kept_id]
    assert result["pagination"]["total"] == 1


def test_scope_all_with_explicit_workspace_id_narrows_to_that_workspace(
    db: CharactersRAGDB,
) -> None:
    """An explicit workspace_id always wins over the 'all' sentinel."""
    target_id = _create_console_workspace_chat(db, workspace_id="ws-target")
    _create_console_workspace_chat(db, title="Other chat", workspace_id="ws-other")
    db.add_conversation({"title": "Seeded conversation"})

    service = ChatConversationService(db)
    result = service.list_conversations(
        scope_type="all",
        workspace_id="ws-target",
        limit=50,
        offset=0,
    )

    assert [item["id"] for item in result["items"]] == [target_id]
    assert result["pagination"]["total"] == 1


def test_search_conversations_page_rejects_all_scope_with_workspace_id(
    db: CharactersRAGDB,
) -> None:
    with pytest.raises(InputError):
        db.search_conversations_page(None, scope_type="all", workspace_id="ws-chats")


def test_get_all_conversation_ids_matches_library_scope_all(db: CharactersRAGDB) -> None:
    """Chatbook export resolves from the same all-scope population the Library lists."""
    console_id = _create_console_workspace_chat(db)
    seeded_id = db.add_conversation({"title": "Seeded conversation"})
    deleted_id = db.add_conversation({"title": "Deleted conversation"})
    row = db.get_conversation_by_id(deleted_id)
    db.soft_delete_conversation(deleted_id, expected_version=int(row["version"]))

    assert set(db.get_all_conversation_ids()) == {console_id, seeded_id}
