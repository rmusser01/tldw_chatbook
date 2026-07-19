"""PersonasScreen dictionary attach picker must list workspace-scoped chats.

Regression suite: ``_list_attachable_conversations`` called
``db.search_conversations_page(query="", limit=200, offset=0)`` without
``scope_type``. The real ``CharactersRAGDB.search_conversations_page``
defaults an omitted ``scope_type`` to ``"global"`` (via ``_normalize_scope``),
so every workspace-scoped conversation -- the common case for conversations
created inside a workspace session -- was silently excluded from the attach
picker. These tests exercise the method against a REAL ``CharactersRAGDB``
seeded with both a global- and a workspace-scoped conversation, mirroring the
seeding pattern in ``Tests/Library/test_library_conversations_visibility.py``.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


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
    return CharactersRAGDB(":memory:", "personas-attach-picker-client")


def _create_workspace_chat(
    db: CharactersRAGDB,
    *,
    title: str = "Workspace chat",
    workspace_id: str = "ws-attach",
) -> str:
    """Persist a conversation exactly the way Console does for a workspace session."""
    persistence = ChatPersistenceService(db, workspace_registry=_StaticWorkspaceRegistry())
    return persistence.create_conversation(
        assistant_kind="generic",
        assistant_id="console",
        conversation_title=title,
        workspace_id=workspace_id,
        scope_type="workspace",
    )


def _stub_screen(db: CharactersRAGDB) -> Any:
    """Lightweight stand-in for PersonasScreen: the method only reads
    self.app_instance.chachanotes_db."""
    return types.SimpleNamespace(app_instance=types.SimpleNamespace(chachanotes_db=db))


def test_attach_picker_lists_both_global_and_workspace_conversations(db: CharactersRAGDB) -> None:
    global_id = db.add_conversation({"title": "Global conversation"})
    workspace_id = _create_workspace_chat(db, title="Workspace conversation")

    stub = _stub_screen(db)
    rows = PersonasScreen._list_attachable_conversations(stub)

    by_id = {row["conversation_id"]: row for row in rows}
    assert global_id in by_id
    assert workspace_id in by_id, "workspace-scoped conversation missing from attach picker"
    assert by_id[global_id]["title"] == "Global conversation"
    assert by_id[workspace_id]["title"] == "Workspace conversation"


def test_attach_picker_rows_are_string_ids_with_expected_shape(db: CharactersRAGDB) -> None:
    conversation_id = _create_workspace_chat(db)

    stub = _stub_screen(db)
    rows = PersonasScreen._list_attachable_conversations(stub)

    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {"conversation_id", "title"}
    assert isinstance(row["conversation_id"], str)
    assert row["conversation_id"] == conversation_id
    assert isinstance(row["title"], str)
