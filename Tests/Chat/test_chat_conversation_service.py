from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService


@dataclass
class FakeDB:
    conversations_page_rows: list[dict[str, Any]] = field(default_factory=list)
    conversations_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    keywords_by_conversation: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    keyword_lookup: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    keyword_add_results: dict[str, int | None] = field(default_factory=dict)
    message_counts: dict[str, int] = field(default_factory=dict)
    message_count_by_conversation: dict[str, int] = field(default_factory=dict)
    root_counts: dict[str, int] = field(default_factory=dict)
    root_messages: dict[tuple[str, int, int, str], list[dict[str, Any]]] = field(default_factory=dict)
    child_messages: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = field(default_factory=dict)
    latest_message: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    replaced_keyword_ids: list[tuple[str, list[int]]] = field(default_factory=list)
    updates: list[tuple[str, dict[str, Any], int]] = field(default_factory=list)

    def search_conversations_page(self, query, **kwargs):
        self.calls.append(("search_conversations_page", (query,), kwargs))
        offset = kwargs.get("offset", 0) or 0
        limit = kwargs.get("limit", len(self.conversations_page_rows)) or len(self.conversations_page_rows)
        rows = self.conversations_page_rows[offset : offset + limit]
        return rows, len(self.conversations_page_rows), 0.0

    def count_messages_for_conversations(self, conversation_ids, **kwargs):
        self.calls.append(("count_messages_for_conversations", (tuple(conversation_ids),), kwargs))
        return {conversation_id: self.message_counts.get(conversation_id, 0) for conversation_id in conversation_ids}

    def count_messages_for_conversation(self, conversation_id, **kwargs):
        self.calls.append(("count_messages_for_conversation", (conversation_id,), kwargs))
        return self.message_count_by_conversation.get(conversation_id, 0)

    def get_keywords_for_conversations(self, conversation_ids):
        self.calls.append(("get_keywords_for_conversations", (tuple(conversation_ids),), {}))
        return {conversation_id: self.keywords_by_conversation.get(conversation_id, []) for conversation_id in conversation_ids}

    def get_keywords_for_conversation(self, conversation_id):
        self.calls.append(("get_keywords_for_conversation", (conversation_id,), {}))
        return self.keywords_by_conversation.get(conversation_id, [])

    def get_keyword_by_text(self, keyword_text):
        self.calls.append(("get_keyword_by_text", (keyword_text,), {}))
        return self.keyword_lookup.get(keyword_text)

    def add_keyword(self, keyword_text):
        self.calls.append(("add_keyword", (keyword_text,), {}))
        return self.keyword_add_results.get(keyword_text)

    def replace_keywords_for_conversation(self, conversation_id, keyword_ids):
        self.calls.append(("replace_keywords_for_conversation", (conversation_id, list(keyword_ids)), {}))
        self.replaced_keyword_ids.append((conversation_id, list(keyword_ids)))
        return True

    def get_conversation_by_id(self, conversation_id, include_deleted=False):
        self.calls.append(("get_conversation_by_id", (conversation_id,), {"include_deleted": include_deleted}))
        return self.conversations_by_id.get(conversation_id)

    def update_conversation(self, conversation_id, update_data, expected_version):
        self.calls.append(("update_conversation", (conversation_id,), {"update_data": update_data, "expected_version": expected_version}))
        self.updates.append((conversation_id, update_data, expected_version))
        return True

    def count_root_messages_for_conversation(self, conversation_id, include_deleted_conversation=False):
        self.calls.append(("count_root_messages_for_conversation", (conversation_id,), {"include_deleted_conversation": include_deleted_conversation}))
        return self.root_counts.get(conversation_id, 0)

    def get_root_messages_for_conversation(self, conversation_id, limit, offset, order_by_timestamp="ASC", include_deleted_conversation=False):
        self.calls.append(
            (
                "get_root_messages_for_conversation",
                (conversation_id, limit, offset),
                {
                    "order_by_timestamp": order_by_timestamp,
                    "include_deleted_conversation": include_deleted_conversation,
                },
            )
        )
        return self.root_messages.get((conversation_id, limit, offset, order_by_timestamp), [])

    def get_messages_for_conversation_by_parent_ids(self, conversation_id, parent_ids, order_by_timestamp="ASC", include_deleted_conversation=False):
        self.calls.append(
            (
                "get_messages_for_conversation_by_parent_ids",
                (conversation_id, tuple(parent_ids)),
                {
                    "order_by_timestamp": order_by_timestamp,
                    "include_deleted_conversation": include_deleted_conversation,
                },
            )
        )
        return self.child_messages.get((conversation_id, tuple(parent_ids), order_by_timestamp), [])


def test_normalize_conversation_and_message_rows_preserve_stable_shape():
    service = ChatConversationService(FakeDB())

    conversation = service.normalize_conversation_row(
        {
            "id": "conv-1",
            "assistant_kind": "character",
            "character_id": 7,
            "assistant_id": "7",
            "persona_memory_mode": None,
            "title": None,
            "state": "Resolved",
            "topic_label": " billing ",
            "topic_label_source": "manual",
            "scope_type": "global",
            "workspace_id": None,
            "created_at": "2026-04-19T00:00:00Z",
            "last_modified": "2026-04-19T00:01:00Z",
            "version": 4,
        }
    )
    assert conversation["title"] == "Chat with Character 7"
    assert conversation["scope_type"] == "global"
    assert conversation["state"] == "resolved"
    assert conversation["topic_label"] == "billing"
    assert conversation["runtime_backend"] == "local"
    assert conversation["discovery_owner"] == "general_chat"
    assert conversation["discovery_entity_id"] is None
    assert conversation["keywords"] == []
    assert conversation["message_count"] == 0
    assert service.derive_conversation_title({"assistant_kind": None, "title": None}) == "New Chat"

    message = service.normalize_message_row(
        {
            "id": "msg-1",
            "conversation_id": "conv-1",
            "parent_message_id": "msg-root",
            "sender": "assistant",
            "content": "hello",
            "timestamp": "2026-04-19T00:02:00Z",
            "last_modified": "2026-04-19T00:03:00Z",
            "role": "assistant",
            "variant_of": "msg-base",
            "variant_number": 2,
            "is_selected_variant": 1,
            "total_variants": 3,
        }
    )
    assert message["parent_message_id"] == "msg-root"
    assert message["topology"]["parent_message_id"] == "msg-root"
    assert message["variant"]["variant_of"] == "msg-base"
    assert message["variant"]["is_selected_variant"] is True


def test_legacy_character_conversation_defaults_missing_assistant_id_to_character_id():
    service = ChatConversationService(FakeDB())

    conversation = service.normalize_conversation_row(
        {
            "id": "conv-legacy",
            "assistant_kind": "character",
            "character_id": 9,
            "assistant_id": None,
            "persona_memory_mode": None,
            "title": None,
            "state": "in-progress",
            "scope_type": "global",
            "workspace_id": None,
            "created_at": "2026-04-19T00:00:00Z",
            "last_modified": "2026-04-19T00:01:00Z",
            "version": 1,
        }
    )

    assert conversation["assistant_id"] == "9"


def test_list_conversations_normalizes_pagination_and_enforces_global_defaults():
    db = FakeDB(
        conversations_page_rows=[
            {
                "id": "conv-1",
                "assistant_kind": "persona",
                "assistant_id": "persona.alpha",
                "title": None,
                "state": "in-progress",
                "topic_label": "billing",
                "scope_type": "global",
                "workspace_id": None,
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:01:00Z",
                "version": 1,
            },
            {
                "id": "conv-2",
                "assistant_kind": None,
                "assistant_id": None,
                "title": "Kept title",
                "state": "resolved",
                "topic_label": None,
                "scope_type": "global",
                "workspace_id": None,
                "created_at": "2026-04-18T00:00:00Z",
                "last_modified": "2026-04-18T00:01:00Z",
                "version": 2,
            },
        ],
        keywords_by_conversation={
            "conv-1": [{"keyword": "alpha"}, {"keyword": "beta"}],
            "conv-2": [{"keyword": "gamma"}],
        },
        message_counts={"conv-1": 3, "conv-2": 1},
    )
    service = ChatConversationService(db)

    result = service.list_conversations(query="billing", limit=1, offset=0, state="resolved", topic_label="billing")

    assert result["pagination"] == {"limit": 1, "offset": 0, "total": 2, "has_more": True}
    assert [item["id"] for item in result["items"]] == ["conv-1"]
    assert result["items"][0]["title"] == "Chat with Persona persona.alpha"
    assert result["items"][0]["keywords"] == ["alpha", "beta"]
    assert result["items"][0]["message_count"] == 3
    assert result["items"][0]["runtime_backend"] == "local"
    assert result["items"][0]["discovery_owner"] == "general_chat"
    assert result["items"][0]["discovery_entity_id"] is None

    search_call = next(call for call in db.calls if call[0] == "search_conversations_page")
    assert search_call[2]["scope_type"] == "global"
    assert search_call[2]["include_deleted"] is False
    assert search_call[2]["deleted_only"] is False

    workspace_result = service.list_conversations(scope_type="workspace", workspace_id="ws-99", include_deleted=True)
    assert workspace_result["pagination"]["limit"] == 50
    workspace_call = [call for call in db.calls if call[0] == "search_conversations_page"][-1]
    assert workspace_call[2]["scope_type"] == "workspace"
    assert workspace_call[2]["workspace_id"] == "ws-99"
    assert workspace_call[2]["include_deleted"] is True

    deleted_only_result = service.list_conversations(deleted_only=True)
    assert deleted_only_result["pagination"]["limit"] == 50
    deleted_only_call = [call for call in db.calls if call[0] == "search_conversations_page"][-1]
    assert deleted_only_call[2]["deleted_only"] is True
    assert deleted_only_call[2]["include_deleted"] is False


def test_replace_conversation_keywords_resolves_ids_before_replacing():
    db = FakeDB(
        keyword_lookup={
            "alpha": {"id": 11, "keyword": "alpha"},
            "beta": None,
        },
        keyword_add_results={"beta": 22},
    )
    service = ChatConversationService(db)

    result = service.replace_conversation_keywords("conv-1", [" alpha ", "beta", "ALPHA", ""])

    assert result == ["alpha", "beta"]
    assert db.replaced_keyword_ids == [("conv-1", [11, 22])]
    assert [call[0] for call in db.calls].count("get_keyword_by_text") == 2
    assert [call[0] for call in db.calls].count("add_keyword") == 1


def test_invalid_state_values_are_rejected_by_the_service_seam():
    service = ChatConversationService(FakeDB())

    with pytest.raises(ValueError, match="Invalid state 'archived'"):
        service.normalize_conversation_row(
            {
                "id": "conv-9",
                "title": None,
                "state": "archived",
                "scope_type": "global",
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:00:00Z",
            }
        )


def test_mixed_case_assistant_kind_normalizes_on_read_and_write():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "assistant_kind": "Character",
                "assistant_id": "17",
                "character_id": 17,
                "scope_type": "global",
                "workspace_id": None,
                "title": None,
                "state": "in-progress",
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:01:00Z",
                "version": 4,
            }
        }
    )
    service = ChatConversationService(db)

    normalized = service.get_conversation_metadata("conv-1")
    assert normalized["assistant_kind"] == "character"
    assert normalized["title"] == "Chat with Character 17"

    service.update_conversation_metadata(
        "conv-1",
        {"assistant_kind": "Persona", "assistant_id": "persona.beta"},
        expected_version=4,
    )

    assert db.updates[-1][1]["assistant_kind"] == "persona"


def test_get_and_update_conversation_metadata_routes_normalized_fields():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "assistant_kind": "character",
                "assistant_id": "17",
                "character_id": 17,
                "persona_memory_mode": None,
                "scope_type": "global",
                "workspace_id": None,
                "title": None,
                "state": "backlog",
                "topic_label": "ops",
                "topic_label_source": "manual",
                "source": "import",
                "external_ref": "ref-1",
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:01:00Z",
                "version": 9,
            }
        },
        keywords_by_conversation={"conv-1": [{"keyword": "ops"}, {"keyword": "urgent"}]},
    )
    service = ChatConversationService(db)

    metadata = service.get_conversation_metadata("conv-1")
    assert metadata["title"] == "Chat with Character 17"
    assert metadata["keywords"] == ["ops", "urgent"]
    assert metadata["topic_label"] == "ops"

    result = service.update_conversation_metadata(
        "conv-1",
        {
            "assistant_kind": None,
            "assistant_id": None,
            "character_id": None,
            "persona_memory_mode": None,
            "scope_type": "workspace",
            "workspace_id": "ws-1",
            "state": "resolved",
            "topic_label": "billing",
            "topic_label_source": "auto",
            "source": "sync",
            "external_ref": "ref-2",
        },
        expected_version=9,
    )

    assert result is True
    assert db.updates == [
        (
            "conv-1",
            {
                "assistant_kind": None,
                "assistant_id": None,
                "character_id": None,
                "persona_memory_mode": None,
                "scope_type": "workspace",
                "workspace_id": "ws-1",
                "state": "resolved",
                "topic_label": "billing",
                "topic_label_source": "auto",
                "source": "sync",
                "external_ref": "ref-2",
            },
            9,
        )
    ]


def test_update_conversation_metadata_merges_scope_from_current_state_before_validation():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "scope_type": "workspace",
                "workspace_id": "ws-1",
                "title": "Existing",
                "last_modified": "2026-04-19T00:01:00Z",
                "created_at": "2026-04-19T00:00:00Z",
                "version": 3,
            }
        }
    )
    service = ChatConversationService(db)

    service.update_conversation_metadata("conv-1", {"scope_type": "workspace"}, expected_version=3)
    service.update_conversation_metadata("conv-1", {"workspace_id": "ws-2"}, expected_version=3)

    assert db.updates[0][1] == {"scope_type": "workspace", "workspace_id": "ws-1"}
    assert db.updates[1][1] == {"scope_type": "workspace", "workspace_id": "ws-2"}


def test_update_conversation_metadata_rejects_workspace_id_clears_without_explicit_scope_change():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "scope_type": "workspace",
                "workspace_id": "ws-1",
                "title": "Existing",
                "last_modified": "2026-04-19T00:01:00Z",
                "created_at": "2026-04-19T00:00:00Z",
                "version": 3,
            }
        }
    )
    service = ChatConversationService(db)

    with pytest.raises(ValueError, match="workspace_id is required"):
        service.update_conversation_metadata("conv-1", {"workspace_id": None}, expected_version=3)


def test_get_conversation_metadata_uses_real_message_count_when_missing_from_row():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "assistant_kind": None,
                "assistant_id": None,
                "character_id": None,
                "title": None,
                "scope_type": "global",
                "workspace_id": None,
                "state": "in-progress",
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:01:00Z",
                "version": 1,
            }
        },
        message_count_by_conversation={"conv-1": 7},
    )
    service = ChatConversationService(db)

    metadata = service.get_conversation_metadata("conv-1")

    assert metadata["message_count"] == 7
    assert any(call[0] == "count_messages_for_conversation" for call in db.calls)


def test_title_updates_are_trimmed_and_whitespace_only_titles_collapse_to_none():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "scope_type": "global",
                "workspace_id": None,
                "title": "Existing",
                "last_modified": "2026-04-19T00:01:00Z",
                "created_at": "2026-04-19T00:00:00Z",
                "version": 3,
            }
        }
    )
    service = ChatConversationService(db)

    service.update_conversation_metadata("conv-1", {"title": "  Fresh Title  "}, expected_version=3)
    service.update_conversation_metadata("conv-1", {"title": "   "}, expected_version=3)

    assert db.updates[0][1]["title"] == "Fresh Title"
    assert db.updates[1][1]["title"] is None


def test_get_conversation_tree_wraps_root_and_child_rows():
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "assistant_kind": None,
                "assistant_id": None,
                "character_id": None,
                "title": None,
                "scope_type": "global",
                "workspace_id": None,
                "state": "in-progress",
                "topic_label": None,
                "created_at": "2026-04-19T00:00:00Z",
                "last_modified": "2026-04-19T00:01:00Z",
                "version": 1,
            }
        },
        root_counts={"conv-1": 2},
        root_messages={
            ("conv-1", 50, 0, "ASC"): [
                {
                    "id": "msg-root-1",
                    "conversation_id": "conv-1",
                    "parent_message_id": None,
                    "sender": "user",
                    "content": "root one",
                    "timestamp": "2026-04-19T00:02:00Z",
                    "role": "user",
                    "variant_of": None,
                    "variant_number": None,
                    "is_selected_variant": None,
                    "total_variants": None,
                },
                {
                    "id": "msg-root-2",
                    "conversation_id": "conv-1",
                    "parent_message_id": None,
                    "sender": "assistant",
                    "content": "root two",
                    "timestamp": "2026-04-19T00:03:00Z",
                    "role": "assistant",
                    "variant_of": None,
                    "variant_number": None,
                    "is_selected_variant": None,
                    "total_variants": None,
                },
            ]
        },
        child_messages={
            (
                "conv-1",
                ("msg-root-1",),
                "ASC",
            ): [
                {
                    "id": "msg-child-1",
                    "conversation_id": "conv-1",
                    "parent_message_id": "msg-root-1",
                    "sender": "assistant",
                    "content": "child",
                    "timestamp": "2026-04-19T00:04:00Z",
                    "role": "assistant",
                    "variant_of": "msg-root-1",
                    "variant_number": 2,
                    "is_selected_variant": 1,
                    "total_variants": 2,
                }
            ]
        },
    )
    service = ChatConversationService(db)

    tree = service.get_conversation_tree("conv-1")

    assert tree["conversation"]["title"] == "New Chat"
    assert tree["pagination"] == {"limit": 50, "offset": 0, "total_root_threads": 2, "has_more": False}
    assert [node["id"] for node in tree["root_threads"]] == ["msg-root-1", "msg-root-2"]
    assert tree["root_threads"][0]["children"][0]["id"] == "msg-child-1"
    assert tree["root_threads"][0]["children"][0]["variant"]["variant_number"] == 2
