from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.citation_legacy_migration import LegacyCitationReadState
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationReaderState,
    select_conversation,
    settle_conversation_continuation,
    settle_conversation_page,
)


@dataclass
class FakeDB:
    conversations_page_rows: list[dict[str, Any]] = field(default_factory=list)
    conversations_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    keywords_by_conversation: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    keyword_lookup: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    keyword_add_results: dict[str, int | None] = field(default_factory=dict)
    message_counts: dict[str, int] = field(default_factory=dict)
    message_count_by_conversation: dict[str, int] = field(default_factory=dict)
    root_counts: dict[str, int] = field(default_factory=dict)
    root_messages: dict[tuple[str, int, int, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    child_messages: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = (
        field(default_factory=dict)
    )
    tree_rows: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    images_by_message_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_message: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    messages_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages_by_conversation: dict[tuple[str, int, int, str], list[dict[str, Any]]] = (
        field(default_factory=dict)
    )
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    replaced_keyword_ids: list[tuple[str, list[int]]] = field(default_factory=list)
    updates: list[tuple[str, dict[str, Any], int]] = field(default_factory=list)
    deletes: list[tuple[str, int]] = field(default_factory=list)
    restores: list[tuple[str, int]] = field(default_factory=list)
    created_conversations: list[dict[str, Any]] = field(default_factory=list)
    located_page: dict[str, Any] | None = None

    def add_conversation(self, conversation_data):
        self.calls.append(("add_conversation", (conversation_data,), {}))
        self.created_conversations.append(conversation_data)
        return "conv-created"

    def search_conversations_page(self, query, **kwargs):
        self.calls.append(("search_conversations_page", (query,), kwargs))
        offset = kwargs.get("offset", 0) or 0
        limit = kwargs.get("limit", len(self.conversations_page_rows)) or len(
            self.conversations_page_rows
        )
        rows = self.conversations_page_rows[offset : offset + limit]
        return rows, len(self.conversations_page_rows), 0.0

    def locate_conversation_page(self, conversation_id, **kwargs):
        self.calls.append(("locate_conversation_page", (conversation_id,), kwargs))
        return self.located_page

    def count_messages_for_conversations(self, conversation_ids, **kwargs):
        self.calls.append(
            ("count_messages_for_conversations", (tuple(conversation_ids),), kwargs)
        )
        return {
            conversation_id: self.message_counts.get(conversation_id, 0)
            for conversation_id in conversation_ids
        }

    def count_messages_for_conversation(self, conversation_id, **kwargs):
        self.calls.append(
            ("count_messages_for_conversation", (conversation_id,), kwargs)
        )
        return self.message_count_by_conversation.get(conversation_id, 0)

    def get_keywords_for_conversations(self, conversation_ids):
        self.calls.append(
            ("get_keywords_for_conversations", (tuple(conversation_ids),), {})
        )
        return {
            conversation_id: self.keywords_by_conversation.get(conversation_id, [])
            for conversation_id in conversation_ids
        }

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
        self.calls.append(
            (
                "replace_keywords_for_conversation",
                (conversation_id, list(keyword_ids)),
                {},
            )
        )
        self.replaced_keyword_ids.append((conversation_id, list(keyword_ids)))
        return True

    def get_conversation_by_id(self, conversation_id, include_deleted=False):
        self.calls.append(
            (
                "get_conversation_by_id",
                (conversation_id,),
                {"include_deleted": include_deleted},
            )
        )
        return self.conversations_by_id.get(conversation_id)

    def update_conversation(self, conversation_id, update_data, expected_version):
        self.calls.append(
            (
                "update_conversation",
                (conversation_id,),
                {"update_data": update_data, "expected_version": expected_version},
            )
        )
        self.updates.append((conversation_id, update_data, expected_version))
        return True

    def soft_delete_conversation(self, conversation_id, expected_version):
        self.calls.append(
            (
                "soft_delete_conversation",
                (conversation_id,),
                {"expected_version": expected_version},
            )
        )
        self.deletes.append((conversation_id, expected_version))
        return True

    def restore_conversation(self, conversation_id, expected_version):
        self.calls.append(
            (
                "restore_conversation",
                (conversation_id,),
                {"expected_version": expected_version},
            )
        )
        self.restores.append((conversation_id, expected_version))
        return True

    def count_root_messages_for_conversation(
        self,
        conversation_id,
        include_deleted_conversation=False,
    ):
        self.calls.append(
            (
                "count_root_messages_for_conversation",
                (conversation_id,),
                {"include_deleted_conversation": include_deleted_conversation},
            )
        )
        return self.root_counts.get(conversation_id, 0)

    def get_root_messages_for_conversation(
        self,
        conversation_id,
        limit,
        offset,
        order_by_timestamp="ASC",
        include_deleted_conversation=False,
    ):
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
        return self.root_messages.get(
            (conversation_id, limit, offset, order_by_timestamp), []
        )

    def get_messages_for_conversation_by_parent_ids(
        self,
        conversation_id,
        parent_ids,
        order_by_timestamp="ASC",
        include_deleted_conversation=False,
    ):
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
        return self.child_messages.get(
            (conversation_id, tuple(parent_ids), order_by_timestamp), []
        )

    def get_message_tree_rows_for_conversation(
        self,
        conversation_id,
        order_by_timestamp="ASC",
        include_deleted_conversation=False,
    ):
        self.calls.append(
            (
                "get_message_tree_rows_for_conversation",
                (conversation_id,),
                {
                    "order_by_timestamp": order_by_timestamp,
                    "include_deleted_conversation": include_deleted_conversation,
                },
            )
        )
        return self.tree_rows.get((conversation_id, order_by_timestamp), [])

    def get_message_images_by_ids(self, message_ids):
        self.calls.append(("get_message_images_by_ids", (tuple(message_ids),), {}))
        return {
            message_id: dict(self.images_by_message_id[message_id])
            for message_id in message_ids
            if message_id in self.images_by_message_id
        }

    def get_message_by_id(self, message_id):
        self.calls.append(("get_message_by_id", (message_id,), {}))
        return self.messages_by_id.get(message_id)

    def get_messages_for_conversation(
        self, conversation_id, limit=100, offset=0, order_by_timestamp="ASC"
    ):
        self.calls.append(
            (
                "get_messages_for_conversation",
                (conversation_id,),
                {
                    "limit": limit,
                    "offset": offset,
                    "order_by_timestamp": order_by_timestamp,
                },
            )
        )
        return self.messages_by_conversation.get(
            (conversation_id, limit, offset, order_by_timestamp), []
        )


def test_normalize_conversation_and_message_rows_preserve_stable_shape():
    service = ChatConversationService(FakeDB())

    conversation = service.normalize_conversation_row(
        {
            "id": "conv-1",
            "assistant_kind": "character",
            "character_id": 7,
            "assistant_id": "7",
            "assistant_authority_id": "local-authority-7",
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
            "system_prompt": "  Be terse.  ",
        }
    )
    assert conversation["title"] == "Chat with Character 7"
    assert conversation["scope_type"] == "global"
    assert conversation["state"] == "resolved"
    assert conversation["topic_label"] == "billing"
    assert conversation["runtime_backend"] == "local"
    assert conversation["assistant_authority_id"] == "local-authority-7"
    assert conversation["discovery_owner"] == "general_chat"
    assert conversation["discovery_entity_id"] is None
    assert conversation["keywords"] == []
    assert conversation["message_count"] == 0
    assert conversation["system_prompt"] == "Be terse."
    assert (
        service.derive_conversation_title({"assistant_kind": None, "title": None})
        == "New Chat"
    )


def test_normalize_conversation_row_defaults_missing_system_prompt_to_none():
    """A conversation row from a pre-migration DB or without one set has no system prompt."""
    service = ChatConversationService(FakeDB())

    conversation = service.normalize_conversation_row({"id": "conv-1"})

    assert conversation["system_prompt"] is None

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
            "provider_continuation_json": '{"schema_version":1}',
        }
    )
    assert message["parent_message_id"] == "msg-root"
    assert message["topology"]["parent_message_id"] == "msg-root"
    assert message["variant"]["variant_of"] == "msg-base"
    assert message["variant"]["is_selected_variant"] is True
    assert message["provider_continuation_json"] == '{"schema_version":1}'


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
    assert conversation["assistant_authority_id"] is None


def test_create_conversation_exposes_and_threads_assistant_authority_id():
    assert (
        "assistant_authority_id"
        in inspect.signature(ChatConversationService.create_conversation).parameters
    )
    db = FakeDB()
    service = ChatConversationService(db)

    conversation_id = service.create_conversation(
        assistant_kind="character",
        assistant_id="server-character-7",
        assistant_authority_id="server-user-v1:" + ("a" * 64),
        runtime_backend="server",
    )

    assert conversation_id == "conv-created"
    assert db.created_conversations[0]["assistant_authority_id"] == (
        "server-user-v1:" + ("a" * 64)
    )


def test_create_conversation_documents_public_contract():
    method = ChatConversationService.create_conversation
    doc = inspect.getdoc(method)

    assert doc is not None
    assert "Args:" in doc
    for name in inspect.signature(method).parameters:
        if name != "self":
            assert f"{name}:" in doc
    assert "Omitting" in doc
    assert "``None`` explicitly" in doc
    assert "conversation_title or title" in doc
    assert "Raw truthy" in doc
    assert "Returns:" in doc
    assert "Raises:" in doc


def test_create_conversation_preserves_omitted_and_explicit_null_authority(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "conversation-authority.sqlite", "test-client")
    try:
        character_id = db.add_character_card({"name": "Authority Character"})
        service = ChatConversationService(db)

        inferred_conversation_id = service.create_conversation(
            character_id=character_id,
            assistant_kind="character",
            assistant_id=str(character_id),
            runtime_backend="local",
        )
        unproven_conversation_id = service.create_conversation(
            character_id=character_id,
            assistant_kind="character",
            assistant_id=str(character_id),
            assistant_authority_id=None,
            runtime_backend="local",
        )

        inferred = db.get_conversation_by_id(inferred_conversation_id)
        unproven = db.get_conversation_by_id(unproven_conversation_id)
        assert inferred["assistant_authority_id"] == db.get_local_authority_id()
        assert unproven["assistant_authority_id"] is None
    finally:
        db.close_connection()


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

    result = service.list_conversations(
        query="billing", limit=1, offset=0, state="resolved", topic_label="billing"
    )

    assert result["pagination"] == {
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
    }
    assert [item["id"] for item in result["items"]] == ["conv-1"]
    assert result["items"][0]["title"] == "Chat with Persona persona.alpha"
    assert result["items"][0]["keywords"] == ["alpha", "beta"]
    assert result["items"][0]["message_count"] == 3
    assert result["items"][0]["runtime_backend"] == "local"
    assert result["items"][0]["discovery_owner"] == "general_chat"
    assert result["items"][0]["discovery_entity_id"] is None

    search_call = next(
        call for call in db.calls if call[0] == "search_conversations_page"
    )
    assert search_call[2]["scope_type"] == "global"
    assert search_call[2]["include_deleted"] is False
    assert search_call[2]["deleted_only"] is False

    workspace_result = service.list_conversations(
        scope_type="workspace", workspace_id="ws-99", include_deleted=True
    )
    assert workspace_result["pagination"]["limit"] == 50
    workspace_call = [
        call for call in db.calls if call[0] == "search_conversations_page"
    ][-1]
    assert workspace_call[2]["scope_type"] == "workspace"
    assert workspace_call[2]["workspace_id"] == "ws-99"
    assert workspace_call[2]["include_deleted"] is True


def test_list_conversations_scope_all_passes_through_without_workspace_filter():
    """The Library snapshot's ``scope_type='all'`` must reach the DB unnarrowed."""
    db = FakeDB(
        conversations_page_rows=[
            {
                "id": "conv-ws",
                "title": "Console chat",
                "scope_type": "workspace",
                "workspace_id": "ws-chats",
                "version": 1,
            },
        ],
    )
    service = ChatConversationService(db)

    service.list_conversations(scope_type="all", limit=50, offset=0)

    search_call = [call for call in db.calls if call[0] == "search_conversations_page"][
        -1
    ]
    assert search_call[2]["scope_type"] == "all"
    assert search_call[2]["workspace_id"] is None

    # An explicit workspace_id always wins over the 'all' sentinel.
    service.list_conversations(scope_type="all", workspace_id="ws-chats")
    narrowed_call = [
        call for call in db.calls if call[0] == "search_conversations_page"
    ][-1]
    assert narrowed_call[2]["scope_type"] == "workspace"
    assert narrowed_call[2]["workspace_id"] == "ws-chats"

    deleted_only_result = service.list_conversations(deleted_only=True)
    assert deleted_only_result["pagination"]["limit"] == 50
    deleted_only_call = [
        call for call in db.calls if call[0] == "search_conversations_page"
    ][-1]
    assert deleted_only_call[2]["deleted_only"] is True
    assert deleted_only_call[2]["include_deleted"] is False


def test_list_conversations_passes_multiple_workspace_ids_to_database():
    db = FakeDB(conversations_page_rows=[])
    service = ChatConversationService(db)

    service.list_conversations(
        scope_type="all",
        workspace_ids=("ws-roleplay", "ws-research"),
        limit=50,
        offset=0,
    )

    search_call = [call for call in db.calls if call[0] == "search_conversations_page"][
        -1
    ]
    assert search_call[2]["workspace_ids"] == ("ws-roleplay", "ws-research")


def test_list_conversations_passes_global_scope_union_to_database():
    db = FakeDB(conversations_page_rows=[])
    service = ChatConversationService(db)

    service.list_conversations(
        scope_type="all",
        workspace_ids=("ws-default",),
        include_global_scope=True,
        limit=50,
        offset=0,
    )

    search_call = [call for call in db.calls if call[0] == "search_conversations_page"][
        -1
    ]
    assert search_call[2]["workspace_ids"] == ("ws-default",)
    assert search_call[2]["include_global_scope"] is True


def test_list_conversations_passes_query_workspace_union_to_database():
    db = FakeDB(conversations_page_rows=[])
    service = ChatConversationService(db)

    service.list_conversations(
        query="Roleplay Tavern",
        scope_type="all",
        query_workspace_ids=("ws-roleplay",),
        query_include_global_scope=False,
        limit=50,
        offset=0,
    )

    search_call = [call for call in db.calls if call[0] == "search_conversations_page"][
        -1
    ]
    assert search_call[2]["query_workspace_ids"] == ("ws-roleplay",)
    assert search_call[2]["query_include_global_scope"] is False


def test_list_conversations_retains_the_exact_ordinary_page_envelope():
    db = FakeDB(
        conversations_page_rows=[
            {"id": f"conv-{index}", "scope_type": "global", "version": 1}
            for index in range(45)
        ]
    )
    service = ChatConversationService(db)

    result = service.list_conversations(limit=20, offset=20)

    assert result["pagination"] == {
        "limit": 20,
        "offset": 20,
        "total": 45,
        "has_more": True,
    }
    assert [item["id"] for item in result["items"]] == [
        f"conv-{index}" for index in range(20, 40)
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": -1}, "limit"),
        ({"limit": 0}, "limit"),
        ({"limit": True}, "limit"),
        ({"limit": 1.5}, "limit"),
        ({"limit": "20"}, "limit"),
        ({"limit": 2**63}, "limit"),
        ({"offset": -1}, "offset"),
        ({"offset": True}, "offset"),
        ({"offset": 1.5}, "offset"),
        ({"offset": "0"}, "offset"),
        ({"offset": 2**63}, "offset"),
    ],
)
def test_list_conversations_rejects_invalid_coordinates_before_db_call(
    kwargs, message
):
    db = FakeDB(conversations_page_rows=[{"id": "conv-1"}])
    service = ChatConversationService(db)

    with pytest.raises(ValueError, match=message):
        service.list_conversations(**kwargs)

    assert db.calls == []


def test_list_conversations_accepts_sqlite_integer_max_without_real_sql():
    sqlite_integer_max = (1 << 63) - 1
    db = FakeDB(conversations_page_rows=[])
    service = ChatConversationService(db)

    result = service.list_conversations(
        limit=sqlite_integer_max, offset=sqlite_integer_max
    )

    assert result["pagination"] == {
        "limit": sqlite_integer_max,
        "offset": sqlite_integer_max,
        "total": 0,
        "has_more": False,
    }
    search_call = next(
        call for call in db.calls if call[0] == "search_conversations_page"
    )
    assert search_call[2]["limit"] == sqlite_integer_max
    assert search_call[2]["offset"] == sqlite_integer_max


def test_locate_conversation_page_normalizes_the_bounded_owning_page():
    rows = [
        {"id": f"conv-{index}", "scope_type": "global", "version": 1}
        for index in range(20, 40)
    ]
    rows[4]["title"] = None
    db = FakeDB(
        located_page={"rows": rows, "offset": 20, "target_index": 24, "total": 45},
        keywords_by_conversation={"conv-24": [{"keyword": "located"}]},
        message_counts={"conv-24": 3},
    )
    service = ChatConversationService(db)

    result = service.locate_conversation_page(
        "conv-24", scope_type="all", limit=20
    )

    assert result["pagination"] == {
        "limit": 20,
        "offset": 20,
        "page": 2,
        "total": 45,
        "target_index": 24,
        "has_more": True,
    }
    assert len(result["items"]) == 20
    assert result["items"][4]["id"] == "conv-24"
    assert result["items"][4]["keywords"] == ["located"]
    assert result["items"][4]["message_count"] == 3
    locate_call = next(call for call in db.calls if call[0] == "locate_conversation_page")
    assert locate_call[2]["scope_type"] == "all"
    assert locate_call[2]["workspace_id"] is None


@pytest.mark.parametrize(
    "located_page, match",
    [
        (
            {"rows": [{"id": "conv-24"}], "offset": 0, "target_index": 24, "total": 45},
            "aligned",
        ),
        (
            {"rows": [{"id": "wrong"}], "offset": 20, "target_index": 20, "total": 21},
            "target",
        ),
        (
            {"rows": [], "offset": 20, "target_index": 20, "total": 45},
            "bounded page",
        ),
    ],
)
def test_locate_conversation_page_rejects_malformed_coordinates(
    located_page, match
):
    service = ChatConversationService(FakeDB(located_page=located_page))

    with pytest.raises(ValueError, match=match):
        service.locate_conversation_page("conv-24", scope_type="all", limit=20)


def test_locate_conversation_page_returns_none_when_target_is_unavailable():
    service = ChatConversationService(FakeDB(located_page=None))

    assert service.locate_conversation_page("conv-missing", limit=20) is None


@pytest.mark.parametrize("limit", [19, 21, True, -1, 1_000_000])
def test_locate_conversation_page_requires_fixed_limit_before_db_call(limit):
    db = FakeDB(located_page=None)
    service = ChatConversationService(db)

    with pytest.raises(ValueError, match="limit"):
        service.locate_conversation_page("conv-target", limit=limit)

    assert db.calls == []


def test_replace_conversation_keywords_resolves_ids_before_replacing():
    db = FakeDB(
        keyword_lookup={
            "alpha": {"id": 11, "keyword": "alpha"},
            "beta": None,
        },
        keyword_add_results={"beta": 22},
    )
    service = ChatConversationService(db)

    result = service.replace_conversation_keywords(
        "conv-1", [" alpha ", "beta", "ALPHA", ""]
    )

    assert result == ["alpha", "beta"]
    assert db.replaced_keyword_ids == [("conv-1", [11, 22])]
    assert [call[0] for call in db.calls].count("get_keyword_by_text") == 2
    assert [call[0] for call in db.calls].count("add_keyword") == 1


def test_restore_conversation_delegates_to_db_restore():
    db = FakeDB()
    service = ChatConversationService(db)

    assert service.restore_conversation("conv-1", expected_version=4) is True
    assert db.restores == [("conv-1", 4)]


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
    authority_id = "11111111-1111-4111-8111-111111111111"
    db = FakeDB(
        conversations_by_id={
            "conv-1": {
                "id": "conv-1",
                "assistant_kind": "character",
                "assistant_id": "17",
                "assistant_authority_id": authority_id,
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
        keywords_by_conversation={
            "conv-1": [{"keyword": "ops"}, {"keyword": "urgent"}]
        },
    )
    service = ChatConversationService(db)

    metadata = service.get_conversation_metadata("conv-1")
    assert metadata["title"] == "Chat with Character 17"
    assert metadata["keywords"] == ["ops", "urgent"]
    assert metadata["topic_label"] == "ops"
    assert metadata["assistant_authority_id"] == authority_id

    authority_result = service.update_conversation_metadata(
        "conv-1",
        {"assistant_authority_id": f"  {authority_id}  "},
        expected_version=9,
    )
    assert authority_result is True

    result = service.update_conversation_metadata(
        "conv-1",
        {
            "assistant_kind": None,
            "assistant_id": None,
            "assistant_authority_id": None,
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
            {"assistant_authority_id": f"  {authority_id}  "},
            9,
        ),
        (
            "conv-1",
            {
                "assistant_kind": None,
                "assistant_id": None,
                "assistant_authority_id": None,
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


def test_update_conversation_authority_leaves_validation_to_database(tmp_path):
    db = CharactersRAGDB(tmp_path / "authority-update.sqlite", "test-client")
    try:
        character_id = db.add_character_card({"name": "Authority Character"})
        conversation_id = db.add_conversation(
            {
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "runtime_backend": "local",
            }
        )
        authority_id = db.get_local_authority_id()
        service = ChatConversationService(db)

        created = db.get_conversation_by_id(conversation_id)
        assert service.update_conversation_metadata(
            conversation_id,
            {"assistant_authority_id": f"  {authority_id}  "},
            expected_version=created["version"],
        )
        canonical = db.get_conversation_by_id(conversation_id)
        assert canonical["assistant_authority_id"] == authority_id

        assert service.update_conversation_metadata(
            conversation_id,
            {"assistant_authority_id": None},
            expected_version=canonical["version"],
        )
        unproven = db.get_conversation_by_id(conversation_id)
        assert unproven["assistant_authority_id"] is None

        with pytest.raises(InputError, match="non-empty"):
            service.update_conversation_metadata(
                conversation_id,
                {"assistant_authority_id": "   "},
                expected_version=unproven["version"],
            )
        after_blank = db.get_conversation_by_id(conversation_id)
        assert after_blank["version"] == unproven["version"]

        with pytest.raises(InputError, match="must be text"):
            service.update_conversation_metadata(
                conversation_id,
                {"assistant_authority_id": 123},
                expected_version=after_blank["version"],
            )
        assert (
            db.get_conversation_by_id(conversation_id)["version"]
            == after_blank["version"]
        )
    finally:
        db.close_connection()


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

    service.update_conversation_metadata(
        "conv-1", {"scope_type": "workspace"}, expected_version=3
    )
    service.update_conversation_metadata(
        "conv-1", {"workspace_id": "ws-2"}, expected_version=3
    )

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
        service.update_conversation_metadata(
            "conv-1", {"workspace_id": None}, expected_version=3
        )


def test_delete_conversation_metadata_routes_soft_delete():
    db = FakeDB()
    service = ChatConversationService(db)

    result = service.delete_conversation("conv-1", expected_version=4)

    assert result is True
    assert db.deletes == [("conv-1", 4)]
    assert db.calls[-1] == (
        "soft_delete_conversation",
        ("conv-1",),
        {"expected_version": 4},
    )


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

    service.update_conversation_metadata(
        "conv-1", {"title": "  Fresh Title  "}, expected_version=3
    )
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
        # TASK-22206: the tree is assembled from ONE conversation-scoped
        # fetch (timestamp order, roots and children interleaved) instead of
        # the old per-parent query fan-out.
        tree_rows={
            ("conv-1", "ASC"): [
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
                },
            ]
        },
    )
    service = ChatConversationService(db)

    tree = service.get_conversation_tree("conv-1")

    assert tree["conversation"]["title"] == "New Chat"
    assert tree["pagination"] == {
        "limit": 50,
        "offset": 0,
        "total_root_threads": 2,
        "has_more": False,
    }
    assert [node["id"] for node in tree["root_threads"]] == ["msg-root-1", "msg-root-2"]
    assert tree["root_threads"][0]["children"][0]["id"] == "msg-child-1"
    assert tree["root_threads"][0]["children"][0]["variant"]["variant_number"] == 2


def test_get_conversation_tree_carries_system_prompt_through_real_db(tmp_path):
    """Full production path: real DB row -> get_conversation_tree()["conversation"].

    This is the exact seam Console's resume handler
    (``ConsoleWorkspaceController._resume_console_workspace_conversation``)
    reads from. A
    fake/static conversation-tree service in a UI-level test would hide a
    regression here (``normalize_conversation_row`` silently drops any key
    it doesn't explicitly allow-list), so this exercises the real
    ``CharactersRAGDB`` + ``ChatConversationService`` stack end to end.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test-client")
    try:
        conversation_id = db.add_conversation(
            {
                "title": "Saved chat",
                "system_prompt": "Answer only in French.",
            }
        )
        service = ChatConversationService(db)

        tree = service.get_conversation_tree(conversation_id)

        assert tree["conversation"]["system_prompt"] == "Answer only in French."
    finally:
        db.close_connection()


def test_get_conversation_tree_carries_metadata_through_real_db(tmp_path):
    """Full production path: real DB row -> get_conversation_tree()["conversation"].

    Console's pinned-response-prefill feature stores
    ``{"pinned_response_prefill": "..."}`` in the ``conversations.metadata``
    JSON column and reads it back via
    ``ChatScreen._console_session_settings_for_resume`` ->
    ``pinned_prefill_from_conversation_metadata(conversation.get("metadata"))``.
    A whitelist projection in ``normalize_conversation_row`` that omits
    ``metadata`` would silently break resume, so this exercises the real
    ``CharactersRAGDB`` + ``ChatConversationService`` stack end to end and
    asserts the raw JSON string round-trips unchanged.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test-client")
    try:
        conversation_id = db.add_conversation({"title": "Saved chat"})
        record = db.get_conversation_by_id(conversation_id)
        metadata_json = json.dumps({"pinned_response_prefill": "Understood:"})
        db.update_conversation(
            conversation_id,
            {"metadata": metadata_json},
            expected_version=record["version"],
        )
        service = ChatConversationService(db)

        tree = service.get_conversation_tree(conversation_id)

        assert tree["conversation"]["metadata"] == metadata_json
    finally:
        db.close_connection()


def test_get_conversation_tree_metadata_is_none_for_new_conversation(tmp_path):
    """A conversation with no metadata written yields an explicit ``None``.

    Consumers use ``.get("metadata")``, but the key must still be present on
    the normalized dict so the contract is explicit rather than accidental.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test-client")
    try:
        conversation_id = db.add_conversation({"title": "Fresh chat"})
        service = ChatConversationService(db)

        tree = service.get_conversation_tree(conversation_id)

        assert "metadata" in tree["conversation"]
        assert tree["conversation"]["metadata"] is None
    finally:
        db.close_connection()


def test_local_rag_context_adjuncts_are_persisted_and_reloaded(tmp_path):
    message = {
        "id": "msg-1",
        "conversation_id": "conv-1",
        "parent_message_id": None,
        "sender": "assistant",
        "content": "Answer with citation",
        "timestamp": "2026-04-19T00:02:00Z",
        "role": "assistant",
    }
    db = FakeDB(
        messages_by_id={"msg-1": message},
        messages_by_conversation={("conv-1", 10, 0, "ASC"): [message]},
    )
    store_path = tmp_path / "chat_rag_context.json"
    service = ChatConversationService(db, rag_context_store_path=store_path)

    saved = service.record_message_rag_context(
        "conv-1",
        "msg-1",
        rag_context={"search_query": "alpha", "chunks": [{"source_id": "note-1"}]},
        citations=[{"id": "cite-1", "source_id": "note-1", "quote": "fact"}],
    )
    messages = service.get_messages_with_context("conv-1", limit=10)
    citations = service.get_citations("conv-1")

    reloaded = ChatConversationService(db, rag_context_store_path=store_path)
    reloaded_messages = reloaded.get_messages_with_context("conv-1", limit=10)
    reloaded_citations = reloaded.get_citations("conv-1")

    assert saved["conversation_id"] == "conv-1"
    assert saved["message_id"] == "msg-1"
    assert messages[0]["id"] == "msg-1"
    assert messages[0]["rag_context"]["search_query"] == "alpha"
    assert messages[0]["citations"][0]["message_id"] == "msg-1"
    assert citations == {
        "conversation_id": "conv-1",
        "citations": [
            {
                "id": "cite-1",
                "source_id": "note-1",
                "quote": "fact",
                "message_id": "msg-1",
            }
        ],
        "total_count": 1,
    }
    assert reloaded_messages[0]["rag_context"]["chunks"] == [{"source_id": "note-1"}]
    assert reloaded_citations == citations


def test_local_rag_context_rejects_message_conversation_mismatches(tmp_path):
    db = FakeDB(
        messages_by_id={
            "msg-1": {
                "id": "msg-1",
                "conversation_id": "conv-2",
                "sender": "assistant",
                "content": "Wrong conversation",
            }
        }
    )
    service = ChatConversationService(
        db, rag_context_store_path=tmp_path / "chat_rag_context.json"
    )

    with pytest.raises(ValueError, match="message does not belong to conversation"):
        service.record_message_rag_context(
            "conv-1", "msg-1", rag_context={"search_query": "alpha"}
        )


def test_canonical_mode_rejects_deprecated_sidecar_writes(tmp_path):
    migration = SimpleNamespace(writes_enabled=True)
    service = ChatConversationService(
        FakeDB(),
        rag_context_store_path=tmp_path / "chat_rag_context.json",
        citation_legacy_migration=migration,
    )

    with pytest.raises(RuntimeError, match="legacy_rag_context_writes_disabled"):
        service.record_message_rag_context(
            "conv-1",
            "msg-1",
            citations=[{"evidence_id": "1", "source_id": "note-1"}],
        )
    assert not service.rag_context_store_path.exists()


def test_canonical_reader_never_merges_changed_legacy_records(tmp_path):
    message = {
        "id": "msg-1",
        "conversation_id": "conv-1",
        "sender": "assistant",
        "content": "Answer [1].",
        "version": 1,
    }
    db = FakeDB(
        messages_by_conversation={("conv-1", 10, 0, "ASC"): [message]},
    )

    class Migration:
        writes_enabled = True

        @staticmethod
        def read_conversation(conversation_id, *, verify_canonical):
            assert conversation_id == "conv-1"
            assert verify_canonical is True
            return SimpleNamespace(
                state=LegacyCitationReadState.DIVERGED,
                records={},
            )

    service = ChatConversationService(
        db,
        rag_context_store_path=tmp_path / "chat_rag_context.json",
        citation_legacy_migration=Migration(),
    )
    service.rag_context_store_path.write_text(
        json.dumps(
            {
                "conversations": {
                    "conv-1": {
                        "msg-1": {"citations": [{"evidence_id": "stale-sidecar"}]}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    messages = service.get_messages_with_context("conv-1", limit=10)
    citations = service.get_citations("conv-1")

    assert messages[0]["citation_provenance_state"] == "diverged"
    assert messages[0]["citations"] == []
    assert citations["state"] == "diverged"
    assert citations["citations"] == []



class TestLibraryConversationSeams:
    """task-1337 (plan Task 4): thin agent-facing delegates over the
    additive DB library read seams. The service forwards pagination/window
    arguments untouched and echoes the list/search envelope shape shared by
    the other Library domains (items/total/offset/limit)."""

    def test_message_projection_preserves_an_already_string_timestamp(self):
        db = object.__new__(CharactersRAGDB)

        item = db._library_message_item(
            {
                "id": "message-1",
                "sender": "user",
                "timestamp": "preserve-this-timestamp",
                "version": 1,
                "total_chars": 4,
                "text": "body",
            },
            char_start=0,
        )

        assert item["timestamp"] == "preserve-this-timestamp"

    def test_list_delegates_and_echoes_pagination(self):
        class FakeLibraryDB:
            def __init__(self):
                self.calls = []

            def list_library_conversations_page(self, *, limit, offset):
                self.calls.append(("list", limit, offset))
                return {"items": [{"id": "conv-1"}], "total": 7}

        db = FakeLibraryDB()
        service = ChatConversationService(db)

        result = service.list_library_conversations(limit=3, offset=6)

        assert db.calls == [("list", 3, 6)]
        assert result == {
            "items": [{"id": "conv-1"}],
            "total": 7,
            "offset": 6,
            "limit": 3,
        }

    def test_search_delegates_and_echoes_pagination(self):
        class FakeLibraryDB:
            def __init__(self):
                self.calls = []

            def search_library_conversations_page(self, *, query, limit, offset):
                self.calls.append(("search", query, limit, offset))
                return {"items": [{"id": "conv-2", "matched_fields": ["title"]}], "total": 1}

        db = FakeLibraryDB()
        service = ChatConversationService(db)

        result = service.search_library_conversations(query="needle", limit=5, offset=10)

        assert db.calls == [("search", "needle", 5, 10)]
        assert result == {
            "items": [{"id": "conv-2", "matched_fields": ["title"]}],
            "total": 1,
            "offset": 10,
            "limit": 5,
        }

    def test_get_messages_forwards_window_arguments(self):
        class FakeLibraryDB:
            def __init__(self):
                self.captured = None

            def get_library_conversation_messages(self, conversation_id, **kwargs):
                self.captured = (conversation_id, kwargs)
                return {"id": conversation_id, "messages": []}

        db = FakeLibraryDB()
        service = ChatConversationService(db)

        result = service.get_library_conversation_messages(
            "conv-9",
            message_offset=4,
            message_limit=5,
            max_chars=100,
            message_id="msg-1",
            char_start=50,
        )

        assert db.captured == (
            "conv-9",
            {
                "message_offset": 4,
                "message_limit": 5,
                "max_chars": 100,
                "message_id": "msg-1",
                "char_start": 50,
            },
        )
        assert result == {"id": "conv-9", "messages": []}

    def test_get_messages_missing_conversation_returns_none(self):
        class FakeLibraryDB:
            def get_library_conversation_messages(self, conversation_id, **kwargs):
                return None

        service = ChatConversationService(FakeLibraryDB())

        assert service.get_library_conversation_messages("missing") is None

    def test_rag_context_sidecar_never_reaches_library_messages(self, tmp_path):
        """RAG context is a JSON sidecar adjunct keyed by conversation/message;
        the library message seam must not join or surface it."""
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(
                db, rag_context_store_path=tmp_path / "rag_ctx.json"
            )
            conv_id = db.add_conversation({"title": "ctx"})
            msg_id = db.add_message(
                {"conversation_id": conv_id, "sender": "user", "content": "hello"}
            )
            service.record_message_rag_context(
                conv_id,
                msg_id,
                rag_context={"pipeline": "websearch"},
                citations=[{"source": "example"}],
            )

            detail = service.get_library_conversation_messages(conv_id)

            assert detail["include_rag_context"] is False
            assert detail["message_total"] == 1
            assert all(
                "rag_context" not in message and "citations" not in message
                for message in detail["messages"]
            )
        finally:
            db.close_connection()

    def test_real_service_pages_are_bounded_exact_and_chronological(self, tmp_path):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            conv_id = db.add_conversation({"title": "paged"})
            message_ids = [
                db.add_message(
                    {
                        "conversation_id": conv_id,
                        "sender": "user",
                        "content": f"body-{index}",
                    }
                )
                for index in range(5)
            ]

            first = service.get_library_conversation_messages(
                conv_id, message_offset=0, message_limit=2, max_chars=4
            )
            middle = service.get_library_conversation_messages(
                conv_id, message_offset=2, message_limit=2, max_chars=4
            )
            last = service.get_library_conversation_messages(
                conv_id, message_offset=4, message_limit=2, max_chars=4
            )
            tiny = service.get_library_conversation_messages(
                conv_id, message_offset=0, message_limit=1, max_chars=4
            )
            repeated_first = service.get_library_conversation_messages(
                conv_id, message_offset=0, message_limit=2, max_chars=4
            )

            assert [
                first["message_offset"],
                middle["message_offset"],
                last["message_offset"],
            ] == [0, 2, 4]
            assert [
                first["message_total"],
                middle["message_total"],
                last["message_total"],
                tiny["message_total"],
            ] == [5, 5, 5, 5]
            messages = first["messages"] + middle["messages"] + last["messages"]
            assert [message["id"] for message in messages] == message_ids
            assert [message["text"] for message in messages] == ["body"] * 5
            assert all(message["returned_chars"] <= 4 for message in messages)
            assert all(message["revision"] for message in messages)
            assert [
                (message["id"], message["revision"])
                for message in repeated_first["messages"]
            ] == [(message["id"], message["revision"]) for message in first["messages"]]
            assert first["version"] == middle["version"] == last["version"] == 1
            assert (
                first["message_epoch"]
                == middle["message_epoch"]
                == last["message_epoch"]
            )
            assert all(
                isinstance(message["timestamp"], str) and message["timestamp"]
                for message in messages
            )
            assert last["has_more"] is False
        finally:
            db.close_connection()

    def test_real_service_epoch_rejects_interleaved_page_and_preserves_iso_timestamp(
        self, tmp_path
    ):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            conv_id = db.add_conversation({"title": "interleaved"})
            first_id = db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "user",
                    "content": "old first",
                    "timestamp": "2026-08-24T12:00:00Z",
                }
            )
            db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "assistant",
                    "content": "second",
                    "timestamp": "2026-08-24T12:01:00Z",
                }
            )
            first_page = service.get_library_conversation_messages(
                conv_id, message_offset=0, message_limit=1
            )
            pending, request = select_conversation(
                ConversationReaderState(), conv_id, version=1
            )
            request = replace(request, message_limit=1)
            partial = settle_conversation_page(pending, request, first_page)
            assert partial.messages[0].timestamp == "2026-08-24T12:00:00Z"

            db.update_message(first_id, {"content": "edited first"}, 1)
            second_page = service.get_library_conversation_messages(
                conv_id, message_offset=1, message_limit=1
            )
            mixed = settle_conversation_page(
                partial,
                replace(request, message_offset=1, message_limit=1),
                second_page,
            )

            assert first_page["message_epoch"] != second_page["message_epoch"]
            assert mixed is partial
            assert not mixed.complete and not mixed.loaded_actions_eligible
        finally:
            db.close_connection()

    def test_real_service_epoch_rejects_interleaved_continuation(self, tmp_path):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            conv_id = db.add_conversation({"title": "continuation epoch"})
            message_id = db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "user",
                    "content": "prefix and suffix",
                }
            )
            first_page = service.get_library_conversation_messages(
                conv_id, message_limit=1, max_chars=7
            )
            pending, request = select_conversation(
                ConversationReaderState(), conv_id, version=1
            )
            partial = settle_conversation_page(pending, request, first_page)

            db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "assistant",
                    "content": "interleaving message",
                }
            )
            continuation = service.get_library_conversation_messages(
                conv_id,
                message_id=message_id,
                char_start=7,
                max_chars=100,
            )
            mixed = settle_conversation_continuation(partial, request, continuation)

            assert first_page["message_epoch"] != continuation["message_epoch"]
            assert mixed is partial
            assert not mixed.complete and not mixed.loaded_actions_eligible
        finally:
            db.close_connection()

    def test_real_service_epoch_ignores_local_usage_and_metadata(self, tmp_path):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            conv_id = db.add_conversation({"title": "local adjuncts"})
            message_id = db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "assistant",
                    "content": "stable transcript",
                }
            )
            before = service.get_library_conversation_messages(conv_id)

            assert db.update_message_usage_local(message_id, '{"total_tokens": 3}')
            assert db.update_message_metadata_local(
                message_id, '{"interrupted": false}'
            )
            after = service.get_library_conversation_messages(conv_id)

            assert before["message_epoch"] == after["message_epoch"]
            assert before["messages"] == after["messages"]
        finally:
            db.close_connection()

    def test_real_service_long_message_continuations_reassemble_once(self, tmp_path):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            conv_id = db.add_conversation({"title": "long"})
            content = "0123456789" * 4 + "tail"
            message_id = db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "user",
                    "content": content,
                }
            )
            assembled = ""
            revisions = set()
            while len(assembled) < len(content):
                detail = service.get_library_conversation_messages(
                    conv_id,
                    message_id=message_id,
                    char_start=len(assembled),
                    max_chars=7,
                )
                message = detail["messages"][0]
                assert message["char_start"] == len(assembled)
                assert 0 < message["returned_chars"] <= 7
                revisions.add(message["revision"])
                assembled += message["text"]

            assert assembled == content
            assert revisions == {detail["messages"][0]["revision"]}
            assert detail["message_total"] == 1
            assert detail["messages"][0]["has_more"] is False
        finally:
            db.close_connection()

    def test_real_service_empty_missing_deleted_and_unavailable_behavior(
        self, tmp_path
    ):
        db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
        try:
            service = ChatConversationService(db)
            empty_id = db.add_conversation({"title": "empty"})

            empty = service.get_library_conversation_messages(empty_id, message_limit=1)

            assert empty["message_total"] == 0
            assert empty["messages"] == []
            assert empty["has_more"] is False
            assert service.get_library_conversation_messages("missing") is None

            db.soft_delete_conversation(empty_id, expected_version=1)
            assert service.get_library_conversation_messages(empty_id) is None

            unavailable_id = db.add_conversation({"title": "unavailable"})
            with db.transaction() as conn:
                conn.execute("DROP TABLE messages")
            with pytest.raises(CharactersRAGDBError):
                service.get_library_conversation_messages(unavailable_id)
        finally:
            db.close_connection()
