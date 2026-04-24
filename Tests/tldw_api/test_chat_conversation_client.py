"""
Tests for chat conversation endpoint wiring on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.chat_conversation_schemas import (
    ChatAnalyticsResponse,
    ChatCommandsListResponse,
    ChatQueueActivityResponse,
    ChatQueueStatusResponse,
    ConversationShareLinkCreateRequest,
    ConversationUpdateRequest,
    KnowledgeSaveRequest,
    KnowledgeSaveResponse,
    RagContext,
    RagContextPersistRequest,
    ValidateDictionaryRequest,
    ValidateDictionaryResponse,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


@pytest.mark.asyncio
class TestChatConversationClient:
    """Verify endpoint wiring for conversation client methods."""

    @pytest.mark.parametrize("order_by", ["recency", "bm25", "hybrid", "topic"])
    async def test_list_chat_conversations_forwards_filters_and_rankings(self, monkeypatch, order_by):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"items": [], "pagination": {}})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_chat_conversations(
            query="alpha",
            state="resolved",
            include_deleted=True,
            deleted_only=False,
            order_by=order_by,
            limit=25,
            offset=5,
            scope_type="workspace",
            workspace_id="ws-1",
        )

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/chat/conversations",
            {
                "params": {
                    "query": "alpha",
                    "state": "resolved",
                    "include_deleted": "true",
                    "deleted_only": "false",
                    "order_by": order_by,
                    "limit": 25,
                    "offset": 5,
                    "date_field": "last_modified",
                    "scope_type": "workspace",
                    "workspace_id": "ws-1",
                },
            },
        )

    async def test_get_chat_conversation_forwards_scope_params(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": "conv-1"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_chat_conversation("conv-1", scope_type="workspace", workspace_id="ws-1")

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/chat/conversations/conv-1",
            {"params": {"scope_type": "workspace", "workspace_id": "ws-1"}},
        )

    async def test_update_chat_conversation_patches_normalized_payload(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": "conv-1"})
        monkeypatch.setattr(client, "_request", mocked)

        request_data = ConversationUpdateRequest(
            version=7,
            state=" resolved ",
            keywords=[" Alpha ", "beta", "alpha", None, ""],
        )

        await client.update_chat_conversation(
            "conv-1",
            request_data,
            scope_type="workspace",
            workspace_id="ws-1",
        )

        _assert_request_call(
            mocked.await_args,
            "PATCH",
            "/api/v1/chat/conversations/conv-1",
            {
                "json_data": {
                    "version": 7,
                    "state": "resolved",
                    "keywords": ["Alpha", "beta"],
                },
                "params": {"scope_type": "workspace", "workspace_id": "ws-1"},
            },
        )

    async def test_get_chat_conversation_tree_forwards_pagination_and_scope(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"conversation": {}, "root_threads": [], "pagination": {}})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_chat_conversation_tree(
            "conv-1",
            limit=25,
            offset=5,
            max_depth=3,
            scope_type="workspace",
            workspace_id="ws-1",
        )

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/chat/conversations/conv-1/tree",
            {
                "params": {
                    "limit": 25,
                    "offset": 5,
                    "max_depth": 3,
                    "scope_type": "workspace",
                    "workspace_id": "ws-1",
                }
            },
        )

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs",
        [
            ("list_chat_conversations", (), {"scope_type": "workspace"}),
            ("get_chat_conversation", ("conv-1",), {"scope_type": "workspace"}),
            ("update_chat_conversation", ("conv-1", ConversationUpdateRequest(version=1)), {"scope_type": "workspace"}),
            ("get_chat_conversation_tree", ("conv-1",), {"scope_type": "workspace"}),
        ],
    )
    async def test_workspace_scope_requires_workspace_id_before_request_dispatch(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        with pytest.raises(ValueError, match="workspace_id is required"):
            await getattr(client, method_name)(*call_args, **call_kwargs)

        assert mocked.await_count == 0

    async def test_list_chat_conversations_rejects_invalid_state_before_request_dispatch(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        with pytest.raises(ValueError, match="Allowed: in-progress, resolved, backlog, non-viable"):
            await client.list_chat_conversations(state="maybe")

        assert mocked.await_count == 0

    async def test_chat_conversation_adjunct_methods_route_server_paths(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                {
                    "share_id": "share-1",
                    "permission": "view",
                    "created_at": "2026-04-23T18:00:00Z",
                    "expires_at": "2026-04-23T19:00:00Z",
                    "token": "token-1",
                    "share_path": "/knowledge/shared/token-1",
                },
                {
                    "conversation_id": "conv-1",
                    "links": [
                        {
                            "id": "share-1",
                            "permission": "view",
                            "created_at": "2026-04-23T18:00:00Z",
                            "expires_at": "2026-04-23T19:00:00Z",
                            "share_path": "/knowledge/shared/token-1",
                            "token": "token-1",
                        }
                    ],
                },
                {"success": True, "share_id": "share-1"},
                {
                    "conversation_id": "conv-1",
                    "title": "Shared",
                    "source": None,
                    "permission": "view",
                    "shared_by_user_id": "7",
                    "expires_at": "2026-04-23T19:00:00Z",
                    "messages": [{"id": "msg-1", "sender": "user"}],
                },
                {"success": True, "message_id": "msg-1"},
                {"rag_context": {"search_query": "alpha"}},
                [{"id": "msg-1", "conversation_id": "conv-1", "sender": "user", "rag_context": {}}],
                {"conversation_id": "conv-1", "citations": [{"title": "Source"}], "total_count": 1},
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        await client.create_chat_conversation_share_link(
            "conv-1",
            ConversationShareLinkCreateRequest(ttl_seconds=600, label="Review"),
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.list_chat_conversation_share_links("conv-1", scope_type="workspace", workspace_id="ws-1")
        await client.revoke_chat_conversation_share_link("conv-1", "share-1", scope_type="workspace", workspace_id="ws-1")
        await client.resolve_shared_chat_conversation("token-1", limit=50)
        await client.persist_chat_message_rag_context(
            "msg-1",
            RagContextPersistRequest(
                message_id="msg-1",
                rag_context=RagContext(search_query="alpha"),
            ),
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.get_chat_message_rag_context("msg-1", scope_type="workspace", workspace_id="ws-1")
        await client.get_chat_conversation_messages_with_context(
            "conv-1",
            limit=25,
            offset=5,
            include_rag_context=False,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.get_chat_conversation_citations("conv-1")

        assert [call.args for call in mocked.await_args_list] == [
            ("POST", "/api/v1/chat/conversations/conv-1/share-links"),
            ("GET", "/api/v1/chat/conversations/conv-1/share-links"),
            ("DELETE", "/api/v1/chat/conversations/conv-1/share-links/share-1"),
            ("GET", "/api/v1/chat/shared/conversations/token-1"),
            ("POST", "/api/v1/chat/messages/msg-1/rag-context"),
            ("GET", "/api/v1/chat/messages/msg-1/rag-context"),
            ("GET", "/api/v1/chat/conversations/conv-1/messages-with-context"),
            ("GET", "/api/v1/chat/conversations/conv-1/citations"),
        ]
        assert mocked.await_args_list[0].kwargs["json_data"] == {
            "permission": "view",
            "ttl_seconds": 600,
            "label": "Review",
        }
        assert mocked.await_args_list[0].kwargs["params"] == {"scope_type": "workspace", "workspace_id": "ws-1"}
        assert mocked.await_args_list[3].kwargs["params"] == {"limit": 50}
        assert mocked.await_args_list[4].kwargs["json_data"] == {
            "message_id": "msg-1",
            "rag_context": {
                "search_query": "alpha",
                "search_mode": "hybrid",
                "retrieved_documents": [],
            },
        }
        assert mocked.await_args_list[6].kwargs["params"] == {
            "limit": 25,
            "offset": 5,
            "include_rag_context": False,
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }

    async def test_chat_server_adjunct_methods_route_current_contract(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                {"commands": [{"name": "search", "description": "Search", "args": ["query"]}]},
                {
                    "ok": False,
                    "schema_version": 1,
                    "errors": [{"code": "missing", "field": "entries", "message": "Missing entries"}],
                    "warnings": [],
                    "entry_stats": {"entries": 0},
                    "suggested_fixes": ["Add entries"],
                    "partial": False,
                },
                {"enabled": True, "queued": 2},
                {"enabled": True, "limit": 5, "activity": [{"run_id": "run-1"}]},
                {
                    "note_id": "note-1",
                    "flashcard_id": "card-1",
                    "conversation_id": "conv-1",
                    "message_id": "msg-1",
                    "export_status": "not_requested",
                },
                {
                    "buckets": [
                        {
                            "bucket_start": "2026-04-23T00:00:00Z",
                            "topic_label": "sync",
                            "state": "in-progress",
                            "count": 3,
                        }
                    ],
                    "pagination": {"limit": 10, "offset": 0, "total": 1, "has_more": False},
                    "bucket_granularity": "day",
                },
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        commands = await client.list_chat_commands()
        validation = await client.validate_chat_dictionary(
            ValidateDictionaryRequest(data={"entries": []}, schema_version=1, strict=True)
        )
        queue_status = await client.get_chat_queue_status()
        queue_activity = await client.get_chat_queue_activity(limit=5)
        knowledge = await client.save_chat_knowledge(
            KnowledgeSaveRequest(
                conversation_id="conv-1",
                message_id="msg-1",
                snippet="Durable note",
                tags=["sync"],
                make_flashcard=True,
            )
        )
        analytics = await client.get_chat_analytics(
            start_date="2026-04-01T00:00:00Z",
            end_date="2026-04-23T00:00:00Z",
            bucket_granularity="day",
            limit=10,
            offset=0,
        )

        assert isinstance(commands, ChatCommandsListResponse)
        assert isinstance(validation, ValidateDictionaryResponse)
        assert isinstance(queue_status, ChatQueueStatusResponse)
        assert isinstance(queue_activity, ChatQueueActivityResponse)
        assert isinstance(knowledge, KnowledgeSaveResponse)
        assert isinstance(analytics, ChatAnalyticsResponse)
        assert [call.args[:2] for call in mocked.await_args_list] == [
            ("GET", "/api/v1/chat/commands"),
            ("POST", "/api/v1/chat/dictionaries/validate"),
            ("GET", "/api/v1/chat/queue/status"),
            ("GET", "/api/v1/chat/queue/activity"),
            ("POST", "/api/v1/chat/knowledge/save"),
            ("GET", "/api/v1/chat/analytics"),
        ]
        assert mocked.await_args_list[1].kwargs["json_data"] == {
            "data": {"entries": []},
            "schema_version": 1,
            "strict": True,
        }
        assert mocked.await_args_list[3].kwargs["params"] == {"limit": 5}
        assert mocked.await_args_list[4].kwargs["json_data"]["tags"] == ["sync"]
        assert mocked.await_args_list[5].kwargs["params"] == {
            "start_date": "2026-04-01T00:00:00Z",
            "end_date": "2026-04-23T00:00:00Z",
            "bucket_granularity": "day",
            "limit": 10,
            "offset": 0,
        }
