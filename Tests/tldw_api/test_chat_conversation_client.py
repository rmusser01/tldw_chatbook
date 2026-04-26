"""
Tests for chat conversation endpoint wiring on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.chat_loop_schemas import ChatLoopApprovalDecisionRequest, ChatLoopStartRequest
from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.chat_conversation_schemas import ConversationUpdateRequest


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

    async def test_get_chat_conversation_messages_with_context_forwards_scope_and_context_flag(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value=[])
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_chat_conversation_messages_with_context(
            "conv-1",
            limit=20,
            offset=10,
            include_rag_context=False,
            scope_type="workspace",
            workspace_id="ws-1",
        )

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/chat/conversations/conv-1/messages-with-context",
            {
                "params": {
                    "limit": 20,
                    "offset": 10,
                    "include_rag_context": "false",
                    "scope_type": "workspace",
                    "workspace_id": "ws-1",
                }
            },
        )

    async def test_get_chat_conversation_citations_forwards_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"conversation_id": "conv-1", "citations": [], "total_count": 0})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_chat_conversation_citations("conv-1")

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/chat/conversations/conv-1/citations",
            {},
        )

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs",
        [
            ("list_chat_conversations", (), {"scope_type": "workspace"}),
            ("get_chat_conversation", ("conv-1",), {"scope_type": "workspace"}),
            ("update_chat_conversation", ("conv-1", ConversationUpdateRequest(version=1)), {"scope_type": "workspace"}),
            ("get_chat_conversation_tree", ("conv-1",), {"scope_type": "workspace"}),
            ("get_chat_conversation_messages_with_context", ("conv-1",), {"scope_type": "workspace"}),
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

    async def test_chat_loop_client_routes_run_events_approval_and_cancel(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                {"run_id": "run_123"},
                {"run_id": "run_123", "events": []},
                {"ok": True},
                {"ok": True},
                {"ok": True},
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        started = await client.start_chat_loop_run(ChatLoopStartRequest(messages=[{"role": "user", "content": "Hi"}]))
        events = await client.list_chat_loop_events("run_123", after_seq=4)
        approved = await client.approve_chat_loop_call(
            "run_123",
            ChatLoopApprovalDecisionRequest(approval_id="approval-1", decision="approve"),
        )
        rejected = await client.reject_chat_loop_call(
            "run_123",
            ChatLoopApprovalDecisionRequest(approval_id="approval-2", decision="reject"),
        )
        cancelled = await client.cancel_chat_loop_run("run_123")

        _assert_request_call(
            mocked.await_args_list[0],
            "POST",
            "/api/v1/chat/loop/start",
            {"json_data": {"messages": [{"role": "user", "content": "Hi"}]}},
        )
        _assert_request_call(
            mocked.await_args_list[1],
            "GET",
            "/api/v1/chat/loop/run_123/events",
            {"params": {"after_seq": 4}},
        )
        _assert_request_call(
            mocked.await_args_list[2],
            "POST",
            "/api/v1/chat/loop/run_123/approve",
            {"json_data": {"approval_id": "approval-1", "decision": "approve"}},
        )
        _assert_request_call(
            mocked.await_args_list[3],
            "POST",
            "/api/v1/chat/loop/run_123/reject",
            {"json_data": {"approval_id": "approval-2", "decision": "reject"}},
        )
        _assert_request_call(mocked.await_args_list[4], "POST", "/api/v1/chat/loop/run_123/cancel", {})
        assert started.run_id == "run_123"
        assert events.events == []
        assert approved.ok is True
        assert rejected.ok is True
        assert cancelled.ok is True
