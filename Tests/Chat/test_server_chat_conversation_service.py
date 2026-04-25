from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService


@dataclass
class FakeChatClient:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def list_chat_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_chat_conversations", (), kwargs))
        return {"items": [{"id": "conv-1"}], "pagination": {"total": 1, "limit": 50, "offset": 0, "has_more": False}}

    async def get_chat_conversation(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_chat_conversation", (conversation_id,), kwargs))
        return {"id": conversation_id, "version": 3}

    async def update_chat_conversation(self, conversation_id: str, request_data: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update_chat_conversation", (conversation_id, request_data), kwargs))
        return {"id": conversation_id, "version": 4}

    async def get_chat_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_chat_conversation_tree", (conversation_id,), kwargs))
        return {"conversation": {"id": conversation_id}, "root_threads": [], "pagination": {}, "depth_cap": 4}

    async def get_chat_conversation_messages_with_context(self, conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("get_chat_conversation_messages_with_context", (conversation_id,), kwargs))
        return [{"id": "msg-1", "conversation_id": conversation_id, "rag_context": {"search_query": "alpha"}}]

    async def get_chat_conversation_citations(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_chat_conversation_citations", (conversation_id,), {}))
        return {"conversation_id": conversation_id, "citations": [{"id": "doc-1"}], "total_count": 1}


@dataclass
class RecordingPolicy:
    denied: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if action_id in self.denied:
            raise RuntimeError(f"denied:{action_id}")


@pytest.mark.asyncio
async def test_server_chat_conversation_service_enforces_policy_and_routes_client_calls():
    client = FakeChatClient()
    policy = RecordingPolicy()
    service = ServerChatConversationService(client, policy_enforcer=policy)

    await service.list_conversations(query="billing", scope_type="workspace", workspace_id="ws-1")
    await service.get_conversation("conv-1", scope_type="workspace", workspace_id="ws-1")
    await service.update_conversation("conv-1", {"version": 3, "state": "resolved"}, scope_type="workspace", workspace_id="ws-1")
    await service.get_conversation_tree("conv-1", limit=10, offset=5, max_depth=3, scope_type="workspace", workspace_id="ws-1")
    await service.get_messages_with_context("conv-1", limit=5, include_rag_context=False, scope_type="workspace", workspace_id="ws-1")
    await service.get_citations("conv-1")

    assert policy.calls == [
        "chat.list.server",
        "chat.detail.server",
        "chat.update.server",
        "chat.detail.server",
        "chat.detail.server",
        "chat.detail.server",
    ]
    assert client.calls[0] == (
        "list_chat_conversations",
        (),
        {"query": "billing", "scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[1] == (
        "get_chat_conversation",
        ("conv-1",),
        {"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    update_request = client.calls[2][1][1]
    assert update_request.version == 3
    assert update_request.state == "resolved"
    assert client.calls[3] == (
        "get_chat_conversation_tree",
        ("conv-1",),
        {"limit": 10, "offset": 5, "max_depth": 3, "scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[4] == (
        "get_chat_conversation_messages_with_context",
        ("conv-1",),
        {"limit": 5, "include_rag_context": False, "scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[5] == (
        "get_chat_conversation_citations",
        ("conv-1",),
        {},
    )


@pytest.mark.asyncio
async def test_server_chat_conversation_service_fails_closed_before_client_call_when_policy_denies():
    client = FakeChatClient()
    policy = RecordingPolicy(denied={"chat.update.server"})
    service = ServerChatConversationService(client, policy_enforcer=policy)

    with pytest.raises(RuntimeError, match="denied:chat.update.server"):
        await service.update_conversation("conv-1", {"version": 3, "state": "resolved"})

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_chat_conversation_create_and_delete_are_explicit_unsupported_boundaries():
    policy = RecordingPolicy()
    service = ServerChatConversationService(FakeChatClient(), policy_enforcer=policy)

    with pytest.raises(NotImplementedError, match="does not expose first-class conversation create"):
        await service.create_conversation(title="Remote draft")
    with pytest.raises(NotImplementedError, match="does not expose conversation delete"):
        await service.delete_conversation("conv-1", expected_version=1)

    assert policy.calls == ["chat.create.server", "chat.delete.server"]
