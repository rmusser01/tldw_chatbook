from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService


@dataclass
class FakeConversationService:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async_mode: bool = False

    def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_conversations", (), kwargs))
        return {"items": [{"id": "local-conv"}], "pagination": {"total": 1}}

    def get_conversation_metadata(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_conversation_metadata", (conversation_id,), {}))
        return {"id": conversation_id, "version": 2}

    def update_conversation_metadata(self, conversation_id: str, update_data: dict[str, Any], expected_version: int) -> bool:
        self.calls.append(("update_conversation_metadata", (conversation_id, update_data, expected_version), {}))
        return True

    def replace_conversation_keywords(self, conversation_id: str, keywords: list[str]) -> list[str]:
        self.calls.append(("replace_conversation_keywords", (conversation_id, keywords), {}))
        return keywords

    def get_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation_tree", (conversation_id,), kwargs))
        return {"conversation": {"id": conversation_id}, "root_threads": []}

    def create_conversation(self, **kwargs: Any) -> str:
        self.calls.append(("create_conversation", (), kwargs))
        return "local-created"

    def delete_conversation(self, conversation_id: str, expected_version: int) -> bool:
        self.calls.append(("delete_conversation", (conversation_id, expected_version), {}))
        return True


@dataclass
class FakeServerConversationService:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_conversations", (), kwargs))
        return {"items": [{"id": "server-conv"}], "pagination": {"total": 1}}

    async def get_conversation(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation", (conversation_id,), kwargs))
        return {"id": conversation_id, "version": 7}

    async def update_conversation(self, conversation_id: str, update_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update_conversation", (conversation_id, update_data), kwargs))
        return {"id": conversation_id, "version": 8}

    async def get_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation_tree", (conversation_id,), kwargs))
        return {"conversation": {"id": conversation_id}, "root_threads": []}

    async def get_messages_with_context(self, conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("get_messages_with_context", (conversation_id,), kwargs))
        return [{"id": "msg-1", "conversation_id": conversation_id}]

    async def get_citations(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_citations", (conversation_id,), {}))
        return {"conversation_id": conversation_id, "citations": [], "total_count": 0}

    async def create_conversation(self, **kwargs: Any) -> str:
        self.calls.append(("create_conversation", (), kwargs))
        return "server-created"

    async def delete_conversation(self, conversation_id: str, expected_version: int) -> bool:
        self.calls.append(("delete_conversation", (conversation_id, expected_version), {}))
        return True


@dataclass
class RecordingPolicy:
    calls: list[str] = field(default_factory=list)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


@pytest.mark.asyncio
async def test_scope_service_routes_local_and_server_conversation_reads_with_policy():
    local = FakeConversationService()
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    local_result = await service.list_conversations(mode="local", query="alpha")
    server_result = await service.list_conversations(mode="server", query="beta", scope_type="workspace", workspace_id="ws-1")
    tree = await service.get_conversation_tree("conv-1", mode="server", limit=10, offset=2, max_depth=3)

    assert local_result["items"][0]["id"] == "local-conv"
    assert server_result["items"][0]["id"] == "server-conv"
    assert tree["conversation"]["id"] == "conv-1"
    assert policy.calls == ["chat.list.local", "chat.list.server", "chat.detail.server"]
    assert local.calls == [("list_conversations", (), {"query": "alpha"})]
    assert server.calls == [
        ("list_conversations", (), {"query": "beta", "scope_type": "workspace", "workspace_id": "ws-1"}),
        ("get_conversation_tree", ("conv-1",), {"limit": 10, "offset": 2, "max_depth": 3}),
    ]


@pytest.mark.asyncio
async def test_scope_service_maps_server_style_tree_pagination_to_local_tree_arguments():
    local = FakeConversationService()
    service = ChatConversationScopeService(local_service=local, server_service=FakeServerConversationService())

    await service.get_conversation_tree("conv-1", mode="local", limit=25, offset=10, max_depth=6)

    assert local.calls == [
        ("get_conversation_tree", ("conv-1",), {"root_limit": 25, "root_offset": 10, "depth_cap": 6}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_context_and_citations_with_local_unsupported_boundaries():
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=server,
        policy_enforcer=policy,
    )

    messages = await service.get_messages_with_context("conv-1", mode="server", limit=10)
    citations = await service.get_citations("conv-1", mode="server")

    assert messages == [{"id": "msg-1", "conversation_id": "conv-1"}]
    assert citations["conversation_id"] == "conv-1"
    assert policy.calls == ["chat.detail.server", "chat.detail.server"]
    assert server.calls == [
        ("get_messages_with_context", ("conv-1",), {"limit": 10}),
        ("get_citations", ("conv-1",), {}),
    ]

    with pytest.raises(NotImplementedError, match="Local RAG-context conversation adjuncts"):
        await service.get_messages_with_context("conv-1", mode="local")
    with pytest.raises(NotImplementedError, match="Local RAG-context conversation adjuncts"):
        await service.get_citations("conv-1", mode="local")


@pytest.mark.asyncio
async def test_scope_service_local_update_preserves_keyword_persistence_separately():
    local = FakeConversationService()
    service = ChatConversationScopeService(local_service=local, server_service=FakeServerConversationService())

    result = await service.update_conversation(
        "conv-1",
        {"version": 3, "state": "resolved", "keywords": ["alpha", "beta"]},
        mode="local",
    )

    assert result is True
    assert local.calls == [
        ("update_conversation_metadata", ("conv-1", {"state": "resolved"}, 3), {}),
        ("replace_conversation_keywords", ("conv-1", ["alpha", "beta"]), {}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_create_delete_and_requires_local_or_server_mode():
    local = FakeConversationService()
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    created = await service.create_conversation(mode="local", title="Local draft")
    deleted = await service.delete_conversation("conv-1", expected_version=3, mode="local")

    assert created == "local-created"
    assert deleted is True
    assert policy.calls == ["chat.create.local", "chat.delete.local"]

    with pytest.raises(ValueError, match="mode must be 'local' or 'server'"):
        await service.list_conversations(mode="mixed")
