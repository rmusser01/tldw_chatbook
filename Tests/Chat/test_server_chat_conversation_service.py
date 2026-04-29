from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api.chat_loop_schemas import ChatLoopActionResponse, ChatLoopEventsResponse, ChatLoopStartResponse


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

    async def list_chat_commands(self) -> Any:
        self.calls.append(("list_chat_commands", (), {}))
        return {"commands": [{"name": "/search"}]}

    async def save_chat_knowledge(self, request_data: Any) -> Any:
        self.calls.append(("save_chat_knowledge", (request_data,), {}))
        return {"note_id": 9, "conversation_id": "conv-1"}

    async def create_chat_conversation_share_link(self, conversation_id: str, request_data: Any, **kwargs: Any) -> Any:
        self.calls.append(("create_chat_conversation_share_link", (conversation_id, request_data), kwargs))
        return {"share_id": "share-1"}

    async def list_chat_conversation_share_links(self, conversation_id: str, **kwargs: Any) -> Any:
        self.calls.append(("list_chat_conversation_share_links", (conversation_id,), kwargs))
        return {"conversation_id": conversation_id, "links": []}

    async def revoke_chat_conversation_share_link(self, conversation_id: str, share_id: str, **kwargs: Any) -> Any:
        self.calls.append(("revoke_chat_conversation_share_link", (conversation_id, share_id), kwargs))
        return {"success": True, "share_id": share_id}

    async def resolve_chat_conversation_share_token(self, share_token: str, *, limit: int = 200) -> Any:
        self.calls.append(("resolve_chat_conversation_share_token", (share_token,), {"limit": limit}))
        return {"conversation_id": "conv-1", "messages": []}

    async def get_chat_analytics(self, **kwargs: Any) -> Any:
        self.calls.append(("get_chat_analytics", (), kwargs))
        return {"buckets": [], "pagination": {"total": 0}}

    async def start_chat_loop_run(self, request_data: Any) -> ChatLoopStartResponse:
        self.calls.append(("start_chat_loop_run", (request_data,), {}))
        return ChatLoopStartResponse(run_id="run_123")

    async def list_chat_loop_events(self, run_id: str, *, after_seq: int = 0) -> ChatLoopEventsResponse:
        self.calls.append(("list_chat_loop_events", (run_id,), {"after_seq": after_seq}))
        return ChatLoopEventsResponse(run_id=run_id, events=[])

    async def approve_chat_loop_call(self, run_id: str, request_data: Any) -> ChatLoopActionResponse:
        self.calls.append(("approve_chat_loop_call", (run_id, request_data), {}))
        return ChatLoopActionResponse(ok=True)

    async def reject_chat_loop_call(self, run_id: str, request_data: Any) -> ChatLoopActionResponse:
        self.calls.append(("reject_chat_loop_call", (run_id, request_data), {}))
        return ChatLoopActionResponse(ok=True)

    async def cancel_chat_loop_run(self, run_id: str) -> ChatLoopActionResponse:
        self.calls.append(("cancel_chat_loop_run", (run_id,), {}))
        return ChatLoopActionResponse(ok=True)


@dataclass
class RecordingPolicy:
    denied: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if action_id in self.denied:
            raise RuntimeError(f"denied:{action_id}")


class FakeCachingProvider:
    def __init__(self, client_factory):
        self.client_factory = client_factory
        self.client = None
        self.build_calls = 0
        self.constructed_clients = 0

    def build_client(self):
        self.build_calls += 1
        if self.client is None:
            self.client = self.client_factory()
            self.constructed_clients += 1
        return self.client


class ExplodingProvider:
    def __init__(self):
        self.calls = 0

    def build_client(self):
        self.calls += 1
        raise AssertionError("provider should not be used")


@pytest.mark.asyncio
async def test_server_chat_conversation_service_from_config_builds_and_reuses_client_lazily(monkeypatch):
    sentinel_client = FakeChatClient()
    build_client_calls: list[dict[str, Any]] = []

    def build_client(app_config):
        build_client_calls.append(app_config)
        return sentinel_client

    monkeypatch.setattr(
        "tldw_chatbook.Chat.server_chat_conversation_service.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerChatConversationService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert service.client is None
    assert service.client_provider is not None
    assert build_client_calls == []

    result = await service.list_conversations(query="billing")
    detail = await service.get_conversation("conv-1")

    assert result["items"][0]["id"] == "conv-1"
    assert detail["id"] == "conv-1"
    assert build_client_calls == [{"tldw_api": {"base_url": "https://example.com"}}]
    assert sentinel_client.calls == [
        ("list_chat_conversations", (), {"query": "billing"}),
        ("get_chat_conversation", ("conv-1",), {}),
    ]


@pytest.mark.asyncio
async def test_server_chat_conversation_service_reuses_provider_cached_client_across_operations():
    provider = FakeCachingProvider(FakeChatClient)
    service = ServerChatConversationService.from_server_context_provider(provider)

    await service.list_conversations(query="billing")
    await service.get_conversation("conv-1")

    assert provider.build_calls == 2
    assert provider.constructed_clients == 1
    assert provider.client.calls == [
        ("list_chat_conversations", (), {"query": "billing"}),
        ("get_chat_conversation", ("conv-1",), {}),
    ]


@pytest.mark.asyncio
async def test_server_chat_conversation_service_direct_client_takes_precedence_over_provider():
    client = FakeChatClient()
    provider = ExplodingProvider()
    service = ServerChatConversationService(client=client, client_provider=provider)

    await service.list_conversations()

    assert provider.calls == 0
    assert client.calls == [("list_chat_conversations", (), {})]


@pytest.mark.asyncio
async def test_server_chat_conversation_service_denied_policy_does_not_build_provider_client():
    policy = RecordingPolicy()
    policy.require_allowed = None
    policy.require_ui_action_allowed = lambda action_id: PolicyDecision(
        allowed=False,
        reason_code="wrong_source",
        user_message="Blocked.",
        effective_source="server",
        authority_owner="server",
    )
    provider = ExplodingProvider()
    service = ServerChatConversationService.from_server_context_provider(provider, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError):
        await service.list_conversations()

    assert provider.calls == 0


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


@pytest.mark.asyncio
async def test_server_chat_conversation_service_routes_chat_loop_with_policy_actions():
    client = FakeChatClient()
    policy = RecordingPolicy()
    service = ServerChatConversationService(client, policy_enforcer=policy)

    started = await service.start_loop(messages=[{"role": "user", "content": "Hi"}])
    events = await service.list_loop_events("run_123", after_seq=3)
    approved = await service.approve_loop_call("run_123", approval_id="approval-1")
    rejected = await service.reject_loop_call("run_123", approval_id="approval-2")
    cancelled = await service.cancel_loop("run_123")

    assert started.run_id == "run_123"
    assert events.run_id == "run_123"
    assert approved.ok is True
    assert rejected.ok is True
    assert cancelled.ok is True
    assert policy.calls == [
        "chat.loop.launch.server",
        "chat.loop.observe.server",
        "chat.loop.approve.server",
        "chat.loop.approve.server",
        "chat.loop.cancel.server",
    ]
    assert client.calls[-5][0] == "start_chat_loop_run"
    assert client.calls[-4] == ("list_chat_loop_events", ("run_123",), {"after_seq": 3})
    assert client.calls[-3][0] == "approve_chat_loop_call"
    assert client.calls[-2][0] == "reject_chat_loop_call"
    assert client.calls[-1] == ("cancel_chat_loop_run", ("run_123",), {})


@pytest.mark.asyncio
async def test_server_chat_conversation_service_routes_chat_adjunct_controls_with_policy():
    client = FakeChatClient()
    policy = RecordingPolicy()
    service = ServerChatConversationService(client, policy_enforcer=policy)

    await service.list_commands()
    await service.save_knowledge(conversation_id="conv-1", snippet="Important", tags=["alpha"])
    await service.create_share_link("conv-1", {"label": "Reviewer"}, scope_type="workspace", workspace_id="ws-1")
    await service.list_share_links("conv-1", scope_type="workspace", workspace_id="ws-1")
    await service.revoke_share_link("conv-1", "share-1", scope_type="workspace", workspace_id="ws-1")
    await service.resolve_share_token("token", limit=25)
    await service.get_analytics(start_date="2026-04-01T00:00:00Z", end_date="2026-04-26T00:00:00Z")

    assert policy.calls[-7:] == [
        "chat.commands.list.server",
        "chat.knowledge.create.server",
        "chat.share_links.create.server",
        "chat.share_links.list.server",
        "chat.share_links.revoke.server",
        "chat.share_links.detail.server",
        "chat.analytics.observe.server",
    ]
    assert client.calls[-7][0] == "list_chat_commands"
    knowledge_request = client.calls[-6][1][0]
    assert knowledge_request.conversation_id == "conv-1"
    assert knowledge_request.snippet == "Important"
    assert client.calls[-5] == (
        "create_chat_conversation_share_link",
        ("conv-1", service._share_link_create_request({"label": "Reviewer"})),
        {"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[-4] == (
        "list_chat_conversation_share_links",
        ("conv-1",),
        {"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[-3] == (
        "revoke_chat_conversation_share_link",
        ("conv-1", "share-1"),
        {"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert client.calls[-2] == ("resolve_chat_conversation_share_token", ("token",), {"limit": 25})
    assert client.calls[-1] == (
        "get_chat_analytics",
        (),
        {"start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-26T00:00:00Z"},
    )
