from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationBackend,
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService


class FakePolicyEnforcer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeLocalConversationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_conversations", (), kwargs))
        return {
            "items": [
                {
                    "id": "local-conv-1",
                    "scope_type": kwargs.get("scope_type") or "global",
                    "workspace_id": kwargs.get("workspace_id"),
                    "runtime_backend": "local",
                    "discovery_owner": "general_chat",
                }
            ],
            "pagination": {"limit": kwargs["limit"], "offset": kwargs["offset"], "total": 1, "has_more": False},
        }

    def get_conversation_metadata(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_conversation_metadata", (conversation_id,), {}))
        return {
            "id": conversation_id,
            "scope_type": "workspace",
            "workspace_id": "ws-1",
            "runtime_backend": "local",
            "discovery_owner": "general_chat",
        }

    def update_conversation_metadata(
        self,
        conversation_id: str,
        update_data: dict[str, Any],
        expected_version: int,
    ) -> bool:
        self.calls.append(("update_conversation_metadata", (conversation_id, update_data, expected_version), {}))
        return True

    def delete_conversation(self, conversation_id: str, expected_version: int) -> bool:
        self.calls.append(("delete_conversation", (conversation_id, expected_version), {}))
        return True

    def get_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation_tree", (conversation_id,), kwargs))
        return {
            "conversation": {"id": conversation_id, "scope_type": "global"},
            "root_threads": [],
            "pagination": {
                "limit": kwargs["root_limit"],
                "offset": kwargs["root_offset"],
                "total_root_threads": 0,
                "has_more": False,
            },
            "depth_cap": kwargs["depth_cap"],
        }


class FakeServerConversationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_conversations", (), kwargs))
        return {
            "items": [{"id": "server-conv-1", "scope_type": "workspace", "workspace_id": "ws-1"}],
            "pagination": {"limit": kwargs["limit"], "offset": kwargs["offset"], "total": 1, "has_more": False},
        }

    async def get_conversation_metadata(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation_metadata", (conversation_id,), kwargs))
        return {"id": conversation_id, "scope_type": kwargs.get("scope_type") or "global"}

    async def update_conversation_metadata(
        self,
        conversation_id: str,
        update_data: dict[str, Any],
        expected_version: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(("update_conversation_metadata", (conversation_id, update_data, expected_version), kwargs))
        return {"id": conversation_id, "version": expected_version + 1}

    async def get_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_conversation_tree", (conversation_id,), kwargs))
        return {"conversation": {"id": conversation_id}, "root_threads": [], "pagination": {}, "depth_cap": kwargs["depth_cap"]}

    async def create_share_link(self, conversation_id: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_share_link", (conversation_id, payload), kwargs))
        return {"share_id": "share-1", "conversation_id": conversation_id}

    async def list_share_links(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_share_links", (conversation_id,), kwargs))
        return {"conversation_id": conversation_id, "links": []}

    async def revoke_share_link(self, conversation_id: str, share_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("revoke_share_link", (conversation_id, share_id), kwargs))
        return {"success": True, "share_id": share_id}

    async def resolve_share_token(self, share_token: str, *, limit: int = 200) -> dict[str, Any]:
        self.calls.append(("resolve_share_token", (share_token,), {"limit": limit}))
        return {"conversation_id": "conv-1", "permission": "view", "messages": []}

    async def persist_message_rag_context(self, message_id: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("persist_message_rag_context", (message_id, payload), kwargs))
        return {"success": True, "message_id": message_id}

    async def get_message_rag_context(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_message_rag_context", (message_id,), kwargs))
        return {"rag_context": {"search_query": "alpha"}}

    async def get_messages_with_context(self, conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("get_messages_with_context", (conversation_id,), kwargs))
        return [{"id": "msg-1", "conversation_id": conversation_id}]

    async def get_conversation_citations(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_conversation_citations", (conversation_id,), {}))
        return {"conversation_id": conversation_id, "citations": [], "total_count": 0}


@pytest.mark.asyncio
async def test_scope_service_defaults_to_local_list_and_enforces_policy() -> None:
    local = FakeLocalConversationService()
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    result = await service.list_conversations(query="alpha", limit=10, offset=5)

    assert result["items"][0]["id"] == "local-conv-1"
    assert local.calls == [
        (
            "list_conversations",
            (),
            {
                "query": "alpha",
                "limit": 10,
                "offset": 5,
                "scope_type": None,
                "workspace_id": None,
                "include_deleted": False,
                "deleted_only": False,
                "state": None,
                "topic_label": None,
                "character_id": None,
            },
        )
    ]
    assert policy.calls == ["chat.list.local"]


@pytest.mark.asyncio
async def test_scope_service_routes_server_list_without_local_fallback() -> None:
    local = FakeLocalConversationService()
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    result = await service.list_conversations(
        mode=ChatConversationBackend.SERVER,
        scope_type="workspace",
        workspace_id="ws-1",
        state="resolved",
        limit=25,
    )

    assert result["items"][0]["id"] == "server-conv-1"
    assert local.calls == []
    assert server.calls == [
        (
            "list_conversations",
            (),
            {
                "query": None,
                "limit": 25,
                "offset": 0,
                "scope_type": "workspace",
                "workspace_id": "ws-1",
                "include_deleted": False,
                "deleted_only": False,
                "state": "resolved",
                "topic_label": None,
                "character_id": None,
            },
        )
    ]
    assert policy.calls == ["chat.list.server"]


@pytest.mark.asyncio
async def test_scope_service_filters_local_detail_by_requested_scope() -> None:
    local = FakeLocalConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=None, policy_enforcer=policy)

    hidden = await service.get_conversation("conv-workspace", scope_type="global")
    visible = await service.get_conversation("conv-workspace", scope_type="workspace", workspace_id="ws-1")

    assert hidden is None
    assert visible["workspace_id"] == "ws-1"
    assert policy.calls == ["chat.detail.local", "chat.detail.local"]


@pytest.mark.asyncio
async def test_scope_service_updates_server_conversation_with_policy_and_scope() -> None:
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    result = await service.update_conversation(
        "conv-1",
        {"state": "resolved"},
        expected_version=7,
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
    )

    assert result == {"id": "conv-1", "version": 8}
    assert server.calls == [
        (
            "update_conversation_metadata",
            ("conv-1", {"state": "resolved"}, 7),
            {"scope_type": "workspace", "workspace_id": "ws-1"},
        )
    ]
    assert policy.calls == ["chat.update.server"]


@pytest.mark.asyncio
async def test_scope_service_deletes_local_conversation_with_policy() -> None:
    local = FakeLocalConversationService()
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    result = await service.delete_conversation("conv-1", expected_version=7)

    assert result is True
    assert local.calls == [("delete_conversation", ("conv-1", 7), {})]
    assert server.calls == []
    assert policy.calls == ["chat.delete.local"]


@pytest.mark.asyncio
async def test_scope_service_rejects_server_delete_until_contract_exists() -> None:
    local = FakeLocalConversationService()
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    with pytest.raises(ValueError, match="Server chat conversation delete is not available"):
        await service.delete_conversation("conv-1", expected_version=7, mode="server")

    assert local.calls == []
    assert server.calls == []
    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_conversation_adjuncts_with_policy() -> None:
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    await service.create_share_link(
        "conv-1",
        {"ttl_seconds": 600},
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
    )
    await service.list_share_links("conv-1", mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.revoke_share_link("conv-1", "share-1", mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.resolve_share_token("token-1", mode="server", limit=50)
    await service.persist_message_rag_context(
        "msg-1",
        {"message_id": "msg-1", "rag_context": {"search_query": "alpha"}},
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
    )
    await service.get_message_rag_context("msg-1", mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.get_messages_with_context(
        "conv-1",
        mode="server",
        limit=25,
        offset=5,
        include_rag_context=False,
        scope_type="workspace",
        workspace_id="ws-1",
    )
    await service.get_conversation_citations("conv-1", mode="server")

    assert server.calls == [
        (
            "create_share_link",
            ("conv-1", {"ttl_seconds": 600}),
            {"scope_type": "workspace", "workspace_id": "ws-1"},
        ),
        ("list_share_links", ("conv-1",), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        ("revoke_share_link", ("conv-1", "share-1"), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        ("resolve_share_token", ("token-1",), {"limit": 50}),
        (
            "persist_message_rag_context",
            ("msg-1", {"message_id": "msg-1", "rag_context": {"search_query": "alpha"}}),
            {"scope_type": "workspace", "workspace_id": "ws-1"},
        ),
        ("get_message_rag_context", ("msg-1",), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        (
            "get_messages_with_context",
            ("conv-1",),
            {
                "limit": 25,
                "offset": 5,
                "include_rag_context": False,
                "scope_type": "workspace",
                "workspace_id": "ws-1",
            },
        ),
        ("get_conversation_citations", ("conv-1",), {}),
    ]
    assert policy.calls == [
        "chat.update.server",
        "chat.detail.server",
        "chat.update.server",
        "chat.detail.server",
        "chat.update.server",
        "chat.detail.server",
        "chat.detail.server",
        "chat.detail.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_local_conversation_adjuncts_before_policy() -> None:
    local = FakeLocalConversationService()
    server = FakeServerConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    with pytest.raises(ValueError, match="requires server mode"):
        await service.create_share_link("conv-1", {"ttl_seconds": 600}, mode="local")

    assert local.calls == []
    assert server.calls == []
    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_tree_uses_detail_policy_and_pagination_mapping() -> None:
    local = FakeLocalConversationService()
    policy = FakePolicyEnforcer()
    service = ChatConversationScopeService(local_service=local, server_service=None, policy_enforcer=policy)

    tree = await service.get_conversation_tree("conv-1", root_limit=3, root_offset=6, max_depth=2)

    assert tree["pagination"]["limit"] == 3
    assert tree["pagination"]["offset"] == 6
    assert tree["depth_cap"] == 2
    assert local.calls == [
        (
            "get_conversation_tree",
            ("conv-1",),
            {"root_limit": 3, "root_offset": 6, "depth_cap": 2},
        )
    ]
    assert policy.calls == ["chat.detail.local"]


@pytest.mark.asyncio
async def test_server_service_adapts_client_methods_and_update_request() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        async def list_chat_conversations(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("list_chat_conversations", (), kwargs))
            return {"items": [], "pagination": {}}

        async def get_chat_conversation(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("get_chat_conversation", (conversation_id,), kwargs))
            return {"id": conversation_id}

        async def update_chat_conversation(self, conversation_id: str, request_data: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("update_chat_conversation", (conversation_id, request_data), kwargs))
            return {"id": conversation_id, "state": request_data.state, "version": request_data.version + 1}

        async def get_chat_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("get_chat_conversation_tree", (conversation_id,), kwargs))
            return {"conversation": {"id": conversation_id}, "root_threads": [], "pagination": {}}

    client = FakeClient()
    service = ServerChatConversationService(client=client)

    await service.list_conversations(scope_type="workspace", workspace_id="ws-1", limit=10)
    await service.get_conversation_metadata("conv-1", scope_type="workspace", workspace_id="ws-1")
    update = await service.update_conversation_metadata(
        "conv-1",
        {"state": "resolved", "keywords": ["alpha"]},
        4,
        scope_type="workspace",
        workspace_id="ws-1",
    )
    await service.get_conversation_tree("conv-1", root_limit=2, root_offset=1, depth_cap=3)

    assert update == {"id": "conv-1", "state": "resolved", "version": 5}
    assert client.calls[0] == (
        "list_chat_conversations",
        (),
        {"scope_type": "workspace", "workspace_id": "ws-1", "limit": 10},
    )
    assert client.calls[1] == (
        "get_chat_conversation",
        ("conv-1",),
        {"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    method, args, kwargs = client.calls[2]
    assert method == "update_chat_conversation"
    assert args[0] == "conv-1"
    assert args[1].model_dump(exclude_none=True, mode="json") == {
        "version": 4,
        "state": "resolved",
        "keywords": ["alpha"],
    }
    assert kwargs == {"scope_type": "workspace", "workspace_id": "ws-1"}
    assert client.calls[3] == (
        "get_chat_conversation_tree",
        ("conv-1",),
        {"limit": 2, "offset": 1, "max_depth": 3},
    )
