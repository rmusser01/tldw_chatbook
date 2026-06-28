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

    def get_messages_with_context(self, conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("get_messages_with_context", (conversation_id,), kwargs))
        return [{"id": "local-msg-1", "conversation_id": conversation_id, "rag_context": {"query": "local"}}]

    def get_citations(self, conversation_id: str) -> dict[str, Any]:
        self.calls.append(("get_citations", (conversation_id,), {}))
        return {"conversation_id": conversation_id, "citations": [{"id": "local-cite-1"}], "total_count": 1}

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
        if kwargs.get("character_id") is None and not (kwargs.get("assistant_kind") and kwargs.get("assistant_id")):
            raise ValueError("Server conversation create requires character_id or assistant_kind + assistant_id.")
        return "server-created"

    async def delete_conversation(self, conversation_id: str, expected_version: int, **kwargs: Any) -> bool:
        self.calls.append(("delete_conversation", (conversation_id, expected_version), kwargs))
        return True

    async def start_loop(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("start_loop", (), kwargs))
        return {"run_id": "run_123"}

    async def list_loop_events(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_loop_events", (run_id,), kwargs))
        return {"run_id": run_id, "events": []}

    async def approve_loop_call(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("approve_loop_call", (run_id,), kwargs))
        return {"ok": True}

    async def reject_loop_call(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("reject_loop_call", (run_id,), kwargs))
        return {"ok": True}

    async def cancel_loop(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("cancel_loop", (run_id,), {}))
        return {"ok": True}

    async def list_commands(self) -> dict[str, Any]:
        self.calls.append(("list_commands", (), {}))
        return {"commands": [{"name": "/search"}]}

    async def save_knowledge(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("save_knowledge", (), kwargs))
        return {"note_id": 9, "conversation_id": kwargs["conversation_id"]}

    async def create_share_link(self, conversation_id: str, request_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_share_link", (conversation_id, request_data), kwargs))
        return {"share_id": "share-1"}

    async def list_share_links(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_share_links", (conversation_id,), kwargs))
        return {"conversation_id": conversation_id, "links": []}

    async def revoke_share_link(self, conversation_id: str, share_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("revoke_share_link", (conversation_id, share_id), kwargs))
        return {"success": True, "share_id": share_id}

    async def resolve_share_token(self, share_token: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("resolve_share_token", (share_token,), kwargs))
        return {"conversation_id": "conv-1", "messages": []}

    async def get_analytics(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_analytics", (), kwargs))
        return {"buckets": [], "pagination": {"total": 0}}


@dataclass
class RecordingPolicy:
    calls: list[str] = field(default_factory=list)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeSyncScopeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_dry_run_mirror_report(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"backend": "server", "domain": kwargs["domain"]}


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
async def test_scope_service_routes_server_chat_loop_and_rejects_local_loop():
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=server,
        policy_enforcer=policy,
    )

    started = await service.start_loop(mode="server", messages=[{"role": "user", "content": "Hi"}])
    events = await service.list_loop_events("run_123", mode="server", after_seq=3)
    approved = await service.approve_loop_call("run_123", mode="server", approval_id="approval-1")
    rejected = await service.reject_loop_call("run_123", mode="server", approval_id="approval-2")
    cancelled = await service.cancel_loop("run_123", mode="server")

    assert started["run_id"] == "run_123"
    assert events["events"] == []
    assert approved["ok"] is True
    assert rejected["ok"] is True
    assert cancelled["ok"] is True
    assert policy.calls == [
        "chat.loop.launch.server",
        "chat.loop.observe.server",
        "chat.loop.approve.server",
        "chat.loop.approve.server",
        "chat.loop.cancel.server",
    ]
    assert server.calls == [
        ("start_loop", (), {"messages": [{"role": "user", "content": "Hi"}]}),
        ("list_loop_events", ("run_123",), {"after_seq": 3}),
        ("approve_loop_call", ("run_123",), {"approval_id": "approval-1"}),
        ("reject_loop_call", ("run_123",), {"approval_id": "approval-2"}),
        ("cancel_loop", ("run_123",), {}),
    ]

    with pytest.raises(NotImplementedError, match="Local chat loop runs are not implemented"):
        await service.start_loop(mode="local", messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_scope_service_rejects_all_local_chat_loop_controls_before_backend_dispatch():
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(NotImplementedError, match="Local chat loop runs are not implemented"):
        await service.list_loop_events("run_123", mode="local", after_seq=3)
    with pytest.raises(NotImplementedError, match="Local chat loop runs are not implemented"):
        await service.approve_loop_call("run_123", mode="local", approval_id="approval-1")
    with pytest.raises(NotImplementedError, match="Local chat loop runs are not implemented"):
        await service.reject_loop_call("run_123", mode="local", approval_id="approval-2")
    with pytest.raises(NotImplementedError, match="Local chat loop runs are not implemented"):
        await service.cancel_loop("run_123", mode="local")

    assert server.calls == []
    assert policy.calls == [
        "chat.loop.observe.local",
        "chat.loop.approve.local",
        "chat.loop.approve.local",
        "chat.loop.cancel.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_maps_server_style_tree_pagination_to_local_tree_arguments():
    local = FakeConversationService()
    service = ChatConversationScopeService(local_service=local, server_service=FakeServerConversationService())

    await service.get_conversation_tree("conv-1", mode="local", limit=25, offset=10, max_depth=6)

    assert local.calls == [
        ("get_conversation_tree", ("conv-1",), {"root_limit": 25, "root_offset": 10, "depth_cap": 6}),
    ]


def test_scope_service_routes_chat_metadata_sync_mirror_report_to_sync_scope():
    sync_scope = FakeSyncScopeService()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=FakeServerConversationService(),
        sync_scope_service=sync_scope,
    )

    result = service.record_sync_mirror_report(
        mode="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        local_records=[{"id": "local-conv-1"}],
        remote_records=[{"id": "remote-conv-1"}],
    )

    assert result == {"backend": "server", "domain": "chat_metadata"}
    assert sync_scope.calls == [
        {
            "mode": "server",
            "domain": "chat_metadata",
            "entity_type": "conversation",
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
            "source_scope": "workspace",
            "local_records": [{"id": "local-conv-1"}],
            "remote_records": [{"id": "remote-conv-1"}],
        }
    ]


def test_scope_service_chat_metadata_mirror_report_does_not_add_local_marks():
    sync_scope = FakeSyncScopeService()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=FakeServerConversationService(),
        sync_scope_service=sync_scope,
    )

    service.record_sync_mirror_report(
        mode="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        local_records=[{"id": "conv-1", "title": "Plain"}],
        remote_records=[],
    )

    local_records = sync_scope.calls[0]["local_records"]
    assert local_records == [{"id": "conv-1", "title": "Plain"}]
    assert "starred" not in local_records[0]
    assert "marks" not in local_records[0]
    assert "local_marks" not in local_records[0]


def test_scope_service_rejects_local_chat_metadata_sync_mirror_report():
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=FakeServerConversationService(),
        sync_scope_service=FakeSyncScopeService(),
    )

    with pytest.raises(ValueError, match="Chat metadata mirror reports require server mode"):
        service.record_sync_mirror_report(
            mode="local",
            server_profile_id="server-a",
        )


@pytest.mark.asyncio
async def test_scope_service_routes_local_and_server_context_and_citations():
    server = FakeServerConversationService()
    local = FakeConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    messages = await service.get_messages_with_context("conv-1", mode="server", limit=10)
    citations = await service.get_citations("conv-1", mode="server")
    local_messages = await service.get_messages_with_context("conv-2", mode="local", limit=5)
    local_citations = await service.get_citations("conv-2", mode="local")

    assert messages == [{"id": "msg-1", "conversation_id": "conv-1"}]
    assert citations["conversation_id"] == "conv-1"
    assert local_messages[0]["rag_context"]["query"] == "local"
    assert local_citations["total_count"] == 1
    assert policy.calls == ["chat.detail.server", "chat.detail.server", "chat.detail.local", "chat.detail.local"]
    assert server.calls == [
        ("get_messages_with_context", ("conv-1",), {"limit": 10}),
        ("get_citations", ("conv-1",), {}),
    ]
    assert local.calls == [
        ("get_messages_with_context", ("conv-2",), {"limit": 5}),
        ("get_citations", ("conv-2",), {}),
    ]


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
    server_created = await service.create_conversation(
        mode="server",
        title="Server draft",
        character_id=7,
        scope_type="workspace",
        workspace_id="ws-1",
    )
    server_deleted = await service.delete_conversation(
        "server-conv-1",
        expected_version=4,
        mode="server",
        hard_delete=True,
        scope_type="workspace",
        workspace_id="ws-1",
    )

    assert created == "local-created"
    assert deleted is True
    assert server_created == "server-created"
    assert server_deleted is True
    assert policy.calls == ["chat.create.local", "chat.delete.local", "chat.create.server", "chat.delete.server"]
    assert server.calls == [
        (
            "create_conversation",
            (),
            {
                "title": "Server draft",
                "character_id": 7,
                "scope_type": "workspace",
                "workspace_id": "ws-1",
            },
        ),
        (
            "delete_conversation",
            ("server-conv-1", 4),
            {"hard_delete": True, "scope_type": "workspace", "workspace_id": "ws-1"},
        ),
    ]

    with pytest.raises(ValueError, match="mode must be 'local' or 'server'"):
        await service.list_conversations(mode="mixed")


@pytest.mark.asyncio
async def test_scope_service_surfaces_server_create_validation_without_local_dispatch():
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="requires character_id or assistant_kind"):
        await service.create_conversation(mode="server", title="Server draft")

    assert server.calls == [("create_conversation", (), {"title": "Server draft"})]
    assert policy.calls == ["chat.create.server"]


@pytest.mark.asyncio
async def test_scope_service_routes_server_chat_adjunct_controls_and_rejects_local_mode():
    server = FakeServerConversationService()
    policy = RecordingPolicy()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=server,
        policy_enforcer=policy,
    )

    await service.list_commands(mode="server")
    await service.save_knowledge(mode="server", conversation_id="conv-1", snippet="Important", tags=["alpha"])
    await service.create_share_link("conv-1", {"label": "Reviewer"}, mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.list_share_links("conv-1", mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.revoke_share_link("conv-1", "share-1", mode="server", scope_type="workspace", workspace_id="ws-1")
    await service.resolve_share_token("token", mode="server", limit=25)
    await service.get_analytics(mode="server", start_date="2026-04-01T00:00:00Z", end_date="2026-04-26T00:00:00Z")

    assert policy.calls[-7:] == [
        "chat.commands.list.server",
        "chat.knowledge.create.server",
        "chat.share_links.create.server",
        "chat.share_links.list.server",
        "chat.share_links.revoke.server",
        "chat.share_links.detail.server",
        "chat.analytics.observe.server",
    ]
    assert server.calls[-7:] == [
        ("list_commands", (), {}),
        ("save_knowledge", (), {"conversation_id": "conv-1", "snippet": "Important", "tags": ["alpha"]}),
        ("create_share_link", ("conv-1", {"label": "Reviewer"}), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        ("list_share_links", ("conv-1",), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        ("revoke_share_link", ("conv-1", "share-1"), {"scope_type": "workspace", "workspace_id": "ws-1"}),
        ("resolve_share_token", ("token",), {"limit": 25}),
        ("get_analytics", (), {"start_date": "2026-04-01T00:00:00Z", "end_date": "2026-04-26T00:00:00Z"}),
    ]

    with pytest.raises(NotImplementedError, match="Server chat commands are unavailable in local mode"):
        await service.list_commands(mode="local")
    with pytest.raises(NotImplementedError, match="Server chat analytics are unavailable in local mode"):
        await service.get_analytics(mode="local")


def test_scope_service_reports_known_chat_conversation_capability_gaps():
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=FakeServerConversationService(),
    )

    local_report = service.list_unsupported_capabilities(mode="local")
    server_report = service.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "chat.loop.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local chat loop run control is not implemented in Chatbook yet.",
            "affected_action_ids": [
                "chat.loop.launch.local",
                "chat.loop.observe.local",
                "chat.loop.approve.local",
                "chat.loop.cancel.local",
            ],
        },
        {
            "operation_id": "chat.adjunct_controls.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Server chat adjunct controls are not available in local Chatbook mode.",
            "affected_action_ids": [
                "chat.analytics.observe.local",
                "chat.commands.list.local",
                "chat.knowledge.create.local",
                "chat.share_links.create.local",
                "chat.share_links.detail.local",
                "chat.share_links.list.local",
                "chat.share_links.revoke.local",
            ],
        },
    ]
    assert server_report == []
