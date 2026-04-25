import pytest

from tldw_chatbook.Sharing_Interop.sharing_scope_service import SharingScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeSharingService:
    def __init__(self):
        self.calls = []

    async def create_link(self, **kwargs):
        self.calls.append(("create_link", kwargs))
        return {"id": 10, "resource_type": kwargs["resource_type"], "resource_id": kwargs["resource_id"]}

    async def list_links(self):
        self.calls.append(("list_links",))
        return {"tokens": [{"id": 9, "resource_id": "ws-1"}], "total": 1}

    async def list_shared_with_me(self):
        self.calls.append(("list_shared_with_me",))
        return {"items": [{"share_id": 7, "workspace_id": "ws-1"}], "total": 1}

    async def list_shared_workspace_sources(self, share_id):
        self.calls.append(("list_shared_workspace_sources", share_id))
        return [{"id": "src-1", "workspace_id": "ws-1"}]

    async def chat_with_shared_workspace(self, share_id, **kwargs):
        self.calls.append(("chat_with_shared_workspace", share_id, kwargs))
        return {"answer": "ok"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_sharing_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeSharingService()
    policy = FakePolicyEnforcer()
    scope = SharingScopeService(server_service=server, policy_enforcer=policy)

    links = await scope.list_links(mode="server")
    created = await scope.create_link(mode="server", resource_type="workspace", resource_id="ws-1")
    shared = await scope.list_shared_with_me(mode="server")
    sources = await scope.list_shared_workspace_sources(mode="server", share_id=7)
    chat = await scope.chat_with_shared_workspace(mode="server", share_id=7, query="summarize")

    assert links["tokens"][0]["record_id"] == "server:sharing_token:9"
    assert links["tokens"][0]["backend"] == "server"
    assert created["record_id"] == "server:sharing_token:10"
    assert shared["items"][0]["record_id"] == "server:shared_workspace:7"
    assert sources[0]["record_id"] == "server:shared_workspace_source:src-1"
    assert chat["backend"] == "server"
    assert server.calls == [
        ("list_links",),
        ("create_link", {"resource_type": "workspace", "resource_id": "ws-1"}),
        ("list_shared_with_me",),
        ("list_shared_workspace_sources", 7),
        ("chat_with_shared_workspace", 7, {"query": "summarize"}),
    ]
    assert policy.calls == [
        "sharing.links.list.server",
        "sharing.links.create.server",
        "sharing.links.list.server",
        "sharing.links.inspect.server",
        "sharing.links.launch.server",
    ]


@pytest.mark.asyncio
async def test_sharing_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeSharingService()
    scope = SharingScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Sharing is a server-only capability"):
        await scope.list_links(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_sharing_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeSharingService()
    scope = SharingScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_link(mode="server", resource_type="workspace", resource_id="ws-1")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_sharing_scope_service_reports_known_unsupported_capabilities():
    scope = SharingScopeService(server_service=FakeSharingService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "sharing.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Sharing is unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
    assert server_report == [
        {
            "operation_id": "sharing.links.observe.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server sharing API does not expose share-link observation events.",
            "affected_action_ids": ["sharing.links.observe.server"],
        }
    ]
