import pytest

from tldw_chatbook.MCP_Governance_Interop.mcp_governance_scope_service import MCPGovernanceScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerMCPGovernanceService:
    def __init__(self):
        self.calls = []

    async def list_external_servers(self, **kwargs):
        self.calls.append(("list_external_servers", kwargs))
        return [{"server_id": "github", "owner_scope_type": "team", "owner_scope_id": 7}]

    async def create_external_server(self, request_data):
        self.calls.append(("create_external_server", request_data))
        return {"server_id": "github", **request_data}

    async def update_external_server(self, server_id, request_data):
        self.calls.append(("update_external_server", {"server_id": server_id, "request_data": request_data}))
        return {"server_id": server_id, **request_data}

    async def list_team_tool_catalogs(self, *, team_id):
        self.calls.append(("list_team_tool_catalogs", {"team_id": team_id}))
        return [{"id": 3, "team_id": team_id, "name": "Team Tools"}]

    async def create_team_tool_catalog(self, *, team_id, request_data):
        self.calls.append(("create_team_tool_catalog", {"team_id": team_id, "request_data": request_data}))
        return {"id": 5, "team_id": team_id, **request_data}

    async def add_team_catalog_entry(self, *, team_id, catalog_id, request_data):
        self.calls.append(("add_team_catalog_entry", {"team_id": team_id, "catalog_id": catalog_id, "request_data": request_data}))
        return {"catalog_id": catalog_id, **request_data}

    async def list_org_tool_catalogs(self, *, org_id):
        self.calls.append(("list_org_tool_catalogs", {"org_id": org_id}))
        return [{"id": 4, "org_id": org_id, "name": "Org Tools"}]

    async def get_effective_policy(self, **kwargs):
        self.calls.append(("get_effective_policy", kwargs))
        return {"policy": {"allowed_tools": ["search"]}, "provenance": []}

    async def stream_events(self, **kwargs):
        self.calls.append(("stream_events", kwargs))
        yield {
            "event_id": "evt-2",
            "event_type": "mcp_hub.external_server.created",
            "resource_type": "mcp_external_server",
            "resource_id": "github",
            "action": "external_server.created",
        }


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
async def test_mcp_governance_scope_service_routes_server_actions_and_normalizes_records():
    server = FakeServerMCPGovernanceService()
    policy = FakePolicyEnforcer()
    scope = MCPGovernanceScopeService(server_service=server, policy_enforcer=policy)

    servers = await scope.list_external_servers(mode="server", owner_scope_type="team", owner_scope_id=7)
    team_catalogs = await scope.list_tool_catalogs(mode="server", scope_type="team", scope_id=7)
    org_catalogs = await scope.list_tool_catalogs(mode="server", scope_type="org", scope_id=2)
    effective = await scope.get_effective_policy(mode="server", team_id=7)

    assert servers[0]["backend"] == "server"
    assert servers[0]["record_id"] == "server:mcp_external_server:github"
    assert team_catalogs[0]["record_id"] == "server:mcp_tool_catalog:team:3"
    assert org_catalogs[0]["record_id"] == "server:mcp_tool_catalog:org:4"
    assert effective["record_id"] == "server:mcp_effective_policy:team:7"
    assert server.calls == [
        ("list_external_servers", {"owner_scope_type": "team", "owner_scope_id": 7}),
        ("list_team_tool_catalogs", {"team_id": 7}),
        ("list_org_tool_catalogs", {"org_id": 2}),
        ("get_effective_policy", {"persona_id": None, "group_id": None, "org_id": None, "team_id": 7}),
    ]
    assert policy.calls == [
        "mcp.governance.external_servers.list.server",
        "mcp.governance.catalogs.list.server",
        "mcp.governance.catalogs.list.server",
        "mcp.governance.effective_policy.detail.server",
    ]


@pytest.mark.asyncio
async def test_mcp_governance_scope_service_exposes_server_crud_through_source_boundary():
    server = FakeServerMCPGovernanceService()
    policy = FakePolicyEnforcer()
    scope = MCPGovernanceScopeService(server_service=server, policy_enforcer=policy)

    created = await scope.create_external_server(
        {"server_id": "github", "name": "GitHub", "transport": "stdio"},
        mode="server",
    )
    updated = await scope.update_external_server("github", {"enabled": False}, mode="server")
    catalog = await scope.create_tool_catalog({"name": "Team Tools"}, mode="server", scope_type="team", scope_id=7)
    entry = await scope.add_catalog_entry(
        {"tool_name": "search"},
        mode="server",
        scope_type="team",
        scope_id=7,
        catalog_id=5,
    )

    assert created["record_id"] == "server:mcp_external_server:github"
    assert updated["enabled"] is False
    assert catalog["record_id"] == "server:mcp_tool_catalog:team:5"
    assert entry["record_id"] == "server:mcp_tool_catalog_entry:search"
    assert server.calls == [
        ("create_external_server", {"server_id": "github", "name": "GitHub", "transport": "stdio"}),
        ("update_external_server", {"server_id": "github", "request_data": {"enabled": False}}),
        ("create_team_tool_catalog", {"team_id": 7, "request_data": {"name": "Team Tools"}}),
        ("add_team_catalog_entry", {"team_id": 7, "catalog_id": 5, "request_data": {"tool_name": "search"}}),
    ]
    assert policy.calls == [
        "mcp.governance.external_servers.create.server",
        "mcp.governance.external_servers.update.server",
        "mcp.governance.catalogs.create.server",
        "mcp.governance.catalog_entries.create.server",
    ]


@pytest.mark.asyncio
async def test_mcp_governance_scope_service_honestly_rejects_local_mode():
    server = FakeServerMCPGovernanceService()
    scope = MCPGovernanceScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Remote MCP governance is server-only"):
        await scope.list_external_servers(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_mcp_governance_scope_service_requires_explicit_catalog_scope():
    scope = MCPGovernanceScopeService(server_service=FakeServerMCPGovernanceService())

    with pytest.raises(ValueError, match="scope_type must be 'org' or 'team'"):
        await scope.list_tool_catalogs(mode="server", scope_type="global", scope_id=1)


@pytest.mark.asyncio
async def test_mcp_governance_scope_service_blocks_denied_policy_before_dispatch():
    server = FakeServerMCPGovernanceService()
    scope = MCPGovernanceScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_external_servers(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_mcp_governance_scope_service_reports_local_unavailable_and_server_followups():
    scope = MCPGovernanceScopeService(server_service=FakeServerMCPGovernanceService())

    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "mcp_governance.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Remote MCP governance is unavailable in local/offline mode; use Chatbook's local MCP runtime controls instead.",
            "affected_action_ids": [],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="server") == []


@pytest.mark.asyncio
async def test_mcp_governance_scope_service_observes_server_events():
    server = FakeServerMCPGovernanceService()
    policy = FakePolicyEnforcer()
    scope = MCPGovernanceScopeService(server_service=server, policy_enforcer=policy)

    events = [
        event
        async for event in scope.observe_events(
            mode="server",
            after_event_id="evt-1",
            event_types=["mcp_hub.external_server.created"],
            replay=True,
        )
    ]

    assert events == [
        {
            "event_id": "evt-2",
            "event_type": "mcp_hub.external_server.created",
            "resource_type": "mcp_external_server",
            "resource_id": "github",
            "action": "external_server.created",
            "backend": "server",
            "record_id": "server:mcp_governance_event:evt-2",
        }
    ]
    assert server.calls == [
        (
            "stream_events",
            {
                "after_event_id": "evt-1",
                "event_types": ["mcp_hub.external_server.created"],
                "replay": True,
            },
        )
    ]
    assert policy.calls == ["mcp.governance.events.observe.server"]
