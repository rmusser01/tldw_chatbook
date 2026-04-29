from unittest.mock import Mock

import pytest

from tldw_chatbook.MCP_Governance_Interop import ServerMCPGovernanceService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeMCPGovernanceClient:
    def __init__(self):
        self.calls = []

    async def list_mcp_external_servers(self, **kwargs):
        self.calls.append(("list_mcp_external_servers", kwargs))
        return [{"server_id": "github", "owner_scope_type": "team", "owner_scope_id": 7}]

    async def create_mcp_external_server(self, request_data):
        self.calls.append(("create_mcp_external_server", request_data.model_dump(mode="json")))
        return {"server_id": "github", "name": "GitHub"}

    async def list_mcp_team_tool_catalogs(self, team_id):
        self.calls.append(("list_mcp_team_tool_catalogs", team_id))
        return [{"id": 3, "team_id": team_id, "name": "Team Tools"}]

    async def create_mcp_approval_decision(self, request_data):
        self.calls.append(("create_mcp_approval_decision", request_data.model_dump(mode="json")))
        return {"id": 5, "decision": "approved"}

    async def get_mcp_effective_policy(self, **kwargs):
        self.calls.append(("get_mcp_effective_policy", kwargs))
        return {"policy": {"allowed_tools": ["search"]}, "provenance": []}


@pytest.mark.asyncio
async def test_server_mcp_governance_service_enforces_action_level_policy_and_routes_core_surfaces():
    client = FakeMCPGovernanceClient()
    policy = Mock()
    service = ServerMCPGovernanceService(client=client, policy_enforcer=policy)

    servers = await service.list_external_servers(owner_scope_type="team", owner_scope_id=7)
    created = await service.create_external_server(
        {
            "server_id": "github",
            "name": "GitHub",
            "transport": "stdio",
            "config": {"command": "gh"},
            "owner_scope_type": "team",
            "owner_scope_id": 7,
        }
    )
    catalogs = await service.list_team_tool_catalogs(team_id=7)
    decision = await service.create_approval_decision(
        {"context_key": "user:1:tool:search", "tool_name": "search", "decision": "approved", "duration": "once"}
    )
    effective = await service.get_effective_policy(team_id=7)

    assert servers[0]["server_id"] == "github"
    assert created["server_id"] == "github"
    assert catalogs[0]["team_id"] == 7
    assert decision["decision"] == "approved"
    assert effective["policy"] == {"allowed_tools": ["search"]}
    assert client.calls == [
        ("list_mcp_external_servers", {"owner_scope_type": "team", "owner_scope_id": 7}),
        (
            "create_mcp_external_server",
            {
                "server_id": "github",
                "name": "GitHub",
                "transport": "stdio",
                "config": {"command": "gh"},
                "owner_scope_type": "team",
                "owner_scope_id": 7,
                "enabled": True,
            },
        ),
        ("list_mcp_team_tool_catalogs", 7),
        (
            "create_mcp_approval_decision",
            {
                "approval_policy_id": None,
                "context_key": "user:1:tool:search",
                "conversation_id": None,
                "tool_name": "search",
                "scope_key": "default",
                "decision": "approved",
                "duration": "once",
            },
        ),
        ("get_mcp_effective_policy", {"persona_id": None, "group_id": None, "org_id": None, "team_id": 7}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "mcp.governance.external_servers.list.server",
        "mcp.governance.external_servers.create.server",
        "mcp.governance.catalogs.list.server",
        "mcp.governance.approval_decisions.approve.server",
        "mcp.governance.effective_policy.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_mcp_governance_service_hard_stops_denied_policy_before_dispatch():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="authority_denied",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeMCPGovernanceClient()
    service = ServerMCPGovernanceService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_external_servers(owner_scope_type="team", owner_scope_id=7)

    assert exc.value.reason_code == "authority_denied"
    assert client.calls == []


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


class FreshClientProvider:
    def __init__(self):
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = object()
        self.clients.append(client)
        return client


def test_server_mcp_governance_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerMCPGovernanceService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_mcp_governance_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerMCPGovernanceService.from_server_context_provider(provider)

    assert isinstance(service, ServerMCPGovernanceService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_mcp_governance_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerMCPGovernanceService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_mcp_governance_service_from_config_returns_provider_backed_service():
    service = ServerMCPGovernanceService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerMCPGovernanceService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
