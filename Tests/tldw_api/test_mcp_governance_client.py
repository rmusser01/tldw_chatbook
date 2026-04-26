from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    MCPApprovalDecisionCreate,
    MCPApprovalPolicyCreate,
    MCPApprovalPolicyUpdate,
    MCPCapabilityMappingCreate,
    MCPCapabilityMappingUpdate,
    MCPCatalogCreate,
    MCPCatalogEntryCreate,
    MCPExternalServerCreate,
    MCPExternalServerUpdate,
    MCPGovernanceObject,
    MCPSecretSetRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_mcp_governance_client_routes_external_server_and_policy_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [{"server_id": "github", "name": "GitHub", "owner_scope_type": "team", "owner_scope_id": 7}],
            {"server_id": "github", "name": "GitHub", "owner_scope_type": "team", "owner_scope_id": 7},
            {"server_id": "github", "enabled": False},
            {"ok": True},
            {"server_id": "github", "secret_set": True},
            [{"id": 11, "name": "Restricted", "owner_scope_type": "team", "owner_scope_id": 7}],
            {"id": 11, "name": "Restricted", "owner_scope_type": "team", "owner_scope_id": 7, "policy_document": {"allowed_tools": ["search"]}},
            {"id": 11, "is_active": False},
            {"ok": True},
            [{"id": 12, "name": "Ask", "mode": "ask_every_time"}],
            {"id": 12, "name": "Ask", "mode": "ask_every_time"},
            {"id": 12, "mode": "allow_silently"},
            {"ok": True},
            {"id": 99, "decision": "approved", "duration": "once"},
            {"policy": {"allowed_tools": ["search"]}, "provenance": []},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    servers = await client.list_mcp_external_servers(owner_scope_type="team", owner_scope_id=7)
    created_server = await client.create_mcp_external_server(
        MCPExternalServerCreate(
            server_id="github",
            name="GitHub",
            transport="stdio",
            config={"command": "gh"},
            owner_scope_type="team",
            owner_scope_id=7,
        )
    )
    updated_server = await client.update_mcp_external_server("github", MCPExternalServerUpdate(enabled=False))
    deleted_server = await client.delete_mcp_external_server("github")
    secret = await client.set_mcp_external_server_secret("github", MCPSecretSetRequest(secret="token"))
    profiles = await client.list_mcp_permission_profiles(owner_scope_type="team", owner_scope_id=7)
    created_policy = await client.create_mcp_permission_profile(
        {
            "name": "Restricted",
            "owner_scope_type": "team",
            "owner_scope_id": 7,
            "policy_document": {"allowed_tools": ["search"]},
        }
    )
    updated_policy = await client.update_mcp_permission_profile(11, {"is_active": False})
    deleted_policy = await client.delete_mcp_permission_profile(11)
    approvals = await client.list_mcp_approval_policies(owner_scope_type="team", owner_scope_id=7)
    created_approval = await client.create_mcp_approval_policy(
        MCPApprovalPolicyCreate(name="Ask", owner_scope_type="team", owner_scope_id=7, mode="ask_every_time")
    )
    updated_approval = await client.update_mcp_approval_policy(12, MCPApprovalPolicyUpdate(mode="allow_silently"))
    deleted_approval = await client.delete_mcp_approval_policy(12)
    decision = await client.create_mcp_approval_decision(
        MCPApprovalDecisionCreate(
            context_key="user:1:tool:search",
            tool_name="search",
            decision="approved",
            duration="once",
        )
    )
    effective = await client.get_mcp_effective_policy(persona_id="persona-1", team_id=7)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/mcp/hub/external-servers")
    assert mocked.await_args_list[0].kwargs["params"] == {"owner_scope_type": "team", "owner_scope_id": 7}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/mcp/hub/external-servers")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "server_id": "github",
        "name": "GitHub",
        "transport": "stdio",
        "config": {"command": "gh"},
        "owner_scope_type": "team",
        "owner_scope_id": 7,
        "enabled": True,
    }
    assert mocked.await_args_list[2].args[:2] == ("PUT", "/api/v1/mcp/hub/external-servers/github")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"enabled": False}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/mcp/hub/external-servers/github")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/mcp/hub/external-servers/github/secret")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"secret": "token"}
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/mcp/hub/permission-profiles")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/mcp/hub/permission-profiles")
    assert mocked.await_args_list[7].args[:2] == ("PUT", "/api/v1/mcp/hub/permission-profiles/11")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/mcp/hub/permission-profiles/11")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/mcp/hub/approval-policies")
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/mcp/hub/approval-policies")
    assert mocked.await_args_list[11].args[:2] == ("PUT", "/api/v1/mcp/hub/approval-policies/12")
    assert mocked.await_args_list[12].args[:2] == ("DELETE", "/api/v1/mcp/hub/approval-policies/12")
    assert mocked.await_args_list[13].args[:2] == ("POST", "/api/v1/mcp/hub/approval-decisions")
    assert mocked.await_args_list[14].args[:2] == ("GET", "/api/v1/mcp/hub/effective-policy")
    assert mocked.await_args_list[14].kwargs["params"] == {"persona_id": "persona-1", "team_id": 7}

    assert isinstance(servers[0], MCPGovernanceObject)
    assert created_server.server_id == "github"
    assert updated_server.enabled is False
    assert deleted_server["ok"] is True
    assert secret.secret_set is True
    assert profiles[0].id == 11
    assert created_policy.policy_document == {"allowed_tools": ["search"]}
    assert updated_policy.is_active is False
    assert deleted_policy["ok"] is True
    assert approvals[0].id == 12
    assert created_approval.mode == "ask_every_time"
    assert updated_approval.mode == "allow_silently"
    assert deleted_approval["ok"] is True
    assert decision.decision == "approved"
    assert effective.policy == {"allowed_tools": ["search"]}


@pytest.mark.asyncio
async def test_mcp_governance_client_routes_catalog_and_registry_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [{"name": "search", "module_id": "web"}],
            [{"module_id": "web", "tool_count": 1}],
            {"entries": [], "modules": []},
            [{"id": 3, "name": "Team Tools", "team_id": 7}],
            {"id": 3, "name": "Team Tools", "team_id": 7},
            {"id": 4, "name": "Org Tools", "org_id": 2},
            {"catalog_id": 3, "tool_name": "search"},
            {"message": "Entry deleted", "catalog_id": 3, "tool_name": "search"},
            {"message": "Catalog deleted", "id": 3},
            [{"id": 21, "mapping_id": "filesystem-read"}],
            {"mapping_id": "filesystem-read", "normalized_mapping": {"capability_name": "filesystem.read"}},
            {"id": 21, "mapping_id": "filesystem-read"},
            {"id": 21, "is_active": False},
            {"ok": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    entries = await client.list_mcp_tool_registry()
    modules = await client.list_mcp_tool_registry_modules()
    summary = await client.get_mcp_tool_registry_summary()
    team_catalogs = await client.list_mcp_team_tool_catalogs(7)
    team_catalog = await client.create_mcp_team_tool_catalog(7, MCPCatalogCreate(name="Team Tools"))
    org_catalog = await client.create_mcp_org_tool_catalog(2, MCPCatalogCreate(name="Org Tools"))
    entry = await client.add_mcp_team_catalog_entry(7, 3, MCPCatalogEntryCreate(tool_name="search"))
    entry_deleted = await client.delete_mcp_team_catalog_entry(7, 3, "search")
    catalog_deleted = await client.delete_mcp_team_tool_catalog(7, 3)
    mappings = await client.list_mcp_capability_mappings(owner_scope_type="team", owner_scope_id=7)
    preview = await client.preview_mcp_capability_mapping(
        MCPCapabilityMappingCreate(mapping_id="filesystem-read", capability_name="filesystem.read")
    )
    created = await client.create_mcp_capability_mapping(
        MCPCapabilityMappingCreate(mapping_id="filesystem-read", capability_name="filesystem.read")
    )
    updated = await client.update_mcp_capability_mapping(21, MCPCapabilityMappingUpdate(is_active=False))
    deleted = await client.delete_mcp_capability_mapping(21)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/mcp/hub/tool-registry")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/mcp/hub/tool-registry/modules")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/mcp/hub/tool-registry/summary")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/teams/7/mcp/tool_catalogs")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/teams/7/mcp/tool_catalogs")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/orgs/2/mcp/tool_catalogs")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/teams/7/mcp/tool_catalogs/3/entries")
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/teams/7/mcp/tool_catalogs/3/entries/search")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/teams/7/mcp/tool_catalogs/3")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/mcp/hub/capability-mappings")
    assert mocked.await_args_list[9].kwargs["params"] == {"owner_scope_type": "team", "owner_scope_id": 7}
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/mcp/hub/capability-mappings/preview")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/mcp/hub/capability-mappings")
    assert mocked.await_args_list[12].args[:2] == ("PUT", "/api/v1/mcp/hub/capability-mappings/21")
    assert mocked.await_args_list[13].args[:2] == ("DELETE", "/api/v1/mcp/hub/capability-mappings/21")

    assert entries[0].name == "search"
    assert modules[0].module_id == "web"
    assert summary.entries == []
    assert team_catalogs[0].id == 3
    assert team_catalog.name == "Team Tools"
    assert org_catalog.name == "Org Tools"
    assert entry.tool_name == "search"
    assert entry_deleted["message"] == "Entry deleted"
    assert catalog_deleted["message"] == "Catalog deleted"
    assert mappings[0].mapping_id == "filesystem-read"
    assert preview.normalized_mapping == {"capability_name": "filesystem.read"}
    assert created.id == 21
    assert updated.is_active is False
    assert deleted["ok"] is True
