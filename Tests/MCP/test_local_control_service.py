from __future__ import annotations

import pytest

from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
from tldw_chatbook.MCP.server import describe_local_mcp_capabilities


class FakeLocalStore:
    def __init__(self) -> None:
        self.profiles = {
            "profile-a": LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                args=("-m", "demo.server"),
                env_placeholders={"API_KEY": "${API_KEY}"},
                env_literals={"LOG_LEVEL": "debug"},
            )
        }
        self.discovery_snapshots = {}
        self.governance_rules = []

    def list_profiles(self):
        return list(self.profiles.values())

    def get_profile(self, profile_id: str):
        return self.profiles.get(profile_id)

    def save_profile(self, profile):
        self.profiles[profile.profile_id] = profile
        return profile

    def save_discovery_snapshot(self, profile_id: str, snapshot):
        self.discovery_snapshots[profile_id] = snapshot
        return snapshot

    def get_discovery_snapshot(self, profile_id: str):
        return self.discovery_snapshots.get(profile_id)

    def list_governance_rules(self):
        return list(self.governance_rules)

    def save_governance_rule(self, rule):
        for index, existing in enumerate(self.governance_rules):
            if existing.rule_id == rule.rule_id:
                self.governance_rules[index] = rule
                return rule
        self.governance_rules.append(rule)
        return rule


class FakeMCPClient:
    def __init__(self) -> None:
        self.connected = []

    async def connect_to_server(self, server_id: str, command: str, args=None, env=None):
        self.connected.append(
            {"server_id": server_id, "command": command, "args": args or [], "env": env or {}}
        )
        return True

    async def describe_server(self, server_id: str):
        return {
            "server_id": server_id,
            "tools": [{"name": "remote_tool"}],
            "resources": [{"uri": "remote://resource"}],
            "prompts": [{"name": "remote_prompt"}],
        }


def test_local_control_service_builds_inventory_from_local_manifest_without_loopback():
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        manifest_provider=lambda: {
            "tools": [{"name": "search_notes"}],
            "resources": [{"uri": "note://{note_id}"}],
            "prompts": [{"name": "summarize_conversation"}],
        },
    )

    inventory = service.get_inventory()

    assert inventory["tools"][0]["name"] == "search_notes"
    assert inventory["resources"][0]["uri"] == "note://{note_id}"
    assert inventory["prompts"][0]["name"] == "summarize_conversation"


def test_local_control_service_uses_real_local_manifest_helper_by_default():
    service = LocalMCPControlService(store=FakeLocalStore())

    inventory = service.get_inventory()
    manifest = describe_local_mcp_capabilities()

    assert inventory == manifest
    assert any(tool["name"] == "search_rag" for tool in inventory["tools"])
    assert any(tool["description"] == "Perform RAG search across ingested media." for tool in inventory["tools"])
    assert any(resource["uri"] == "note://{note_id}" for resource in inventory["resources"])
    assert any(prompt["name"] == "summarize_conversation" for prompt in inventory["prompts"])


def test_local_control_service_exposes_overview_external_servers_and_governance():
    store = FakeLocalStore()
    service = LocalMCPControlService(
        store=store,
        client=FakeMCPClient(),
        manifest_provider=lambda: {
            "tools": [{"name": "search_notes"}, {"name": "search_rag"}],
            "resources": [{"uri": "note://{note_id}"}],
            "prompts": [{"name": "summarize_conversation"}],
        },
    )

    service.save_governance_rule(
        {
            "rule_id": "rule-a",
            "capability_id": "mcp.governance.list.local",
            "decision": "allow",
        }
    )

    overview = service.get_overview()
    external_servers = service.get_external_servers()
    governance = service.get_governance()

    assert overview["inventory"]["tools"] == 2
    assert overview["external_servers"]["profiles"] == 1
    assert overview["governance"]["rules"] == 1
    assert external_servers[0]["profile_id"] == "profile-a"
    assert external_servers[0]["env"]["API_KEY"] == "${API_KEY}"
    assert external_servers[0]["env"]["LOG_LEVEL"] == "debug"
    assert governance[0]["capability_id"] == "mcp.governance.list.local"


@pytest.mark.asyncio
async def test_local_control_service_connects_profile_and_persists_discovery_snapshot():
    store = FakeLocalStore()
    client = FakeMCPClient()
    service = LocalMCPControlService(store=store, client=client, manifest_provider=lambda: {})

    snapshot = await service.connect_profile("profile-a")

    assert client.connected[0]["server_id"] == "profile-a"
    assert store.discovery_snapshots["profile-a"]["tools"][0]["name"] == "remote_tool"
    assert snapshot["prompts"][0]["name"] == "remote_prompt"


@pytest.mark.asyncio
async def test_local_control_service_describes_connected_server_from_client_cache():
    from tldw_chatbook.MCP.client import MCPClient

    client = MCPClient.__new__(MCPClient)
    client.name = "test-client"
    client.sessions = {}
    client.servers = {
        "profile-a": {
            "command": "python",
            "args": ["-m", "demo.server"],
            "connected_at": "2026-04-22T10:00:00Z",
            "tools": [type("Tool", (), {"name": "remote_tool", "description": "Remote tool", "inputSchema": {}})()],
            "resources": [type("Resource", (), {"uri": "remote://resource", "name": "Remote Resource", "description": "Remote resource", "mimeType": "text/plain"})()],
            "prompts": [type("Prompt", (), {"name": "remote_prompt", "description": "Remote prompt", "arguments": []})()],
        }
    }

    description = await client.describe_server("profile-a")

    assert description["tools"][0]["name"] == "remote_tool"
    assert description["resources"][0]["uri"] == "remote://resource"
    assert description["prompts"][0]["name"] == "remote_prompt"
