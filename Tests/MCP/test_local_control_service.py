from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import patch

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


class EmptySnapshotClient(FakeMCPClient):
    async def describe_server(self, server_id: str):
        return {
            "server_id": server_id,
            "tools": [],
            "resources": [],
            "prompts": [],
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
    assert any(tool["description"] == "Search the RAG database for relevant content." for tool in inventory["tools"])
    assert any(resource["uri"] == "note://{note_id}" for resource in inventory["resources"])
    assert any(prompt["name"] == "summarize_conversation" for prompt in inventory["prompts"])


def test_local_manifest_helper_stays_aligned_with_registered_server_surface():
    server_path = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "MCP" / "server.py"
    module_node = ast.parse(server_path.read_text(encoding="utf-8"))

    def _registered_entries(method_name: str, decorator_name: str) -> list[dict[str, str]]:
        for node in module_node.body:
            if isinstance(node, ast.ClassDef) and node.name == "TldwMCPServer":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == method_name:
                        entries: list[dict[str, str]] = []
                        for nested in child.body:
                            if not isinstance(nested, ast.AsyncFunctionDef):
                                continue
                            for decorator in nested.decorator_list:
                                if not isinstance(decorator, ast.Call):
                                    continue
                                func = decorator.func
                                if not isinstance(func, ast.Attribute) or func.attr != decorator_name:
                                    continue
                                entry = {
                                    "name": nested.name,
                                    "description": (ast.get_docstring(nested) or "").strip().splitlines()[0].strip(),
                                }
                                if decorator_name == "resource":
                                    entry["uri"] = decorator.args[0].value
                                entries.append(entry)
                                break
                        return entries
        return []

    helper_manifest = describe_local_mcp_capabilities()
    expected_tools = _registered_entries("_register_tools", "tool")
    expected_resources = _registered_entries("_register_resources", "resource")
    expected_prompts = _registered_entries("_register_prompts", "prompt")

    assert helper_manifest["tools"] == expected_tools
    assert helper_manifest["resources"] == expected_resources
    assert helper_manifest["prompts"] == expected_prompts
    assert any(item["uri"] == "note://{note_id}" for item in helper_manifest["resources"])
    assert any(item["description"] == "Search notes by content or title." for item in helper_manifest["tools"])
    assert any(item["description"] == "Generate a prompt to summarize a conversation." for item in helper_manifest["prompts"])


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

    with patch.dict(os.environ, {"API_KEY": "resolved-api-key", "PATH": "/usr/bin"}, clear=True):
        snapshot = await service.connect_profile("profile-a")

    assert client.connected[0]["server_id"] == "profile-a"
    assert client.connected[0]["env"]["API_KEY"] == "resolved-api-key"
    assert client.connected[0]["env"]["LOG_LEVEL"] == "debug"
    assert client.connected[0]["env"]["PATH"] == "/usr/bin"
    assert store.discovery_snapshots["profile-a"]["tools"][0]["name"] == "remote_tool"
    assert snapshot["prompts"][0]["name"] == "remote_prompt"


@pytest.mark.asyncio
async def test_local_control_service_rejects_empty_capability_snapshots():
    store = FakeLocalStore()
    client = EmptySnapshotClient()
    service = LocalMCPControlService(store=store, client=client, manifest_provider=lambda: {})

    with patch.dict(os.environ, {"API_KEY": "resolved-api-key"}, clear=True):
        with pytest.raises(RuntimeError):
            await service.connect_profile("profile-a")

    assert "profile-a" not in store.discovery_snapshots


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
