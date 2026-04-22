from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tldw_chatbook.MCP.client as mcp_client_module
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
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


def test_local_control_service_rejects_legacy_env_literal_bypass_on_profile_save():
    store = FakeLocalStore()
    service = LocalMCPControlService(store=store, client=FakeMCPClient(), manifest_provider=lambda: {})

    saved = service.save_external_profile(
        {
            "profile_id": "profile-b",
            "command": "python",
            "args": ["-m", "demo.server"],
            "env_literals": {"LOG_LEVEL": "debug"},
            "legacy_env_literals": {
                "SERVICE_ALIAS": "example-service-prod",
                "MODEL_NAME": "gpt-4o-mini",
                "SOCKET_PATH": "/tmp/mcp-demo.sock",
            },
        }
    )

    stored = store.get_profile("profile-b")

    assert saved["env"]["LOG_LEVEL"] == "debug"
    assert "SERVICE_ALIAS" not in saved["env"]
    assert "MODEL_NAME" not in saved["env"]
    assert "SOCKET_PATH" not in saved["env"]
    assert stored is not None
    assert stored.legacy_env_literals == {}


def test_local_control_service_rejects_legacy_env_literal_bypass_on_profile_object_save():
    store = FakeLocalStore()
    service = LocalMCPControlService(store=store, client=FakeMCPClient(), manifest_provider=lambda: {})
    profile = LocalExternalMCPProfile(
        profile_id="profile-c",
        command="python",
        args=("-m", "demo.server"),
        env_literals={"LOG_LEVEL": "debug"},
        legacy_env_literals={
            "SERVICE_ALIAS": "example-service-prod",
            "MODEL_NAME": "gpt-4o-mini",
            "SOCKET_PATH": "/tmp/mcp-demo.sock",
        },
    )

    saved = service.save_external_profile(profile)
    stored = store.get_profile("profile-c")

    assert saved["env"]["LOG_LEVEL"] == "debug"
    assert "SERVICE_ALIAS" not in saved["env"]
    assert "MODEL_NAME" not in saved["env"]
    assert "SOCKET_PATH" not in saved["env"]
    assert stored is not None
    assert stored.legacy_env_literals == {}


def test_local_control_service_rejects_invalid_profile_writes(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(store=store, client=FakeMCPClient(), manifest_provider=lambda: {})

    with pytest.raises(ValueError, match="profile_id"):
        service.save_external_profile(
            {
                "profile_id": "",
                "command": "python",
            }
        )

    with pytest.raises(ValueError, match="command"):
        service.save_external_profile(
            {
                "profile_id": "profile-b",
                "command": "",
            }
        )

    assert store.get_profile("profile-b") is None


def test_local_control_service_rejects_invalid_governance_rule_writes(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(store=store, client=FakeMCPClient(), manifest_provider=lambda: {})

    with pytest.raises(ValueError, match="rule_id"):
        service.save_governance_rule(
            {
                "rule_id": "",
                "capability_id": "mcp.governance.list.local",
                "decision": "allow",
            }
        )

    with pytest.raises(ValueError, match="capability_id"):
        service.save_governance_rule(
            {
                "rule_id": "rule-b",
                "capability_id": "",
                "decision": "allow",
            }
        )

    with pytest.raises(ValueError, match="decision"):
        service.save_governance_rule(
            {
                "rule_id": "rule-b",
                "capability_id": "mcp.governance.list.local",
                "decision": "",
            }
        )

    assert store.list_governance_rules() == []


@pytest.mark.asyncio
async def test_local_control_service_connects_profile_and_persists_discovery_snapshot():
    store = FakeLocalStore()
    client = FakeMCPClient()
    service = LocalMCPControlService(store=store, client=client, manifest_provider=lambda: {})

    with patch.dict(
        os.environ,
        {
            "API_KEY": "resolved-api-key",
            "PATH": "/usr/bin",
            "HOME": "/tmp/demo-home",
            "LANG": "C.UTF-8",
            "UNRELATED_SECRET": "should-not-leak",
        },
        clear=True,
    ):
        snapshot = await service.connect_profile("profile-a")

    assert client.connected[0]["server_id"] == "profile-a"
    assert client.connected[0]["env"] == {
        "API_KEY": "resolved-api-key",
        "LOG_LEVEL": "debug",
        "PATH": "/usr/bin",
        "HOME": "/tmp/demo-home",
        "LANG": "C.UTF-8",
    }
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


@pytest.mark.asyncio
async def test_mcp_client_connect_to_server_uses_stdio_transport_flow(monkeypatch):
    call_log = []

    class FakeServerParams:
        def __init__(self, command, args, env):
            self.command = command
            self.args = args
            self.env = env
            call_log.append(("server_params", command, list(args), env))

    class FakeTransportContext:
        def __init__(self, server_params):
            self.server_params = server_params

        async def __aenter__(self):
            call_log.append(("transport_enter", self.server_params.command))
            return ("read-stream", "write-stream")

        async def __aexit__(self, exc_type, exc, tb):
            call_log.append(("transport_exit", self.server_params.command))

    def fake_stdio_client(server_params):
        call_log.append(("stdio_client", server_params.command))
        return FakeTransportContext(server_params)

    class FakeSession:
        def __init__(self, read_stream, write_stream):
            self.read_stream = read_stream
            self.write_stream = write_stream
            call_log.append(("session_init", read_stream, write_stream))

        async def __aenter__(self):
            call_log.append(("session_enter", self.read_stream, self.write_stream))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            call_log.append(("session_exit", exc_type is None))

        async def initialize(self):
            call_log.append(("initialize", self.read_stream, self.write_stream))

        async def list_tools(self):
            call_log.append(("list_tools",))
            return SimpleNamespace(
                tools=[SimpleNamespace(name="remote_tool", description="Remote tool", inputSchema={})]
            )

        async def list_resources(self):
            call_log.append(("list_resources",))
            return SimpleNamespace(
                resources=[
                    SimpleNamespace(
                        uri="remote://resource",
                        name="Remote Resource",
                        description="Remote resource",
                        mimeType="text/plain",
                    )
                ]
            )

        async def list_prompts(self):
            call_log.append(("list_prompts",))
            return SimpleNamespace(
                prompts=[SimpleNamespace(name="remote_prompt", description="Remote prompt", arguments=[])]
            )

    monkeypatch.setattr(mcp_client_module, "MCP_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(mcp_client_module, "StdioServerParameters", FakeServerParams, raising=False)
    monkeypatch.setattr(mcp_client_module, "stdio_client", fake_stdio_client, raising=False)
    monkeypatch.setattr(mcp_client_module, "ClientSession", FakeSession, raising=False)

    client = mcp_client_module.MCPClient(name="test-client")

    connected = await client.connect_to_server(
        "profile-a",
        "python",
        args=["-m", "demo.server"],
        env={"API_KEY": "resolved-api-key"},
    )

    assert connected is True
    assert call_log[:8] == [
        ("server_params", "python", ["-m", "demo.server"], {"API_KEY": "resolved-api-key"}),
        ("stdio_client", "python"),
        ("transport_enter", "python"),
        ("session_init", "read-stream", "write-stream"),
        ("session_enter", "read-stream", "write-stream"),
        ("initialize", "read-stream", "write-stream"),
        ("list_tools",),
        ("list_resources",),
    ]
    assert call_log[8] == ("list_prompts",)
    assert "profile-a" in client.sessions
    assert client.servers["profile-a"]["command"] == "python"
    assert client.servers["profile-a"]["args"] == ["-m", "demo.server"]
    assert client.servers["profile-a"]["tools"][0].name == "remote_tool"
    assert client.servers["profile-a"]["resources"][0].uri == "remote://resource"
    assert client.servers["profile-a"]["prompts"][0].name == "remote_prompt"


@pytest.mark.asyncio
async def test_mcp_client_connect_to_server_cleans_up_existing_connection_for_same_server_id(monkeypatch):
    call_log = []
    session_counter = {"value": 0}

    class FakeServerParams:
        def __init__(self, command, args, env):
            self.command = command
            self.args = args
            self.env = env

    class FakeTransportContext:
        def __init__(self, server_params):
            self.server_params = server_params
            self.session_id = None

        async def __aenter__(self):
            self.session_id = session_counter["value"] + 1
            call_log.append(("transport_enter", self.session_id, self.server_params.command))
            return (f"read-{self.session_id}", f"write-{self.session_id}")

        async def __aexit__(self, exc_type, exc, tb):
            call_log.append(("transport_exit", self.session_id, self.server_params.command))

    def fake_stdio_client(server_params):
        return FakeTransportContext(server_params)

    class FakeSession:
        def __init__(self, read_stream, write_stream):
            session_counter["value"] += 1
            self.session_id = session_counter["value"]
            self.read_stream = read_stream
            self.write_stream = write_stream
            call_log.append(("session_init", self.session_id, read_stream, write_stream))

        async def __aenter__(self):
            call_log.append(("session_enter", self.session_id))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            call_log.append(("session_exit", self.session_id))

        async def initialize(self):
            call_log.append(("initialize", self.session_id))

        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name=f"remote_tool_{self.session_id}", description="Remote tool", inputSchema={})]
            )

        async def list_resources(self):
            return SimpleNamespace(
                resources=[
                    SimpleNamespace(
                        uri=f"remote://resource/{self.session_id}",
                        name="Remote Resource",
                        description="Remote resource",
                        mimeType="text/plain",
                    )
                ]
            )

        async def list_prompts(self):
            return SimpleNamespace(
                prompts=[SimpleNamespace(name=f"remote_prompt_{self.session_id}", description="Remote prompt", arguments=[])]
            )

    monkeypatch.setattr(mcp_client_module, "MCP_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(mcp_client_module, "StdioServerParameters", FakeServerParams, raising=False)
    monkeypatch.setattr(mcp_client_module, "stdio_client", fake_stdio_client, raising=False)
    monkeypatch.setattr(mcp_client_module, "ClientSession", FakeSession, raising=False)

    client = mcp_client_module.MCPClient(name="test-client")

    first_connected = await client.connect_to_server("profile-a", "python", args=["-m", "demo.one"])
    second_connected = await client.connect_to_server("profile-a", "python", args=["-m", "demo.two"])

    assert first_connected is True
    assert second_connected is True
    assert ("session_exit", 1) in call_log
    assert ("transport_exit", 1, "python") in call_log
    assert client.sessions["profile-a"].session_id == 2
    assert client.servers["profile-a"]["args"] == ["-m", "demo.two"]
    assert client.servers["profile-a"]["tools"][0].name == "remote_tool_2"
    assert client.servers["profile-a"]["resources"][0].uri == "remote://resource/2"
    assert client.servers["profile-a"]["prompts"][0].name == "remote_prompt_2"
