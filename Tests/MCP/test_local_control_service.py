from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import tldw_chatbook.MCP.client as mcp_client_module
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
from tldw_chatbook.MCP.server import (
    _signature_to_input_schema,
    _signature_to_prompt_arguments,
    describe_local_mcp_capabilities,
)


# Fix Round H (PR-T3 review), Item 2c: not a fake-vs-real key-set pin
# candidate -- every method below stores/returns REAL domain dataclasses
# already imported from `local_store.py` (`LocalExternalMCPProfile`,
# `LocalGovernanceRule`, ...), not independently-invented shapes, so a
# signature or field drift in the real `LocalMCPStore` raises a loud
# TypeError/AttributeError at the first call -- the opposite of the
# silent-drift failure mode a pin exists to catch. See the POLICY comment
# above `FakeLocalMCPControlService` in
# Tests/MCP/test_unified_control_plane_service.py for the full policy and
# contrast with `FakeLocalRuntimeDelegate` below, which IS pinned.
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
        self.approval_requests = []
        self.runtime_activity = []

    def list_profiles(self):
        return list(self.profiles.values())

    def get_profile(self, profile_id: str):
        return self.profiles.get(profile_id)

    def save_profile(self, profile):
        self.profiles[profile.profile_id] = profile
        return profile

    def delete_profile(self, profile_id: str):
        if profile_id not in self.profiles:
            return False
        self.profiles.pop(profile_id, None)
        self.discovery_snapshots.pop(profile_id, None)
        return True

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

    def list_approval_requests(self):
        return list(self.approval_requests)

    def save_approval_request(self, request):
        for index, existing in enumerate(self.approval_requests):
            if existing.request_id == request.request_id:
                self.approval_requests[index] = request
                return request
        self.approval_requests.append(request)
        return request

    def resolve_approval_request(self, request_id: str, status: str):
        for index, existing in enumerate(self.approval_requests):
            if existing.request_id != request_id:
                continue
            updated = type(existing)(
                request_id=existing.request_id,
                action_name=existing.action_name,
                resolved_action_id=existing.resolved_action_id,
                registry_capability_id=existing.registry_capability_id,
                payload=existing.payload,
                payload_fingerprint=existing.payload_fingerprint,
                status=status,
                matched_rule_id=existing.matched_rule_id,
                notes=existing.notes,
                created_at=existing.created_at,
                updated_at=existing.updated_at,
                resolved_at=existing.resolved_at,
            )
            self.approval_requests[index] = updated
            return updated
        return None

    def delete_approval_request(self, request_id: str):
        original_count = len(self.approval_requests)
        self.approval_requests = [
            request
            for request in self.approval_requests
            if request.request_id != request_id
        ]
        return len(self.approval_requests) != original_count

    def list_runtime_activity(self, limit: int = 20):
        normalized_limit = max(1, int(limit or 20))
        return list(reversed(self.runtime_activity[-normalized_limit:]))

    def record_runtime_activity(self, entry, limit: int = 50):
        saved_entry = dict(entry)
        saved_entry.setdefault(
            "activity_id", f"activity-{len(self.runtime_activity) + 1}"
        )
        self.runtime_activity.append(saved_entry)
        if len(self.runtime_activity) > limit:
            self.runtime_activity = self.runtime_activity[-limit:]
        return self.runtime_activity[-1]


class FakeLocalRuntimeDelegate:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_status(self):
        payload = {
            "server_id": "local:tldw_chatbook",
            "server_label": "tldw_chatbook local MCP",
            "mcp_sdk_available": False,
            "tool_count": 2,
            "resource_count": 1,
            "prompt_count": 1,
        }
        self.calls.append(("status.get", payload))
        return payload

    def get_protocol_capabilities(self):
        return {
            "adapter": "direct_in_process",
            "supports_batch": True,
            "request_methods": [
                "initialize",
                "status/get",
                "tools/list",
                "resources/list",
                "prompts/list",
                "tools/call",
                "resources/read",
                "prompts/get",
            ],
            # Item 4 (PR-T3 fix round D): mirrors the real delegate's new
            # `unavailable_request_methods` field (`_UNAVAILABLE_DIRECT_
            # METHODS`) -- this fake's own shape must not drift from it.
            "unavailable_request_methods": ["tools/call"],
        }

    def get_protocol_diagnostics(self):
        return {
            "adapter": "direct_in_process",
            "protocol_version": "2025-03-26",
            "transport": "in_process",
            "mcp_sdk_available": False,
            "supports_batch": True,
            "methods": [
                {"name": "tools/list", "supported": True},
                # Fix Round C, Item 1: `tools/call` is refused
                # unconditionally by the real delegate's `request()`, so
                # diagnostics must not advertise it as callable -- this
                # fake mirrors that shape (`_UNAVAILABLE_DIRECT_METHODS`).
                {"name": "tools/call", "supported": False},
            ],
            "manifest": {"tools": 2, "resources": 1, "prompts": 1},
            "implementation": {
                "tools": {
                    "implemented": ["search_notes"],
                    "unavailable": ["chat_with_llm"],
                    "missing": [],
                },
                "resources": {
                    "supported_uri_prefixes": ["note://"],
                },
                "prompts": {
                    "implemented": ["summarize_conversation"],
                    "missing": [],
                },
            },
        }

    def get_runtime_health(self):
        return {
            "state": "ready",
            "adapter": "direct_in_process",
            "transport": "in_process",
            "mcp_sdk_available": False,
            "initialized_at": "2026-04-23T00:00:00+00:00",
            "uptime_seconds": 12.5,
            "manifest": {
                "loaded": True,
                "tools": 2,
                "resources": 1,
                "prompts": 1,
            },
            "component_cache": {
                "tools_loaded": False,
                "resources_loaded": False,
                "prompts_loaded": False,
            },
            "issues": [],
        }

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, object] | None = None
    ):
        payload = {"tool_name": tool_name, "arguments": dict(arguments or {})}
        self.calls.append(("tool.execute", payload))
        return {"ok": True, **payload}

    async def read_resource(self, resource_uri: str):
        payload = {"resource_uri": resource_uri}
        self.calls.append(("resource.read", payload))
        return {
            "uri": resource_uri,
            "mimeType": "text/plain",
            "content": "resource-body",
        }

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, object] | None = None
    ):
        payload = {"prompt_name": prompt_name, "arguments": dict(arguments or {})}
        self.calls.append(("prompt.get", payload))
        return [{"role": "assistant", "content": f"prompt:{prompt_name}"}]

    async def request(self, method: str, params: dict[str, object] | None = None):
        payload = {"method": method, "params": dict(params or {})}
        self.calls.append(("runtime.request", payload))
        return {"method": method, "echo": dict(params or {})}

    async def batch(self, requests: list[dict[str, object]]):
        payload = {"requests": [dict(item) for item in requests]}
        self.calls.append(("runtime.batch", payload))
        return [
            {
                "index": index,
                "method": request.get("method"),
                "ok": True,
            }
            for index, request in enumerate(requests)
        ]


def test_fake_local_runtime_delegate_protocol_shapes_match_the_real_delegate():
    """Item 3 (PR-T3 fix round F). `FakeLocalRuntimeDelegate.get_protocol_
    capabilities()`/`get_protocol_diagnostics()` above are hand-maintained
    to mirror `LocalMCPRuntimeDelegate`'s real methods of the same name --
    enforced by nothing but a comment at each ("this fake's own shape must
    not drift", "fake mirrors that shape"). Not hypothetical on this
    branch: an earlier round's two fake-based tests stayed GREEN while the
    real-delegate test went RED for the identical scenario, and the
    fake-based assertions were briefly mistaken (by a reviewer, and in
    relaying it) for real-delegate coverage. A fake that drifts from
    reality is how a suite ends up certifying a lie.

    Pins the top-level key SETS only, not values -- the fake's values are
    hardcoded literals (`protocol_version="2025-03-26"`,
    `mcp_sdk_available=False`, ...) while the real ones are computed
    (`self._PROTOCOL_VERSION`, `MCP_AVAILABLE`, a manifest-derived
    `methods`/`implementation`), so those legitimately differ and this
    test does not compare them. What must never drift is WHICH FIELDS
    exist: a future change to either real method (a renamed or added key)
    that this fake does not also get would otherwise stay invisible to
    every test written against the fake, exactly as it did before.

    Fix Round H (PR-T3 review), Item 2a: `get_runtime_health()` is a THIRD
    hand-mirrored method of this same fake/real pair, previously pinned by
    nothing at all -- not even the comment-only "must not drift" discipline
    the other two had. Verified before adding this line: temporarily adding
    a bogus key to the real delegate's `get_runtime_health()` left all 43
    OTHER tests across this file and `test_unified_control_plane_service.py`
    green (44 collected total; only this NEW assertion caught it) -- the
    raw-JSON Advanced dump renders whatever this method returns verbatim to
    the user, so a silently-diverged fake would certify a rendering
    scenario a real run could never produce. Same policy as the two lines
    above: key SETS only."""
    real_delegate = LocalMCPRuntimeDelegate(manifest_provider=lambda: {})
    fake_delegate = FakeLocalRuntimeDelegate()

    assert set(fake_delegate.get_protocol_capabilities().keys()) == set(
        real_delegate.get_protocol_capabilities().keys()
    )
    assert set(fake_delegate.get_protocol_diagnostics().keys()) == set(
        real_delegate.get_protocol_diagnostics().keys()
    )
    assert set(fake_delegate.get_runtime_health().keys()) == set(
        real_delegate.get_runtime_health().keys()
    )


class FakeMCPClient:
    def __init__(self) -> None:
        self.connected = []
        self.disconnected = []
        self.describe_calls = []
        self.sessions = {}

    async def connect_to_server(
        self, server_id: str, command: str, args=None, env=None
    ):
        self.connected.append(
            {
                "server_id": server_id,
                "command": command,
                "args": args or [],
                "env": env or {},
            }
        )
        self.sessions[server_id] = {
            "server_id": server_id,
            "command": command,
            "args": list(args or []),
            "env": dict(env or {}),
        }
        return True

    async def describe_server(self, server_id: str):
        self.describe_calls.append(server_id)
        return {
            "server_id": server_id,
            "tools": [{"name": "remote_tool"}],
            "resources": [{"uri": "remote://resource"}],
            "prompts": [{"name": "remote_prompt"}],
        }

    async def disconnect_from_server(self, server_id: str):
        self.disconnected.append(server_id)
        self.sessions.pop(server_id, None)
        return True


class EmptySnapshotClient(FakeMCPClient):
    async def describe_server(self, server_id: str):
        return {
            "server_id": server_id,
            "tools": [],
            "resources": [],
            "prompts": [],
        }


class FakeJSONRPCStdout:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._queue.get()

    async def push_message(self, payload: dict[str, object]) -> None:
        await self._queue.put(json.dumps(payload).encode("utf-8") + b"\n")

    async def close(self) -> None:
        await self._queue.put(b"")


class FakeJSONRPCStdin:
    def __init__(self, process: "FakeJSONRPCProcess") -> None:
        self.process = process
        self._buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)

    async def drain(self) -> None:
        while b"\n" in self._buffer:
            raw_line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            if not raw_line:
                continue
            message = json.loads(raw_line.decode("utf-8"))
            self.process.client_messages.append(message)
            await self.process.handle_client_message(message)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeJSONRPCProcess:
    def __init__(self, process_id: int) -> None:
        self.process_id = process_id
        self.client_messages: list[dict[str, object]] = []
        self.stdout = FakeJSONRPCStdout()
        self.stderr = FakeJSONRPCStdout()
        self.stdin = FakeJSONRPCStdin(self)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    async def handle_client_message(self, message: dict[str, object]) -> None:
        if "id" in message and "method" in message:
            method = message["method"]
            request_id = message["id"]
            params = message.get("params", {})

            if method == "initialize":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {
                                "tools": {"listChanged": True},
                                "resources": {"listChanged": True},
                                "prompts": {"listChanged": True},
                            },
                            "serverInfo": {
                                "name": f"fake-server-{self.process_id}",
                                "version": "1.0.0",
                            },
                        },
                    }
                )
                return

            if method == "tools/list":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": f"remote_tool_{self.process_id}",
                                    "description": "Remote tool",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                )
                return

            if method == "resources/list":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "resources": [
                                {
                                    "uri": f"remote://resource/{self.process_id}",
                                    "name": "Remote Resource",
                                    "description": "Remote resource",
                                    "mimeType": "text/plain",
                                }
                            ]
                        },
                    }
                )
                return

            if method == "prompts/list":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "prompts": [
                                {
                                    "name": f"remote_prompt_{self.process_id}",
                                    "description": "Remote prompt",
                                    "arguments": [],
                                }
                            ]
                        },
                    }
                )
                return

            if method == "tools/call":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"tool-result-{params['name']}",
                                }
                            ]
                        },
                    }
                )
                return

            if method == "resources/read":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "contents": [
                                {
                                    "uri": params["uri"],
                                    "mimeType": "text/plain",
                                    "text": "resource-body",
                                }
                            ]
                        },
                    }
                )
                return

            if method == "prompts/get":
                await self.stdout.push_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": {
                                        "type": "text",
                                        "text": "prompt-body",
                                    },
                                }
                            ]
                        },
                    }
                )
                return

            await self.stdout.push_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }
            )
            return

        if message.get("method") == "notifications/initialized":
            await self.stdout.push_message(
                {
                    "jsonrpc": "2.0",
                    "id": f"server-ping-{self.process_id}",
                    "method": "ping",
                    "params": {},
                }
            )
            return

        if message.get("id") == f"server-ping-{self.process_id}":
            return

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self.stdin.close()
        asyncio.create_task(self.stdout.close())

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdin.close()
        asyncio.create_task(self.stdout.close())


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
    assert any(
        tool["description"] == "Search the RAG database for relevant content."
        for tool in inventory["tools"]
    )
    assert any(
        resource["uri"] == "note://{note_id}" for resource in inventory["resources"]
    )
    assert any(
        prompt["name"] == "summarize_conversation" for prompt in inventory["prompts"]
    )


def test_local_manifest_helper_stays_aligned_with_registered_server_surface():
    server_path = (
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "MCP" / "server.py"
    )
    module_node = ast.parse(server_path.read_text(encoding="utf-8"))

    def _registered_entries(
        method_name: str, decorator_name: str
    ) -> list[dict[str, object]]:
        for node in module_node.body:
            if isinstance(node, ast.ClassDef) and node.name == "TldwMCPServer":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == method_name:
                        entries: list[dict[str, object]] = []
                        for nested in child.body:
                            if not isinstance(nested, ast.AsyncFunctionDef):
                                continue
                            for decorator in nested.decorator_list:
                                if not isinstance(decorator, ast.Call):
                                    continue
                                func = decorator.func
                                if (
                                    not isinstance(func, ast.Attribute)
                                    or func.attr != decorator_name
                                ):
                                    continue
                                entry: dict[str, object] = {
                                    "name": nested.name,
                                    "description": (ast.get_docstring(nested) or "")
                                    .strip()
                                    .splitlines()[0]
                                    .strip(),
                                }
                                if decorator_name == "resource":
                                    entry["uri"] = decorator.args[0].value
                                if decorator_name == "tool":
                                    # Reuse the real mapper here -- this
                                    # mirror's job is pinning the AST *walk*
                                    # (does it find the same
                                    # functions/decorators), not
                                    # re-deriving the schema mapping rules a
                                    # second time.
                                    entry["inputSchema"] = _signature_to_input_schema(
                                        nested
                                    )
                                elif decorator_name == "prompt":
                                    entry["arguments"] = _signature_to_prompt_arguments(
                                        nested
                                    )
                                entries.append(entry)
                                break
                        return entries
        return []

    helper_manifest = describe_local_mcp_capabilities()
    expected_tools = _registered_entries("_register_tools", "tool")
    expected_resources = _registered_entries("_register_resources", "resource")
    expected_prompts = _registered_entries("_register_prompts", "prompt")

    # task-1337 (plan Task 9): the manifest now appends the 18
    # descriptor-backed Library tools after the AST-derived legacy entries.
    # This mirror keeps owning the legacy AST-walk alignment (first N tools,
    # resources, prompts); Tests/MCP/test_library_tools.py owns the
    # descriptor tail.
    assert helper_manifest["tools"][: len(expected_tools)] == expected_tools
    assert helper_manifest["resources"] == expected_resources
    assert helper_manifest["prompts"] == expected_prompts
    assert any(
        item["uri"] == "note://{note_id}" for item in helper_manifest["resources"]
    )
    assert any(
        item["description"] == "Search notes by content or title."
        for item in helper_manifest["tools"]
    )
    assert any(
        item["description"] == "Generate a prompt to summarize a conversation."
        for item in helper_manifest["prompts"]
    )


def test_manifest_tools_carry_input_schema():
    manifest = describe_local_mcp_capabilities()
    tools = {t["name"]: t for t in manifest["tools"]}
    # search_rag: query is required str; media_types is Optional[List[str]]
    schema = tools["search_rag"]["inputSchema"]
    assert schema["type"] == "object"
    assert schema["properties"]["query"] == {"type": "string"}
    assert "query" in schema["required"]
    mt = schema["properties"]["media_types"]
    assert mt["type"] == ["array", "null"] and mt["items"] == {"type": "string"}
    assert "media_types" not in schema["required"]


def test_every_builtin_tool_has_object_schema():
    for tool in describe_local_mcp_capabilities()["tools"]:
        assert tool["inputSchema"]["type"] == "object", tool["name"]


def test_schema_defaults_recorded():
    tools = {t["name"]: t for t in describe_local_mcp_capabilities()["tools"]}
    # search_rag's limit param defaults to 10 (int, literal default)
    prop = tools["search_rag"]["inputSchema"]["properties"]["limit"]
    assert prop.get("default") is not None


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
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )

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
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )
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
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )

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


def test_local_control_service_rejects_colon_bearing_profile_id(tmp_path):
    """I3: `save_external_profile()` is the single entry point both the
    profile-save form and the mcpServers import batch route through
    (`UnifiedMCPControlPlaneService.save_local_profile()`) -- a colon in
    profile_id would corrupt `HubTool.tool_id` routing
    (`"local:a::b::search"` partitions ambiguously), so both `"a::b"` and
    `"a:b"` must be rejected here, at the real choke point, not just at
    `LocalMCPStore.save_profile()` in isolation."""
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )

    with pytest.raises(ValueError, match="profile_id"):
        service.save_external_profile({"profile_id": "a::b", "command": "python"})

    with pytest.raises(ValueError, match="profile_id"):
        service.save_external_profile({"profile_id": "a:b", "command": "python"})

    assert store.get_profile("a::b") is None
    assert store.get_profile("a:b") is None


def test_local_control_service_rejects_invalid_governance_rule_writes(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )

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


def test_local_control_service_deletes_governance_rules(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-a",
            "capability_id": "mcp.governance.list.local",
            "decision": "allow",
        }
    )

    deleted = service.delete_governance_rule("rule-a")

    assert deleted is True
    assert store.list_governance_rules() == []


def test_local_control_service_previews_governance_decisions(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    service = LocalMCPControlService(
        store=store, client=FakeMCPClient(), manifest_provider=lambda: {}
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-a",
            "capability_id": "mcp.inventory.list.local",
            "decision": "allow",
            "notes": "Inventory is allowed locally.",
        }
    )

    matched = service.preview_governance_decision("mcp.inventory.list.local")
    unmatched = service.preview_governance_decision("mcp.advanced.observe.local")

    assert matched == {
        "source": "local",
        "capability_id": "mcp.inventory.list.local",
        "decision": "allow",
        "matched_rule_id": "rule-a",
        "notes": "Inventory is allowed locally.",
    }
    assert unmatched == {
        "source": "local",
        "capability_id": "mcp.advanced.observe.local",
        "decision": "inherit",
        "matched_rule_id": None,
        "notes": None,
    }


@pytest.mark.asyncio
async def test_local_control_service_executes_local_inventory_runtime_actions():
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )

    tool_result = await service.execute_tool("search_notes", {"query": "roadmap"})
    resource_result = await service.read_resource("note://123")
    prompt_result = await service.get_prompt(
        "summarize_conversation", {"conversation_id": 4}
    )

    assert tool_result == {
        "source": "local",
        "tool_name": "search_notes",
        "result": {
            "ok": True,
            "tool_name": "search_notes",
            "arguments": {"query": "roadmap"},
        },
        "governance": {
            "resolved_action_id": "notes.list.local",
            "registry_capability_id": "notes_workspaces",
            "decision": "inherit",
            "matched_rule_id": None,
            "notes": None,
        },
    }
    assert resource_result == {
        "source": "local",
        "resource_uri": "note://123",
        "result": {
            "uri": "note://123",
            "mimeType": "text/plain",
            "content": "resource-body",
        },
        "governance": {
            "resolved_action_id": "notes.detail.local",
            "registry_capability_id": "notes_workspaces",
            "decision": "inherit",
            "matched_rule_id": None,
            "notes": None,
        },
    }
    assert prompt_result == {
        "source": "local",
        "prompt_name": "summarize_conversation",
        "arguments": {"conversation_id": 4},
        "messages": [{"role": "assistant", "content": "prompt:summarize_conversation"}],
        "governance": {
            "resolved_action_id": "prompts.preview.local",
            "registry_capability_id": "prompts_chatbooks",
            "decision": "inherit",
            "matched_rule_id": None,
            "notes": None,
        },
    }
    assert runtime_delegate.calls == [
        (
            "tool.execute",
            {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
        ),
        ("resource.read", {"resource_uri": "note://123"}),
        (
            "prompt.get",
            {
                "prompt_name": "summarize_conversation",
                "arguments": {"conversation_id": 4},
            },
        ),
    ]


@pytest.mark.asyncio
async def test_local_control_service_exposes_runtime_status_and_protocol_helpers():
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )

    advanced = service.get_advanced()
    status = service.get_runtime_status()
    health = service.get_runtime_health()
    diagnostics = service.get_runtime_protocol_diagnostics()
    request_result = await service.run_runtime_request("tools/list", {"scope": "local"})
    batch_result = await service.run_runtime_batch(
        [
            {"method": "tools/list", "params": {}},
            {"method": "prompts/list", "params": {}},
        ]
    )

    assert advanced == {
        "source": "local",
        "section": "advanced",
        "runtime_status": {
            "server_id": "local:tldw_chatbook",
            "server_label": "tldw_chatbook local MCP",
            "mcp_sdk_available": False,
            "tool_count": 2,
            "resource_count": 1,
            "prompt_count": 1,
        },
        "runtime_health": {
            "state": "ready",
            "adapter": "direct_in_process",
            "transport": "in_process",
            "mcp_sdk_available": False,
            "initialized_at": "2026-04-23T00:00:00+00:00",
            "uptime_seconds": 12.5,
            "manifest": {
                "loaded": True,
                "tools": 2,
                "resources": 1,
                "prompts": 1,
            },
            "component_cache": {
                "tools_loaded": False,
                "resources_loaded": False,
                "prompts_loaded": False,
            },
            "issues": [],
        },
        "protocol": {
            "adapter": "direct_in_process",
            "supports_batch": True,
            "request_methods": [
                "initialize",
                "status/get",
                "tools/list",
                "resources/list",
                "prompts/list",
                "tools/call",
                "resources/read",
                "prompts/get",
            ],
            # Item 4 (PR-T3 fix round D): the fixture's `get_protocol_
            # capabilities()` (above) now returns this field too -- kept in
            # sync here since `get_advanced()` passes it through verbatim.
            "unavailable_request_methods": ["tools/call"],
        },
        "protocol_diagnostics": {
            "adapter": "direct_in_process",
            "protocol_version": "2025-03-26",
            "transport": "in_process",
            "mcp_sdk_available": False,
            "supports_batch": True,
            "methods": [
                {"name": "tools/list", "supported": True},
                # Fix Round C, Item 1: `tools/call` is refused
                # unconditionally by the real delegate's `request()`, so
                # diagnostics must not advertise it as callable -- this
                # fake mirrors that shape (`_UNAVAILABLE_DIRECT_METHODS`).
                {"name": "tools/call", "supported": False},
            ],
            "manifest": {"tools": 2, "resources": 1, "prompts": 1},
            "implementation": {
                "tools": {
                    "implemented": ["search_notes"],
                    "unavailable": ["chat_with_llm"],
                    "missing": [],
                },
                "resources": {
                    "supported_uri_prefixes": ["note://"],
                },
                "prompts": {
                    "implemented": ["summarize_conversation"],
                    "missing": [],
                },
            },
        },
        "governance": {
            "rules": 0,
            "deny_rules": 0,
            "allow_rules": 0,
        },
    }
    assert status == {
        "source": "local",
        "status": {
            "server_id": "local:tldw_chatbook",
            "server_label": "tldw_chatbook local MCP",
            "mcp_sdk_available": False,
            "tool_count": 2,
            "resource_count": 1,
            "prompt_count": 1,
        },
    }
    assert health == {
        "source": "local",
        "health": advanced["runtime_health"],
    }
    assert diagnostics == {
        "source": "local",
        "diagnostics": advanced["protocol_diagnostics"],
    }
    assert request_result == {
        "source": "local",
        "method": "tools/list",
        "params": {"scope": "local"},
        "result": {"method": "tools/list", "echo": {"scope": "local"}},
        "governance": {
            "resolved_action_id": "mcp.inventory.list.local",
            "registry_capability_id": "local_mcp_runtime",
            "decision": "inherit",
            "matched_rule_id": None,
            "notes": None,
        },
    }
    assert batch_result == {
        "source": "local",
        "results": [
            {
                "index": 0,
                "method": "tools/list",
                "ok": True,
                "result": {"method": "tools/list", "echo": {}},
                "governance": {
                    "resolved_action_id": "mcp.inventory.list.local",
                    "registry_capability_id": "local_mcp_runtime",
                    "decision": "inherit",
                    "matched_rule_id": None,
                    "notes": None,
                },
            },
            {
                "index": 1,
                "method": "prompts/list",
                "ok": True,
                "result": {"method": "prompts/list", "echo": {}},
                "governance": {
                    "resolved_action_id": "mcp.inventory.list.local",
                    "registry_capability_id": "local_mcp_runtime",
                    "decision": "inherit",
                    "matched_rule_id": None,
                    "notes": None,
                },
            },
        ],
    }
    assert runtime_delegate.calls == [
        (
            "status.get",
            {
                "server_id": "local:tldw_chatbook",
                "server_label": "tldw_chatbook local MCP",
                "mcp_sdk_available": False,
                "tool_count": 2,
                "resource_count": 1,
                "prompt_count": 1,
            },
        ),
        (
            "status.get",
            {
                "server_id": "local:tldw_chatbook",
                "server_label": "tldw_chatbook local MCP",
                "mcp_sdk_available": False,
                "tool_count": 2,
                "resource_count": 1,
                "prompt_count": 1,
            },
        ),
        ("runtime.request", {"method": "tools/list", "params": {"scope": "local"}}),
        ("runtime.request", {"method": "tools/list", "params": {}}),
        ("runtime.request", {"method": "prompts/list", "params": {}}),
    ]


def test_local_runtime_delegate_builds_protocol_diagnostics_from_manifest():
    delegate = LocalMCPRuntimeDelegate(
        manifest_provider=lambda: {
            "server_id": "local:test",
            "server_label": "Test MCP",
            "tools": [
                {"name": "search_notes"},
                {"name": "chat_with_llm"},
                {"name": "missing_tool"},
            ],
            "resources": [
                {"uri": "note://{id}"},
                {"uri": "conversation://{id}"},
            ],
            "prompts": [
                {"name": "summarize_conversation"},
                {"name": "missing_prompt"},
            ],
        }
    )

    diagnostics = delegate.get_protocol_diagnostics()

    assert diagnostics["protocol_version"] == "2025-03-26"
    assert diagnostics["transport"] == "in_process"
    assert diagnostics["manifest"] == {"tools": 3, "resources": 2, "prompts": 2}
    assert diagnostics["methods"][0] == {"name": "initialize", "supported": True}
    assert diagnostics["implementation"]["tools"] == {
        "implemented": ["search_notes"],
        "unavailable": ["chat_with_llm"],
        "missing": ["missing_tool"],
    }
    assert diagnostics["implementation"]["resources"]["supported_uri_prefixes"] == [
        "conversation://",
        "note://",
    ]
    assert diagnostics["implementation"]["prompts"] == {
        "implemented": ["summarize_conversation"],
        "missing": ["missing_prompt"],
    }


def test_local_runtime_delegate_reports_runtime_health_from_lifecycle():
    delegate = LocalMCPRuntimeDelegate(
        manifest_provider=lambda: {
            "tools": [{"name": "search_notes"}],
            "resources": [{"uri": "note://{id}"}],
            "prompts": [{"name": "summarize_conversation"}],
        }
    )

    health = delegate.get_runtime_health()

    assert health["state"] == "ready"
    assert health["adapter"] == "direct_in_process"
    assert health["transport"] == "in_process"
    assert health["manifest"] == {
        "loaded": True,
        "tools": 1,
        "resources": 1,
        "prompts": 1,
    }
    assert health["component_cache"] == {
        "tools_loaded": False,
        "resources_loaded": False,
        "prompts_loaded": False,
    }
    assert isinstance(health["initialized_at"], str)
    assert health["uptime_seconds"] >= 0
    assert health["issues"] == []


@pytest.mark.asyncio
async def test_governance_denial_raises_the_typed_mcp_governance_denied_error():
    """task-2537 (PR-T3 fix round B, item 3): `_require_runtime_
    governance_allowed()`'s deny branch raises the dedicated
    `MCPGovernanceDenied` subclass, not a bare `PermissionError` -- so
    `mcp_workbench._is_permission_refusal()` can tell "governance refused
    this call" apart from an unrelated `PermissionError` a tool's own body
    might raise. Also proves the existing `except PermissionError`
    contract still holds (the type IS a `PermissionError`)."""
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-deny-notes-list",
            "capability_id": "notes.list.local",
            "decision": "deny",
            "notes": "Local note listing is blocked.",
        }
    )

    from tldw_chatbook.MCP.local_control_service import MCPGovernanceDenied

    with pytest.raises(MCPGovernanceDenied) as exc_info:
        await service.execute_tool("search_notes", {"query": "roadmap"})

    assert isinstance(exc_info.value, PermissionError)
    assert str(exc_info.value) == "Denied by local governance: notes.list.local"


@pytest.mark.asyncio
async def test_local_control_service_previews_and_enforces_runtime_governance():
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-deny-notes-list",
            "capability_id": "notes.list.local",
            "decision": "deny",
            "notes": "Local note listing is blocked.",
        }
    )

    preview = service.preview_runtime_access(
        "tool.execute",
        {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
    )

    assert preview == {
        "source": "local",
        "action_name": "tool.execute",
        "resolved_action_id": "notes.list.local",
        "registry_capability_id": "notes_workspaces",
        "decision": "deny",
        "matched_rule_id": "rule-deny-notes-list",
        "notes": "Local note listing is blocked.",
    }

    with pytest.raises(PermissionError, match="notes.list.local"):
        await service.execute_tool("search_notes", {"query": "roadmap"})

    with pytest.raises(PermissionError, match="notes.list.local"):
        await service.run_runtime_request(
            "tools/call",
            {"name": "search_notes", "arguments": {"query": "roadmap"}},
        )

    batch_result = await service.run_runtime_batch(
        [
            {
                "method": "tools/call",
                "params": {"name": "search_notes", "arguments": {"query": "roadmap"}},
            },
            {"method": "tools/list", "params": {}},
        ]
    )

    assert batch_result == {
        "source": "local",
        "results": [
            {
                "index": 0,
                "method": "tools/call",
                "ok": False,
                "blocked": True,
                "error": "Denied by local governance: notes.list.local",
                "governance": {
                    "resolved_action_id": "notes.list.local",
                    "registry_capability_id": "notes_workspaces",
                    "decision": "deny",
                    "matched_rule_id": "rule-deny-notes-list",
                    "notes": "Local note listing is blocked.",
                },
            },
            {
                "index": 1,
                "method": "tools/list",
                "ok": True,
                "result": {"method": "tools/list", "echo": {}},
                "governance": {
                    "resolved_action_id": "mcp.inventory.list.local",
                    "registry_capability_id": "local_mcp_runtime",
                    "decision": "inherit",
                    "matched_rule_id": None,
                    "notes": None,
                },
            },
        ],
    }
    assert runtime_delegate.calls == [
        ("runtime.request", {"method": "tools/list", "params": {}}),
    ]


@pytest.mark.asyncio
async def test_local_control_service_creates_and_resolves_local_runtime_approvals():
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-ask-notes-list",
            "capability_id": "notes.list.local",
            "decision": "ask",
            "notes": "Approval required for local note listing.",
        }
    )

    initial_preview = service.preview_runtime_access(
        "tool.execute",
        {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
    )

    with pytest.raises(PermissionError, match="Approval required"):
        await service.execute_tool("search_notes", {"query": "roadmap"})

    approval_requests = service.list_approval_requests()
    pending_preview = service.preview_runtime_access(
        "tool.execute",
        {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
    )

    approved_request = service.approve_approval_request(
        approval_requests[0]["request_id"]
    )
    tool_result = await service.execute_tool("search_notes", {"query": "roadmap"})

    assert initial_preview == {
        "source": "local",
        "action_name": "tool.execute",
        "resolved_action_id": "notes.list.local",
        "registry_capability_id": "notes_workspaces",
        "decision": "ask",
        "matched_rule_id": "rule-ask-notes-list",
        "notes": "Approval required for local note listing.",
        "approval_request_id": None,
        "approval_status": None,
    }
    assert approval_requests[0]["status"] == "pending"
    assert pending_preview["approval_status"] == "pending"
    assert pending_preview["approval_request_id"] == approval_requests[0]["request_id"]
    assert approved_request["status"] == "approved"
    assert (
        service.list_approval_requests(status="approved")[0]["request_id"]
        == approval_requests[0]["request_id"]
    )
    assert service.list_approval_requests(status="pending") == []
    assert (
        service.list_approval_requests(resolved_action_id="notes.list.local")[0][
            "status"
        ]
        == "approved"
    )
    assert tool_result["governance"] == {
        "resolved_action_id": "notes.list.local",
        "registry_capability_id": "notes_workspaces",
        "decision": "ask",
        "matched_rule_id": "rule-ask-notes-list",
        "notes": "Approval required for local note listing.",
        "approval_request_id": approval_requests[0]["request_id"],
        "approval_status": "approved",
    }
    assert runtime_delegate.calls == [
        (
            "tool.execute",
            {"tool_name": "search_notes", "arguments": {"query": "roadmap"}},
        ),
    ]
    assert service.delete_approval_request(approval_requests[0]["request_id"]) is True
    assert service.list_approval_requests() == []


@pytest.mark.asyncio
async def test_local_control_service_exposes_recent_runtime_activity():
    runtime_delegate = FakeLocalRuntimeDelegate()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=runtime_delegate,
    )
    service.save_governance_rule(
        {
            "rule_id": "rule-deny-notes-list",
            "capability_id": "notes.list.local",
            "decision": "deny",
            "notes": "Local note listing is blocked.",
        }
    )

    await service.run_runtime_request("tools/list", {})

    with pytest.raises(PermissionError, match="Denied by local governance"):
        await service.execute_tool("search_notes", {"query": "roadmap"})

    activity = service.get_runtime_activity(limit=5)
    advanced = service.get_advanced()

    assert activity == {
        "source": "local",
        "limit": 5,
        "entries": [
            {
                "activity_id": "activity-2",
                "action_name": "tool.execute",
                "target": "search_notes",
                "ok": False,
                "blocked": True,
                "error": "Denied by local governance: notes.list.local",
                "resolved_action_id": "notes.list.local",
                "decision": "deny",
                "matched_rule_id": "rule-deny-notes-list",
                "approval_request_id": None,
                "approval_status": None,
                "occurred_at": activity["entries"][0]["occurred_at"],
            },
            {
                "activity_id": "activity-1",
                "action_name": "runtime.request",
                "target": "tools/list",
                "ok": True,
                "blocked": False,
                "error": None,
                "resolved_action_id": "mcp.inventory.list.local",
                "decision": "inherit",
                "matched_rule_id": None,
                "approval_request_id": None,
                "approval_status": None,
                "occurred_at": activity["entries"][1]["occurred_at"],
            },
        ],
    }
    assert advanced["recent_activity_count"] == 2
    assert len(advanced["recent_activity"]) == 2
    assert advanced["recent_activity"][0]["action_name"] == "tool.execute"
    assert advanced["recent_activity"][1]["action_name"] == "runtime.request"


@pytest.mark.asyncio
async def test_local_control_service_persists_runtime_activity_across_instances(
    tmp_path,
):
    store_path = tmp_path / "local_mcp_store.json"
    first_service = LocalMCPControlService(
        store=LocalMCPStore(store_path),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=FakeLocalRuntimeDelegate(),
    )

    await first_service.run_runtime_request("tools/list", {})

    second_service = LocalMCPControlService(
        store=LocalMCPStore(store_path),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=FakeLocalRuntimeDelegate(),
    )

    activity = second_service.get_runtime_activity(limit=5)
    advanced = second_service.get_advanced()

    assert activity["entries"] == [
        {
            "activity_id": activity["entries"][0]["activity_id"],
            "action_name": "runtime.request",
            "target": "tools/list",
            "ok": True,
            "blocked": False,
            "error": None,
            "resolved_action_id": "mcp.inventory.list.local",
            "decision": "inherit",
            "matched_rule_id": None,
            "approval_request_id": None,
            "approval_status": None,
            "occurred_at": activity["entries"][0]["occurred_at"],
        }
    ]
    assert advanced["recent_activity_count"] == 1
    assert advanced["recent_activity"][0]["action_name"] == "runtime.request"


@pytest.mark.asyncio
async def test_local_control_service_connects_profile_and_persists_discovery_snapshot():
    store = FakeLocalStore()
    client = FakeMCPClient()
    service = LocalMCPControlService(
        store=store, client=client, manifest_provider=lambda: {}
    )

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
    service = LocalMCPControlService(
        store=store, client=client, manifest_provider=lambda: {}
    )

    with patch.dict(os.environ, {"API_KEY": "resolved-api-key"}, clear=True):
        with pytest.raises(RuntimeError):
            await service.connect_profile("profile-a")

    assert "profile-a" not in store.discovery_snapshots


@pytest.mark.asyncio
async def test_local_control_service_tests_refreshes_disconnects_and_deletes_external_profiles():
    store = FakeLocalStore()
    client = FakeMCPClient()
    service = LocalMCPControlService(
        store=store, client=client, manifest_provider=lambda: {}
    )

    with patch.dict(
        os.environ, {"API_KEY": "resolved-api-key", "PATH": "/usr/bin"}, clear=True
    ):
        test_result = await service.test_external_profile("profile-a")
        await service.connect_profile("profile-a")
        refreshed = await service.refresh_external_profile("profile-a")
        disconnected = await service.disconnect_profile("profile-a")

    deleted = service.delete_external_profile("profile-a")

    assert test_result["ok"] is True
    assert test_result["profile_id"] == "profile-a"
    assert refreshed["tools"][0]["name"] == "remote_tool"
    assert disconnected is True
    assert deleted is True
    assert client.disconnected.count("profile-a") >= 2
    assert store.get_profile("profile-a") is None
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
            "tools": [
                type(
                    "Tool",
                    (),
                    {
                        "name": "remote_tool",
                        "description": "Remote tool",
                        "inputSchema": {},
                    },
                )()
            ],
            "resources": [
                type(
                    "Resource",
                    (),
                    {
                        "uri": "remote://resource",
                        "name": "Remote Resource",
                        "description": "Remote resource",
                        "mimeType": "text/plain",
                    },
                )()
            ],
            "prompts": [
                type(
                    "Prompt",
                    (),
                    {
                        "name": "remote_prompt",
                        "description": "Remote prompt",
                        "arguments": [],
                    },
                )()
            ],
        }
    }

    description = await client.describe_server("profile-a")

    assert description["tools"][0]["name"] == "remote_tool"
    assert description["resources"][0]["uri"] == "remote://resource"
    assert description["prompts"][0]["name"] == "remote_prompt"


@pytest.mark.asyncio
async def test_mcp_client_connect_to_server_uses_stdio_jsonrpc_flow(monkeypatch):
    created_processes: list[FakeJSONRPCProcess] = []

    async def fake_create_subprocess_exec(command, *args, **kwargs):
        process = FakeJSONRPCProcess(process_id=len(created_processes) + 1)
        process.spawn = {
            "command": command,
            "args": list(args),
            "env": kwargs.get("env"),
        }
        created_processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = mcp_client_module.MCPClient(name="test-client")

    connected = await client.connect_to_server(
        "profile-a",
        "python",
        args=["-m", "demo.server"],
        env={"API_KEY": "resolved-api-key"},
    )

    assert connected is True
    assert len(created_processes) == 1
    process = created_processes[0]
    assert process.spawn == {
        "command": "python",
        "args": ["-m", "demo.server"],
        "env": {"API_KEY": "resolved-api-key"},
    }
    methods = [
        message["method"] for message in process.client_messages if "method" in message
    ]
    assert methods[:2] == ["initialize", "notifications/initialized"]
    assert "tools/list" in methods
    assert "resources/list" in methods
    assert "prompts/list" in methods
    assert process.client_messages[0]["params"]["clientInfo"]["name"] == "test-client"
    assert process.client_messages[1] == {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    ping_response = next(
        message
        for message in process.client_messages
        if message.get("id") == "server-ping-1"
    )
    assert ping_response["id"] == "server-ping-1"
    assert ping_response["result"] == {}
    assert client.servers["profile-a"]["command"] == "python"
    assert client.servers["profile-a"]["args"] == ["-m", "demo.server"]
    assert client.servers["profile-a"]["tools"][0].name == "remote_tool_1"
    assert client.servers["profile-a"]["resources"][0].uri == "remote://resource/1"
    assert client.servers["profile-a"]["prompts"][0].name == "remote_prompt_1"


@pytest.mark.asyncio
async def test_mcp_client_connect_to_server_cleans_up_existing_connection_for_same_server_id(
    monkeypatch,
):
    created_processes: list[FakeJSONRPCProcess] = []

    async def fake_create_subprocess_exec(command, *args, **kwargs):
        process = FakeJSONRPCProcess(process_id=len(created_processes) + 1)
        process.spawn = {
            "command": command,
            "args": list(args),
            "env": kwargs.get("env"),
        }
        created_processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = mcp_client_module.MCPClient(name="test-client")

    first_connected = await client.connect_to_server(
        "profile-a", "python", args=["-m", "demo.one"]
    )
    second_connected = await client.connect_to_server(
        "profile-a", "python", args=["-m", "demo.two"]
    )

    assert first_connected is True
    assert second_connected is True
    assert len(created_processes) == 2
    assert created_processes[0].terminated is True
    assert created_processes[0].stdin.closed is True
    assert created_processes[1].terminated is False
    assert client.servers["profile-a"]["args"] == ["-m", "demo.two"]
    assert client.servers["profile-a"]["tools"][0].name == "remote_tool_2"
    assert client.servers["profile-a"]["resources"][0].uri == "remote://resource/2"
    assert client.servers["profile-a"]["prompts"][0].name == "remote_prompt_2"


@pytest.mark.asyncio
async def test_mcp_client_tool_resource_and_prompt_calls_use_jsonrpc_requests(
    monkeypatch,
):
    created_processes: list[FakeJSONRPCProcess] = []

    async def fake_create_subprocess_exec(command, *args, **kwargs):
        process = FakeJSONRPCProcess(process_id=len(created_processes) + 1)
        process.spawn = {
            "command": command,
            "args": list(args),
            "env": kwargs.get("env"),
        }
        created_processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = mcp_client_module.MCPClient(name="test-client")
    connected = await client.connect_to_server(
        "profile-a", "python", args=["-m", "demo.server"]
    )

    assert connected is True

    tool_result = await client.call_tool(
        "profile-a", "remote_tool_1", {"topic": "news"}
    )
    resource_result = await client.read_resource("profile-a", "remote://resource/1")
    prompt_result = await client.get_prompt(
        "profile-a", "remote_prompt_1", {"topic": "news"}
    )

    assert tool_result == {
        "result": [{"type": "text", "text": "tool-result-remote_tool_1"}]
    }
    assert resource_result == {
        "uri": "remote://resource/1",
        "content": "resource-body",
        "mimeType": "text/plain",
        "_meta": {},
    }
    assert prompt_result == [{"role": "assistant", "content": "prompt-body"}]

    methods = [
        message["method"]
        for message in created_processes[0].client_messages
        if "method" in message
    ]
    assert "tools/call" in methods
    assert "resources/read" in methods
    assert "prompts/get" in methods


# -- task-1337 (plan Task 9): Library tool policy mapping ----------------------
#
# Each of the 18 descriptor-backed ``library_*`` tools resolves to a read
# action owned by its Library item type (list/search -> the type's ``list``
# action, get -> its ``detail`` action). Collections get the dedicated
# local-only ``library.collections`` resource -- never
# ``collections.reading_list.*``. Both ``tool.execute`` previews and
# ``tools/call`` runtime requests resolve through the same mapping, with the
# generic MCP trigger retained as the fallback seam for unknown names.

_LIBRARY_TOOL_POLICY_EXPECTATIONS = {
    "library_list_media": ("media.reading.list.local", "media_reading_ingestion_sources"),
    "library_get_media": ("media.reading.detail.local", "media_reading_ingestion_sources"),
    "library_search_media": ("media.reading.list.local", "media_reading_ingestion_sources"),
    "library_list_notes": ("notes.list.local", "notes_workspaces"),
    "library_get_note": ("notes.detail.local", "notes_workspaces"),
    "library_search_notes": ("notes.list.local", "notes_workspaces"),
    "library_list_prompts": ("prompts.list.local", "prompts_chatbooks"),
    "library_get_prompt": ("prompts.detail.local", "prompts_chatbooks"),
    "library_search_prompts": ("prompts.list.local", "prompts_chatbooks"),
    "library_list_skills": ("skills.list.local", "server_skills"),
    "library_get_skill": ("skills.detail.local", "server_skills"),
    "library_search_skills": ("skills.list.local", "server_skills"),
    "library_list_conversations": ("chat.list.local", "chat"),
    "library_get_conversation": ("chat.detail.local", "chat"),
    "library_search_conversations": ("chat.list.local", "chat"),
    "library_list_collections": ("library.collections.list.local", "library_collections"),
    "library_get_collection": ("library.collections.detail.local", "library_collections"),
    "library_search_collections": ("library.collections.list.local", "library_collections"),
    # chunking-agent-tools siblings: the read tools ride the existing media
    # read path (spec §6 "no new verbs") at the matching LEVEL -- structure
    # and chunk fetch are single-item detail reads (by id, like get), spec
    # list is a browse-level listing. spec_save resolves to its OWN write
    # action (Task 4's deadline carry: the moment the save handler went
    # live, the provisional derived READ mapping had to stop -- a live
    # write must never resolve to a read action under policy).
    "library_get_media_structure": ("media.reading.detail.local", "media_reading_ingestion_sources"),
    "library_get_media_chunk": ("media.reading.detail.local", "media_reading_ingestion_sources"),
    "library_list_chunk_specs": ("media.reading.list.local", "media_reading_ingestion_sources"),
    "library_save_chunk_spec": ("library.templates.save.local", "library_collections"),
    "library_rechunk_media": ("library.media.rechunk.local", "library_collections"),
    # student-workflow (Task 1, spec §4): the note write resolves to its OWN
    # local Library write action -- never the derived notes read a note-typed
    # tool would otherwise fall to.
    "library_save_note": ("library.notes.save.local", "library_collections"),
}


def _library_policy_control_service() -> LocalMCPControlService:
    return LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        runtime_delegate=FakeLocalRuntimeDelegate(),
    )


@pytest.mark.parametrize(
    "tool_name,expected",
    sorted(_LIBRARY_TOOL_POLICY_EXPECTATIONS.items()),
    ids=sorted(_LIBRARY_TOOL_POLICY_EXPECTATIONS),
)
def test_library_tools_resolve_to_type_owned_read_actions(tool_name, expected):
    service = _library_policy_control_service()
    expected_action_id, expected_capability_id = expected

    preview = service.preview_runtime_access(
        "tool.execute", {"tool_name": tool_name, "arguments": {}}
    )

    assert preview["resolved_action_id"] == expected_action_id
    assert preview["registry_capability_id"] == expected_capability_id
    assert preview["decision"] == "inherit"


@pytest.mark.parametrize(
    "tool_name,expected",
    sorted(_LIBRARY_TOOL_POLICY_EXPECTATIONS.items()),
    ids=sorted(_LIBRARY_TOOL_POLICY_EXPECTATIONS),
)
def test_library_tools_call_requests_use_the_same_mapping(tool_name, expected):
    service = _library_policy_control_service()
    expected_action_id, expected_capability_id = expected

    preview = service.preview_runtime_access(
        "runtime.request",
        {
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": {}},
        },
    )

    assert preview["resolved_action_id"] == expected_action_id
    assert preview["registry_capability_id"] == expected_capability_id


def test_unknown_tools_keep_the_generic_mcp_trigger_fallback():
    service = _library_policy_control_service()

    preview = service.preview_runtime_access(
        "tool.execute", {"tool_name": "not_a_real_tool", "arguments": {}}
    )

    assert preview["resolved_action_id"] == "mcp.runtime.trigger.local"
    assert preview["registry_capability_id"] == "local_mcp_runtime"


@pytest.mark.asyncio
async def test_library_collections_deny_rule_blocks_execute_and_tools_call():
    service = _library_policy_control_service()
    service.save_governance_rule(
        {
            "rule_id": "rule-deny-library-collections",
            "capability_id": "library.collections.list.local",
            "decision": "deny",
            "notes": "Library collection listing is blocked.",
        }
    )

    with pytest.raises(PermissionError, match="library.collections.list.local"):
        await service.execute_tool("library_list_collections", {})

    with pytest.raises(PermissionError, match="library.collections.list.local"):
        await service.run_runtime_request(
            "tools/call",
            {"name": "library_search_collections", "arguments": {"query": "x"}},
        )


def test_control_service_forwards_its_policy_enforcer_to_the_default_delegate():
    """Task 5 (chunking-agent-tools, spec §6): the control service's
    enforcer rides into the runtime delegate it builds by default, so the
    lazily-composed shared Library service (and through it the chunk tools)
    is gated on the local MCP surface."""
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    enforcer = object()
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=lambda: {},
        policy_enforcer=enforcer,
    )

    assert service.runtime_delegate._policy_enforcer is enforcer
