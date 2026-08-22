"""Composition, wire-protocol, and process coverage for standalone MCP stdio."""

from __future__ import annotations

import asyncio
import ast
from collections import deque
from collections.abc import Callable
import io
import json
import os
from pathlib import Path
from queue import Empty, Queue
import runpy
from types import SimpleNamespace
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayLimits = gateway.GatewayLimits
GatewayProtocolConnection = gateway.GatewayProtocolConnection
GatewayRequestContext = gateway.GatewayRequestContext
serve_stdio = gateway.serve_stdio

from tldw_chatbook.Agents.agent_models import ToolResult  # noqa: E402
from tldw_chatbook.config import CLI_APP_CLIENT_ID  # noqa: E402
from tldw_chatbook.Library.library_tool_contract import (  # noqa: E402
    LIBRARY_TOOL_DESCRIPTORS,
)
from tldw_chatbook.MCP import client as client_module  # noqa: E402
from tldw_chatbook.MCP import server as server_module  # noqa: E402
from tldw_chatbook.MCP.gateway_runtime import (  # noqa: E402
    ChatbookGatewayRuntime,
)
from tldw_chatbook.MCP.local_server_tools import (  # noqa: E402
    LocalToolRegistration,
)
from tldw_chatbook.MCP.client import MCPClient  # noqa: E402
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT_PATH = REPO_ROOT / "tldw_chatbook" / "MCP" / "__main__.py"
SHARED_VENV_PYTHON = Path(sys.executable).absolute()
LEGACY_VERSIONS = ("2025-03-26", "2025-11-25")
CURRENT_VERSION = "2026-07-28"
PROTOCOL_VERSIONS = (*LEGACY_VERSIONS, CURRENT_VERSION)
BUILTIN_TOOL_NAMES = [
    "chat_with_llm",
    "chat_with_character",
    "search_rag",
    "search_conversations",
    "create_note",
    "search_notes",
    "list_characters",
    "get_conversation_history",
    "export_conversation",
]
RESOURCE_TEMPLATES = [
    "conversation://{conversation_id}",
    "note://{note_id}",
    "character://{character_id}",
    "media://{media_id}",
    "rag-chunk://{chunk_uuid}",
]
PROMPT_NAMES = [
    "summarize_conversation",
    "generate_document",
    "analyze_media",
    "search_and_synthesize",
    "character_writing",
]
LONG_RESOURCE_TEXT = "é" * 140_000
_SUBPROCESS_HOST_ENV_KEYS = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TZ",
    "WINDIR",
)
_PROCESS_EOF = object()


class _FakeTools:
    async def list_available_characters(self) -> list[dict[str, Any]]:
        return [{"id": 7, "name": "Ada"}]

    async def get_conversation_history(self, **_arguments: Any) -> str:
        return "plain-result"

    async def export_conversation(self, **_arguments: Any) -> dict[str, Any]:
        return {"format": "markdown", "content": "# Fixture"}


class _FakeResources:
    async def get_conversation_resource(self, conversation_id: str) -> dict[str, Any]:
        return {
            "uri": f"conversation://{conversation_id}",
            "name": "Long fixture",
            "mimeType": "text/plain",
            "content": LONG_RESOURCE_TEXT,
            "metadata": {"kind": "fixture"},
        }

    async def get_note_resource(self, note_id: str) -> dict[str, Any]:
        return self._small_resource("note", note_id)

    async def get_character_resource(self, character_id: str) -> dict[str, Any]:
        return self._small_resource("character", character_id)

    async def get_media_resource(self, media_id: str) -> dict[str, Any]:
        return self._small_resource("media", media_id)

    async def get_rag_chunk_resource(self, chunk_uuid: str) -> dict[str, Any]:
        return self._small_resource("rag-chunk", chunk_uuid)

    async def list_recent_conversations(self, *, limit: int) -> list[dict[str, Any]]:
        assert limit == 5
        return [
            {
                "uri": "conversation://recent",
                "name": "Recent conversation",
                "description": "Fixture conversation",
                "mimeType": "text/markdown",
            }
        ]

    async def list_recent_notes(self, *, limit: int) -> list[dict[str, Any]]:
        assert limit == 5
        return [
            {
                "uri": "note://recent",
                "name": "Recent note",
                "mimeType": "text/markdown",
            }
        ]

    @staticmethod
    def _small_resource(scheme: str, identifier: str) -> dict[str, Any]:
        return {
            "uri": f"{scheme}://{identifier}",
            "name": f"{scheme} fixture",
            "mimeType": "text/plain",
            "content": f"{scheme}:{identifier}",
        }


class _FakePrompts:
    async def summarize_conversation_prompt(self, **arguments: Any):
        return [{"role": "user", "content": f"summary:{arguments}"}]

    async def generate_document_prompt(self, **arguments: Any):
        return [{"role": "user", "content": f"document:{arguments}"}]

    async def analyze_media_prompt(self, **arguments: Any):
        return [{"role": "user", "content": f"analysis:{arguments}"}]

    async def search_and_synthesize_prompt(self, **arguments: Any):
        return [{"role": "user", "content": f"search:{arguments}"}]

    async def character_writing_prompt(self, **arguments: Any):
        return [{"role": "user", "content": f"character:{arguments}"}]


def _compose_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    local_registration: LocalToolRegistration | None = None,
) -> server_module.TldwMCPServer:
    from tldw_chatbook.MCP import local_server_tools, permission_store, prompts
    from tldw_chatbook.MCP import resources, tools
    import tldw_chatbook.config as config

    def init_without_databases(instance: server_module.TldwMCPServer) -> None:
        instance.chachanotes_db = object()
        instance.media_db = object()
        instance.notes_service = SimpleNamespace()

    monkeypatch.setattr(
        server_module.TldwMCPServer, "_init_databases", init_without_databases
    )
    monkeypatch.setattr(tools, "MCPTools", lambda *_args: _FakeTools())
    monkeypatch.setattr(resources, "MCPResources", lambda *_args: _FakeResources())
    monkeypatch.setattr(prompts, "MCPPrompts", lambda *_args: _FakePrompts())
    monkeypatch.setattr(
        local_server_tools,
        "local_tools_exposure_enabled",
        lambda: local_registration is not None,
    )
    if local_registration is not None:
        monkeypatch.setattr(
            local_server_tools, "resolve_server_workspace_root", lambda: tmp_path
        )
        monkeypatch.setattr(
            local_server_tools,
            "build_server_local_provider",
            lambda _root, _store: object(),
        )
        monkeypatch.setattr(
            local_server_tools,
            "_local_agent_tool_registrations",
            lambda _provider: [local_registration],
        )
        monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)
        monkeypatch.setattr(
            permission_store, "MCPPermissionStore", lambda _path: object()
        )
    return server_module.TldwMCPServer(version="6.0.0")


class _ProtocolSession:
    def __init__(
        self,
        runtime: ChatbookGatewayRuntime,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages: list[Any] = []
        self.connection = GatewayProtocolConnection(
            runtime, self._write, metadata=metadata
        )

    async def _write(self, value: Any) -> None:
        self.messages.append(json.loads(json.dumps(value)))

    async def request(self, payload: Any) -> list[Any]:
        start = len(self.messages)
        await self.connection.receive(payload)
        await self.connection.wait_for_idle()
        return self.messages[start:]

    async def close(self) -> None:
        await self.connection.shutdown()


def _modern_meta(**extra: Any) -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": CURRENT_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "chatbook-tests",
            "version": "1.0",
        },
        **extra,
    }


def _request(
    version: str,
    request_id: str | int,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    request_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = dict(params or {})
    if version == CURRENT_VERSION:
        values["_meta"] = _modern_meta(**(request_meta or {}))
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": values,
    }


async def _open_session(
    runtime: ChatbookGatewayRuntime,
    version: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> _ProtocolSession:
    session = _ProtocolSession(runtime, metadata=metadata)
    if version in LEGACY_VERSIONS:
        [response] = await session.request(
            {
                "jsonrpc": "2.0",
                "id": "initialize",
                "method": "initialize",
                "params": {
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": "chatbook-tests", "version": "1.0"},
                },
            }
        )
        assert response["result"]["protocolVersion"] == version
        before = len(session.messages)
        await session.connection.receive(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        await asyncio.sleep(0)
        assert len(session.messages) == before
    else:
        [response] = await session.request(
            _request(version, "discover", "server/discover")
        )
        assert CURRENT_VERSION in response["result"]["supportedVersions"]
        assert response["result"]["resultType"] == "complete"
    return session


def _one_response(messages: list[Any]) -> dict[str, Any]:
    assert len(messages) == 1
    response = messages[0]
    assert isinstance(response, dict)
    assert response.get("jsonrpc") == "2.0"
    return response


@pytest.mark.asyncio
async def test_constructor_finalizes_the_exact_default_standalone_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    context = GatewayRequestContext(request_id="construction")

    assert isinstance(server.mcp, ChatbookGatewayRuntime)
    assert [item["name"] for item in await server.mcp.list_tools(context)] == (
        BUILTIN_TOOL_NAMES
    )
    assert [
        item["uriTemplate"]
        for item in await server.mcp.list_resource_templates(context)
    ] == RESOURCE_TEMPLATES
    assert [item["name"] for item in await server.mcp.list_prompts(context)] == (
        PROMPT_NAMES
    )
    assert await server.mcp.list_resources(context) == [
        {
            "uri": "conversation://recent",
            "name": "Recent conversation",
            "description": "Fixture conversation",
            "mimeType": "text/markdown",
        },
        {
            "uri": "note://recent",
            "name": "Recent note",
            "mimeType": "text/markdown",
        },
    ]
    with pytest.raises(RuntimeError, match="finalized"):
        server.mcp.tool()


@pytest.mark.asyncio
async def test_run_and_main_propagate_public_serve_stdio_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object()
    instance = server_module.TldwMCPServer.__new__(server_module.TldwMCPServer)
    instance.mcp = runtime
    served: list[object] = []

    async def fake_serve_stdio(value: object) -> int:
        served.append(value)
        return 37

    monkeypatch.setattr(server_module, "serve_stdio", fake_serve_stdio)
    assert await instance.run() == 37
    assert served == [runtime]
    with pytest.raises(NotImplementedError, match="Only stdio transport is supported"):
        await instance.run("http")

    class _Server:
        async def run(self, transport: str) -> int:
            assert transport == "stdio"
            return 41

    monkeypatch.setattr(server_module, "TldwMCPServer", _Server)
    assert await server_module.main() == 41


def test_module_entrypoint_propagates_status_without_mutating_path_or_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_path = list(sys.path)

    def fake_run(coroutine: Any) -> int:
        coroutine.close()
        return 23

    monkeypatch.setattr(asyncio, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(ENTRYPOINT_PATH), run_name="__main__")

    assert exc_info.value.code == 23
    assert sys.path == original_path
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_module_entrypoint_uses_only_fixed_stderr_diagnostics() -> None:
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "sys.path" not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and not any(
            keyword.arg == "file"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "sys"
            and keyword.value.attr == "stderr"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )
    assert "{e}" not in source and "{exc}" not in source


@pytest.mark.parametrize(
    ("error", "status", "diagnostic"),
    [
        (KeyboardInterrupt(), 130, "MCP server interrupted.\n"),
        (RuntimeError("payload-sentinel"), 1, "MCP server failed.\n"),
    ],
)
def test_module_entrypoint_emits_fixed_failure_diagnostics(
    error: BaseException,
    status: int,
    diagnostic: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(coroutine: Any) -> int:
        coroutine.close()
        raise error

    monkeypatch.setattr(asyncio, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(ENTRYPOINT_PATH), run_name="__main__")

    assert exc_info.value.code == status
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == diagnostic
    assert "payload-sentinel" not in captured.err


@pytest.mark.parametrize("version", PROTOCOL_VERSIONS)
@pytest.mark.asyncio
async def test_real_protocol_core_flow_and_revision_projection(
    version: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    session = await _open_session(server.mcp, version)
    try:
        tools_response = _one_response(
            await session.request(_request(version, "tools", "tools/list"))
        )
        assert {tool["name"] for tool in tools_response["result"]["tools"]} == set(
            BUILTIN_TOOL_NAMES
        )

        templates_response = _one_response(
            await session.request(
                _request(version, "templates", "resources/templates/list")
            )
        )
        assert {
            item["uriTemplate"]
            for item in templates_response["result"]["resourceTemplates"]
        } == set(RESOURCE_TEMPLATES)

        resources_response = _one_response(
            await session.request(_request(version, "resources", "resources/list"))
        )
        assert [item["uri"] for item in resources_response["result"]["resources"]] == [
            "conversation://recent",
            "note://recent",
        ]

        tool_cases = [
            (
                "dict",
                "export_conversation",
                {"conversation_id": 1},
                {"format": "markdown", "content": "# Fixture"},
            ),
            ("list", "list_characters", {}, [{"id": 7, "name": "Ada"}]),
            (
                "string",
                "get_conversation_history",
                {"conversation_id": 1},
                "plain-result",
            ),
        ]
        for request_id, tool_name, arguments, expected in tool_cases:
            response = _one_response(
                await session.request(
                    _request(
                        version,
                        request_id,
                        "tools/call",
                        {"name": tool_name, "arguments": arguments},
                    )
                )
            )
            result = response["result"]
            assert len(result["content"]) == 1
            assert json.loads(result["content"][0]["text"]) == expected
            if version == "2025-03-26":
                assert "structuredContent" not in result
            elif version == "2025-11-25":
                if isinstance(expected, dict):
                    assert result["structuredContent"] == expected
                else:
                    assert "structuredContent" not in result
            else:
                assert result["structuredContent"] == expected

        first_read = _one_response(
            await session.request(
                _request(
                    version,
                    "read-1",
                    "resources/read",
                    {"uri": "conversation://chat-one"},
                )
            )
        )["result"]
        assert len(first_read["contents"]) == 1
        first_block = first_read["contents"][0]
        assert first_block["uri"] == "conversation://chat-one"
        first_meta = first_read["_meta"]
        expected_meta_keys = {
            "tldw.chatbook/continuation",
            "tldw.chatbook/resource",
        }
        if version == CURRENT_VERSION:
            expected_meta_keys.add("io.modelcontextprotocol/serverInfo")
            assert first_meta["io.modelcontextprotocol/serverInfo"] == {
                "name": "tldw_chatbook",
                "version": "6.0.0",
            }
        assert set(first_meta) == expected_meta_keys
        assert first_meta["tldw.chatbook/resource"] == {"kind": "fixture"}
        continuation = first_meta["tldw.chatbook/continuation"]
        assert continuation == {
            "startChar": 0,
            "endChar": len(first_block["text"]),
            "totalChars": len(LONG_RESOURCE_TEXT),
            "totalBytes": len(LONG_RESOURCE_TEXT.encode("utf-8")),
            "returnedBytes": len(first_block["text"].encode("utf-8")),
            "hasMore": True,
            "nextUri": continuation["nextUri"],
        }
        assert continuation["nextUri"].startswith(
            "conversation://chat-one?tldw_continue="
        )
        assert (
            len(json.dumps(first_read, ensure_ascii=False).encode("utf-8"))
            <= GatewayLimits().max_output_line_bytes
        )

        second_read = _one_response(
            await session.request(
                _request(
                    version,
                    "read-2",
                    "resources/read",
                    {"uri": continuation["nextUri"]},
                )
            )
        )["result"]
        assert len(second_read["contents"]) == 1
        assert second_read["contents"][0]["uri"] == "conversation://chat-one"
        assert (
            first_block["text"] + second_read["contents"][0]["text"]
            == LONG_RESOURCE_TEXT
        )
        assert second_read["_meta"]["tldw.chatbook/continuation"]["hasMore"] is False

        prompts_response = _one_response(
            await session.request(_request(version, "prompts", "prompts/list"))
        )
        assert {
            prompt["name"] for prompt in prompts_response["result"]["prompts"]
        } == set(PROMPT_NAMES)
        prompt_response = _one_response(
            await session.request(
                _request(
                    version,
                    "prompt",
                    "prompts/get",
                    {
                        "name": "summarize_conversation",
                        "arguments": {"conversation_id": "7"},
                    },
                )
            )
        )
        assert prompt_response["result"]["messages"]
        assert prompt_response["result"]["messages"][0]["role"] == "user"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_current_request_meta_reaches_context_without_overwriting_reserved_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    observed: list[GatewayRequestContext] = []
    original_call_tool = server.mcp.call_tool

    async def recording_call_tool(
        name: str, arguments: dict[str, Any], context: GatewayRequestContext
    ) -> Any:
        observed.append(context)
        return await original_call_tool(name, arguments, context)

    server.mcp.call_tool = recording_call_tool  # type: ignore[method-assign]
    session = await _open_session(
        server.mcp,
        CURRENT_VERSION,
        metadata={
            "host": {"trace": "base"},
            "method": "forged-host-method",
            "transport": "forged-host-transport",
        },
    )
    try:
        response = _one_response(
            await session.request(
                _request(
                    CURRENT_VERSION,
                    "meta",
                    "tools/call",
                    {"name": "list_characters", "arguments": {}},
                    request_meta={
                        "com.example/request": {"trace": "request"},
                        "method": "forged-client-method",
                        "transport": "forged-client-transport",
                        "io.modelcontextprotocol/serverInfo": {"name": "forged"},
                    },
                )
            )
        )
        assert "result" in response
        assert len(observed) == 1
        context = observed[0]
        assert context.protocol_version == CURRENT_VERSION
        assert context.protocol_era == "modern"
        assert context.metadata == {
            "host": {"trace": "base"},
            "com.example/request": {"trace": "request"},
            "method": "tools/call",
            "transport": "stdio",
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_current_profile_rejects_unsupported_version_deterministically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    session = _ProtocolSession(server.mcp)
    try:
        [response] = await session.request(
            {
                "jsonrpc": "2.0",
                "id": "unsupported",
                "method": "tools/list",
                "params": {
                    "_meta": {
                        **_modern_meta(),
                        "io.modelcontextprotocol/protocolVersion": "2099-01-01",
                    }
                },
            }
        )
        assert response == {
            "jsonrpc": "2.0",
            "id": "unsupported",
            "error": {
                "code": -32022,
                "message": "Unsupported protocol version",
                "data": {
                    "supported": list(gateway.PROTOCOL_PROFILES),
                    "requested": "2099-01-01",
                },
            },
        }
    finally:
        await session.close()


@pytest.mark.parametrize("version", PROTOCOL_VERSIONS)
@pytest.mark.asyncio
async def test_batches_are_accepted_only_by_2025_03_26(
    version: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    session = await _open_session(server.mcp, version)
    try:
        messages = await session.request(
            [
                _request(version, "ping-1", "ping"),
                _request(version, "ping-2", "ping"),
            ]
        )
        assert len(messages) == 1
        response = messages[0]
        if version == "2025-03-26":
            assert isinstance(response, list)
            assert [item["id"] for item in response] == ["ping-1", "ping-2"]
            assert all(item["result"] == {} for item in response)
        else:
            assert response == {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": "Request rejected"},
            }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cancelling_blocked_local_thread_suppresses_late_response_without_rollback_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    side_effects: list[str] = []

    def blocked_handler(_arguments: dict[str, Any]) -> ToolResult:
        side_effects.append("started")
        started.set()
        release.wait(timeout=5)
        side_effects.append("completed")
        finished.set()
        return ToolResult(ok=True, content={"status": "completed"})

    registration = LocalToolRegistration(
        name="local_blocked",
        description="Block until the cancellation test releases the worker.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=blocked_handler,
    )
    server = _compose_server(monkeypatch, tmp_path, local_registration=registration)
    session = await _open_session(server.mcp, "2025-03-26")
    try:
        call_id = "blocked-call"
        await session.connection.receive(
            _request(
                "2025-03-26",
                call_id,
                "tools/call",
                {"name": "local_blocked", "arguments": {}},
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        await session.connection.receive(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": call_id, "reason": "test cancellation"},
            }
        )
        await session.connection.wait_for_idle()
        assert not any(message.get("id") == call_id for message in session.messages)

        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0.05)
        assert not any(message.get("id") == call_id for message in session.messages)
        assert side_effects == ["started", "completed"]
        assert "rolled_back" not in side_effects
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_wire_excludes_library_tools_while_in_process_manifest_keeps_all_18(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    session = await _open_session(server.mcp, "2025-03-26")
    try:
        response = _one_response(
            await session.request(_request("2025-03-26", "tools", "tools/list"))
        )
        wire_names = {item["name"] for item in response["result"]["tools"]}
        manifest_names = {
            item["name"]
            for item in server_module.describe_local_mcp_capabilities()["tools"]
        }
        library_names = set(LIBRARY_TOOL_DESCRIPTORS)

        assert len(library_names) == 22
        assert library_names <= manifest_names
        assert wire_names == set(BUILTIN_TOOL_NAMES)
        assert wire_names.isdisjoint(library_names)
    finally:
        await session.close()


class _BytesReader:
    def __init__(self, *lines: bytes) -> None:
        self.lines = deque(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self.lines.popleft() if self.lines else b""


class _BytesWriter:
    def __init__(self, *, write_error: BaseException | None = None) -> None:
        self.data = bytearray()
        self.write_error = write_error

    def write(self, data: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.data.extend(data)

    async def drain(self) -> None:
        return None


class _HangingDrainWriter(_BytesWriter):
    async def drain(self) -> None:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_public_stdio_returns_zero_on_clean_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    writer = _BytesWriter()

    assert (
        await serve_stdio(
            server.mcp,
            input_stream=_BytesReader(b""),
            output_stream=writer,
        )
        == 0
    )
    assert writer.data == b""


@pytest.mark.parametrize(
    "writer_factory",
    [
        pytest.param(
            lambda: _BytesWriter(write_error=BrokenPipeError("closed")),
            id="broken-output",
        ),
        pytest.param(_HangingDrainWriter, id="bounded-drain"),
    ],
)
@pytest.mark.asyncio
async def test_public_stdio_output_failures_return_nonzero_with_bounded_shutdown(
    writer_factory: Callable[[], _BytesWriter],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server = _compose_server(monkeypatch, tmp_path)
    ping = (
        json.dumps(
            {"jsonrpc": "2.0", "id": "ping", "method": "ping", "params": {}}
        ).encode()
        + b"\n"
    )
    started = time.monotonic()
    status = await asyncio.wait_for(
        serve_stdio(
            server.mcp,
            input_stream=_BytesReader(ping, b""),
            output_stream=writer_factory(),
            limits=GatewayLimits(graceful_shutdown_timeout_seconds=0.05),
        ),
        timeout=1,
    )

    assert status == 1
    assert time.monotonic() - started < 1


def _isolated_subprocess_environment(profile: Path) -> dict[str, str]:
    home = profile / "home"
    config_home = profile / "config"
    data_home = profile / "data"
    temp_dir = profile / "tmp"
    for directory in (home, config_home, data_home, temp_dir):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    environment = {
        key: os.environ[key] for key in _SUBPROCESS_HOST_ENV_KEYS if key in os.environ
    }
    environment.update(
        {
            "APPDATA": str(config_home),
            "HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "LOCALAPPDATA": str(data_home),
            "LOGURU_LEVEL": "ERROR",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "PYTHONWARNINGS": "ignore",
            "TEMP": str(temp_dir),
            "TLDW_CONFIG_PATH": str(config_home / "config.toml"),
            "TLDW_TEST_MODE": "1",
            "TMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_DATA_HOME": str(data_home),
        }
    )
    return environment


class _ProcessLineReader:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._items: Queue[object] = Queue()
        self._eof_seen = False
        self._thread = threading.Thread(
            target=self._read_lines,
            name="mcp-subprocess-stdout",
            daemon=True,
        )
        self._thread.start()

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def _read_lines(self) -> None:
        try:
            while line := self._stream.readline():
                self._items.put(line)
        except BaseException as error:
            self._items.put(error)
        finally:
            self._items.put(_PROCESS_EOF)

    def _next(self, timeout: float) -> object:
        try:
            return self._items.get(timeout=timeout)
        except Empty:
            raise TimeoutError("standalone MCP server produced no response") from None

    def readline(self, *, timeout: float = 20) -> str:
        item = self._next(timeout)
        if item is _PROCESS_EOF:
            self._eof_seen = True
            return ""
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, str)
        return item

    def finish(self, *, timeout: float = 5) -> list[str]:
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        failure: BaseException | None = None
        while not self._eof_seen:
            item = self._next(max(0, deadline - time.monotonic()))
            if item is _PROCESS_EOF:
                self._eof_seen = True
            elif isinstance(item, BaseException):
                failure = item
            else:
                assert isinstance(item, str)
                lines.append(item)
        self._thread.join(max(0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("standalone MCP stdout reader did not stop")
        if failure is not None:
            raise failure
        return lines

    def close(self, *, timeout: float = 5) -> None:
        self._stream.close()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("standalone MCP stdout reader did not stop")
        while True:
            try:
                self._items.get_nowait()
            except Empty:
                return


def test_isolated_subprocess_environment_does_not_inherit_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited_secrets = {
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "DATABASE_URL": "postgresql://secret@host/db",
        "CONFLUENCE_PASSWORD": "confluence-secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/credentials.json",
        "OPENAI_API_KEY": "openai-secret",
        "GITHUB_TOKEN": "github-secret",
        "PYTHONPATH": "/secret/import/path",
        "TASK2512_UNRELATED_HOST_VALUE": "host-only",
    }
    for key, value in inherited_secrets.items():
        monkeypatch.setenv(key, value)

    environment = _isolated_subprocess_environment(tmp_path / "profile")

    assert set(environment).isdisjoint(inherited_secrets)


def test_subprocess_harness_does_not_assume_selectable_pipes() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "selectors" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "DefaultSelector"
        for node in ast.walk(tree)
    )


def test_process_line_reader_supports_non_selectable_pipe_and_joins() -> None:
    reader = _ProcessLineReader(io.StringIO("first\nsecond\n"))

    assert reader.readline(timeout=1) == "first\n"
    assert reader.finish(timeout=1) == ["second\n"]
    assert not reader.thread_alive


def test_process_line_reader_propagates_failure_and_joins() -> None:
    class _FailingPipe:
        def readline(self) -> str:
            raise OSError("read failed")

        def close(self) -> None:
            return None

    reader = _ProcessLineReader(_FailingPipe())

    with pytest.raises(OSError, match="read failed"):
        reader.readline(timeout=1)
    reader.finish(timeout=1)
    assert not reader.thread_alive


def test_process_line_reader_timeout_can_be_closed_without_leaking_thread() -> None:
    release = threading.Event()

    class _BlockingPipe:
        def readline(self) -> str:
            release.wait(timeout=5)
            return ""

        def close(self) -> None:
            release.set()

    reader = _ProcessLineReader(_BlockingPipe())

    with pytest.raises(TimeoutError, match="produced no response"):
        reader.readline(timeout=0.01)
    reader.close(timeout=1)
    assert not reader.thread_alive


def _exchange_process_request(
    process: subprocess.Popen[str],
    reader: _ProcessLineReader,
    payload: dict[str, Any],
) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = reader.readline(timeout=20)
    assert line, "standalone MCP server closed stdout before responding"
    return json.loads(line)


def test_real_module_subprocess_is_protocol_clean_and_exits_on_eof(
    tmp_path: Path,
) -> None:
    assert SHARED_VENV_PYTHON.is_absolute()
    assert SHARED_VENV_PYTHON.is_file()
    payload_sentinel = "https://payload-sentinel.invalid/private"
    process = subprocess.Popen(
        [str(SHARED_VENV_PYTHON), "-m", "tldw_chatbook.MCP"],
        cwd=REPO_ROOT,
        env=_isolated_subprocess_environment(tmp_path / "profile"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    reader = _ProcessLineReader(process.stdout)
    responses: list[dict[str, Any]] = []
    try:
        responses.append(
            _exchange_process_request(
                process,
                reader,
                {
                    "jsonrpc": "2.0",
                    "id": "initialize",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "process-test", "version": "1.0"},
                    },
                },
            )
        )
        assert process.stdin is not None
        process.stdin.write(
            '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n'
        )
        process.stdin.flush()
        requests = [
            _request("2025-03-26", "tools", "tools/list"),
            _request("2025-03-26", "resources", "resources/list"),
            _request("2025-03-26", "prompts", "prompts/list"),
            _request(
                "2025-03-26",
                "call",
                "tools/call",
                {
                    "name": "search_notes",
                    "arguments": {"query": payload_sentinel},
                },
            ),
            _request(
                "2025-03-26",
                "read",
                "resources/read",
                {"uri": "conversation://missing"},
            ),
            _request(
                "2025-03-26",
                "get",
                "prompts/get",
                {
                    "name": "summarize_conversation",
                    "arguments": {"conversation_id": "999"},
                },
            ),
        ]
        responses.extend(
            _exchange_process_request(process, reader, request) for request in requests
        )

        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=20)
        assert process.stdout is not None and process.stderr is not None
        trailing_stdout = "".join(reader.finish(timeout=5))
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        reader.close(timeout=5)

    assert not reader.thread_alive
    assert process.returncode == 0
    assert process.poll() == 0
    assert trailing_stdout == ""
    assert payload_sentinel not in stderr
    assert '"jsonrpc"' not in stderr
    assert '"media_id"' not in stderr
    assert [response["id"] for response in responses] == [
        "initialize",
        "tools",
        "resources",
        "prompts",
        "call",
        "read",
        "get",
    ]
    assert all("result" in response for response in responses)


@pytest.mark.asyncio
async def test_legacy_client_spawn_uses_gateway_output_line_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    process = object()

    async def fake_create_subprocess_exec(command: str, *args: str, **kwargs: Any):
        captured.update(command=command, args=list(args), kwargs=kwargs)
        return process

    async def no_op(*_args: Any) -> None:
        return None

    connection = SimpleNamespace(
        process=process,
        server_info={},
        server_capabilities={},
        protocol_version="2025-03-26",
        initialize=no_op,
        close=no_op,
    )

    def fake_connection(
        created_process: object, *, client_name: str
    ) -> SimpleNamespace:
        assert created_process is process
        assert client_name == "line-limit-client"
        return connection

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", fake_connection)

    client = client_module.MCPClient(name="line-limit-client")
    monkeypatch.setattr(client, "_discover_server_capabilities", no_op)
    try:
        assert await client.connect_to_server(
            "standalone",
            "python",
            args=["-m", "tldw_chatbook.MCP"],
            env={"PROFILE": "isolated"},
        )
    finally:
        await client.disconnect_all()

    assert captured["command"] == "python"
    assert captured["args"] == ["-m", "tldw_chatbook.MCP"]
    assert captured["kwargs"]["env"] == {"PROFILE": "isolated"}
    assert (
        captured["kwargs"]["limit"]
        == GatewayLimits().max_output_line_bytes
        == 1_048_576
    )


@pytest.mark.asyncio
async def test_real_legacy_client_command_discovers_calls_continues_and_stops_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert SHARED_VENV_PYTHON.is_absolute()
    assert SHARED_VENV_PYTHON.is_file()
    monkeypatch.chdir(REPO_ROOT)
    assert Path.cwd() == REPO_ROOT
    profile = tmp_path / "client-profile"
    environment = _isolated_subprocess_environment(profile)
    database_path = (
        profile
        / "home"
        / ".local"
        / "share"
        / "tldw_cli"
        / "default_user"
        / "tldw_chatbook_ChaChaNotes.db"
    )
    database_path.parent.mkdir(parents=True, mode=0o700)
    database = CharactersRAGDB(database_path, client_id=CLI_APP_CLIENT_ID)
    conversation_id = database.add_conversation({"title": "Client flow fixture"})
    assert conversation_id is not None
    long_body = "é" * 300_000
    assert database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": long_body,
        }
    )
    database.close()

    client = MCPClient(name="task-2512-real-client")
    process: asyncio.subprocess.Process | None = None
    try:
        connected = await asyncio.wait_for(
            client.connect_to_server(
                "standalone",
                str(SHARED_VENV_PYTHON),
                args=["-m", "tldw_chatbook.MCP"],
                env=environment,
            ),
            timeout=30,
        )
        assert connected is True
        session = client.sessions["standalone"]
        process = session.process
        assert session.protocol_version == "2025-03-26"
        assert client.servers["standalone"]["protocol_version"] == "2025-03-26"
        assert client.servers["standalone"]["command"] == str(SHARED_VENV_PYTHON)
        assert client.servers["standalone"]["args"] == [
            "-m",
            "tldw_chatbook.MCP",
        ]
        assert {item["name"] for item in client.get_server_tools("standalone")} == set(
            BUILTIN_TOOL_NAMES
        )
        assert [item["uri"] for item in client.get_server_resources("standalone")] == [
            f"conversation://{conversation_id}"
        ]
        assert {
            item["name"] for item in client.get_server_prompts("standalone")
        } == set(PROMPT_NAMES)

        tool_result = await client.call_tool("standalone", "list_characters", {})
        assert "error" not in tool_result
        assert tool_result["result"]

        resource_uri = f"conversation://{conversation_id}"
        resource_parts: list[str] = []
        for _ in range(10):
            resource_result = await client.read_resource("standalone", resource_uri)
            assert set(resource_result) == {"uri", "content", "mimeType", "_meta"}
            assert resource_result["uri"] == resource_uri
            resource_parts.append(resource_result["content"])
            continuation = resource_result["_meta"]["tldw.chatbook/continuation"]
            if not continuation["hasMore"]:
                break
            resource_uri = continuation["nextUri"]
        else:
            pytest.fail("resource continuation did not terminate within ten reads")
        assert long_body in "".join(resource_parts)
        assert len(resource_parts) > 1

        prompt_result = await client.get_prompt(
            "standalone",
            "summarize_conversation",
            {"conversation_id": "999"},
        )
        assert len(prompt_result) == 1
        assert prompt_result[0]["role"] == "user"
        assert prompt_result[0]["content"]
    finally:
        await asyncio.wait_for(client.disconnect_all(), timeout=10)

    assert process is not None
    assert process.returncode is not None


def test_real_module_import_failure_is_fixed_and_payload_free(tmp_path: Path) -> None:
    profile = tmp_path / "failure-profile"
    secret = "import-secret-sentinel"
    environment = _isolated_subprocess_environment(profile)
    invalid_config = profile / "config" / f"{secret}.toml"
    invalid_config.mkdir()
    environment["TLDW_CONFIG_PATH"] = str(invalid_config)

    result = subprocess.run(
        [str(SHARED_VENV_PYTHON), "-m", "tldw_chatbook.MCP"],
        cwd=REPO_ROOT,
        env=environment,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "MCP server failed.\n"
    assert secret not in result.stderr
    assert str(invalid_config) not in result.stderr
    assert "Traceback" not in result.stderr


def test_database_init_failure_log_is_fixed_and_payload_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.config as config

    secret = "database-secret-sentinel"

    class RecordingLogger:
        def __init__(self) -> None:
            self.contexts: list[dict[str, str]] = []
            self.messages: list[str] = []

        def bind(self, **context: str) -> "RecordingLogger":
            self.contexts.append(context)
            return self

        def error(self, message: str) -> None:
            self.messages.append(message)

    recorder = RecordingLogger()

    def fail_path_resolution() -> Path:
        raise RuntimeError(secret)

    monkeypatch.setattr(config, "get_chachanotes_db_path", fail_path_resolution)
    monkeypatch.setattr(server_module, "logger", recorder)
    server = server_module.TldwMCPServer.__new__(server_module.TldwMCPServer)

    with pytest.raises(RuntimeError, match=secret):
        server._init_databases()

    assert recorder.contexts == [{"operation": "initialize_standalone_mcp_databases"}]
    assert recorder.messages == ["Standalone MCP database initialization failed."]
    assert secret not in json.dumps(
        {"contexts": recorder.contexts, "messages": recorder.messages}
    )
