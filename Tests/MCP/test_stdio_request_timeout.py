"""TASK-38582: stdio session deadlines honor the configured tool-call timeout.

A wire deadline with a healthy transport must fail only the timed-out
request; teardown stays reserved for transport uncertainty. The configured
``[mcp] tool_call_timeout_seconds`` must reach the session construction
through ``MCPClient.connect_to_server``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.MCP import client as client_module
from tldw_chatbook.MCP.client import MCPClient

FIXTURE = Path(__file__).parent / "fixtures" / "stdio_timeout_server.py"


async def _spawn_fixture() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        str(FIXTURE),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


@pytest.mark.asyncio
async def test_slow_tool_deadline_fails_only_that_call_and_keeps_session():
    """A deadline with a live transport is a typed timeout, not a teardown."""
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    process = await _spawn_fixture()
    connection = _StdioJSONRPCConnection(
        process, client_name="deadline-client", request_timeout_seconds=0.5
    )
    try:
        slow = await connection.call_tool("slow", {"seconds": 3})

        assert slow.transport_error == "mcp_tool_timeout"
        assert process.returncode is None, "deadline must not kill the server"
        assert not connection._reader_unavailable
        assert not connection._cleanup_complete

        fast = await connection.call_tool("fast", {})
        assert fast.transport_error is None
        assert fast.content[0]["text"] == "fast final"
    finally:
        await connection.close()
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_genuine_transport_death_still_tears_down_the_session():
    """Reader EOF mid-request keeps the settle/teardown path intact."""
    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    process = await _spawn_fixture()
    connection = _StdioJSONRPCConnection(
        process, client_name="death-client", request_timeout_seconds=10.0
    )
    try:
        held = asyncio.create_task(connection.call_tool("slow", {"seconds": 30}))
        await asyncio.sleep(0.15)  # let the request reach the server
        process.kill()
        result = await held
        assert result.transport_error == "mcp_transport_unavailable"
        assert connection._reader_unavailable
    finally:
        await connection.close()
        if process.returncode is None:
            process.kill()
            await process.wait()


class _RecordingSession:
    protocol_version = "2025-03-26"
    server_info: dict[str, Any] = {}
    server_capabilities: dict[str, Any] = {}

    def __init__(
        self,
        process: object,
        *,
        client_name: str,
        request_timeout_seconds: float = 10.0,
        server_request_dispatcher: object | None = None,
    ) -> None:
        self.process = process
        self.request_timeout_seconds = request_timeout_seconds
        self._on_transport_failure = None

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_configured_timeout_reaches_stdio_session(monkeypatch: pytest.MonkeyPatch):
    """connect_to_server resolves [mcp] tool_call_timeout_seconds for the wire."""
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key, default=None: 42.5,
    )

    class _Process:
        stdin = None
        stdout = None
        stderr = None
        returncode = 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return 0

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return _Process()

    async def discover(_server_id: str) -> None:
        return None

    client = MCPClient(name="timeout-plumb-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", _RecordingSession)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)
    try:
        assert await client.connect_to_server("plumbed", "plumbed-command")
        assert client.sessions["plumbed"].request_timeout_seconds == 42.5

        assert await client.connect_to_server(
            "explicit", "explicit-command", request_timeout_seconds=7.0
        )
        assert client.sessions["explicit"].request_timeout_seconds == 7.0
    finally:
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_configured_timeout_helper_degrades_to_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unreadable or unparsable knob falls back to the 60 s control-plane default."""
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module, "get_cli_setting", lambda *args, **kwargs: "not-a-number"
    )
    assert client_module._configured_request_timeout_seconds() == 60.0

    def raising(*_args: Any, **_kwargs: Any) -> float:
        raise RuntimeError("config admission unavailable")

    monkeypatch.setattr(config_module, "get_cli_setting", raising)
    assert client_module._configured_request_timeout_seconds() == 60.0

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key, default=None: 90,
    )
    assert client_module._configured_request_timeout_seconds() == 90.0
