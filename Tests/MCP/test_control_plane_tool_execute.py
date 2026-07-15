from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
import tldw_chatbook.MCP.unified_control_plane_service as control_plane_module


class FakeToolClient:
    """Stands in for MCPClient: session-gated call_tool + connect bookkeeping."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.connect_calls: list[str] = []
        self.call_tool_calls: list[tuple[str, str, dict]] = []
        self.call_tool_response: dict = {"result": {"content": [{"type": "text", "text": "ok"}]}}
        self.call_tool_error: str | None = None
        self.call_tool_delay: float = 0.0

    async def connect_to_server(self, server_id, command, args=None, env=None):
        self.connect_calls.append(server_id)
        self.sessions[server_id] = {"server_id": server_id}
        return True

    async def describe_server(self, server_id):
        return {"server_id": server_id, "tools": [{"name": "t"}], "resources": [], "prompts": []}

    async def disconnect_from_server(self, server_id):
        self.sessions.pop(server_id, None)
        return True

    async def call_tool(self, server_id, tool_name, arguments):
        self.call_tool_calls.append((server_id, tool_name, dict(arguments)))
        if self.call_tool_delay:
            await asyncio.sleep(self.call_tool_delay)
        if self.call_tool_error is not None:
            return {"error": self.call_tool_error}
        return self.call_tool_response


class FakeLocalService:
    """Mirrors test_control_plane_lifecycle.py's FakeLocalService style: a
    coarse stand-in for LocalMCPControlService, wired to a *real* store and
    a real LocalMCPControlService instance for the external-tool path (so
    execute_external_tool's connect-if-needed/call_tool/error-raise logic is
    genuinely exercised), plus a hand-written builtin execute_tool fake.
    """

    def __init__(self, store: LocalMCPStore, client: FakeToolClient) -> None:
        self.store = store
        self.client = client
        self._real = LocalMCPControlService(store=store, client=client, manifest_provider=lambda: {})
        self.execute_tool_calls: list[tuple[str, dict]] = []
        self.builtin_result: dict = {"source": "local", "result": "builtin-ok"}
        self.builtin_error: Exception | None = None

    async def execute_external_tool(self, profile_id, tool_name, arguments=None):
        return await self._real.execute_external_tool(profile_id, tool_name, arguments)

    async def execute_tool(self, tool_name, arguments=None):
        self.execute_tool_calls.append((tool_name, dict(arguments or {})))
        if self.builtin_error is not None:
            raise self.builtin_error
        return self.builtin_result


def _service(tmp_path: Path) -> tuple[UnifiedMCPControlPlaneService, FakeLocalService, FakeToolClient, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(LocalExternalMCPProfile(profile_id="docs", command="python", args=("-m", "demo")))
    client = FakeToolClient()
    fake = FakeLocalService(store, client)
    service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    return service, fake, client, store


def _log_records(store: LocalMCPStore) -> list[dict]:
    log_path = Path(store.path).with_name("mcp_execution_log.jsonl")
    return MCPExecutionLog(log_path).read_recent()


@pytest.mark.asyncio
async def test_hub_tool_local_connects_if_needed_and_returns_result(tmp_path):
    service, fake, client, store = _service(tmp_path)
    assert "docs" not in client.sessions

    result = await service.test_hub_tool("local:docs", "search", {"q": "hi"})

    assert client.connect_calls == ["docs"]
    assert client.call_tool_calls == [("docs", "search", {"q": "hi"})]
    assert result == client.call_tool_response

    records = _log_records(store)
    assert records and records[0]["ok"] is True
    assert records[0]["server_key"] == "local:docs"
    assert records[0]["tool_name"] == "search"


@pytest.mark.asyncio
async def test_hub_tool_local_error_response_raises_and_records_failure(tmp_path):
    service, fake, client, store = _service(tmp_path)
    client.call_tool_error = "boom from server"

    with pytest.raises(RuntimeError, match="boom from server"):
        await service.test_hub_tool("local:docs", "search", {})

    records = _log_records(store)
    assert records and records[0]["ok"] is False
    assert "boom from server" in (records[0]["error"] or "")


@pytest.mark.asyncio
async def test_hub_tool_builtin_routes_to_execute_tool(tmp_path):
    service, fake, client, store = _service(tmp_path)

    result = await service.test_hub_tool("builtin:tldw_chatbook", "calculator", {"x": 1})

    assert fake.execute_tool_calls == [("calculator", {"x": 1})]
    assert result == fake.builtin_result
    assert client.call_tool_calls == []


@pytest.mark.asyncio
async def test_hub_tool_unknown_prefix_raises_value_error_mentioning_phase_4(tmp_path):
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(ValueError, match="Phase 4"):
        await service.test_hub_tool("server:remote-1", "search", {})


@pytest.mark.asyncio
async def test_hub_tool_timeout_raises_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            0.05 if key == "hub_lifecycle_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    client.call_tool_delay = 1.0

    with pytest.raises(RuntimeError, match="Timed out"):
        await service.test_hub_tool("local:docs", "slow_tool", {})

    records = _log_records(store)
    assert records and records[0]["ok"] is False
    assert "Timed out" in (records[0]["error"] or "")


@pytest.mark.asyncio
async def test_hub_tool_log_write_failure_does_not_mask_result(tmp_path, monkeypatch):
    class _RaisingExecutionLog(MCPExecutionLog):
        def append(self, record):
            raise OSError("disk full")

    monkeypatch.setattr(control_plane_module, "MCPExecutionLog", _RaisingExecutionLog)
    service, fake, client, store = _service(tmp_path)

    result = await service.test_hub_tool("local:docs", "search", {"q": "x"})

    assert result == client.call_tool_response


@pytest.mark.asyncio
async def test_hub_tool_execution_log_property_raise_does_not_mask_result(tmp_path, monkeypatch):
    """N1: `_record_tool_execution()` used to read `self.execution_log`
    OUTSIDE its own try/except -- if the property itself raised (e.g. a
    `Path(store.path)` oddity), that would escape `_record_tool_execution()`
    entirely and mask the tool result/error being propagated by
    `test_hub_tool()`, violating the "recording is best-effort, never masks
    the result" contract the existing append-failure test already covers."""

    def _raise(self):
        raise RuntimeError("execution_log unavailable")

    monkeypatch.setattr(
        control_plane_module.UnifiedMCPControlPlaneService,
        "execution_log",
        property(_raise),
    )
    service, fake, client, store = _service(tmp_path)

    result = await service.test_hub_tool("local:docs", "search", {"q": "x"})

    assert result == client.call_tool_response


@pytest.mark.asyncio
async def test_hub_tool_result_excerpt_is_redacted_before_disk(tmp_path):
    """I2: `build_record()`/`MCPExecutionLog.append()` only redact
    `arguments`, never `result_excerpt` -- a tool result that happens to
    echo back a secret-shaped key (e.g. an API key in its response payload)
    must never reach the JSONL execution log unredacted."""
    service, fake, client, store = _service(tmp_path)
    client.call_tool_response = {"api_key": "sk-secret123", "data": "ok"}

    result = await service.test_hub_tool("local:docs", "search", {"q": "x"})

    assert result == {"api_key": "sk-secret123", "data": "ok"}  # returned raw, unredacted
    records = _log_records(store)
    assert records and records[0]["ok"] is True
    excerpt = records[0]["result_excerpt"] or ""
    assert "sk-secret123" not in excerpt
    assert "***" in excerpt
    assert "ok" in excerpt  # non-secret fields still recorded
