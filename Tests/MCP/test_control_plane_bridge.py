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
    """Same shape as test_control_plane_tool_execute.py's fake: a real
    LocalMCPControlService backs the external-tool path so connect-if-needed
    and error-raise logic is genuinely exercised, plus a hand-written
    builtin execute_tool fake."""

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


# ---- execute_hub_tool: generalized seam ---------------------------------


@pytest.mark.asyncio
async def test_execute_hub_tool_records_given_initiator_and_decision(tmp_path):
    service, fake, client, store = _service(tmp_path)

    result = await service.execute_hub_tool(
        "local:docs", "search", {"q": "hi"}, initiator="agent", decision="approved",
    )

    assert result == client.call_tool_response
    records = _log_records(store)
    assert records and records[0]["ok"] is True
    assert records[0]["initiator"] == "agent"
    assert records[0]["decision"] == "approved"


@pytest.mark.asyncio
async def test_execute_hub_tool_default_initiator_and_decision(tmp_path):
    service, fake, client, store = _service(tmp_path)

    await service.execute_hub_tool("builtin:tldw_chatbook", "calculator", {"x": 1})

    records = _log_records(store)
    assert records and records[0]["initiator"] == "test"
    assert records[0]["decision"] == "allowed"


@pytest.mark.asyncio
async def test_execute_hub_tool_records_initiator_decision_on_failure(tmp_path):
    service, fake, client, store = _service(tmp_path)
    client.call_tool_error = "boom"

    with pytest.raises(RuntimeError, match="boom"):
        await service.execute_hub_tool(
            "local:docs", "search", {}, initiator="agent", decision="approved",
        )

    records = _log_records(store)
    assert records and records[0]["ok"] is False
    assert records[0]["initiator"] == "agent"
    assert records[0]["decision"] == "approved"


@pytest.mark.asyncio
async def test_execute_hub_tool_unknown_prefix_raises_value_error_mentioning_phase_4(tmp_path):
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(ValueError, match="Phase 4"):
        await service.execute_hub_tool("server:remote-1", "search", {})


@pytest.mark.asyncio
async def test_execute_hub_tool_uses_tool_call_timeout_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            0.05 if key == "tool_call_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    client.call_tool_delay = 1.0

    with pytest.raises(RuntimeError, match="Timed out"):
        await service.execute_hub_tool("local:docs", "slow_tool", {})

    records = _log_records(store)
    assert records and records[0]["ok"] is False
    assert "Timed out" in (records[0]["error"] or "")


@pytest.mark.asyncio
async def test_execute_hub_tool_explicit_timeout_seconds_overrides_config(tmp_path, monkeypatch):
    # Even with a generous tool_call_timeout_seconds config, an explicit
    # timeout_seconds= override wins.
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            30.0 if key == "tool_call_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    client.call_tool_delay = 1.0

    with pytest.raises(RuntimeError, match="Timed out"):
        await service.execute_hub_tool("local:docs", "slow_tool", {}, timeout_seconds=0.05)


# ---- test_hub_tool: thin delegate, old behavior preserved ---------------


@pytest.mark.asyncio
async def test_hub_tool_delegates_to_execute_hub_tool_with_test_semantics(tmp_path, monkeypatch):
    service, fake, client, store = _service(tmp_path)

    captured: dict = {}
    real_execute = service.execute_hub_tool

    async def _spy(server_key, tool_name, arguments=None, **kwargs):
        captured.update(kwargs)
        return await real_execute(server_key, tool_name, arguments, **kwargs)

    monkeypatch.setattr(service, "execute_hub_tool", _spy)

    await service.test_hub_tool("local:docs", "search", {"q": "hi"})

    assert captured["initiator"] == "test"
    assert captured["decision"] == "allowed"
    assert captured["timeout_seconds"] == service._lifecycle_timeout()


@pytest.mark.asyncio
async def test_hub_tool_still_uses_lifecycle_timeout_not_tool_call_timeout(tmp_path, monkeypatch):
    # test_hub_tool's timeout knob stays hub_lifecycle_timeout_seconds
    # (pinned by test_control_plane_tool_execute.py); a generous
    # tool_call_timeout_seconds must not rescue a slow call here.
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            0.05 if key == "hub_lifecycle_timeout_seconds"
            else (30.0 if key == "tool_call_timeout_seconds" else default)
        ),
    )
    service, fake, client, store = _service(tmp_path)
    client.call_tool_delay = 1.0

    with pytest.raises(RuntimeError, match="Timed out"):
        await service.test_hub_tool("local:docs", "slow_tool", {})


# ---- timeout config knobs -------------------------------------------------


def test_tool_call_timeout_default(tmp_path):
    service, fake, client, store = _service(tmp_path)
    assert service._tool_call_timeout() == 60.0


def test_tool_call_timeout_reads_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            "12.5" if key == "tool_call_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    assert service._tool_call_timeout() == 12.5


def test_tool_call_timeout_falls_back_on_garbage_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            "not-a-number" if key == "tool_call_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    assert service._tool_call_timeout() == 60.0


def test_approval_timeout_seconds_default(tmp_path):
    service, fake, client, store = _service(tmp_path)
    assert service.approval_timeout_seconds() == 120.0


def test_approval_timeout_seconds_reads_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            "5" if key == "approval_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    assert service.approval_timeout_seconds() == 5.0


def test_approval_timeout_seconds_falls_back_on_garbage_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            "garbage" if key == "approval_timeout_seconds" else default
        ),
    )
    service, fake, client, store = _service(tmp_path)
    assert service.approval_timeout_seconds() == 120.0


# ---- session approvals -----------------------------------------------------


def test_session_approval_set_check_clear(tmp_path):
    service, fake, client, store = _service(tmp_path)

    assert service.is_session_approved("local:docs", "search") is False

    service.approve_for_session("local:docs", "search")
    assert service.is_session_approved("local:docs", "search") is True
    # A different tool on the same server is unaffected.
    assert service.is_session_approved("local:docs", "other_tool") is False
    # A different server with the same tool name is unaffected.
    assert service.is_session_approved("builtin:tldw_chatbook", "search") is False

    service.clear_session_approvals()
    assert service.is_session_approved("local:docs", "search") is False


def test_session_approvals_are_per_instance_not_persisted(tmp_path):
    service, fake, client, store = _service(tmp_path)
    service.approve_for_session("local:docs", "search")

    other_service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    assert other_service.is_session_approved("local:docs", "search") is False


# ---- record_tool_decision --------------------------------------------------


def test_record_tool_decision_writes_denied_record(tmp_path):
    service, fake, client, store = _service(tmp_path)

    service.record_tool_decision(
        "local:docs", "search", decision="denied", initiator="agent", error="user denied the call",
    )

    records = _log_records(store)
    assert records
    record = records[0]
    assert record["server_key"] == "local:docs"
    assert record["tool_name"] == "search"
    assert record["initiator"] == "agent"
    assert record["decision"] == "denied"
    assert record["ok"] is False
    assert record["duration_ms"] == 0
    assert "user denied the call" in (record["error"] or "")


def test_record_tool_decision_defaults_initiator_to_agent(tmp_path):
    service, fake, client, store = _service(tmp_path)

    service.record_tool_decision("local:docs", "search", decision="timeout")

    records = _log_records(store)
    assert records and records[0]["initiator"] == "agent"
    assert records[0]["decision"] == "timeout"


def test_record_tool_decision_survives_log_failure(tmp_path, monkeypatch):
    class _RaisingExecutionLog(MCPExecutionLog):
        def append(self, record):
            raise OSError("disk full")

    monkeypatch.setattr(control_plane_module, "MCPExecutionLog", _RaisingExecutionLog)
    service, fake, client, store = _service(tmp_path)

    # Must not raise.
    service.record_tool_decision("local:docs", "search", decision="denied", error="nope")


def test_record_tool_decision_survives_execution_log_property_raise(tmp_path, monkeypatch):
    def _raise(self):
        raise RuntimeError("execution_log unavailable")

    monkeypatch.setattr(
        control_plane_module.UnifiedMCPControlPlaneService,
        "execution_log",
        property(_raise),
    )
    service, fake, client, store = _service(tmp_path)

    # Must not raise.
    service.record_tool_decision("local:docs", "search", decision="denied")
