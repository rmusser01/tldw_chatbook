from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import MappingProxyType
from typing import get_args, get_type_hints
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.local_tool_provider import (
    LocalProviderTerminal,
    LocalToolInvocationReason,
    LocalToolInvocationResult,
)
from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy
from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.hub_test_execution import (
    LocalHubExecutionOutcome,
    ToolTestAdmissionBlocked,
    ToolTestAdmissionStale,
)
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
import tldw_chatbook.MCP.unified_control_plane_service as control_plane_module
import tldw_chatbook.Agents.mcp_tool_provider as mcp_provider_module
import tldw_chatbook.Agents.local_tool_provider as local_tool_provider_module
import tldw_chatbook.MCP.local_server_tools as local_server_tools_module


class FakeToolClient:
    """Stands in for MCPClient: session-gated call_tool + connect bookkeeping."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.connect_calls: list[str] = []
        self.call_tool_calls: list[tuple[str, str, dict]] = []
        self.call_tool_response: dict = {
            "result": {"content": [{"type": "text", "text": "ok"}]}
        }
        self.call_tool_error: str | None = None
        self.call_tool_delay: float = 0.0

    async def connect_to_server(self, server_id, command, args=None, env=None):
        self.connect_calls.append(server_id)
        self.sessions[server_id] = {"server_id": server_id}
        return True

    async def describe_server(self, server_id):
        return {
            "server_id": server_id,
            "tools": [{"name": "t"}],
            "resources": [],
            "prompts": [],
        }

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
        self._real = LocalMCPControlService(
            store=store, client=client, manifest_provider=lambda: {}
        )
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

    async def run_runtime_request(self, method, params=None):
        """Mirrors `LocalRuntimeDelegate.request()`'s own `tools/call` branch:
        the raw protocol method reaches the SAME `execute_tool` seam the
        `tool.execute` action does -- which is exactly why it needs the same
        refusal at the control-plane boundary."""
        normalized = dict(params or {})
        result: dict = {}
        if method == "tools/call":
            arguments = normalized.get("arguments")
            result = await self.execute_tool(
                str(normalized.get("name") or normalized.get("tool_name") or ""),
                arguments if isinstance(arguments, dict) else {},
            )
        return {
            "source": "local",
            "method": method,
            "params": normalized,
            "result": result,
        }

    async def run_runtime_batch(self, requests):
        # Fix Round H (PR-T3 review), Item 2b: mirrors
        # `LocalMCPControlService.run_runtime_batch()`'s own
        # `dict(request)` coercion (`local_control_service.py:500`) --
        # this used to only accept a literal `dict` (`isinstance(request,
        # dict)`) and silently drop anything else (a `Mapping` that isn't a
        # `dict` subclass, or a list-of-pairs) to `{}`, which is exactly
        # the divergence dimension `test_raw_tools_call_as_a_list_of_
        # pairs_inside_a_batch_is_refused_before_anything_runs` below had
        # to route around by wiring a REAL `LocalMCPControlService`
        # instead of this fake -- this fake could not exercise that bug
        # either way. `dict(request)` raises the SAME TypeError/ValueError
        # the real method would for a genuinely non-coercible item, which
        # is correct: by the time `run_action()` calls this method the
        # requests are already normalized by `_normalize_batch_requests()`
        # one layer up, so in practice every item here is already a plain
        # dict -- this only matters for a caller that reaches this fake
        # directly, bypassing that normalization.
        results = []
        for index, request in enumerate(requests):
            entry = dict(request)
            method = str(entry.get("method") or "")
            if method == "tools/call":
                params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
                arguments = params.get("arguments")
                await self.execute_tool(
                    str(params.get("name") or params.get("tool_name") or ""),
                    arguments if isinstance(arguments, dict) else {},
                )
            results.append({"index": index, "method": method, "ok": True})
        return {"source": "local", "results": results}


def _service(
    tmp_path: Path,
) -> tuple[
    UnifiedMCPControlPlaneService, FakeLocalService, FakeToolClient, LocalMCPStore
]:
    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="docs", command="python", args=("-m", "demo")
        )
    )
    client = FakeToolClient()
    fake = FakeLocalService(store, client)
    service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    return service, fake, client, store


@pytest.mark.asyncio
async def test_fake_local_service_run_runtime_batch_coerces_like_the_real_one(tmp_path):
    """Fix Round H (PR-T3 review), Item 2b. Direct-call proof that
    `FakeLocalService.run_runtime_batch()`'s `dict(request)` coercion now
    matches `LocalMCPControlService.run_runtime_batch()`'s own -- called
    DIRECTLY on the fake, bypassing `run_action()`'s own
    `_normalize_batch_requests()` pre-dispatch scan. Every OTHER batch test
    in this file goes through that scan, which already coerces every item
    to a plain `dict` before the fake ever sees it -- so none of them could
    tell this fake's own coercion apart from the old `isinstance(request,
    dict) else {}` fallback it silently carried. A non-dict `Mapping` and a
    list-of-pairs must both be recognized as a genuine `tools/call` request
    when the fake is exercised on its own: the old code dropped both to
    `{}`, reporting `method=""` and never dispatching to `execute_tool`."""
    _service_ignored, fake, _client, _store = _service(tmp_path)
    non_dict_request = MappingProxyType(
        {"method": "tools/call", "params": {"name": "calculator"}}
    )
    list_of_pairs_request = [
        ["method", "tools/call"], ["params", {"name": "calculator"}]
    ]

    result = await fake.run_runtime_batch([non_dict_request, list_of_pairs_request])

    assert result["results"] == [
        {"index": 0, "method": "tools/call", "ok": True},
        {"index": 1, "method": "tools/call", "ok": True},
    ]
    assert fake.execute_tool_calls == [
        ("calculator", {}),
        ("calculator", {}),
    ]


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
    assert records[0]["status"] == "error"
    assert records[0]["error_category"] == "execution_failed"
    assert records[0]["exception_type"] == "RuntimeError"
    assert "boom from server" not in repr(records[0])


@pytest.mark.asyncio
async def test_hub_tool_builtin_routes_to_execute_tool(tmp_path):
    service, fake, client, store = _service(tmp_path)

    result = await service.test_hub_tool(
        "builtin:tldw_chatbook", "calculator", {"x": 1}
    )

    assert fake.execute_tool_calls == [("calculator", {"x": 1})]
    assert result == fake.builtin_result
    assert client.call_tool_calls == []


@pytest.mark.asyncio
async def test_hub_tool_governance_denial_records_honest_blocked_row(tmp_path):
    """Item 1 (PR-T3 fix round D). Before this item, an `MCPGovernanceDenied`
    raised inside `coro` (here: the builtin `execute_tool()` path) fell into
    `execute_hub_tool()`'s generic `except Exception` branch and was recorded
    as a crashed execution -- `status="error"`, `error_category=
    "execution_failed"`, `duration_ms` measured as if the call ran, and the
    caller's pre-computed `decision="allowed"` left untouched -- three false
    statements about an event that never dispatched at all; governance
    refused it outright, before the tool ever ran. Recorded honestly instead,
    reusing `record_tool_decision()`'s own never-executed vocabulary
    (`status="blocked"`, `duration_ms=0`) and overriding `decision` to
    `"denied"`. `error_category="governance_denied"` is asserted against the
    RECORD READ BACK OFF THE PERSISTED JSONL (`_log_records()`), not the
    in-memory value passed to `_record_tool_execution()` -- proving the token
    survives `safe_metadata_token()` (execution_log.py's `build_record()`)
    unmodified; that sanitizer rejects any value containing whitespace, so a
    token with a space would silently come back as `"invalid"` instead of
    failing loudly."""
    service, fake, client, store = _service(tmp_path)
    fake.builtin_error = control_plane_module.MCPGovernanceDenied(
        "Denied by local governance: tool.execute"
    )

    with pytest.raises(control_plane_module.MCPGovernanceDenied):
        await service.test_hub_tool("builtin:tldw_chatbook", "calculator", {"x": 1})

    records = _log_records(store)
    assert records and records[0]["ok"] is False
    assert records[0]["status"] == "blocked"
    assert records[0]["duration_ms"] == 0
    assert records[0]["decision"] == "denied"
    assert records[0]["exception_type"] == "MCPGovernanceDenied"
    assert records[0]["error_category"] == "governance_denied"


@pytest.mark.asyncio
async def test_hub_tool_unknown_prefix_raises_value_error_display_only(tmp_path):
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(ValueError, match="display-only"):
        await service.test_hub_tool("server:remote-1", "search", {})


@pytest.mark.asyncio
async def test_hub_tool_unknown_prefix_raises_the_typed_display_only_error(tmp_path):
    """task-2539 (PR-T3 fix round B, item 3): drift-proofing at the RAISE
    SITE itself, not just where the message is rendered. Before this item,
    nothing in this suite pinned the exact type/message `execute_hub_tool()`
    raises for a server-source key -- only the UI-side classifier
    (`mcp_workbench._is_permission_refusal()`) pinned its OWN copy of the
    string, so a reword here would have silently reverted that classifier's
    fix (it no longer even LOOKS at the message, but this test exists so a
    future accidental reword is caught at the source, not inferred)."""
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(control_plane_module.MCPServerSourceDisplayOnlyError) as exc_info:
        await service.test_hub_tool("server:remote-1", "search", {})

    assert str(exc_info.value) == "Server-source tools are display-only."


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
    assert records[0]["status"] == "timeout"
    assert records[0]["error_category"] == "timeout"
    assert records[0]["exception_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_hub_tool_cancellation_reraises_and_records_one_terminal_row(tmp_path):
    service, fake, _client, store = _service(tmp_path)
    started = asyncio.Event()

    async def _blocked_builtin(tool_name, arguments=None):
        fake.execute_tool_calls.append((tool_name, dict(arguments or {})))
        started.set()
        await asyncio.Future()

    fake.execute_tool = _blocked_builtin
    pending = asyncio.create_task(
        service.test_hub_tool(
            "builtin:tldw_chatbook",
            "calculator",
            {"x": 1},
            registered_argument_names={"x"},
        )
    )
    await started.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == "cancelled"
    assert records[0]["error_category"] == "cancelled"
    assert records[0]["exception_type"] == "CancelledError"
    assert records[0]["initiator"] == "test"
    assert records[0]["decision"] == "allowed"
    assert records[0]["duration_ms"] >= 0
    assert records[0]["argument_names"] == ["x"]
    assert records[0]["unknown_argument_count"] == 0
    assert records[0]["result_type"] == "none"
    assert records[0]["result_size"] == 0


@pytest.mark.asyncio
async def test_agent_bridge_outer_timeout_has_one_bridge_owned_audit_row(
    tmp_path, monkeypatch
):
    service, fake, _client, store = _service(tmp_path)
    started = asyncio.Event()

    async def _blocked_builtin(tool_name, arguments=None):
        fake.execute_tool_calls.append((tool_name, dict(arguments or {})))
        started.set()
        await asyncio.Future()

    fake.execute_tool = _blocked_builtin
    monkeypatch.setattr(service, "_tool_call_timeout", lambda: 0.5)
    monkeypatch.setattr(mcp_provider_module, "_RESULT_WAIT_SLACK_SECONDS", -0.45)
    tool = HubTool(
        server_key="builtin:tldw_chatbook",
        server_label="tldw_chatbook",
        source="builtin",
        name="calculator",
        description="Calculate",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        tags=(),
        stale=False,
        executable=True,
    )
    provider = MCPToolProvider(
        service=service,
        main_loop=asyncio.get_running_loop(),
    )
    pending = asyncio.create_task(
        asyncio.to_thread(provider._execute, tool, {"x": 1}, decision="allowed")
    )
    await started.wait()

    result = await pending
    await asyncio.sleep(0)

    assert result.ok is False
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["initiator"] == "agent"
    assert records[0]["status"] == "blocked"
    assert records[0]["error_category"] == "execution_bridge_failed"
    assert records[0]["decision"] == "allowed"


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
async def test_hub_tool_execution_log_property_raise_does_not_mask_result(
    tmp_path, monkeypatch
):
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
async def test_hub_tool_result_is_replaced_by_type_and_size_before_disk(tmp_path):
    service, fake, client, store = _service(tmp_path)
    client.call_tool_response = {"api_key": "sk-secret123", "data": "ok"}

    result = await service.test_hub_tool("local:docs", "search", {"q": "x"})

    assert result == {
        "api_key": "sk-secret123",
        "data": "ok",
    }  # returned raw, unredacted
    records = _log_records(store)
    assert records and records[0]["ok"] is True
    assert records[0]["result_type"] == "dict"
    assert records[0]["result_size"] == 2
    assert "sk-secret123" not in repr(records[0])
    assert "result_excerpt" not in records[0]


@pytest.mark.asyncio
async def test_hub_tool_argument_values_are_never_captured_regardless_setting(
    tmp_path, monkeypatch
):
    """The obsolete capture setting cannot re-enable private values."""
    real_get_cli_setting = None
    import tldw_chatbook.MCP.unified_control_plane_service as ucps

    real_get_cli_setting = ucps.get_cli_setting

    def fake_get_cli_setting(section, key, default=None):
        if section == "mcp" and key == "log_tool_arguments":
            return "false"
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(ucps, "get_cli_setting", fake_get_cli_setting)
    service, fake, client, store = _service(tmp_path)

    await service.test_hub_tool("local:docs", "search", {"q": "sensitive input"})

    records = _log_records(store)
    assert records and records[0]["ok"] is True
    assert "arguments" not in records[0]
    assert records[0]["argument_names"] == []
    assert records[0]["unknown_argument_count"] == 1
    assert "sensitive input" not in repr(records[0])


# -- Task 5 (RAG-51): name the permission decision -- `test_hub_tool()` used
# to hardcode `decision="allowed"` regardless of what actually authorized
# the run; the Hub workbench now passes the real decision for an Ask-gated
# tool the user just confirmed.


@pytest.mark.asyncio
async def test_hub_tool_default_decision_is_still_allowed(tmp_path):
    """Every existing caller (no `decision=` kwarg) keeps recording
    "allowed" -- byte-identical to the pre-Task-5 hardcoded value."""
    service, fake, client, store = _service(tmp_path)

    await service.test_hub_tool("local:docs", "search", {"q": "hi"})

    records = _log_records(store)
    assert records and records[0]["decision"] == "allowed"


@pytest.mark.asyncio
async def test_hub_tool_ask_approved_decision_is_recorded_not_allowed(tmp_path):
    """A `decision="approved"` call (the Hub workbench's Ask-then-confirmed
    case) records THAT decision in the execution log entry, not the
    hardcoded "allowed" every run used to get regardless of gate."""
    service, fake, client, store = _service(tmp_path)

    result = await service.test_hub_tool(
        "local:docs", "search", {"q": "hi"}, decision="approved"
    )

    assert result == client.call_tool_response
    records = _log_records(store)
    assert records and records[0]["decision"] == "approved"
    assert records[0]["ok"] is True


@pytest.mark.asyncio
async def test_hub_tool_approved_decision_recorded_on_failure_too(tmp_path):
    """The decision is recorded on a FAILED run too -- it describes why the
    call dispatched, not whether it succeeded."""
    service, fake, client, store = _service(tmp_path)
    client.call_tool_error = "boom from server"

    with pytest.raises(RuntimeError, match="boom from server"):
        await service.test_hub_tool(
            "local:docs", "search", {}, decision="approved"
        )

    records = _log_records(store)
    assert records and records[0]["decision"] == "approved"
    assert records[0]["ok"] is False


# -- Task 4 (PR-T3): `registered_argument_names` threads from `test_hub_tool()`
# through to the execution log -- before this task NO caller in the tree
# supplied it, so every row recorded `argument_names: []` and
# `unknown_argument_count == len(arguments)` regardless of what the tool's
# schema actually registered.


@pytest.mark.asyncio
async def test_hub_tool_registered_argument_names_threads_to_execution_log(tmp_path):
    service, fake, client, store = _service(tmp_path)

    await service.test_hub_tool(
        "local:docs",
        "search",
        {"q": "hi", "surprise": "unexpected arg"},
        registered_argument_names={"q", "limit"},
    )

    records = _log_records(store)
    assert records and records[0]["argument_names"] == ["q"]
    # "surprise" was supplied but never registered on the schema.
    assert records[0]["unknown_argument_count"] == 1


@pytest.mark.asyncio
async def test_hub_tool_omitted_registered_argument_names_keeps_pre_task4_behavior(
    tmp_path,
):
    """A caller that doesn't supply `registered_argument_names` (every
    caller before Task 4) keeps recording NO names and counting every
    supplied argument as unknown -- byte-identical to before this task."""
    service, fake, client, store = _service(tmp_path)

    await service.test_hub_tool("local:docs", "search", {"q": "hi"})

    records = _log_records(store)
    assert records and records[0]["argument_names"] == []
    assert records[0]["unknown_argument_count"] == 1


# -- Task 6 (PR-T3), Route B: the Advanced runner's `tool.execute` hatch.
# `run_action("tool.execute", ...)` used to call `local_service.execute_tool()`
# directly -- bypassing BOTH the Hub's per-tool permission gate (a tool set to
# Off ran anyway) and `_record_tool_execution()` (which lives only inside
# `execute_hub_tool()`), so the run left no trace in the audit trail at all.
# It now resolves the same permission state the Test Tool gate resolves and
# executes through the same shared seam as every other run.


@pytest.mark.asyncio
async def test_advanced_tool_execute_routes_through_execute_hub_tool_and_logs(tmp_path):
    service, fake, client, store = _service(tmp_path)

    result = await service.run_action(
        "tool.execute", {"tool_name": "calculator", "arguments": {"x": 1}}
    )

    # Same underlying call and same returned envelope as before -- the
    # built-in branch of `execute_hub_tool()` IS `local_service.execute_tool`.
    assert fake.execute_tool_calls == [("calculator", {"x": 1})]
    assert result == fake.builtin_result

    records = _log_records(store)
    assert records, "the Advanced execute hatch must leave an audit-trail row"
    assert records[0]["server_key"] == "builtin:tldw_chatbook"
    assert records[0]["tool_name"] == "calculator"
    assert records[0]["ok"] is True
    assert records[0]["initiator"] == "test"


@pytest.mark.asyncio
async def test_advanced_tool_execute_refuses_a_tool_set_to_off(tmp_path):
    """The headline: an Off tool must not run from the Advanced hatch."""
    service, fake, client, store = _service(tmp_path)
    service.set_tool_state("builtin:tldw_chatbook", "calculator", "deny")

    with pytest.raises(PermissionError, match="Off"):
        await service.run_action(
            "tool.execute", {"tool_name": "calculator", "arguments": {"x": 1}}
        )

    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_advanced_tool_execute_refuses_a_tool_set_to_off_with_the_typed_error(
    tmp_path,
):
    """Item 2 (PR-T3 fix round D): drift-proofing at the RAISE SITE, same
    precedent as `test_hub_tool_unknown_prefix_raises_the_typed_display_
    only_error`. `UI/MCP_Modules/mcp_inspector.py`'s Advanced runner
    narrows its own classifier to this TYPE -- if this raise site ever
    reverted to a bare `PermissionError`, that narrowed handler would stop
    recognizing this genuine refusal (falling through to "Action failed:"),
    with nothing in this suite failing to say so were it not for this test
    pinning the type here, at the source."""
    service, fake, client, store = _service(tmp_path)
    service.set_tool_state("builtin:tldw_chatbook", "calculator", "deny")

    with pytest.raises(control_plane_module.MCPHubGateDeniedError):
        await service.run_action(
            "tool.execute", {"tool_name": "calculator", "arguments": {"x": 1}}
        )


@pytest.mark.asyncio
async def test_advanced_tool_execute_refusal_is_recorded_as_denied(tmp_path):
    """A refusal is part of the audit trail too -- a blocked row, not silence.

    Fix Round B, Item 1 (this test's assertion is a PRE-AUTHORIZED contract
    change): Fix Round A's Item 5 made this row carry a reason by passing
    the refusal message through as ``error=``, reusing the SAME
    ``decision="denied"`` + truthy ``error`` -> ``"approval_cancelled"``
    derivation ``test_record_tool_decision_writes_denied_record`` (test_
    control_plane_bridge.py) and ``console_chat_controller.py``'s shutdown-
    mid-approval recorder rely on. That reuse was wrong: those other two
    callers describe a genuine user-cancelled-an-offered-approval outcome,
    while THIS call site is the permission gate denying outright -- no
    approval was ever offered, nobody cancelled anything. Reusing their
    category made the row specific AND FALSE instead of generic and true.
    ``record_tool_decision()`` now takes an explicit ``error_category``
    that, when supplied, is used verbatim instead of being derived --
    this call site now passes the honest ``"gate_denied"`` token instead of
    ``error=``. The message text still never reaches the row --
    ``build_record()``'s ``error_category`` is a sanitized token
    (``safe_metadata_token()``), never exception/free text, by the
    metadata-only design the whole execution log is built on;
    ``error_category`` is the only signal that survives.
    """
    service, fake, client, store = _service(tmp_path)
    service.set_tool_state("builtin:tldw_chatbook", "calculator", "deny")

    with pytest.raises(PermissionError):
        await service.run_action("tool.execute", {"tool_name": "calculator"})

    records = _log_records(store)
    assert records and records[0]["decision"] == "denied"
    assert records[0]["status"] == "blocked"
    assert records[0]["ok"] is False
    assert records[0]["server_key"] == "builtin:tldw_chatbook"
    assert records[0]["tool_name"] == "calculator"
    # Fix Round B, Item 1: an explicit, honest category -- the permission
    # gate denied this outright; nothing was ever offered for the user to
    # cancel. "error" itself (the raw refusal sentence) must never land on
    # the row -- only the derived/explicit category.
    assert records[0]["error_category"] == "gate_denied"
    assert "error" not in records[0]
    assert "Off in Permissions" not in repr(records[0])


@pytest.mark.asyncio
async def test_advanced_tool_execute_records_the_decision_it_ran_under(tmp_path):
    """An Ask-resolved tool (the default posture for a NON-hash-free server
    key, and what `resolve_effective_state_by_key()` collapses every
    non-deny verdict to for those) records "approved" -- the Advanced
    runner's confirm press is the approval, the same vocabulary the agent
    bridge's approved calls use. `builtin:tldw_chatbook` is itself in
    `HASH_FREE_SERVER_KEYS`, so an explicit "ask" override (set here) still
    resolves to "ask"/"approved" same as always -- only an explicit/
    inherited "allow" for a hash-free key is exempt from the collapse (see
    `test_advanced_tool_execute_allow_records_allowed_not_approved` below)."""
    service, fake, client, store = _service(tmp_path)
    service.set_tool_state("builtin:tldw_chatbook", "calculator", "ask")

    await service.run_action("tool.execute", {"tool_name": "calculator"})

    records = _log_records(store)
    assert records and records[0]["decision"] == "approved"


@pytest.mark.asyncio
async def test_advanced_tool_execute_allow_records_allowed_not_approved(tmp_path):
    """Fix Round A, Item 1: `builtin:tldw_chatbook` (the Advanced hatch's
    fixed gate key, `BUILTIN_SERVER_KEY`) is in `HASH_FREE_SERVER_KEYS` --
    in-process code the rug-pull hash guard was never meant to cover (see
    that constant's docstring). Before this fix,
    `resolve_effective_state_by_key()` collapsed EVERY "allow" to "ask"
    unconditionally, so a calculator explicitly set to Allow still recorded
    `decision="approved"` (asked-and-approved) from this Advanced hatch,
    while the SAME tool resolved via a live `HubTool` (the Test Tool panel,
    `resolve_effective_state()`, which already honors the exemption) would
    record `decision="allowed"` (no ask needed) for the identical
    permission state -- a cross-surface split in the audit trail. This also
    exercises the `decision="allowed" if state.state == "allow" else
    "approved"` branch in `execute_advanced_tool()`, which was unreachable
    before this fix since by-key resolution never returned "allow"."""
    service, fake, client, store = _service(tmp_path)
    service.set_tool_state("builtin:tldw_chatbook", "calculator", "allow")

    await service.run_action(
        "tool.execute", {"tool_name": "calculator", "arguments": {"x": 1}}
    )

    records = _log_records(store)
    assert records and records[0]["decision"] == "allowed"
    assert records[0]["ok"] is True


@pytest.mark.asyncio
async def test_advanced_tool_execute_gate_check_exception_uses_honest_copy(tmp_path):
    """Item 1 (PR-T3 fix round F): the THIRD occurrence of one pattern in
    this branch -- a permission-gate check that RAISES must not be told
    to the user as a genuine "Off" verdict. Before this fix,
    `execute_advanced_tool()` synthesized the same fail-closed
    `EffectiveToolState(state="deny", origin="gate_error")` that
    `MCPWorkbench._resolve_test_gate()` (`mcp_workbench.py`) synthesizes
    for the identical case, then fell into the SAME deny branch as a
    genuine "Off" and raised `_ADVANCED_EXECUTE_BLOCKED_MESSAGE` --
    "{tool} is set to Off in Permissions." -- a confident, false claim
    about the user's own configuration when the RESOLVER crashed, not the
    gate. This is what task-2536 (fix round B) already fixed on the Test
    Tool panel's blocked-result body and fix round D polished further;
    this raise site had never branched on `origin` at all. Verified here
    by making `gate_tool_test_by_key` raise directly, mirroring `_resolve_
    test_gate()`'s own mutation precedent for that twin fix."""
    service, fake, client, store = _service(tmp_path)

    def _raise(server_key: str, tool_name: str):
        raise RuntimeError("permission store corrupt")

    service.gate_tool_test_by_key = _raise  # type: ignore[method-assign]

    with pytest.raises(control_plane_module.MCPHubGateDeniedError) as exc_info:
        await service.run_action(
            "tool.execute", {"tool_name": "calculator", "arguments": {"x": 1}}
        )

    assert str(exc_info.value) == "Permission state could not be resolved."
    assert "Off in Permissions" not in str(exc_info.value)
    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_advanced_tool_execute_gate_check_exception_records_gate_error_token(
    tmp_path,
):
    """Item 1 (PR-T3 fix round F), audit-row half: `error_category=
    "gate_denied"` is ALSO false for a gate check that raised -- that
    token means the Hub's OWN Allow/Ask/Off gate genuinely resolved to
    Off, and here the gate never resolved at all. A resolver failure now
    records its own honest token, `"gate_error"` (matching the
    `EffectiveToolState.origin` value that produced it, the same
    vocabulary-consistency precedent `"gate_denied"`'s own comment
    states), read back off the PERSISTED JSONL row -- not the in-memory
    call -- proving it survives `safe_metadata_token()` unmodified, same
    precedent as `test_advanced_tool_execute_refusal_is_recorded_as_
    denied` above."""
    service, fake, client, store = _service(tmp_path)

    def _raise(server_key: str, tool_name: str):
        raise RuntimeError("permission store corrupt")

    service.gate_tool_test_by_key = _raise  # type: ignore[method-assign]

    with pytest.raises(control_plane_module.MCPHubGateDeniedError):
        await service.run_action("tool.execute", {"tool_name": "calculator"})

    records = _log_records(store)
    assert records and records[0]["decision"] == "denied"
    assert records[0]["status"] == "blocked"
    assert records[0]["error_category"] == "gate_error"
    assert records[0]["error_category"] != "gate_denied"
    assert "error" not in records[0]


# -- Task 6 (PR-T3), Route B, second door: `runtime.request` /`runtime.batch`
# are Advanced descriptors too, and the in-process runtime speaks the real
# protocol -- `{"method": "tools/call"}` reached the SAME
# `runtime_delegate.execute_tool()` as the hatch above, with the same two
# holes (no Hub permission gate, no execution-log row). Gating only
# `tool.execute` would have locked the front door and left this one open.


@pytest.mark.asyncio
async def test_raw_tools_call_through_runtime_request_is_refused(tmp_path):
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await service.run_action(
            "runtime.request",
            {"method": "tools/call", "params": {"name": "calculator"}},
        )

    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_raw_tools_call_through_runtime_request_raises_the_typed_error(tmp_path):
    """Item 2 (PR-T3 fix round D): same drift-proofing precedent as
    `test_advanced_tool_execute_refuses_a_tool_set_to_off_with_the_typed_
    error` -- pin the TYPE at the raise site, not just the message, since
    `UI/MCP_Modules/mcp_inspector.py`'s Advanced runner now classifies
    refusals by type. `RawToolCallRefusedError` is shared verbatim with
    `local_runtime_delegate.LocalMCPRuntimeDelegate.request()`'s own raise
    site for the identical refusal (see that type's own docstring)."""
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(control_plane_module.RawToolCallRefusedError):
        await service.run_action(
            "runtime.request",
            {"method": "tools/call", "params": {"name": "calculator"}},
        )


@pytest.mark.asyncio
async def test_raw_tools_call_inside_a_runtime_batch_is_refused(tmp_path):
    """One `tools/call` anywhere in the batch refuses the whole batch --
    partial execution would run the ungated call and report it as a normal
    batch row."""
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await service.run_action(
            "runtime.batch",
            {
                "requests": [
                    {"method": "tools/list"},
                    {"method": "tools/call", "params": {"name": "calculator"}},
                ]
            },
        )

    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_raw_tools_call_inside_a_batch_as_non_dict_mapping_is_refused(tmp_path):
    """Minor #4 (Fix Round A): the pre-dispatch scan used to check
    `isinstance(request, dict)`, while `LocalMCPControlService.
    run_runtime_batch()` accepts any `Mapping` -- a non-dict `Mapping` item
    (unreachable from the UI today, since the payload is always
    `json.loads` output, but worth closing so the scan and the batch runner
    agree on what counts as a request) would silently skip the scan. Widened
    to `Mapping` so it doesn't."""
    service, fake, client, store = _service(tmp_path)
    non_dict_request = MappingProxyType(
        {"method": "tools/call", "params": {"name": "calculator"}}
    )

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await service.run_action("runtime.batch", {"requests": [non_dict_request]})

    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_raw_tools_call_as_a_list_of_pairs_inside_a_batch_is_refused_before_anything_runs(
    tmp_path,
):
    """Item 2 (PR-T3 fix round F). The pre-dispatch scan only ever inspected
    items where `isinstance(request, Mapping)` -- but `LocalMCPControlService.
    run_runtime_batch()` (`local_control_service.py:500`) normalizes every
    item with `dict(request)`, which ALSO accepts a list of `(key, value)`
    pairs. `json.loads` produces exactly a bare `list` for a JSON array, so
    a payload like

        {"requests": [{"method": "tools/list"}, [["method", "tools/call"]]]}

    -- reachable from the Advanced pane's raw-JSON textarea -- has its
    second item skip the scan entirely (it is a `list`, not a `Mapping`)
    while the real dispatcher still normalizes and runs it.

    This falsifies `RawToolCallRefusedError`'s own docstring promise ("a
    refusal ... that never reached the tool") in a subtler way than an
    outright miss: the scan DOES eventually refuse this batch -- but only
    after item 0 has already dispatched for real through the delegate and
    been recorded, because the durable backstop
    (`LocalMCPRuntimeDelegate.request()`) is what actually catches item 1,
    one item late. "It raised" alone is not evidence of the fix -- the
    pre-fix code raises too. The activity log staying non-empty is the
    tell; this test's post-fix assertion is that it stays EMPTY.

    Wired with a REAL `LocalMCPControlService` (default, real
    `LocalMCPRuntimeDelegate`) instead of this file's `FakeLocalService` --
    that fake's own `run_runtime_batch()` neither coerces a list of pairs
    nor reproduces the delegate's `tools/call` backstop, so it cannot
    exercise this bug either way.
    """
    store = LocalMCPStore(tmp_path / "store.json")
    local_service = LocalMCPControlService(
        store=store, client=None, manifest_provider=lambda: {}
    )
    service = UnifiedMCPControlPlaneService(
        local_service=local_service,
        server_service=None,
        target_store=None,
        context_store=None,
    )

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await service.run_action(
            "runtime.batch",
            {
                "requests": [
                    {"method": "tools/list"},
                    [["method", "tools/call"]],
                ]
            },
        )

    activity = local_service.get_runtime_activity()
    assert activity["entries"] == [], (
        "the pre-dispatch scan must refuse the WHOLE batch before any item "
        "dispatches -- a non-empty activity log means an earlier item "
        "(here: the 'tools/list' request) already ran for real"
    )


@pytest.mark.asyncio
async def test_raw_tools_call_as_a_list_of_pairs_alone_in_a_batch_is_refused(tmp_path):
    """Companion control to the test above, isolating the SAME normalization
    gap without the "does item 0 leak first" timing question -- a batch
    whose ONLY item is a list-of-pairs `tools/call` must still be refused
    (not silently treated as an unrecognized, harmless item)."""
    service, fake, client, store = _service(tmp_path)

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await service.run_action(
            "runtime.batch",
            {"requests": [[["method", "tools/call"], ["params", {"name": "calculator"}]]]},
        )

    assert fake.execute_tool_calls == []


@pytest.mark.asyncio
async def test_other_runtime_request_methods_are_untouched(tmp_path):
    """The diagnostic value of the raw request runner survives: only the
    executing method is refused."""
    service, fake, client, store = _service(tmp_path)

    result = await service.run_action(
        "runtime.request", {"method": "tools/list", "params": {}}
    )

    assert result["method"] == "tools/list"


def _prepared_external_tool() -> HubTool:
    return HubTool(
        server_key="local:docs",
        server_label="docs",
        source="local",
        name="search",
        description="Search docs",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
        },
        tags=(),
        stale=False,
        executable=True,
    )


def _install_prepared_external_catalog(fake: FakeLocalService, tool: HubTool) -> None:
    fake.get_external_servers = lambda: [
        {
            "profile_id": "docs",
            "is_connected": True,
            "discovery_snapshot": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    }
                ]
            },
        }
    ]


@pytest.mark.asyncio
async def test_prepared_hub_external_and_builtin_dispatch_through_legacy_test_seam(
    tmp_path,
):
    service, fake, _client, _store = _service(tmp_path)
    external = _prepared_external_tool()
    _install_prepared_external_catalog(fake, external)
    builtin = HubTool(
        server_key="builtin:tldw_chatbook",
        server_label="tldw_chatbook",
        source="builtin",
        name="calculator",
        description="Calculate",
        input_schema=None,
        tags=(),
        stale=False,
        executable=True,
    )
    fake.get_inventory = lambda: {
        "tools": [{"name": builtin.name, "description": builtin.description}]
    }
    service.set_tool_state(external.server_key, external.name, "allow", tool=external)
    service.set_tool_state(builtin.server_key, builtin.name, "allow", tool=builtin)
    legacy = AsyncMock(side_effect=[{"external": True}, {"builtin": True}])
    service.test_hub_tool = legacy

    external_preview = service.prepare_hub_test(external)
    builtin_preview = service.prepare_hub_test(builtin)
    external_result = await service.execute_prepared_hub_test(
        external_preview.nonce, "run", {"q": "original"}
    )
    builtin_result = await service.execute_prepared_hub_test(
        builtin_preview.nonce, "run", {}
    )

    assert external_result == {"external": True}
    assert builtin_result == {"builtin": True}
    assert legacy.await_args_list[0].args == (
        external.server_key,
        external.name,
        {"q": "original"},
    )
    assert legacy.await_args_list[1].args == (builtin.server_key, builtin.name, {})


@pytest.mark.asyncio
async def test_prepared_hub_concurrent_double_click_admits_at_most_once(tmp_path):
    service, fake, _client, _store = _service(tmp_path)
    tool = _prepared_external_tool()
    _install_prepared_external_catalog(fake, tool)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def _legacy(*_args, **_kwargs):
        admitted.set()
        await release.wait()
        return {"ok": True}

    service.test_hub_tool = AsyncMock(side_effect=_legacy)
    preview = service.prepare_hub_test(tool)
    first = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    await admitted.wait()
    second = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    release.set()
    first_result = await first

    assert first_result == {"ok": True}
    assert isinstance(second, ToolTestAdmissionStale)
    assert service.test_hub_tool.await_count == 1


@pytest.mark.asyncio
async def test_prepared_hub_invalid_arguments_fail_before_nonce_consumption_or_audit(
    tmp_path,
):
    service, fake, _client, store = _service(tmp_path)
    tool = _prepared_external_tool()
    _install_prepared_external_catalog(fake, tool)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    service.test_hub_tool = AsyncMock(return_value={"ok": True})
    preview = service.prepare_hub_test(tool)

    with pytest.raises(ValueError, match="JSON object"):
        await service.execute_prepared_hub_test(
            preview.nonce,
            "run",
            ["not", "an", "object"],  # type: ignore[arg-type]
        )

    assert _log_records(store) == []
    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    assert result == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_arguments",
    [
        pytest.param({"custom": object()}, id="custom-value"),
        pytest.param({"number": float("nan")}, id="nan"),
        pytest.param({"number": float("inf")}, id="positive-infinity"),
        pytest.param({"number": float("-inf")}, id="negative-infinity"),
        pytest.param({"nested": [{"number": float("nan")}]}, id="nested-nan"),
        pytest.param({1: "non-string key"}, id="non-string-key"),
        pytest.param({"nested": {2: "non-string key"}}, id="nested-non-string-key"),
    ],
)
async def test_prepared_hub_strict_json_rejection_preserves_nonce_and_audit(
    tmp_path,
    invalid_arguments,
):
    service, fake, _client, store = _service(tmp_path)
    tool = _prepared_external_tool()
    _install_prepared_external_catalog(fake, tool)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    service.test_hub_tool = AsyncMock(return_value={"ok": True})
    preview = service.prepare_hub_test(tool)

    with pytest.raises(ValueError):
        await service.execute_prepared_hub_test(
            preview.nonce,
            "run",
            invalid_arguments,
        )

    assert _log_records(store) == []
    service.test_hub_tool.assert_not_awaited()
    result = await service.execute_prepared_hub_test(
        preview.nonce,
        "run",
        {"q": "valid"},
    )
    assert result == {"ok": True}
    service.test_hub_tool.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["external", "builtin"])
async def test_prepared_hub_real_delegate_appends_exactly_one_terminal_row(
    tmp_path,
    source,
):
    service, fake, client, store = _service(tmp_path)
    if source == "external":
        tool = _prepared_external_tool()
        _install_prepared_external_catalog(fake, tool)
        arguments = {"q": "real delegate"}
    else:
        tool = HubTool(
            server_key="builtin:tldw_chatbook",
            server_label="tldw_chatbook",
            source="builtin",
            name="calculator",
            description="Calculate",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            tags=(),
            stale=False,
            executable=True,
        )
        fake.get_inventory = lambda: {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
            ]
        }
        arguments = {"x": 1}
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    preview = service.prepare_hub_test(tool)

    result = await service.execute_prepared_hub_test(
        preview.nonce,
        "run",
        arguments,
    )

    assert result == (
        client.call_tool_response if source == "external" else fake.builtin_result
    )
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["server_key"] == tool.server_key
    assert records[0]["tool_name"] == tool.name
    assert records[0]["initiator"] == "test"
    assert records[0]["decision"] == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["external", "builtin"])
async def test_prepared_hub_real_delegate_cancellation_records_once_and_reraises(
    tmp_path,
    source,
):
    service, fake, client, store = _service(tmp_path)
    started = asyncio.Event()
    if source == "external":
        tool = _prepared_external_tool()
        _install_prepared_external_catalog(fake, tool)

        async def _blocked_external(server_id, tool_name, arguments):
            client.call_tool_calls.append((server_id, tool_name, dict(arguments)))
            started.set()
            await asyncio.Future()

        client.call_tool = _blocked_external
        arguments = {"q": "cancel me"}
    else:
        tool = HubTool(
            server_key="builtin:tldw_chatbook",
            server_label="tldw_chatbook",
            source="builtin",
            name="calculator",
            description="Calculate",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            tags=(),
            stale=False,
            executable=True,
        )
        fake.get_inventory = lambda: {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
            ]
        }

        async def _blocked_builtin(tool_name, arguments=None):
            fake.execute_tool_calls.append((tool_name, dict(arguments or {})))
            started.set()
            await asyncio.Future()

        fake.execute_tool = _blocked_builtin
        arguments = {"x": 1}
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    preview = service.prepare_hub_test(tool)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", arguments)
    )
    await started.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == "cancelled"
    assert records[0]["error_category"] == "cancelled"
    assert records[0]["exception_type"] == "CancelledError"
    assert records[0]["server_key"] == tool.server_key
    assert records[0]["tool_name"] == tool.name
    assert records[0]["initiator"] == "test"
    assert records[0]["decision"] == "allowed"
    assert records[0]["duration_ms"] >= 0
    assert records[0]["argument_names"] == sorted(arguments)
    assert records[0]["unknown_argument_count"] == 0
    assert records[0]["result_type"] == "none"
    assert records[0]["result_size"] == 0


@pytest.mark.asyncio
async def test_prepared_hub_uses_canonical_copy_when_caller_mutates_during_admission(
    tmp_path,
):
    service, fake, _client, _store = _service(tmp_path)
    tool = _prepared_external_tool()
    _install_prepared_external_catalog(fake, tool)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    arguments = {"q": "original", "nested": {"values": [1]}}
    entered_dispatch = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def _waited_legacy(*args, **kwargs):
        entered_dispatch.set()
        await release_dispatch.wait()
        return {"arguments": args[2]}

    service.test_hub_tool = AsyncMock(side_effect=_waited_legacy)
    preview = service.prepare_hub_test(tool)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", arguments)
    )
    await entered_dispatch.wait()
    arguments["q"] = "mutated"
    arguments["nested"]["values"].append(2)
    release_dispatch.set()
    result = await pending

    assert result["arguments"] == {"q": "original", "nested": {"values": [1]}}


@pytest.mark.asyncio
async def test_prepared_local_malformed_provider_fails_closed_after_admission(
    tmp_path, monkeypatch
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    service, _fake, _client, store = _service(tmp_path)
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read a file",
        input_schema=None,
        tags=("reads",),
        stale=False,
        executable=True,
    )

    class _Provider:
        def hub_tools(self):
            return [tool]

    class _Handle:
        provider = _Provider()

        def __init__(self):
            from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

            self.authority = capture_directory_chain(tmp_path)

        def close(self):
            return None

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _Handle(),
    )
    monkeypatch.setattr(
        local_server_tools, "build_hub_local_provider", lambda *a, **k: _Handle()
    )
    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", lambda: tmp_path
    )
    monkeypatch.setattr(control_plane_module, "get_cli_setting", lambda *a, **k: True)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    preview = service.prepare_hub_test(tool)

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, LocalHubExecutionOutcome)
    assert result.status == "error"
    assert result.error_category == "execution_failed"
    assert result.dispatch_started is False
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["error_category"] == "execution_failed"


def _install_local_hub_execution_provider(
    monkeypatch,
    tmp_path,
    service,
    *,
    invoke_detailed,
    gate="allow",
    policy=ToolExecutionPolicy.BOUNDED_ABANDONABLE,
    timeout_floor=None,
    threads=None,
    eligible=True,
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read one file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )
    captured_authority = capture_directory_chain(tmp_path)

    def _mark(label):
        if threads is not None:
            threads.append((label, threading.get_ident()))

    class _Authority:
        canonical_root = captured_authority.canonical_root
        identities = captured_authority.identities

        def __eq__(self, other):
            _mark("authority_compare")
            return (
                getattr(other, "canonical_root", None) == self.canonical_root
                and getattr(other, "identities", None) == self.identities
            )

    authority = _Authority() if threads is not None else captured_authority
    captured_callbacks = []
    captured_guards = []
    closed = []

    class _Provider:
        def __init__(
            self,
            approval_callback=None,
            dispatch_guard=None,
            *,
            force_eligible=False,
        ):
            self.approval_callback = approval_callback
            self.dispatch_guard = dispatch_guard
            self.force_eligible = force_eligible

        def hub_tools(self):
            _mark("definition_compare")
            current = (
                True
                if self.force_eligible
                else eligible()
                if callable(eligible)
                else eligible
            )
            return [tool] if current else []

        def invoke_detailed(self, tool_id, arguments):
            _mark("invoke_detailed")
            if self.dispatch_guard is not None:
                assert self.dispatch_guard() is True
            return invoke_detailed(self, tool_id, arguments)

        def execution_policy_for(self, tool_id):
            _mark("execution_policy")
            return policy() if callable(policy) else policy

        def timeout_for(self, tool_id):
            _mark("timeout_floor")
            return timeout_floor

    class _Handle:
        def __init__(
            self,
            approval_callback=None,
            dispatch_guard=None,
            *,
            force_eligible=False,
        ):
            self.provider = _Provider(
                approval_callback,
                dispatch_guard,
                force_eligible=force_eligible,
            )
            self.authority = authority

        def close(self):
            closed.append(self)

    def _build(
        *_args,
        approval_callback=None,
        dispatch_guard=None,
        force_eligible=False,
        **_kwargs,
    ):
        _mark("provider_construction")
        captured_callbacks.append(approval_callback)
        captured_guards.append(dispatch_guard)
        return _Handle(
            approval_callback,
            dispatch_guard,
            force_eligible=force_eligible,
        )

    def _build_inspection(*args, **kwargs):
        return _build(*args, force_eligible=True, **kwargs)

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        _build_inspection,
    )
    monkeypatch.setattr(local_server_tools, "build_hub_local_provider", _build)

    def _resolve_root():
        _mark("root_resolution")
        return tmp_path

    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", _resolve_root
    )
    monkeypatch.setattr(control_plane_module, "get_cli_setting", lambda *a, **k: True)

    def _gate(_tool):
        _mark("gate_revalidation")
        current = gate() if callable(gate) else gate
        if isinstance(current, EffectiveToolState):
            return current
        return EffectiveToolState(state=current, origin="tool_override")

    service.gate_tool_test = _gate
    return tool, captured_callbacks, captured_guards, closed


def test_local_hub_outcome_vocabulary_is_closed():
    hints = get_type_hints(LocalHubExecutionOutcome)

    assert set(get_args(hints["decision"])) == {"allowed", "approved", "denied"}
    assert set(get_args(hints["status"])) == {
        "success",
        "blocked",
        "error",
        "timeout",
        "cancelled",
    }
    assert set(get_args(hints["provider_terminal"])) == {
        "not_started",
        "returned",
        "raised",
    }
    assert {
        "allow",
        "ask",
        "deny",
        "gate_error",
        "kill_switch",
        "no_callback",
        "not_checked",
        "timeout",
        "unresolved",
    } == set(get_args(hints["final_gate"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "case",
        "initial_gate",
        "intent",
        "provider_reason",
        "provider_terminal",
        "result_ok",
        "expected_type",
        "expected_status",
        "expected_category",
        "expected_decision",
        "expected_gate",
        "expected_approval",
        "expected_dispatch",
        "expected_handler_calls",
    ),
    [
        (
            "persistent_allow",
            "allow",
            "run",
            LocalToolInvocationReason.HANDLER_RETURNED,
            LocalProviderTerminal.RETURNED,
            True,
            "outcome",
            "success",
            None,
            "allowed",
            "allow",
            False,
            True,
            1,
        ),
        (
            "ask_once",
            "ask",
            "approve_once",
            LocalToolInvocationReason.HANDLER_RETURNED,
            LocalProviderTerminal.RETURNED,
            True,
            "outcome",
            "success",
            None,
            "approved",
            "allow",
            True,
            True,
            1,
        ),
        (
            "configured_off",
            "deny",
            "run",
            None,
            None,
            False,
            "blocked",
            "blocked",
            "permission_denied",
            "denied",
            "deny",
            False,
            False,
            0,
        ),
        (
            "unresolved",
            EffectiveToolState(state="ask", origin="gate_error"),
            "run",
            None,
            None,
            False,
            "blocked",
            "blocked",
            "permission_unresolved",
            "denied",
            "unresolved",
            False,
            False,
            0,
        ),
        (
            "provider_refusal",
            "allow",
            "run",
            LocalToolInvocationReason.PERMISSION_OFF,
            LocalProviderTerminal.NOT_STARTED,
            False,
            "outcome",
            "blocked",
            "permission_off",
            "denied",
            "deny",
            False,
            False,
            0,
        ),
        (
            "eligibility_mismatch",
            "allow",
            "run",
            None,
            None,
            False,
            "stale",
            "stale",
            "local_tool_ineligible",
            "denied",
            "unavailable",
            False,
            False,
            0,
        ),
        (
            "handler_crash",
            "allow",
            "run",
            LocalToolInvocationReason.HANDLER_RAISED,
            LocalProviderTerminal.RAISED,
            False,
            "outcome",
            "error",
            "execution_failed",
            "allowed",
            "allow",
            False,
            True,
            1,
        ),
        (
            "success",
            "allow",
            "run",
            LocalToolInvocationReason.HANDLER_RETURNED,
            LocalProviderTerminal.RETURNED,
            True,
            "outcome",
            "success",
            None,
            "allowed",
            "allow",
            False,
            True,
            1,
        ),
    ],
)
async def test_local_hub_terminal_matrix_for_gate_provider_and_cleanup(
    tmp_path,
    monkeypatch,
    case,
    initial_gate,
    intent,
    provider_reason,
    provider_terminal,
    result_ok,
    expected_type,
    expected_status,
    expected_category,
    expected_decision,
    expected_gate,
    expected_approval,
    expected_dispatch,
    expected_handler_calls,
):
    service, _fake, _client, store = _service(tmp_path)
    provider_calls = []
    handler_calls = []
    eligible = {"value": True}

    def _invoke(provider, tool_id, arguments):
        provider_calls.append((tool_id, dict(arguments)))
        approval_consumed = False
        if provider.approval_callback is not None:
            decision = provider.approval_callback(
                [
                    type(
                        "Gate",
                        (),
                        {
                            "server_key": "local:__local__",
                            "tool_name": "fs_read",
                            "arguments": dict(arguments),
                        },
                    )()
                ]
            )
            approval_consumed = decision == {"fs_read": "approve_once"}
        dispatch_started = provider_terminal in {
            LocalProviderTerminal.RETURNED,
            LocalProviderTerminal.RAISED,
        }
        if dispatch_started:
            handler_calls.append(True)
        return LocalToolInvocationResult(
            result=ToolResult(ok=result_ok, content="safe" if result_ok else ""),
            final_gate=(
                "allow"
                if approval_consumed
                else "deny"
                if provider_reason is LocalToolInvocationReason.PERMISSION_OFF
                else "allow"
            ),
            approval_consumed=approval_consumed,
            reason_code=provider_reason,
            dispatch_started=dispatch_started,
            provider_terminal=provider_terminal,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        gate=initial_gate,
        eligible=lambda: eligible["value"],
    )
    preview = service.prepare_hub_test(tool)
    callbacks.clear()
    closed.clear()
    if case == "eligibility_mismatch":
        eligible["value"] = False

    result = await service.execute_prepared_hub_test(preview.nonce, intent, {})

    if expected_type == "outcome":
        assert isinstance(result, LocalHubExecutionOutcome)
        assert result.status == expected_status
        assert result.error_category == expected_category
        assert result.decision == expected_decision
        assert result.final_gate == expected_gate
        assert result.approval_consumed is expected_approval
        assert result.dispatch_started is expected_dispatch
        assert result.provider_terminal == provider_terminal.value
    elif expected_type == "blocked":
        assert isinstance(result, ToolTestAdmissionBlocked)
        assert result.status == expected_status
        assert result.reason == expected_category
        assert result.refreshed_preview is not None
        assert result.refreshed_preview.rendered_gate == expected_gate
    else:
        assert isinstance(result, ToolTestAdmissionStale)
        assert result.status == expected_status
        assert result.reason == expected_category
        assert result.refreshed_preview is not None
        assert result.refreshed_preview.rendered_gate == expected_gate
    assert len(handler_calls) == expected_handler_calls
    assert len(provider_calls) == (1 if expected_type == "outcome" else 0)
    records = _log_records(store)
    assert len(records) == 1, case
    assert records[0]["error_category"] == expected_category
    assert records[0]["decision"] == expected_decision
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gate", "intent", "expected_decision", "approval_consumed"),
    [
        ("allow", "run", "allowed", False),
        ("ask", "approve_once", "approved", True),
    ],
)
async def test_local_hub_allow_and_ask_invoke_once_with_structured_outcome(
    tmp_path, monkeypatch, gate, intent, expected_decision, approval_consumed
):
    service, _fake, _client, store = _service(tmp_path)
    calls = []

    def _invoke(provider, tool_id, arguments):
        calls.append((tool_id, dict(arguments)))
        consumed = False
        if provider.approval_callback is not None:
            decision = provider.approval_callback(
                [
                    type(
                        "Gate",
                        (),
                        {
                            "server_key": "local:__local__",
                            "tool_name": "fs_read",
                            "arguments": dict(arguments),
                        },
                    )()
                ]
            )
            consumed = decision == {"fs_read": "approve_once"}
        return LocalToolInvocationResult(
            result=ToolResult(
                ok=True,
                content=(
                    '{"token":"secret","path":"'
                    + str(tmp_path / "private-note.txt")
                    + '"}'
                ),
            ),
            final_gate="allow",
            approval_consumed=consumed,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke, gate=gate
    )
    preview = service.prepare_hub_test(tool)

    outcome = await service.execute_prepared_hub_test(
        preview.nonce, intent, {"path": "note.txt", "api_key": "never-store"}
    )

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.decision == expected_decision
    assert outcome.status == "success"
    assert outcome.final_gate == "allow"
    assert outcome.approval_consumed is approval_consumed
    assert outcome.dispatch_started is True
    assert outcome.provider_terminal == "returned"
    assert "secret" not in outcome.result.content
    assert str(tmp_path) not in outcome.result.content
    assert "never-store" not in repr(outcome)
    assert calls == [("fs_read", {"api_key": "never-store", "path": "note.txt"})]
    if gate == "allow":
        assert callbacks[-1] is None
    else:
        assert callable(callbacks[-1])
    assert len(closed) == len(callbacks)
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["decision"] == expected_decision
    assert records[0]["status"] == "success"


@pytest.mark.asyncio
async def test_local_hub_rendered_ask_fresh_allow_uses_final_allowed_decision(
    tmp_path, monkeypatch
):
    service, _fake, _client, store = _service(tmp_path)
    gate = {"value": "ask"}
    calls = []

    def _invoke(provider, tool_id, arguments):
        calls.append((tool_id, dict(arguments)))
        assert provider.approval_callback is None
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="done"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        gate=lambda: gate["value"],
    )
    preview = service.prepare_hub_test(tool)
    callbacks.clear()
    closed.clear()
    gate["value"] = "allow"

    outcome = await service.execute_prepared_hub_test(preview.nonce, "approve_once", {})

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.decision == "allowed"
    assert outcome.approval_consumed is False
    assert outcome.dispatch_started is True
    assert calls == [("fs_read", {})]
    assert callbacks == [None, None, None]
    assert len(closed) == len(callbacks)
    assert _log_records(store)[0]["decision"] == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "terminal", "expected_status", "expected_category"),
    [
        (
            LocalToolInvocationReason.PERMISSION_OFF,
            LocalProviderTerminal.NOT_STARTED,
            "blocked",
            "permission_off",
        ),
        (
            LocalToolInvocationReason.HANDLER_RAISED,
            LocalProviderTerminal.RAISED,
            "error",
            "execution_failed",
        ),
    ],
)
async def test_local_hub_refusal_and_crash_use_structured_facts_not_result_text(
    tmp_path,
    monkeypatch,
    reason,
    terminal,
    expected_status,
    expected_category,
):
    service, _fake, _client, store = _service(tmp_path)
    sentinel = f"looks timed out API_KEY=secret {tmp_path / 'private.txt'}"

    def _invoke(_provider, _tool_id, _arguments):
        return LocalToolInvocationResult(
            result=ToolResult(ok=False, error=sentinel),
            final_gate="off",
            approval_consumed=False,
            reason_code=reason,
            dispatch_started=terminal is LocalProviderTerminal.RAISED,
            provider_terminal=terminal,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    preview = service.prepare_hub_test(tool)

    outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.status == expected_status
    assert outcome.error_category == expected_category
    assert outcome.provider_terminal == terminal.value
    assert outcome.decision == (
        "denied" if terminal is LocalProviderTerminal.NOT_STARTED else "allowed"
    )
    assert "secret" not in outcome.result.error
    assert str(tmp_path) not in outcome.result.error
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == expected_status
    assert records[0]["error_category"] == expected_category
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)


@pytest.mark.asyncio
async def test_local_hub_audit_append_failure_does_not_mask_terminal(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)

    class _RaisingExecutionLog:
        def __init__(self):
            self.calls = 0

        def append(self, _record):
            self.calls += 1
            raise OSError("disk full")

    def _invoke(_provider, _tool_id, _arguments):
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="completed"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    execution_log = _RaisingExecutionLog()
    service._execution_log = execution_log
    preview = service.prepare_hub_test(tool)

    outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.status == "success"
    assert outcome.result.content == "completed"
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)
    assert service.hub_test_active("local:__local__", "fs_read") is False
    assert execution_log.calls == 1


@pytest.mark.asyncio
async def test_local_hub_terminal_audit_append_is_awaited_off_loop_before_release(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    ui_thread = threading.get_ident()
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(
                (
                    threading.get_ident(),
                    service.hub_test_active("local:__local__", "fs_read"),
                )
            )
            release_append.wait(timeout=1)

    def _invoke(_provider, _tool_id, _arguments):
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="done"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    preview = service.prepare_hub_test(tool)
    service._execution_log = _BlockingLog()
    timer = threading.Timer(0.05, release_append.set)
    timer.start()
    try:
        outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    finally:
        timer.cancel()
        release_append.set()

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert attempts == [(attempts[0][0], True)]
    assert attempts[0][0] != ui_thread
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_concurrent_first_audits_share_one_execution_log(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    constructor_started = threading.Event()
    release_constructor = threading.Event()
    instances = []
    rows = []
    rows_lock = threading.Lock()

    class _RacingExecutionLog:
        def __init__(self, path):
            self.path = path
            instances.append(self)
            constructor_started.set()
            release_constructor.wait(timeout=2)

        def append(self, record):
            with rows_lock:
                rows.append(record)

    monkeypatch.setattr(control_plane_module, "MCPExecutionLog", _RacingExecutionLog)
    outcome = LocalHubExecutionOutcome(
        decision="allowed",
        status="success",
        error_category=None,
        final_gate="allow",
        approval_consumed=False,
        dispatch_started=True,
        provider_terminal="returned",
        duration_ms=1,
        result=ToolResult(ok=True, content="done"),
    )

    def _tool(name):
        return HubTool(
            server_key="local:__local__",
            server_label="Local workspace",
            source="local",
            name=name,
            description=name,
            input_schema={"type": "object", "properties": {}},
            tags=(),
            stale=False,
            executable=True,
        )

    audits = [
        asyncio.create_task(
            service._attempt_local_hub_audit(
                lambda tool=_tool(name): service._record_local_hub_outcome(
                    tool, {}, outcome
                ),
                f"audit {name}",
            )
        )
        for name in ("fs_read", "fs_list")
    ]
    assert await asyncio.to_thread(constructor_started.wait, 1)
    await asyncio.sleep(0.03)
    release_constructor.set()
    await asyncio.gather(*audits)

    assert len(instances) == 1
    assert {row.tool_name for row in rows} == {"fs_read", "fs_list"}
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_local_hub_review_audit_append_is_awaited_off_loop(tmp_path, monkeypatch):
    service, _fake, _client, _store = _service(tmp_path)
    ui_thread = threading.get_ident()
    gate = {"value": "allow"}
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(threading.get_ident())
            release_append.wait(timeout=1)

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=lambda *_args: pytest.fail("review reached invocation"),
        gate=lambda: gate["value"],
    )
    preview = service.prepare_hub_test(tool)
    gate["value"] = "deny"
    service._execution_log = _BlockingLog()
    timer = threading.Timer(0.05, release_append.set)
    timer.start()
    try:
        outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    finally:
        timer.cancel()
        release_append.set()

    assert isinstance(outcome, ToolTestAdmissionStale)
    assert outcome.reason == "gate_changed"
    assert len(attempts) == 1
    assert attempts[0] != ui_thread
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_invalid_intent_audit_append_is_awaited_off_loop(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    ui_thread = threading.get_ident()
    attempts = []

    class _RecordingLog:
        def append(self, _record):
            attempts.append(threading.get_ident())

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=lambda *_args: pytest.fail("invalid intent reached invocation"),
    )
    preview = service.prepare_hub_test(tool)
    service._execution_log = _RecordingLog()

    outcome = await service.execute_prepared_hub_test(
        preview.nonce,
        "invalid",
        {},  # type: ignore[arg-type]
    )

    assert isinstance(outcome, ToolTestAdmissionBlocked)
    assert outcome.reason == "intent_invalid"
    assert len(attempts) == 1
    assert attempts[0] != ui_thread


@pytest.mark.asyncio
async def test_local_hub_invalid_intent_cancellation_waits_audit_then_propagates(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    append_started = threading.Event()
    append_finished = threading.Event()
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(True)
            append_started.set()
            release_append.wait(timeout=2)
            append_finished.set()

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=lambda *_args: pytest.fail("invalid intent reached invocation"),
    )
    preview = service.prepare_hub_test(tool)
    service._execution_log = _BlockingLog()
    task = asyncio.create_task(
        service.execute_prepared_hub_test(
            preview.nonce,
            "invalid",
            {},  # type: ignore[arg-type]
        )
    )
    assert await asyncio.to_thread(append_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.01)
    assert task.done() is False
    release_append.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert append_finished.is_set()
    assert attempts == [True]
    retry = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    assert isinstance(retry, ToolTestAdmissionStale)
    assert attempts == [True]
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_owner_cancellation_publishes_review_before_propagating(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    gate = {"value": "allow"}
    append_started = threading.Event()
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(True)
            append_started.set()
            release_append.wait(timeout=2)

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=lambda *_args: pytest.fail("review reached invocation"),
        gate=lambda: gate["value"],
    )
    preview = service.prepare_hub_test(tool)
    gate["value"] = "deny"
    service._execution_log = _BlockingLog()
    caller = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(append_started.wait, 1)
    [owner] = list(service._local_hub_execution._tasks)
    owner.cancel()
    release_append.set()

    outcome = await caller
    await asyncio.sleep(0)

    assert isinstance(outcome, ToolTestAdmissionStale)
    assert outcome.reason == "gate_changed"
    assert owner.cancelled() is True
    assert attempts == [True]
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_duplicate_audit_append_is_awaited_off_loop_once(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    ui_thread = threading.get_ident()
    handler_started = threading.Event()
    release_handler = threading.Event()
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(threading.get_ident())
            if len(attempts) == 1:
                release_append.wait(timeout=1)

    def _invoke(_provider, _tool_id, _arguments):
        handler_started.set()
        release_handler.wait(timeout=2)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="done"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    first_preview = service.prepare_hub_test(tool)
    first = asyncio.create_task(
        service.execute_prepared_hub_test(first_preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(handler_started.wait, 1)
    duplicate_preview = service.prepare_hub_test(tool)
    service._execution_log = _BlockingLog()
    timer = threading.Timer(0.05, release_append.set)
    timer.start()
    try:
        duplicate = await service.execute_prepared_hub_test(
            duplicate_preview.nonce, "run", {}
        )
    finally:
        timer.cancel()
        release_append.set()

    assert isinstance(duplicate, LocalHubExecutionOutcome)
    assert duplicate.error_category == "already_active"
    assert len(attempts) == 1
    assert attempts[0] != ui_thread
    assert service.hub_test_active("local:__local__", "fs_read") is True
    release_handler.set()
    await first
    assert len(attempts) == 2
    assert all(thread_id != ui_thread for thread_id in attempts)
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_duplicate_cancellation_waits_audit_then_propagates(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    handler_started = threading.Event()
    release_handler = threading.Event()
    append_started = threading.Event()
    append_finished = threading.Event()
    release_append = threading.Event()
    attempts = []

    class _BlockingLog:
        def append(self, _record):
            attempts.append(True)
            if len(attempts) == 1:
                append_started.set()
                release_append.wait(timeout=2)
                append_finished.set()

    def _invoke(_provider, _tool_id, _arguments):
        handler_started.set()
        release_handler.wait(timeout=2)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="done"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    first_preview = service.prepare_hub_test(tool)
    first = asyncio.create_task(
        service.execute_prepared_hub_test(first_preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(handler_started.wait, 1)
    duplicate_preview = service.prepare_hub_test(tool)
    service._execution_log = _BlockingLog()
    duplicate = asyncio.create_task(
        service.execute_prepared_hub_test(duplicate_preview.nonce, "run", {})
    )
    try:
        assert await asyncio.to_thread(append_started.wait, 1)
        duplicate.cancel()
        await asyncio.sleep(0.01)
        assert duplicate.done() is False
        release_append.set()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        retry = await service.execute_prepared_hub_test(
            duplicate_preview.nonce, "run", {}
        )
        assert isinstance(retry, ToolTestAdmissionStale)
        assert append_finished.is_set()
        assert attempts == [True]
        assert service.hub_test_active("local:__local__", "fs_read") is True
    finally:
        release_append.set()
        release_handler.set()
        await first

    assert attempts == [True, True]
    assert service.hub_test_active("local:__local__", "fs_read") is False


def test_local_hub_recursive_result_redaction_shares_one_payload_with_audit(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    root = tmp_path.resolve()
    secret = "TOPSECRET"
    raw = ToolResult(
        ok=True,
        content=(
            '[{"nested":[{"api_key":"'
            + secret
            + '","paths":["'
            + str(root / "private.txt")
            + '"]}],"authorization":"Bearer hidden","padding":"'
            + ("x" * 40_000)
            + '"}]'
        ),
        error=(
            '{"outer":[{"credential":"' + secret + '","root":"' + str(root) + '"}]}'
        ),
    )
    detail = LocalToolInvocationResult(
        result=raw,
        final_gate="allow",
        approval_consumed=False,
        reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
        dispatch_started=True,
        provider_terminal=LocalProviderTerminal.RETURNED,
    )
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read",
        input_schema={"type": "object"},
        tags=(),
        stale=False,
        executable=True,
    )
    audited = []

    def _capture(*_args, **kwargs):
        audited.append(kwargs["result"])

    monkeypatch.setattr(service, "_record_tool_execution", _capture)

    outcome = service._local_hub_outcome_from_detail(detail, root, 1)
    service._record_local_hub_outcome(
        tool,
        {
            "api_key": secret,
            "nested": [{"token": secret, "path": str(root / "argument.txt")}],
        },
        outcome,
    )

    assert audited == [outcome.result]
    assert audited[0] is outcome.result
    combined = outcome.result.content + outcome.result.error
    assert secret not in combined
    assert "Bearer hidden" not in combined
    assert str(root) not in combined
    assert '"api_key": "***"' in outcome.result.content
    assert '"credential": "***"' in outcome.result.error
    assert len(outcome.result.content.encode("utf-8")) < 33_000
    assert outcome.result.content.endswith("… [truncated]")


@pytest.mark.parametrize(
    "payload",
    [
        (
            '{"api_key":"TOPSECRET_API","token":"TOPSECRET_TOKEN",'
            '"password":"TOPSECRET_PASSWORD","authorization":'
            '"Bearer multi token secret value","padding":"' + ("x" * 40_000) + '"}'
        ),
        (
            '[{"nested":[{"api_key":"TOPSECRET_API"},{"token":'
            '"TOPSECRET_TOKEN"}],"password":"TOPSECRET_PASSWORD",'
            '"authorization":"Bearer multi token secret value","padding":"'
            + ("y" * 40_000)
            + '"}]'
        ),
        (
            '{"credentials":{"safe":"comma, bracket ]",'
            '"quoted":"escaped \\" bracket ] comma,",'
            '"opaque":"TOPSECRET_CONTAINER"},"padding":"' + ("z" * 40_000) + '"}'
        ),
        (
            '{"api_key_parts":["TOPSECRET_PART_ONE",'
            '"quoted comma, bracket ]",'
            '"escaped \\" bracket ] comma,","TOPSECRET_PART_TWO",'
            '{"opaque":"TOPSECRET_NESTED"},"padding ' + ("w" * 40_000)
        ),
    ],
)
def test_real_local_provider_truncated_json_fragments_are_secret_safe(
    tmp_path, monkeypatch, payload
):
    service, _fake, _client, _store = _service(tmp_path)

    class _Executor:
        def execute(self, _operation, _arguments, *, intent):
            assert intent == "read"
            return payload

    monkeypatch.setattr(
        local_tool_provider_module,
        "WorkspaceToolExecutor",
        lambda _root: _Executor(),
    )
    handle = local_server_tools_module.build_hub_local_provider(
        tmp_path,
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        approval_callback=None,
    )
    try:
        detail = handle.provider.invoke_detailed("fs_read", {"path": "note.txt"})
        root = handle.authority.canonical_root
    finally:
        handle.close()

    assert detail.provider_terminal is LocalProviderTerminal.RETURNED
    assert detail.result.content.endswith("… [truncated]")
    audited = []
    monkeypatch.setattr(
        service,
        "_record_tool_execution",
        lambda *_args, **kwargs: audited.append(kwargs["result"]),
    )
    outcome = service._local_hub_outcome_from_detail(detail, root, 1)
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )
    service._record_local_hub_outcome(tool, {"path": "note.txt"}, outcome)

    assert audited == [outcome.result]
    combined = outcome.result.content + outcome.result.error
    for secret in (
        "TOPSECRET_API",
        "TOPSECRET_TOKEN",
        "TOPSECRET_PASSWORD",
        "multi token secret value",
        "TOPSECRET_CONTAINER",
        "TOPSECRET_PART_ONE",
        "TOPSECRET_PART_TWO",
        "TOPSECRET_NESTED",
    ):
        assert secret not in combined
    assert len(outcome.result.content.encode("utf-8")) < 33_000


@pytest.mark.asyncio
async def test_local_hub_entire_rebuild_gate_compare_policy_and_invoke_are_off_loop(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    ui_thread = threading.get_ident()
    threads = []

    def _invoke(_provider, _tool_id, _arguments):
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="done"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, _callbacks, _guards, _closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        threads=threads,
    )
    preview = service.prepare_hub_test(tool)
    threads.clear()

    outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.status == "success"
    labels = {label for label, _thread in threads}
    assert {
        "provider_construction",
        "root_resolution",
        "authority_compare",
        "definition_compare",
        "gate_revalidation",
        "execution_policy",
        "timeout_floor",
        "invoke_detailed",
    } <= labels
    assert all(thread_id != ui_thread for _label, thread_id in threads)


@pytest.mark.asyncio
async def test_local_hub_already_cancelled_after_nonce_consumption_is_owned_and_audited(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    handler_calls = []

    def _invoke(_provider, _tool_id, _arguments):
        handler_calls.append(True)
        raise AssertionError("already-cancelled request reached handler")

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    preview = service.prepare_hub_test(tool)
    original_consume = service._hub_test_previews.consume
    original_record = service._record_local_hub_outcome
    terminals = []

    def _consume_and_cancel(nonce):
        registered = original_consume(nonce)
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return registered

    def _capture(tool, arguments, outcome):
        terminals.append(outcome)
        return original_record(tool, arguments, outcome)

    monkeypatch.setattr(service._hub_test_previews, "consume", _consume_and_cancel)
    monkeypatch.setattr(service, "_record_local_hub_outcome", _capture)

    with pytest.raises(asyncio.CancelledError):
        await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert handler_calls == []
    assert len(terminals) == 1
    assert isinstance(terminals[0], LocalHubExecutionOutcome)
    assert terminals[0].status == "cancelled"
    assert terminals[0].dispatch_started is False
    assert terminals[0].provider_terminal == "not_started"
    assert original_consume(preview.nonce) is None
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_cancellation_during_owned_review_audits_and_cleans(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)
    handler_calls = []

    def _invoke(_provider, _tool_id, _arguments):
        handler_calls.append(True)
        raise AssertionError("review cancellation reached handler")

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    preview = service.prepare_hub_test(tool)
    callbacks.clear()
    closed.clear()
    original_resolve = service._resolve_hub_test
    original_record = service._record_local_hub_outcome
    review_started = threading.Event()
    release_review = threading.Event()
    terminals = []

    def _blocked_review(server_key, tool_name):
        review_started.set()
        release_review.wait(timeout=2)
        return original_resolve(server_key, tool_name)

    def _capture(tool, arguments, outcome):
        terminals.append(outcome)
        return original_record(tool, arguments, outcome)

    monkeypatch.setattr(service, "_resolve_hub_test", _blocked_review)
    monkeypatch.setattr(service, "_record_local_hub_outcome", _capture)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(review_started.wait, 1)
    assert service.hub_test_active("local:__local__", "fs_read") is True

    pending.cancel()
    release_review.set()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert handler_calls == []
    assert len(terminals) == 1
    assert terminals[0].status == "cancelled"
    assert terminals[0].dispatch_started is False
    assert terminals[0].provider_terminal == "not_started"
    assert len(closed) == len(callbacks) == 2
    assert len({id(handle) for handle in closed}) == 2
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_timeout_seals_once_and_late_worker_is_cleanup_only(
    tmp_path, monkeypatch
):
    service, _fake, _client, store = _service(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def _invoke(_provider, _tool_id, _arguments):
        started.set()
        release.wait(timeout=2)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="late secret"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    monkeypatch.setattr(service, "_lifecycle_timeout", lambda: 0.01)
    preview = service.prepare_hub_test(tool)

    outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert started.is_set()
    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.status == "timeout"
    assert outcome.error_category == "timeout"
    assert outcome.final_gate == "allow"
    assert outcome.approval_consumed is False
    assert outcome.dispatch_started is True
    assert outcome.provider_terminal == "not_started"
    assert service.hub_test_active("local:__local__", "fs_read") is True
    assert len(_log_records(store)) == 1
    duplicate_preview = service.prepare_hub_test(tool)
    duplicate = await service.execute_prepared_hub_test(
        duplicate_preview.nonce, "run", {}
    )
    assert isinstance(duplicate, LocalHubExecutionOutcome)
    assert duplicate.error_category == "already_active"
    assert len(_log_records(store)) == 2
    release.set()
    for _ in range(100):
        if not service.hub_test_active("local:__local__", "fs_read"):
            break
        await asyncio.sleep(0.01)
    assert service.hub_test_active("local:__local__", "fs_read") is False
    assert len(_log_records(store)) == 2
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)


@pytest.mark.asyncio
async def test_local_hub_timeout_floor_longer_than_lifecycle_wins(
    tmp_path, monkeypatch
):
    service, _fake, _client, _store = _service(tmp_path)

    def _invoke(_provider, _tool_id, _arguments):
        threading.Event().wait(0.03)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="within provider floor"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        timeout_floor=0.2,
    )
    monkeypatch.setattr(service, "_lifecycle_timeout", lambda: 0.01)
    preview = service.prepare_hub_test(tool)

    outcome = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(outcome, LocalHubExecutionOutcome)
    assert outcome.status == "success"
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)


@pytest.mark.asyncio
async def test_local_hub_lifecycle_deadline_covers_policy_review_before_dispatch(
    tmp_path, monkeypatch
):
    service, _fake, _client, store = _service(tmp_path)
    policy_started = threading.Event()
    release_policy = threading.Event()
    handler_calls = []

    def _policy():
        policy_started.set()
        release_policy.wait(timeout=2)
        return ToolExecutionPolicy.BOUNDED_ABANDONABLE

    def _invoke(_provider, _tool_id, _arguments):
        handler_calls.append(True)
        raise AssertionError("timed-out review reached handler")

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        policy=_policy,
    )
    monkeypatch.setattr(service, "_lifecycle_timeout", lambda: 0.01)
    preview = service.prepare_hub_test(tool)
    callbacks.clear()
    closed.clear()

    task = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(policy_started.wait, 1)
    await asyncio.sleep(0.04)
    try:
        assert task.done(), "the absolute lifecycle deadline did not seal review"
        outcome = task.result()
        assert isinstance(outcome, LocalHubExecutionOutcome)
        assert outcome.status == "timeout"
        assert outcome.dispatch_started is False
        assert outcome.provider_terminal == "not_started"
        assert handler_calls == []
        assert len(_log_records(store)) == 1
        assert service.hub_test_active("local:__local__", "fs_read") is True
        assert len(closed) + 1 == len(callbacks)
    finally:
        release_policy.set()
    if not task.done():
        await task
    for _ in range(100):
        if not service.hub_test_active("local:__local__", "fs_read"):
            break
        await asyncio.sleep(0.01)
    assert service.hub_test_active("local:__local__", "fs_read") is False
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)
    assert len(_log_records(store)) == 1


@pytest.mark.asyncio
async def test_local_hub_cancellation_during_construction_never_starts_handler(
    tmp_path, monkeypatch
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    service, _fake, _client, store = _service(tmp_path)
    construction_started = threading.Event()
    release_construction = threading.Event()
    handler_calls = []

    def _invoke(_provider, _tool_id, _arguments):
        handler_calls.append(True)
        raise AssertionError("pre-dispatch cancellation reached the handler")

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    build = local_server_tools.build_hub_local_provider

    def _blocked_build(*args, **kwargs):
        if kwargs.get("dispatch_guard") is None:
            return build(*args, **kwargs)
        construction_started.set()
        release_construction.wait(timeout=2)
        return build(*args, **kwargs)

    monkeypatch.setattr(local_server_tools, "build_hub_local_provider", _blocked_build)
    preview = service.prepare_hub_test(tool)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(construction_started.wait, 1)
    assert service.hub_test_active("local:__local__", "fs_read") is True

    pending.cancel()
    release_construction.set()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert handler_calls == []
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == "cancelled"
    for _ in range(100):
        if not service.hub_test_active("local:__local__", "fs_read"):
            break
        await asyncio.sleep(0.01)
    assert service.hub_test_active("local:__local__", "fs_read") is False


@pytest.mark.asyncio
async def test_local_hub_bounded_cancellation_audits_before_detach_and_late_return_is_cleanup_only(
    tmp_path, monkeypatch
):
    service, _fake, _client, store = _service(tmp_path)
    started = threading.Event()
    release = threading.Event()
    terminals = []
    original_record = service._record_local_hub_outcome

    def _capture(tool, arguments, outcome):
        terminals.append(outcome)
        return original_record(tool, arguments, outcome)

    monkeypatch.setattr(service, "_record_local_hub_outcome", _capture)

    def _invoke(_provider, _tool_id, _arguments):
        started.set()
        release.wait(timeout=2)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="late"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch, tmp_path, service, invoke_detailed=_invoke
    )
    preview = service.prepare_hub_test(tool)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(started.wait, 1)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["status"] == "cancelled"
    assert len(terminals) == 1
    assert terminals[0].status == "cancelled"
    assert terminals[0].error_category == "cancelled"
    assert terminals[0].final_gate == "allow"
    assert terminals[0].approval_consumed is False
    assert terminals[0].dispatch_started is True
    assert terminals[0].provider_terminal == "not_started"
    assert service.hub_test_active("local:__local__", "fs_read") is True
    release.set()
    for _ in range(100):
        if not service.hub_test_active("local:__local__", "fs_read"):
            break
        await asyncio.sleep(0.01)
    assert len(_log_records(store)) == 1
    assert service.hub_test_active("local:__local__", "fs_read") is False
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)


@pytest.mark.asyncio
async def test_local_hub_definitive_after_start_detaches_caller_but_audits_actual_terminal(
    tmp_path, monkeypatch
):
    service, _fake, _client, store = _service(tmp_path)
    started = threading.Event()
    release = threading.Event()
    terminals = []
    original_record = service._record_local_hub_outcome

    def _capture(tool, arguments, outcome):
        terminals.append(outcome)
        return original_record(tool, arguments, outcome)

    monkeypatch.setattr(service, "_record_local_hub_outcome", _capture)

    def _invoke(_provider, _tool_id, _arguments):
        started.set()
        release.wait(timeout=2)
        return LocalToolInvocationResult(
            result=ToolResult(ok=True, content="committed"),
            final_gate="allow",
            approval_consumed=False,
            reason_code=LocalToolInvocationReason.HANDLER_RETURNED,
            dispatch_started=True,
            provider_terminal=LocalProviderTerminal.RETURNED,
        )

    tool, callbacks, _guards, closed = _install_local_hub_execution_provider(
        monkeypatch,
        tmp_path,
        service,
        invoke_detailed=_invoke,
        policy=ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )
    monkeypatch.setattr(service, "_lifecycle_timeout", lambda: 0.01)
    preview = service.prepare_hub_test(tool)
    pending = asyncio.create_task(
        service.execute_prepared_hub_test(preview.nonce, "run", {})
    )
    assert await asyncio.to_thread(started.wait, 1)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.sleep(0.03)
    assert _log_records(store) == []
    assert service.hub_test_active("local:__local__", "fs_read") is True

    release.set()
    for _ in range(100):
        records = _log_records(store)
        if records:
            break
        await asyncio.sleep(0.01)
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["error_category"] is None
    assert len(terminals) == 1
    assert terminals[0].status == "success"
    assert terminals[0].error_category is None
    assert terminals[0].final_gate == "allow"
    assert terminals[0].approval_consumed is False
    assert terminals[0].dispatch_started is True
    assert terminals[0].provider_terminal == "returned"
    for _ in range(100):
        if not service.hub_test_active("local:__local__", "fs_read"):
            break
        await asyncio.sleep(0.01)
    assert service.hub_test_active("local:__local__", "fs_read") is False
    assert len(closed) == len(callbacks)
    assert len({id(handle) for handle in closed}) == len(closed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("disabled", "local_tools_disabled"),
        ("configuration", "local_configuration_unavailable"),
        ("provider", "local_provider_unavailable"),
        ("ineligible", "local_tool_ineligible"),
    ],
)
async def test_prepared_local_click_time_configuration_failures_never_reach_handler(
    tmp_path, monkeypatch, failure, expected_reason
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    service, _fake, _client, store = _service(tmp_path)
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read a file",
        input_schema=None,
        tags=("reads",),
        stale=False,
        executable=True,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    authority = capture_directory_chain(tmp_path)
    state = {"enabled": True, "fail": None}

    class _InspectionProvider:
        def hub_tools(self):
            return [tool]

    class _ExecutableProvider:
        def hub_tools(self):
            return [] if state["fail"] == "ineligible" else [tool]

    class _InspectionHandle:
        provider = _InspectionProvider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    class _ExecutableHandle:
        provider = _ExecutableProvider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    def _root():
        return tmp_path

    def _provider(*_args, **_kwargs):
        if state["fail"] == "provider":
            raise RuntimeError("provider unavailable")
        return _ExecutableHandle()

    def _setting(section, key, default=None):
        if state["fail"] == "configuration":
            raise RuntimeError("configuration unavailable")
        if (section, key) == ("console", "local_tools_enabled"):
            return state["enabled"]
        return default

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _InspectionHandle(),
    )
    monkeypatch.setattr(local_server_tools, "build_hub_local_provider", _provider)
    monkeypatch.setattr(local_server_tools, "resolve_server_workspace_root", _root)
    monkeypatch.setattr(control_plane_module, "get_cli_setting", _setting)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    local_handler = AsyncMock(return_value={"should": "not run"})
    nonlocal_handler = AsyncMock(return_value={"should": "not run"})
    service._execute_prepared_local_hub_test = local_handler
    service.test_hub_tool = nonlocal_handler
    preview = service.prepare_hub_test(tool)
    if failure == "disabled":
        state["enabled"] = False
    else:
        state["fail"] = failure

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionStale)
    assert result.reason == expected_reason
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "unavailable"
    local_handler.assert_not_awaited()
    nonlocal_handler.assert_not_awaited()
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["initiator"] == "test"
    assert records[0]["error_category"] == expected_reason
    assert result.refreshed_preview.safe_authority_label is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rendered_state", "expected_reason"),
    [
        ("unavailable", "permission_unresolved"),
        ("unresolved", "permission_unresolved"),
        ("off", "permission_denied"),
    ],
)
async def test_prepared_local_non_actionable_render_wins_over_fresh_unavailable(
    tmp_path, monkeypatch, rendered_state, expected_reason
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    service, _fake, _client, store = _service(tmp_path)
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read a file",
        input_schema=None,
        tags=("reads",),
        stale=False,
        executable=True,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    authority = capture_directory_chain(tmp_path)
    state = {"enabled": rendered_state != "unavailable"}

    class _Provider:
        def hub_tools(self):
            return [tool]

    class _Handle:
        provider = _Provider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _Handle(),
    )
    monkeypatch.setattr(
        local_server_tools, "build_hub_local_provider", lambda *a, **k: _Handle()
    )
    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            state["enabled"]
            if (section, key) == ("console", "local_tools_enabled")
            else default
        ),
    )
    if rendered_state == "unresolved":
        service.gate_tool_test = lambda _tool: EffectiveToolState(
            state="ask", origin="gate_error"
        )
    elif rendered_state == "off":
        service.gate_tool_test = lambda _tool: EffectiveToolState(
            state="deny", origin="tool_override"
        )
    else:
        service.gate_tool_test = lambda _tool: EffectiveToolState(
            state="allow", origin="tool_override"
        )
    local_handler = AsyncMock(return_value={"should": "not run"})
    nonlocal_handler = AsyncMock(return_value={"should": "not run"})
    service._execute_prepared_local_hub_test = local_handler
    service.test_hub_tool = nonlocal_handler
    preview = service.prepare_hub_test(tool)
    state["enabled"] = False

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionBlocked)
    assert result.reason == expected_reason
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "unavailable"
    local_handler.assert_not_awaited()
    nonlocal_handler.assert_not_awaited()
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["initiator"] == "test"
    assert records[0]["error_category"] == expected_reason


@pytest.mark.asyncio
async def test_prepared_local_rendered_unavailable_stays_blocked_when_fresh_allow(
    tmp_path, monkeypatch
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    service, _fake, _client, store = _service(tmp_path)
    tool = HubTool(
        server_key="local:__local__",
        server_label="Local workspace",
        source="local",
        name="fs_read",
        description="Read a file",
        input_schema=None,
        tags=("reads",),
        stale=False,
        executable=True,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    authority = capture_directory_chain(tmp_path)
    enabled = {"value": False}

    class _Provider:
        def hub_tools(self):
            return [tool]

    class _Handle:
        provider = _Provider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _Handle(),
    )
    monkeypatch.setattr(
        local_server_tools, "build_hub_local_provider", lambda *a, **k: _Handle()
    )
    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            enabled["value"]
            if (section, key) == ("console", "local_tools_enabled")
            else default
        ),
    )
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    local_handler = AsyncMock(return_value={"should": "not run"})
    nonlocal_handler = AsyncMock(return_value={"should": "not run"})
    service._execute_prepared_local_hub_test = local_handler
    service.test_hub_tool = nonlocal_handler
    preview = service.prepare_hub_test(tool)
    enabled["value"] = True

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionBlocked)
    assert result.reason == "permission_unresolved"
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "allow"
    local_handler.assert_not_awaited()
    nonlocal_handler.assert_not_awaited()
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["error_category"] == "permission_unresolved"


@pytest.mark.asyncio
async def test_prepared_hub_gate_resolution_error_refreshes_unavailable_without_dispatch(
    tmp_path,
):
    service, fake, _client, _store = _service(tmp_path)
    tool = _prepared_external_tool()
    _install_prepared_external_catalog(fake, tool)
    gate_calls = 0

    def _gate(_tool):
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 1:
            return EffectiveToolState(state="allow", origin="tool_override")
        raise RuntimeError("permission store unavailable")

    service.gate_tool_test = _gate
    service.test_hub_tool = AsyncMock(return_value={"should": "not run"})
    preview = service.prepare_hub_test(tool)

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionStale)
    assert result.reason == "gate_changed"
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "unresolved"
    service.test_hub_tool.assert_not_awaited()
