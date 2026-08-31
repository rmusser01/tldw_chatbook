from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.hub_test_execution import (
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
async def test_prepared_local_default_seam_fails_closed_after_admission(
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

    assert isinstance(result, ToolTestAdmissionBlocked)
    assert result.reason == "local_execution_unavailable"
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["error_category"] == "local_execution_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disabled", "root", "provider"])
async def test_prepared_local_click_time_configuration_failures_never_reach_handler(
    tmp_path, monkeypatch, failure
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

    class _Provider:
        def hub_tools(self):
            return [tool]

    class _Handle:
        provider = _Provider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    def _root():
        if state["fail"] == "root":
            raise OSError("root unavailable")
        return tmp_path

    def _provider(*_args, **_kwargs):
        if state["fail"] == "provider":
            raise RuntimeError("provider unavailable")
        return _Handle()

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _Handle(),
    )
    monkeypatch.setattr(local_server_tools, "build_hub_local_provider", _provider)
    monkeypatch.setattr(local_server_tools, "resolve_server_workspace_root", _root)
    monkeypatch.setattr(
        control_plane_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            state["enabled"]
            if (section, key) == ("console", "local_tools_enabled")
            else default
        ),
    )
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    local_handler = AsyncMock(return_value={"should": "not run"})
    service._execute_prepared_local_hub_test = local_handler
    preview = service.prepare_hub_test(tool)
    if failure == "disabled":
        state["enabled"] = False
    else:
        state["fail"] = failure

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionStale)
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "unavailable"
    local_handler.assert_not_awaited()
    records = _log_records(store)
    assert len(records) == 1
    assert records[0]["initiator"] == "test"


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
