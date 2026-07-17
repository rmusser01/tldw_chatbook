"""Tests for `tldw_chatbook.Agents.mcp_tool_provider.MCPToolProvider` (task-201).

Uses a small fake service whose method signatures mirror
`UnifiedMCPControlPlaneService`'s real ones exactly (no `**kwargs` masking)
so a signature drift in the provider under test would fail loudly here
rather than being hidden by a permissive fake.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import (
    DENY_REFUSAL,
    KILL_SWITCH_REFUSAL,
    MCPPendingCall,
    MCPToolProvider,
    NON_TEXT_PLACEHOLDER,
    TIMEOUT_REFUSAL,
)
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def _catalog_record(profile_id: str, tools: list[dict], *, is_connected: bool = True) -> dict:
    return {
        "profile_id": profile_id,
        "is_connected": is_connected,
        "discovery_snapshot": {"tools": tools},
    }


def _tool_dict(name: str, description: str = "", input_schema: dict | None = None) -> dict:
    raw = {"name": name, "description": description}
    if input_schema is not None:
        raw["inputSchema"] = input_schema
    return raw


class FakeMCPService:
    """Mirrors the real `UnifiedMCPControlPlaneService` seams `MCPToolProvider` consumes."""

    def __init__(
        self,
        *,
        kill_switch: bool = False,
        catalog_records: list[dict] | None = None,
        inventory: dict | None = None,
        states: dict[tuple[str, str], EffectiveToolState] | None = None,
        default_state: EffectiveToolState | None = None,
        tool_call_timeout: float = 5.0,
        execute_result: dict | None = None,
        execute_raises: Exception | None = None,
        execute_delay: float = 0.0,
    ) -> None:
        self.kill_switch = kill_switch
        self.catalog_records = catalog_records or []
        self.inventory = inventory if inventory is not None else {"tools": []}
        self.states = states or {}
        self.default_state = default_state or EffectiveToolState(state="ask", origin="global_default")
        self.tool_call_timeout = tool_call_timeout
        self.execute_result = (
            execute_result if execute_result is not None
            else {"content": [{"type": "text", "text": "ok"}]}
        )
        self.execute_raises = execute_raises
        self.execute_delay = execute_delay

        self.local_service = SimpleNamespace(get_inventory=lambda: self.inventory)
        self.session_approvals: set[tuple[str, str]] = set()
        self.set_tool_state_calls: list[tuple] = []
        self.approve_for_session_calls: list[tuple] = []
        self.record_tool_decision_calls: list[tuple] = []
        self.execute_calls: list[tuple] = []

    def get_kill_switch(self) -> bool:
        return self.kill_switch

    async def local_external_catalog(self) -> list[dict]:
        return self.catalog_records

    def effective_tool_states(self, tools: list[HubTool]) -> dict[tuple[str, str], EffectiveToolState]:
        return {
            (t.server_key, t.name): self.states.get((t.server_key, t.name), self.default_state)
            for t in tools
        }

    def gate_tool_test(self, tool: HubTool) -> EffectiveToolState:
        return self.states.get((tool.server_key, tool.name), self.default_state)

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self.approve_for_session_calls.append((server_key, tool_name))
        self.session_approvals.add((server_key, tool_name))

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self.session_approvals

    def set_tool_state(self, server_key: str, tool_name: str, ui_state, *, tool=None) -> None:
        self.set_tool_state_calls.append((server_key, tool_name, ui_state, tool))

    def record_tool_decision(
        self, server_key: str, tool_name: str, *, decision: str,
        initiator: str = "agent", error: str | None = None,
    ) -> None:
        self.record_tool_decision_calls.append((server_key, tool_name, decision, initiator, error))

    def _tool_call_timeout(self) -> float:
        return self.tool_call_timeout

    async def execute_hub_tool(
        self, server_key: str, tool_name: str, arguments: dict | None = None, *,
        initiator: str = "test", decision: str = "allowed", timeout_seconds: float | None = None,
    ) -> dict:
        self.execute_calls.append((server_key, tool_name, dict(arguments or {}), initiator, decision))
        if self.execute_delay:
            await asyncio.sleep(self.execute_delay)
        if self.execute_raises is not None:
            raise self.execute_raises
        return self.execute_result


@pytest.fixture
def running_loop():
    """A real event loop running forever in a background thread (the "main loop" stand-in)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(timeout=2), "background loop failed to start"
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _compose(provider: MCPToolProvider) -> None:
    asyncio.run(provider.compose_catalog())


# ---------------------------------------------------------------------------
# compose_catalog
# ---------------------------------------------------------------------------

def test_compose_catalog_kill_switch_on_yields_empty_catalog():
    service = FakeMCPService(
        kill_switch=True,
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    assert provider.list_catalog() == []
    assert provider.not_connected_count == 0


def test_compose_catalog_filters_deny_state():
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("keep"), _tool_dict("drop")])],
        states={
            ("local:srv", "keep"): EffectiveToolState(state="allow", origin="tool_override"),
            ("local:srv", "drop"): EffectiveToolState(state="deny", origin="tool_override"),
        },
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    names = {e.name for e in provider.list_catalog()}
    assert any("keep" in n for n in names)
    assert not any("drop" in n for n in names)


def test_compose_catalog_includes_builtin_inventory():
    service = FakeMCPService(inventory={"tools": [{"name": "calc", "description": "adds"}]})
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    names = {e.name for e in provider.list_catalog()}
    assert any("calc" in n for n in names)


def test_compose_catalog_dedupes_colliding_llm_names():
    # "server.1" and "server_1" both sanitize to "server_1" -- a genuine
    # post-sanitization collision, exercising the T1 handoff requirement:
    # names are computed for ALL tools first, then ONE dedupe_names() pass
    # runs over the whole list.
    service = FakeMCPService(catalog_records=[
        _catalog_record("server.1", [_tool_dict("run")]),
        _catalog_record("server_1", [_tool_dict("run")]),
    ])
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    names = [e.name for e in provider.list_catalog()]
    assert len(names) == 2
    assert len(set(names)) == 2
    assert names[0] == "mcp__server_1__run"
    assert names[1] == "mcp__server_1__run_2"


def test_not_connected_count_counts_distinct_eligible_stale_servers():
    service = FakeMCPService(
        catalog_records=[
            _catalog_record("a", [_tool_dict("t1")], is_connected=False),
            _catalog_record("b", [_tool_dict("t2")], is_connected=False),
            _catalog_record("c", [_tool_dict("t3")], is_connected=True),
        ],
        states={("local:b", "t2"): EffectiveToolState(state="deny", origin="tool_override")},
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    # Server "b" is stale but its only tool is denied (ineligible) -> must
    # not count. Only server "a" (stale + eligible) counts.
    assert provider.not_connected_count == 1


# ---------------------------------------------------------------------------
# list_catalog / load_schema
# ---------------------------------------------------------------------------

def test_load_schema_defaults_missing_input_schema_to_empty_object():
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    schema = provider.load_schema(tool_id)
    assert schema.parameters == {"type": "object", "properties": {}}
    assert schema.id == tool_id
    assert schema.name == tool_id


def test_load_schema_passes_through_existing_input_schema():
    raw_schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run", input_schema=raw_schema)])]
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    schema = provider.load_schema(tool_id)
    assert schema.parameters == raw_schema


def test_load_schema_unknown_id_raises_key_error():
    provider = MCPToolProvider(service=FakeMCPService(), main_loop=asyncio.new_event_loop())
    with pytest.raises(KeyError):
        provider.load_schema("mcp__nope__run")


# ---------------------------------------------------------------------------
# apply_batch_decisions / consume_decision
# ---------------------------------------------------------------------------

def test_apply_batch_decisions_then_consume_once():
    provider = MCPToolProvider(service=FakeMCPService(), main_loop=asyncio.new_event_loop())
    provider.apply_batch_decisions({"mcp__srv__run": "deny"})
    assert provider.consume_decision("mcp__srv__run") == "deny"
    assert provider.consume_decision("mcp__srv__run") is None


def test_compose_catalog_clears_stale_stamped_decisions():
    # Finding 3: stale stamps for tools not in the new catalog should be
    # cleared on recompose to prevent auto-approval of old tool names.
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)

    # Stamp a bogus tool name not in the catalog.
    bogus_name = "mcp__srv__nonexistent"
    provider.apply_batch_decisions({bogus_name: "approve_once"})
    assert provider.consume_decision(bogus_name) == "approve_once"

    # Recompose: stale decision should be cleared.
    _compose(provider)
    assert provider.consume_decision(bogus_name) is None


# ---------------------------------------------------------------------------
# pending_gate_for
# ---------------------------------------------------------------------------

def test_pending_gate_for_returns_none_for_allow_and_deny():
    # Both tools start "ask" (the default) so BOTH survive compose_catalog's
    # own deny-filter and land in the catalog. pending_gate_for() re-resolves
    # the gate FRESH per call (never trusts the compose-time cache) --
    # mutating state after compose is exactly what this observes, and
    # mirrors invoke()'s own fresh-gate design.
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("allowed_tool"), _tool_dict("denied_tool")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    by_name = {e.name: e.id for e in provider.list_catalog()}
    allow_id = next(v for k, v in by_name.items() if "allowed_tool" in k)
    deny_id = next(v for k, v in by_name.items() if "denied_tool" in k)

    service.states[("local:srv", "allowed_tool")] = EffectiveToolState(state="allow", origin="tool_override")
    service.states[("local:srv", "denied_tool")] = EffectiveToolState(state="deny", origin="tool_override")

    assert provider.pending_gate_for(allow_id, {}) is None
    assert provider.pending_gate_for(deny_id, {}) is None


def test_pending_gate_for_ask_reports_config_changed_and_risk_floored_reasons():
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("cfg"), _tool_dict("risky")])],
        states={
            ("local:srv", "cfg"): EffectiveToolState(
                state="ask", origin="tool_override", config_changed=True),
            ("local:srv", "risky"): EffectiveToolState(
                state="ask", origin="server_default", risk_floored=True),
        },
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    by_name = {e.name: e.id for e in provider.list_catalog()}
    cfg_id = next(v for k, v in by_name.items() if "cfg" in k)
    risky_id = next(v for k, v in by_name.items() if "risky" in k)

    pending_cfg = provider.pending_gate_for(cfg_id, {"a": 1})
    pending_risky = provider.pending_gate_for(risky_id, {})

    assert isinstance(pending_cfg, MCPPendingCall)
    assert pending_cfg.reason == "config_changed"
    assert pending_cfg.arguments == {"a": 1}
    assert pending_risky.reason == "risk_floored"


def test_pending_gate_for_plain_ask_reason():
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    pending = provider.pending_gate_for(tool_id, {})
    assert pending.reason == "ask"


def test_pending_gate_for_unknown_name_returns_none():
    provider = MCPToolProvider(service=FakeMCPService(), main_loop=asyncio.new_event_loop())
    assert provider.pending_gate_for("mcp__nope__run", {}) is None


def test_pending_gate_for_returns_none_when_session_approved():
    # Finding I1: "Approve for session" must suppress the approval card on
    # the NEXT turn's `pending_gate_for` call -- the real service's
    # `gate_tool_test` resolves from the permission store only (mirrored
    # here: it never consults `session_approvals`), so `pending_gate_for`
    # itself must consult `is_session_approved` -- without this, the tool
    # re-prompts on every turn even after a session approval.
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    service.session_approvals.add(("local:srv", "run"))

    assert provider.pending_gate_for(tool_id, {}) is None


# ---------------------------------------------------------------------------
# invoke() -- unknown tool
# ---------------------------------------------------------------------------

def test_invoke_unknown_tool_id_returns_error():
    provider = MCPToolProvider(service=FakeMCPService(), main_loop=asyncio.new_event_loop())
    result = provider.invoke("mcp__nope__run", {})
    assert result.ok is False


# ---------------------------------------------------------------------------
# invoke() -- allow path
# ---------------------------------------------------------------------------

def test_invoke_allow_executes_and_returns_content(running_loop):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
        execute_result={"content": [{"type": "text", "text": "42"}]},
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {"x": 1})

    assert result.ok is True
    assert "42" in result.content
    assert service.execute_calls == [("local:srv", "run", {"x": 1}, "agent", "allowed")]


def test_invoke_non_text_result_returns_placeholder(running_loop):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
        execute_result={"content": [{"type": "image", "data": "base64=="}]},
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert result.content == NON_TEXT_PLACEHOLDER


def test_invoke_result_truncated_and_redacted(running_loop):
    huge_text = "x" * 5000
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
        execute_result={"content": [{"type": "text", "text": huge_text}], "api_key": "sekrit"},
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert len(result.content) == 4000
    assert "sekrit" not in result.content


# ---------------------------------------------------------------------------
# invoke() -- deny path
# ---------------------------------------------------------------------------

def test_invoke_deny_refuses_and_records_decision(running_loop):
    # Compose with "allow" so the tool actually lands in the catalog, then
    # flip the store to "deny" before invoking -- invoke() must gate FRESH
    # per call (Global Constraints: "Gate at call time, fresh per call"),
    # never trusting the compose-time cached state.
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        default_state=EffectiveToolState(state="allow", origin="tool_override"),
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    service.states[("local:srv", "run")] = EffectiveToolState(state="deny", origin="tool_override")

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == DENY_REFUSAL
    assert service.record_tool_decision_calls == [("local:srv", "run", "denied", "agent", None)]
    assert service.execute_calls == []


# ---------------------------------------------------------------------------
# invoke() -- ask path
# ---------------------------------------------------------------------------

def test_invoke_ask_without_callback_fails_closed(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=running_loop)  # no approval_callback
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == DENY_REFUSAL
    assert service.execute_calls == []
    assert service.record_tool_decision_calls == [("local:srv", "run", "denied", "agent", None)]


def test_invoke_ask_callback_approve_once_executes_and_records_approved(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    captured: dict = {}

    def approval_callback(pending: list[MCPPendingCall]) -> dict[str, str]:
        captured["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    provider = MCPToolProvider(service=service, main_loop=running_loop, approval_callback=approval_callback)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {"q": "hi"})

    assert result.ok is True
    assert len(captured["pending"]) == 1
    assert captured["pending"][0] == MCPPendingCall(
        llm_name=tool_id, server_key="local:srv", tool_name="run",
        server_label="srv", arguments={"q": "hi"}, reason="ask",
    )
    assert service.execute_calls == [("local:srv", "run", {"q": "hi"}, "agent", "approved")]
    # approve_once has no persisted side effect.
    assert service.set_tool_state_calls == []
    assert service.approve_for_session_calls == []


def test_invoke_ask_callback_approve_session_persists_and_short_circuits_next_call(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(
        service=service, main_loop=running_loop,
        approval_callback=lambda pending: {p.llm_name: "approve_session" for p in pending},
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})
    assert result.ok is True
    assert service.approve_for_session_calls == [("local:srv", "run")]

    # A later call with no stamp and NO callback still executes: the
    # service's own is_session_approved() short-circuits the "ask" gate.
    # Finding I1: this fresh-path execution records decision="approved"
    # (a live session approval), NOT "allowed" (a persistent server
    # default) -- the two are different entries in the decision
    # vocabulary and must not collapse together.
    provider._approval_callback = None
    result2 = provider.invoke(tool_id, {})
    assert result2.ok is True
    assert len(service.execute_calls) == 2
    assert service.execute_calls[1][4] == "approved"


def test_invoke_ask_callback_always_allow_sets_tool_state_with_live_hub_tool(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(
        service=service, main_loop=running_loop,
        approval_callback=lambda pending: {p.llm_name: "always_allow" for p in pending},
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert len(service.set_tool_state_calls) == 1
    server_key, tool_name, ui_state, tool = service.set_tool_state_calls[0]
    assert (server_key, tool_name, ui_state) == ("local:srv", "run", "allow")
    assert isinstance(tool, HubTool)
    assert tool.name == "run" and tool.server_key == "local:srv"
    assert service.execute_calls[0][4] == "approved"


def test_invoke_ask_callback_deny_refuses(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(
        service=service, main_loop=running_loop,
        approval_callback=lambda pending: {p.llm_name: "deny" for p in pending},
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == DENY_REFUSAL
    assert service.execute_calls == []


def test_invoke_ask_callback_missing_verdict_fails_closed(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(
        service=service, main_loop=running_loop,
        approval_callback=lambda pending: {},  # decisions dict lacks this call's name
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == DENY_REFUSAL


def test_invoke_ask_callback_raises_returns_error_never_raises(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])

    def bad_callback(pending):
        raise RuntimeError("card crashed")

    provider = MCPToolProvider(service=service, main_loop=running_loop, approval_callback=bad_callback)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert "card crashed" in result.error


# ---------------------------------------------------------------------------
# invoke() -- stamped verdicts (T6's batch-review closure path)
# ---------------------------------------------------------------------------

def test_invoke_stamped_deny_wins_without_fresh_gate_call(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    provider.apply_batch_decisions({tool_id: "deny"})

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == DENY_REFUSAL
    assert service.record_tool_decision_calls[-1][2] == "denied"
    assert service.execute_calls == []
    # The stamp is consumed -- a second call (no new stamp) re-gates fresh
    # and, with the default "ask" state and no callback, also fails closed
    # (but now via the fresh-gate path, proven by a second decision record).
    result2 = provider.invoke(tool_id, {})
    assert result2.ok is False
    assert len(service.record_tool_decision_calls) == 2


def test_invoke_stamped_timeout_uses_exact_model_facing_copy(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    provider.apply_batch_decisions({tool_id: "timeout"})

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == TIMEOUT_REFUSAL
    assert service.record_tool_decision_calls[-1][2] == "denied-timeout"
    assert service.execute_calls == []


def test_invoke_stamped_approve_once_executes(running_loop):
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    provider.apply_batch_decisions({tool_id: "approve_once"})

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert service.execute_calls[0][4] == "approved"


# ---------------------------------------------------------------------------
# invoke() -- execute-path failure modes (must never raise, never hang)
# ---------------------------------------------------------------------------

def test_invoke_execute_timeout_returns_error_within_bound(running_loop):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
        execute_delay=999,
        # bound = tool_call_timeout() + 5s slack; a negative value keeps
        # this test fast (~0.1s) while still exercising the real formula.
        tool_call_timeout=-4.9,
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    started = time.monotonic()
    result = provider.invoke(tool_id, {})
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert result.error  # Finding 1: guarantee non-empty error even for TimeoutError
    assert elapsed < 2.0, f"invoke() blocked for {elapsed:.2f}s -- must be bounded"


def test_invoke_execute_exception_from_coroutine_returns_error(running_loop):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
        execute_raises=RuntimeError("boom from execute_hub_tool"),
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert "boom from execute_hub_tool" in result.error


def test_invoke_execute_on_closed_loop_returns_error_never_raises():
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
    )
    provider = MCPToolProvider(service=service, main_loop=closed_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error  # some message, whatever asyncio's exact wording is


# ---------------------------------------------------------------------------
# invoke() -- kill switch checked at call time (Minor 5)
# ---------------------------------------------------------------------------

def test_invoke_refuses_when_kill_switch_flips_between_compose_and_invoke(running_loop):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    service.kill_switch = True  # flipped AFTER compose_catalog(), before invoke()

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == KILL_SWITCH_REFUSAL
    assert service.execute_calls == []
    assert service.record_tool_decision_calls[-1][2] == "denied"


def test_invoke_refuses_when_kill_switch_flips_even_with_a_stamped_verdict(running_loop):
    """The kill-switch check wins over an earlier-this-turn stamp too --
    the T6 batch-review closure could have stamped `approve_once` before
    the kill switch flipped between review and dispatch."""
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)
    tool_id = provider.list_catalog()[0].id
    provider.apply_batch_decisions({tool_id: "approve_once"})

    service.kill_switch = True

    result = provider.invoke(tool_id, {})

    assert result.ok is False
    assert result.error == KILL_SWITCH_REFUSAL
    assert service.execute_calls == []


def test_invoke_kill_switch_check_survives_getter_exception(running_loop):
    """A read failure on the kill-switch getter must not block execution --
    fails open, same as any other best-effort read in this provider."""
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        states={("local:srv", "run"): EffectiveToolState(state="allow", origin="tool_override")},
    )
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    _compose(provider)  # kill_switch is still False here -- compose must succeed
    tool_id = provider.list_catalog()[0].id

    def _raise():
        raise RuntimeError("kill switch read failed")

    service.get_kill_switch = _raise

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert service.execute_calls
