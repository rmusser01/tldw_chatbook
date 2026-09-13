"""Task 6 (workspace assistant defaults): `MCPToolProvider` profile threading.

`MCPToolProvider.__init__` gains an optional `profile_id_provider` callable
(defaulting to the ``"default"`` profile). `compose_catalog()` resolves the
batched effective states under the ACTIVE profile (so a tool denied only in
the named workspace profile is dropped from the catalog), and
`_apply_verdict`'s always-allow persist path writes the tool-level allow
into the ACTIVE profile rather than the default one.

Task 7 (controller ruling from Task 6's review): the invoke-time FRESH
gates (`pending_gate_for`, `invoke`'s own gate) resolve under the ACTIVE
profile too, so a named-profile "ask" beats a default-profile "allow" at
invoke -- never a silent execution.

The service double below mirrors `UnifiedMCPControlPlaneService`'s
profile-aware seams exactly (no ``**kwargs`` masking) and records every
`profile_id` it was called with, so a drift in which profile the provider
resolves/persists under fails loudly here.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState

#: The workspace-assistant profile id used throughout (never "default", so
#: every assertion distinguishes the named-profile call from the default one).
NAMED_PROFILE = "ws-w-1"


def _catalog_record(profile_id: str, tools: list[dict], *, is_connected=True) -> dict:
    return {
        "profile_id": profile_id,
        "is_connected": is_connected,
        "discovery_snapshot": {"tools": tools},
    }


def _tool_dict(name: str, description: str = "") -> dict:
    return {"name": name, "description": description}


class ProfileTrackingService:
    """Profile-aware double for the service seams `MCPToolProvider` touches.

    `states_by_profile` maps ``profile_id -> {(server_key, tool_name):
    EffectiveToolState}``; anything absent resolves to the fail-closed ask
    default, mirroring the real store's inherit-from-default chain well
    enough for the catalog/persist assertions here.
    """

    def __init__(
        self, states_by_profile: dict[str, dict[tuple[str, str], EffectiveToolState]]
    ) -> None:
        self.states_by_profile = states_by_profile
        self.default_state = EffectiveToolState(state="ask", origin="global_default")
        self.effective_tool_states_calls: list[str] = []
        self.set_tool_state_calls: list[tuple[str, str, object, str]] = []
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})
        self.session_approvals: set[tuple[str, str]] = set()
        self.record_tool_decision_calls: list[tuple] = []
        self.execute_calls: list[tuple] = []

    def _state_for(
        self, server_key: str, tool_name: str, profile_id: str
    ) -> EffectiveToolState:
        states = self.states_by_profile.get(profile_id, {})
        return states.get((server_key, tool_name), self.default_state)

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return [
            _catalog_record("srv", [_tool_dict("run"), _tool_dict("fs_write")])
        ]

    def effective_tool_states(
        self, tools: list[HubTool], *, profile_id: str = "default"
    ) -> dict[tuple[str, str], EffectiveToolState]:
        self.effective_tool_states_calls.append(profile_id)
        return {
            (t.server_key, t.name): self._state_for(
                t.server_key, t.name, profile_id
            )
            for t in tools
        }

    def gate_tool_test(
        self, tool: HubTool, *, profile_id: str = "default"
    ) -> EffectiveToolState:
        return self._state_for(tool.server_key, tool.name, profile_id)

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self.session_approvals.add((server_key, tool_name))

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self.session_approvals

    def set_tool_state(
        self,
        server_key: str,
        tool_name: str,
        ui_state,
        *,
        tool: HubTool | None = None,
        profile_id: str = "default",
    ) -> None:
        self.set_tool_state_calls.append((server_key, tool_name, ui_state, profile_id))

    def record_tool_decision(
        self,
        server_key: str,
        tool_name: str,
        *,
        decision: str,
        initiator: str = "agent",
        error: str | None = None,
    ) -> None:
        self.record_tool_decision_calls.append(
            (server_key, tool_name, decision, initiator, error)
        )

    def _tool_call_timeout(self) -> float:
        return 5.0

    async def execute_hub_tool(
        self,
        server_key: str,
        tool_name: str,
        arguments: dict | None = None,
        *,
        initiator: str = "test",
        decision: str = "allowed",
        timeout_seconds: float | None = None,
        registered_argument_names: set[str] | None = None,
    ) -> dict:
        self.execute_calls.append(
            (server_key, tool_name, dict(arguments or {}), initiator, decision)
        )
        return {"content": [{"type": "text", "text": "ok"}]}


def _named_deny_states() -> dict[str, dict[tuple[str, str], EffectiveToolState]]:
    """fs_write denied ONLY in the named profile (ask everywhere else)."""
    return {
        NAMED_PROFILE: {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="deny", origin="tool_override"
            )
        }
    }


def _compose(provider: MCPToolProvider) -> None:
    asyncio.run(provider.compose_catalog())


def _catalog_names(provider: MCPToolProvider) -> set[str]:
    return {entry.name for entry in provider.list_catalog()}


# -- compose_catalog: profile-aware batch resolution -------------------------


def test_compose_catalog_drops_tool_denied_only_in_named_profile():
    """A tool denied only in the active named profile is dropped from the
    catalog, and the batched resolution ran under that profile id."""
    service = ProfileTrackingService(_named_deny_states())
    provider = MCPToolProvider(
        service=service,
        main_loop=asyncio.new_event_loop(),
        profile_id_provider=lambda: NAMED_PROFILE,
    )

    _compose(provider)

    names = _catalog_names(provider)
    assert any("fs_write" in n for n in names) is False
    assert any("run" in n for n in names) is True
    assert service.effective_tool_states_calls == [NAMED_PROFILE]


def test_compose_catalog_keeps_named_denied_tool_when_profile_is_default():
    """The same store data, but the provider resolves under "default": the
    named-profile deny never applies and the tool stays eligible."""
    service = ProfileTrackingService(_named_deny_states())
    provider = MCPToolProvider(
        service=service,
        main_loop=asyncio.new_event_loop(),
        profile_id_provider=lambda: "default",
    )

    _compose(provider)

    names = _catalog_names(provider)
    assert any("fs_write" in n for n in names) is True
    assert service.effective_tool_states_calls == ["default"]


def test_compose_catalog_without_provider_resolves_default_profile():
    """No `profile_id_provider` wired (every non-Console caller today):
    resolution runs under "default" -- today's behavior, unchanged."""
    service = ProfileTrackingService(_named_deny_states())
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())

    _compose(provider)

    assert any("fs_write" in n for n in _catalog_names(provider)) is True
    assert service.effective_tool_states_calls == ["default"]


# -- Task 7 (HARD REQUIREMENT): invoke-time fresh gates under the profile ----


def test_invoke_resolves_named_ask_over_default_allow(running_loop):
    """Task 7 controller ruling: the invoke-time FRESH gates must resolve
    under the ACTIVE profile. A tool set to "ask" in the named workspace
    profile but "allow" in default composes into the catalog yet produces an
    ask/pending gate at invoke -- an approval round and a refusal, never a
    silent default-profile execution."""
    states = {
        NAMED_PROFILE: {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="ask", origin="tool_override"
            )
        },
        "default": {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="allow", origin="tool_override"
            )
        },
    }
    service = ProfileTrackingService(states)
    requested: list = []

    def _ask(pending):
        requested.extend(pending)
        return {}

    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
        profile_id_provider=lambda: NAMED_PROFILE,
        approval_callback=_ask,
    )
    _compose(provider)
    entry = next(e for e in provider.list_catalog() if "fs_write" in e.name)
    assert entry is not None  # "ask" is catalog-eligible; it composes

    pending = provider.pending_gate_for(entry.id, {})
    assert pending is not None and pending.tool_name == "fs_write"

    result = provider.invoke(entry.id, {})
    assert result.ok is False  # unresolved ask -> refusal, NOT execution
    assert requested and requested[0].tool_name == "fs_write"
    assert service.execute_calls == []  # never silently executed


def test_invoke_without_provider_still_resolves_default_profile(running_loop):
    """The no-``profile_id_provider`` caller keeps today's behavior: the
    fresh gate resolves under "default", where the same tool is "allow" and
    executes."""
    states = {
        "default": {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="allow", origin="tool_override"
            )
        }
    }
    service = ProfileTrackingService(states)
    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
    )
    _compose(provider)
    entry = next(e for e in provider.list_catalog() if "fs_write" in e.name)

    pending = provider.pending_gate_for(entry.id, {})
    assert pending is None  # default-profile "allow" needs no asking
    result = provider.invoke(entry.id, {})
    assert result.ok is True
    assert service.execute_calls != []


# -- _apply_verdict: always-allow persist path -------------------------------


@pytest.fixture
def running_loop():
    """A real event loop running forever in a background thread (the "main
    loop" stand-in `invoke()` submits its execute coroutines onto)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(timeout=2), "background loop failed to start"
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_always_allow_verdict_persists_into_named_profile(running_loop):
    """An "Always allow" card verdict under the named profile writes the
    tool-level allow into THAT profile, not the default one."""
    service = ProfileTrackingService({})
    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
        profile_id_provider=lambda: NAMED_PROFILE,
        approval_callback=lambda pending: {
            p.llm_name: "always_allow" for p in pending
        },
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert service.set_tool_state_calls == [
        ("local:srv", "run", "allow", NAMED_PROFILE)
    ]


def test_always_allow_verdict_without_provider_persists_default_profile(
    running_loop,
):
    """Without a `profile_id_provider`, the persist path stays on the
    default profile -- byte-identical to the pre-profiles behavior."""
    service = ProfileTrackingService({})
    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
        approval_callback=lambda pending: {
            p.llm_name: "always_allow" for p in pending
        },
    )
    _compose(provider)
    tool_id = provider.list_catalog()[0].id

    result = provider.invoke(tool_id, {})

    assert result.ok is True
    assert service.set_tool_state_calls == [
        ("local:srv", "run", "allow", "default")
    ]


# -- Persona require_confirmation floor at invoke (final-review FIX 1) --------


def _confirm_policy():
    from tldw_chatbook.Agents.persona_policy import parse_persona_policy_from_rules

    return parse_persona_policy_from_rules(
        [
            {
                "rule_kind": "mcp_tool",
                "rule_name": "fs_write",
                "allowed": True,
                "require_confirmation": True,
            }
        ]
    )


def test_invoke_persona_floor_lowers_profile_allow_to_ask(running_loop):
    """The persona `require_confirmation` rule floors an MCP tool to "ask"
    at invoke even when the active named profile (and any persisted
    always_allow grant) says "allow": pending gate surfaces, invoke
    refuses rather than silently executing."""
    states = {
        NAMED_PROFILE: {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="allow", origin="tool_override"
            )
        }
    }
    service = ProfileTrackingService(states)
    requested: list = []

    def _ask(pending):
        requested.extend(pending)
        return {}

    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
        profile_id_provider=lambda: NAMED_PROFILE,
        persona_policy_provider=_confirm_policy,
        approval_callback=_ask,
    )
    _compose(provider)
    entry = next(e for e in provider.list_catalog() if "fs_write" in e.name)

    pending = provider.pending_gate_for(entry.id, {})
    assert pending is not None and pending.tool_name == "fs_write"
    result = provider.invoke(entry.id, {})
    assert result.ok is False  # unresolved ask -> refusal, NOT execution
    assert requested and requested[0].tool_name == "fs_write"
    assert service.execute_calls == []  # never silently executed


def test_invoke_persona_floor_absent_policy_unchanged(running_loop):
    """No `persona_policy_provider` wired: same store data, same profile,
    byte-identical pre-feature behavior -- the allow executes silently."""
    states = {
        NAMED_PROFILE: {
            ("local:srv", "fs_write"): EffectiveToolState(
                state="allow", origin="tool_override"
            )
        }
    }
    service = ProfileTrackingService(states)
    provider = MCPToolProvider(
        service=service,
        main_loop=running_loop,
        profile_id_provider=lambda: NAMED_PROFILE,
        approval_callback=lambda pending: {},
    )
    _compose(provider)
    entry = next(e for e in provider.list_catalog() if "fs_write" in e.name)

    assert provider.pending_gate_for(entry.id, {}) is None
    result = provider.invoke(entry.id, {})
    assert result.ok is True
    assert service.execute_calls != []
