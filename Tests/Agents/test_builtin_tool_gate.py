import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate, build_builtin_gate
from tldw_chatbook.Tools.tool_executor import CalculatorTool, Tool

#: PR2a Task 5: the gate keys every per-turn verdict by ``(run_id,
#: tool_name)``, so each of these single-run tests names the one run whose
#: turn it is exercising. The assertions are unchanged -- what a run stamps
#: is what that same run's ``check()`` sees; cross-run isolation is what
#: ``Tests/Agents/test_gate_run_scoping.py`` pins.
RUN = "run-1"


class _Mutating(Tool):
    @property
    def name(self) -> str:
        return "write_thing"

    @property
    def description(self) -> str:
        return "d"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return ("mutates",)

    async def execute(self, **kwargs):
        return {}


class _Networked(Tool):
    @property
    def name(self) -> str:
        return "fetch_thing"

    @property
    def description(self) -> str:
        return "d"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return ("network",)

    async def execute(self, **kwargs):
        return {}


class _FakePermissionStore:
    """Stands in for ``MCPPermissionStore`` -- just enough to count loads."""

    def __init__(self, payload):
        self._payload = payload
        self.loads = 0

    def load(self):
        self.loads += 1
        return self._payload


class _FakeService:
    """Stands in for UnifiedMCPControlPlaneService.

    Real accessor confirmed in ``unified_control_plane_service.py``:
    the service exposes a ``permission_store`` property (``MCPPermissionStore
    | None``) whose ``.load()`` returns the payload dict -- there is no
    ``_load_payload`` method on the service itself.
    """

    def __init__(self, payload=None, kill=False, session=()):
        self.permission_store = _FakePermissionStore(payload or {})
        self._kill = kill
        self._session = set(session)
        self.session_approved = []
        self.session_reads = []

    @property
    def loads(self):
        return self.permission_store.loads

    def get_kill_switch(self):
        return self._kill

    def is_session_approved(self, server_key, tool_name, *, profile_id="default"):
        self.session_reads.append((server_key, tool_name, profile_id))
        return tool_name in self._session

    def approve_for_session(self, server_key, tool_name, *, profile_id="default"):
        self.session_approved.append((tool_name, profile_id))
        self._session.add(tool_name)


class _ServiceWithoutStore:
    """A present, live service whose ``permission_store`` is ``None``.

    Mirrors ``UnifiedMCPControlPlaneService.permission_store``
    (``unified_control_plane_service.py:2423-2428``), which returns
    ``None`` whenever ``local_service.store`` is unset -- a real runtime
    state distinct from "no service at all" (every other method still
    answers; only the store accessor is absent).
    """

    def __init__(self):
        self.permission_store = None

    def get_kill_switch(self):
        return False

    def is_session_approved(self, server_key, tool_name, *, profile_id="default"):
        return False

    def approve_for_session(self, server_key, tool_name, *, profile_id="default"):
        pass


def test_untagged_tool_is_permitted():
    gate = BuiltinToolGate(_FakeService())
    assert gate.check(CalculatorTool(), RUN) is None


def test_mutating_tool_without_approval_fails_closed():
    gate = BuiltinToolGate(_FakeService())
    reason = gate.check(_Mutating(), RUN)
    assert reason is not None and "approval" in reason.lower()


def test_kill_switch_blocks_even_untagged_tools():
    gate = BuiltinToolGate(_FakeService(kill=True))
    reason = gate.check(CalculatorTool(), RUN)
    assert reason is not None and "kill switch" in reason.lower()


def test_stamped_approval_permits_a_mutating_tool():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_once")
    assert gate.check(_Mutating(), RUN) is None


def test_begin_turn_clears_prior_stamps():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_once")
    gate.begin_turn(RUN)
    assert gate.check(_Mutating(), RUN) is not None


def test_session_approval_permits_without_a_stamp():
    gate = BuiltinToolGate(_FakeService(session={"write_thing"}))
    assert gate.check(_Mutating(), RUN) is None


def test_approve_session_records_a_session_approval():
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_session")
    assert gate.check(_Mutating(), RUN) is None
    assert svc.session_approved == [("write_thing", "default")]


def test_named_profile_is_used_for_resolution_and_session_approval():
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "allow"}},
            },
            "research": {
                "servers": {"agent:builtin": {"default": "ask"}},
            },
        }
    }
    svc = _FakeService(payload=payload)
    gate = BuiltinToolGate(svc, profile_id="research")

    state = gate.resolve(CalculatorTool())
    gate.stamp(RUN, "calculator", "approve_session")
    approved = gate.is_session_approved("calculator")

    assert state.state == "ask"
    assert svc.session_approved == [("calculator", "research")]
    assert svc.session_reads == [
        ("agent:builtin", "calculator", "research")
    ]
    assert approved is True


def test_no_service_still_gates_mutating_tools():
    # Constraint 7: a missing service is never allow-everything.
    gate = BuiltinToolGate(None)
    assert gate.check(CalculatorTool(), RUN) is None
    assert gate.check(_Mutating(), RUN) is not None


def test_payload_is_loaded_once_per_turn():
    # Constraint 8: one store load per turn, not per call.
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn(RUN)
    for _ in range(5):
        gate.check(CalculatorTool(), RUN)
    assert svc.loads == 1


def test_no_persistent_state_is_ever_written_for_builtins():
    """Constraint 3: P1 is session-scoped only. `always_allow` must not
    reach `set_tool_state` -- the card does not offer it for built-in
    rows, and the gate must not honor it as a persistent write either."""
    svc = _FakeService()
    svc.set_tool_state = lambda *a, **k: pytest.fail("persistent write")
    gate = BuiltinToolGate(svc)
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "always_allow")
    gate.check(_Mutating(), RUN)  # must not raise via set_tool_state


def test_deny_state_blocks():
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "deny"}},
            }
        }
    }
    gate = BuiltinToolGate(_FakeService(payload=payload))
    assert gate.check(CalculatorTool(), RUN) is not None


def test_service_present_but_permission_store_is_none_still_gates():
    """Finding 1: a service can be present yet have no store to load

    (``permission_store`` returns ``None``) -- this must degrade to the
    same allow-floor behavior as no service at all, never crash and never
    become allow-everything for tagged tools."""
    gate = BuiltinToolGate(_ServiceWithoutStore())
    assert gate.check(CalculatorTool(), RUN) is None
    assert gate.check(_Mutating(), RUN) is not None


def test_build_builtin_gate_with_no_service_still_gates():
    """Finding 2: ``build_builtin_gate()`` with no argument must return a
    gate that still gates -- ``None`` is never "ungated" (Constraint 7)."""
    gate = build_builtin_gate()
    assert isinstance(gate, BuiltinToolGate)
    assert gate.check(CalculatorTool(), RUN) is None
    assert gate.check(_Mutating(), RUN) is not None


def test_build_builtin_gate_uses_the_passed_service():
    """Finding 2: ``build_builtin_gate(service)`` must actually wire the
    given service in -- proven by its kill switch taking effect."""
    svc = _FakeService(kill=True)
    gate = build_builtin_gate(svc)
    reason = gate.check(CalculatorTool(), RUN)
    assert reason is not None and "kill switch" in reason.lower()


def test_effective_deny_wins_over_stamped_approve_once():
    """Finding 1: an effective ``deny`` (Off) must be absolute -- a
    stamped ``approve_once`` for this turn must NOT override it. Before
    the fix, ``check()`` consulted the stamp before the resolved state,
    so a permitting stamp on a tool the user set to Off would still let
    it execute."""
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "deny"}},
            }
        }
    }
    gate = BuiltinToolGate(_FakeService(payload=payload))
    gate.begin_turn(RUN)
    gate.stamp(RUN, "calculator", "approve_once")
    reason = gate.check(CalculatorTool(), RUN)
    assert reason is not None and "off" in reason.lower()


def test_effective_deny_wins_over_live_session_approval():
    """Finding 1: an effective ``deny`` (Off) must be absolute -- a live
    session approval for this turn must NOT override it either. Before
    the fix, a session-approved tool short-circuited before the resolved
    ``deny`` branch was ever reached."""
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "deny"}},
            }
        }
    }
    gate = BuiltinToolGate(_FakeService(payload=payload, session={"calculator"}))
    reason = gate.check(CalculatorTool(), RUN)
    assert reason is not None and "off" in reason.lower()


# --- task-628: nested sub-agent stamp scoping -------------------------------


def test_stamp_scope_restores_stamps_after_a_nested_run():
    """A child run's begin_turn() must not clobber the parent's stamps.

    `spawn_subagent` runs the child's whole loop INLINE on the parent's
    call stack, and the child invokes the SAME shared review hook, whose
    first act is `begin_turn()`. Without a scope the parent's verdicts for
    this turn are wiped before its own remaining same-batch tool calls are
    dispatched. Mirrors `MCPToolProvider.stamp_scope`.

    PR2a Task 5: the nested `begin_turn` here deliberately uses the SAME
    run id as the parent. A real child now has its OWN run id and cannot
    reach this slice at all (that is the new, load-bearing protection --
    see `test_gate_run_scoping.py`); reusing the id is what still
    exercises the scope's own guarantee, which is unchanged: whatever
    happens to this run's slice inside the scope is undone on exit.
    """
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_once")

    with gate.stamp_scope(RUN):
        # Stand in for the child's own turn against the shared gate.
        gate.begin_turn(RUN)
        gate.stamp(RUN, "other_tool", "deny")
        assert gate.check(_Mutating(), RUN) is not None  # child wiped it in-scope

    # Parent's verdict is back, and the child's is gone (restore, not merge).
    assert gate.check(_Mutating(), RUN) is None
    assert gate._stamps == {(RUN, "write_thing"): "approve_once"}


def test_stamp_scope_restores_even_when_the_nested_run_raises():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_once")

    try:
        with gate.stamp_scope(RUN):
            gate.begin_turn(RUN)
            raise RuntimeError("child blew up")
    except RuntimeError:
        pass

    assert gate.check(_Mutating(), RUN) is None


def test_stamp_scope_is_reentrant_for_nested_scopes():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn(RUN)
    gate.stamp(RUN, "write_thing", "approve_once")
    with gate.stamp_scope(RUN):
        gate.begin_turn(RUN)
        gate.stamp(RUN, "write_thing", "deny")
        with gate.stamp_scope(RUN):
            gate.begin_turn(RUN)
        assert gate._stamps == {(RUN, "write_thing"): "deny"}
    assert gate._stamps == {(RUN, "write_thing"): "approve_once"}


# --- task-627 (P2 Task 2): settings-time enumeration -------------------------


def test_builtin_permission_rows_lists_live_tools_with_resolved_state():
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows

    rows = builtin_permission_rows({})  # empty payload -> the allow floor
    by_name = {r.name: r for r in rows}
    assert "calculator" in by_name and "get_current_datetime" in by_name
    # Untagged tools resolve to the built-in floor, not the MCP "ask" default.
    assert by_name["calculator"].effective.state == "allow"
    assert by_name["calculator"].effective.origin == "builtin_default"
    assert by_name["calculator"].orphaned is False
    assert by_name["calculator"].description  # carried for display


def test_builtin_permission_rows_reflects_a_stored_override():
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_TOOL_SERVER_KEY: {
                        "tools": {"calculator": {"state": "deny"}}
                    }
                }
            }
        }
    }
    row = {r.name: r for r in builtin_permission_rows(payload)}["calculator"]
    assert row.effective.state == "deny"
    assert row.effective.origin == "tool_override"


def test_builtin_permission_rows_surfaces_orphaned_stored_entries():
    """A decision stored for a tool a later release removed must still be
    listed, or the user cannot clear it."""
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_TOOL_SERVER_KEY: {
                        "tools": {"tool_that_no_longer_exists": {"state": "allow"}}
                    }
                }
            }
        }
    }
    rows = {r.name: r for r in builtin_permission_rows(payload)}
    assert rows["tool_that_no_longer_exists"].orphaned is True
    assert rows["calculator"].orphaned is False


def test_builtin_permission_rows_needs_no_agent_run():
    """Enumeration must not start a run or build a gate."""
    import tldw_chatbook.Agents.tool_catalog as tc
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows

    calls = []
    original = tc.build_builtin_gate
    tc.build_builtin_gate = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        builtin_permission_rows({})
    finally:
        tc.build_builtin_gate = original
    assert calls == []  # the lazy gate was never built


def test_network_tag_floors_inherited_allow_to_ask():
    """Egress is the exfiltration leg of a prompt-injection chain.

    An untagged read-only fetch would resolve to the built-in allow floor
    and execute silently, so `network` joins `BUILTIN_HIGH_RISK_TAGS` (not
    the shared `HIGH_RISK_TAGS` -- widening that would make remote MCP
    tools start prompting too, which is not this change's call to make).
    """
    from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
    from tldw_chatbook.MCP.permission_store import resolve_builtin_state

    state = resolve_builtin_state({}, tool_ref(_Networked()))

    assert state.state == "ask"
    assert state.risk_floored is True


# --- task-3240: unified [tools]/[console] gate enumerator -------------------
#
# Seam namespace note (spec review, Minor 6): `all_tool_gates()` reads
# `tldw_chatbook.config.get_cli_setting` via a FUNCTION-LOCAL import (like
# `BuiltinToolProvider.__init__`), so these tests patch that module
# attribute directly -- patching `mcp_workbench`'s (or any other caller's)
# own imported name would not reach it.


def _no_override_get_cli_setting(section, key=None, default=None):
    """A `get_cli_setting` fake that always returns the caller's own
    default -- gives every gate a deterministic, environment-independent
    `enabled=False` baseline regardless of the real config.toml on disk."""
    return default


def test_all_tool_gates_enumerates_every_gate_with_sections_and_groups(monkeypatch):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import ToolGate, all_tool_gates
    from tldw_chatbook.Agents.local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    monkeypatch.setattr(config_module, "get_cli_setting", _no_override_get_cli_setting)

    gates = all_tool_gates()
    # Derived, not a literal (TASK-16174): the arity is "every builtin row
    # plus the local group's two", so adding a gateable built-in must not
    # make this test the thing that fails.
    assert len(gates) == len(_GATEABLE_BUILTINS) + 2
    assert all(isinstance(gate, ToolGate) for gate in gates)

    # The _GATEABLE_BUILTINS rows come first, in registration order,
    # constants-not-literals (a re-typed key here would drift silently the
    # moment _GATEABLE_BUILTINS gains or reorders an entry).
    builtin_gates = gates[: len(_GATEABLE_BUILTINS)]
    assert [g.key for g in builtin_gates] == [e.gate_key for e in _GATEABLE_BUILTINS]
    assert [g.tool_name for g in builtin_gates] == [
        e.tool_name for e in _GATEABLE_BUILTINS
    ]
    assert all(g.section == "tools" and g.group == "builtin" for g in builtin_gates)
    assert all(
        g.description for g in builtin_gates
    )  # real tool descriptions, never blank
    assert all(g.enabled is False for g in builtin_gates)  # no override -> all off

    # The local group: master switch FIRST, then web_deep_search.
    local_gates = gates[len(_GATEABLE_BUILTINS) :]
    assert len(local_gates) == 2
    assert local_gates[0].section == "console"
    assert local_gates[0].key == "local_tools_enabled"
    assert local_gates[0].group == "local"
    assert local_gates[0].enabled is True  # missing key -> available by default
    assert "workspace, web, and Watchlists" in local_gates[0].description
    assert "standard web research" not in local_gates[0].description
    assert local_gates[1].section == "tools"
    assert local_gates[1].key == WEB_DEEP_SEARCH_GATE_KEY
    assert local_gates[1].group == "local"
    assert all(g.description for g in local_gates)


def test_all_tool_gates_enabled_is_coerced_not_raw_truthy(monkeypatch):
    """A quoted "false" must read as OFF -- the same class of bug
    task-3240's tool_catalog.py fix closed at the registration layer;
    this pins it at the enumerator layer too."""
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    target_key = _GATEABLE_BUILTINS[0].gate_key

    def fake_get_cli_setting(section, key=None, default=None):
        if key == target_key:
            return "false"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    gates = {gate.key: gate for gate in all_tool_gates()}
    assert gates[target_key].enabled is False


def test_all_tool_gates_enabled_coerces_quoted_true(monkeypatch):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    target_key = _GATEABLE_BUILTINS[0].gate_key

    def fake_get_cli_setting(section, key=None, default=None):
        if key == target_key:
            return "true"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    gates = {gate.key: gate for gate in all_tool_gates()}
    assert gates[target_key].enabled is True


def test_all_tool_gates_enabled_passes_through_real_bools(monkeypatch):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
    from tldw_chatbook.Agents.local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY

    def fake_get_cli_setting(section, key=None, default=None):
        if key == "local_tools_enabled":
            return True
        if key == WEB_DEEP_SEARCH_GATE_KEY:
            return False
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    gates = {gate.key: gate for gate in all_tool_gates()}
    assert gates["local_tools_enabled"].enabled is True
    assert gates[WEB_DEEP_SEARCH_GATE_KEY].enabled is False


def test_web_deep_search_gate_key_is_the_relocated_constant_not_a_literal(monkeypatch):
    """Spec: 'the two hand-listed entries don't drift from their source
    constants (assert WEB_DEEP_SEARCH_GATE_KEY equality rather than a
    re-typed literal)'."""
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
    from tldw_chatbook.Agents.local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY

    monkeypatch.setattr(config_module, "get_cli_setting", _no_override_get_cli_setting)
    gate = next(g for g in all_tool_gates() if g.tool_name == "web_deep_search")
    assert gate.key == WEB_DEEP_SEARCH_GATE_KEY


def test_tool_gate_breadcrumb_absent_when_all_gates_are_on(monkeypatch):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import tool_gate_breadcrumb

    monkeypatch.setattr(
        config_module, "get_cli_setting", lambda section, key=None, default=None: True
    )
    assert tool_gate_breadcrumb() is None


def test_tool_gate_breadcrumb_names_the_off_count(monkeypatch):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import (
        all_tool_gates,
        tool_gate_breadcrumb,
    )

    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    monkeypatch.setattr(config_module, "get_cli_setting", _no_override_get_cli_setting)
    gates = all_tool_gates()  # master defaults on; every other gate defaults off
    off_count = len(_GATEABLE_BUILTINS) + 1  # + web_deep_search, - the master
    text = tool_gate_breadcrumb(gates)
    assert text is not None
    assert str(off_count) in text
    assert "Tools mode" in text
    assert "web_search, web_fetch, web_crawl" not in text


def test_tool_gate_breadcrumb_names_expanded_principal_when_master_is_off(
    monkeypatch,
):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_tool_gate import (
        LOCAL_TOOLS_MASTER_KEY,
        tool_gate_breadcrumb,
    )

    def only_master_off(section, key=None, default=None):
        return False if key == LOCAL_TOOLS_MASTER_KEY else True

    monkeypatch.setattr(config_module, "get_cli_setting", only_master_off)

    text = tool_gate_breadcrumb()
    assert text is not None
    assert "Enable 'Local workspace, web, and Watchlists tools'" in text
    assert "local/web" not in text


def test_gate_key_pairs_and_all_tool_gates_can_never_drift(monkeypatch):
    """Qodo PR #1453: the count path (`_gate_key_pairs`) and the full
    enumerator (`all_tool_gates`) hand-build the same key set in two
    places — this pin makes silent divergence impossible."""
    from tldw_chatbook.Agents.builtin_tool_gate import _gate_key_pairs, all_tool_gates

    assert [(g.section, g.key) for g in all_tool_gates()] == _gate_key_pairs()


def test_count_off_tool_gates_constructs_no_tools(monkeypatch):
    """The breadcrumb's count-only path must never instantiate a Tool
    (it runs on every Permissions-mode resync; construction also spams
    warnings for optional tools missing on this system)."""
    from tldw_chatbook.Agents import builtin_tool_gate, tool_catalog
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    def explode(entry):
        raise AssertionError(f"count path constructed a tool: {entry}")

    monkeypatch.setattr(tool_catalog, "build_gateable_tool", explode)
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda s, k, d=None: d)
    off_count = len(_GATEABLE_BUILTINS) + 1  # + web_deep_search, - the master
    assert builtin_tool_gate.count_off_tool_gates() == off_count
    breadcrumb = builtin_tool_gate.tool_gate_breadcrumb()
    assert breadcrumb is not None
    assert f"{off_count} tool gate(s)" in breadcrumb
    assert "workspace, web, and Watchlists master switch in Tools mode" in breadcrumb
    assert "local/web" not in breadcrumb
    assert "built-in server detail" in breadcrumb


def test_tool_gate_breadcrumb_reads_each_config_gate_once(monkeypatch):
    """Count and master-state messaging share one coherent config snapshot."""
    from tldw_chatbook.Agents import builtin_tool_gate

    reads: list[tuple[str, str]] = []

    def read_gate(section, key, default=None):
        reads.append((section, key))
        return default

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", read_gate)

    breadcrumb = builtin_tool_gate.tool_gate_breadcrumb()

    assert breadcrumb is not None
    assert reads == builtin_tool_gate._gate_key_pairs()
