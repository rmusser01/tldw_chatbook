import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate, build_builtin_gate
from tldw_chatbook.Tools.tool_executor import CalculatorTool, Tool


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

    @property
    def loads(self):
        return self.permission_store.loads

    def get_kill_switch(self):
        return self._kill

    def is_session_approved(self, server_key, tool_name):
        return tool_name in self._session

    def approve_for_session(self, server_key, tool_name):
        self.session_approved.append(tool_name)
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

    def is_session_approved(self, server_key, tool_name):
        return False

    def approve_for_session(self, server_key, tool_name):
        pass


def test_untagged_tool_is_permitted():
    gate = BuiltinToolGate(_FakeService())
    assert gate.check(CalculatorTool()) is None


def test_mutating_tool_without_approval_fails_closed():
    gate = BuiltinToolGate(_FakeService())
    reason = gate.check(_Mutating())
    assert reason is not None and "approval" in reason.lower()


def test_kill_switch_blocks_even_untagged_tools():
    gate = BuiltinToolGate(_FakeService(kill=True))
    reason = gate.check(CalculatorTool())
    assert reason is not None and "kill switch" in reason.lower()


def test_stamped_approval_permits_a_mutating_tool():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")
    assert gate.check(_Mutating()) is None


def test_begin_turn_clears_prior_stamps():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")
    gate.begin_turn()
    assert gate.check(_Mutating()) is not None


def test_session_approval_permits_without_a_stamp():
    gate = BuiltinToolGate(_FakeService(session={"write_thing"}))
    assert gate.check(_Mutating()) is None


def test_approve_session_records_a_session_approval():
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    gate.stamp("write_thing", "approve_session")
    assert gate.check(_Mutating()) is None
    assert svc.session_approved == ["write_thing"]


def test_no_service_still_gates_mutating_tools():
    # Constraint 7: a missing service is never allow-everything.
    gate = BuiltinToolGate(None)
    assert gate.check(CalculatorTool()) is None
    assert gate.check(_Mutating()) is not None


def test_payload_is_loaded_once_per_turn():
    # Constraint 8: one store load per turn, not per call.
    svc = _FakeService()
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    for _ in range(5):
        gate.check(CalculatorTool())
    assert svc.loads == 1


def test_no_persistent_state_is_ever_written_for_builtins():
    """Constraint 3: P1 is session-scoped only. `always_allow` must not
    reach `set_tool_state` -- the card does not offer it for built-in
    rows, and the gate must not honor it as a persistent write either."""
    svc = _FakeService()
    svc.set_tool_state = lambda *a, **k: pytest.fail("persistent write")
    gate = BuiltinToolGate(svc)
    gate.begin_turn()
    gate.stamp("write_thing", "always_allow")
    gate.check(_Mutating())          # must not raise via set_tool_state


def test_deny_state_blocks():
    payload = {
        "profiles": {
            "default": {
                "servers": {"agent:builtin": {"default": "deny"}},
            }
        }
    }
    gate = BuiltinToolGate(_FakeService(payload=payload))
    assert gate.check(CalculatorTool()) is not None


def test_service_present_but_permission_store_is_none_still_gates():
    """Finding 1: a service can be present yet have no store to load

    (``permission_store`` returns ``None``) -- this must degrade to the
    same allow-floor behavior as no service at all, never crash and never
    become allow-everything for tagged tools."""
    gate = BuiltinToolGate(_ServiceWithoutStore())
    assert gate.check(CalculatorTool()) is None
    assert gate.check(_Mutating()) is not None


def test_build_builtin_gate_with_no_service_still_gates():
    """Finding 2: ``build_builtin_gate()`` with no argument must return a
    gate that still gates -- ``None`` is never "ungated" (Constraint 7)."""
    gate = build_builtin_gate()
    assert isinstance(gate, BuiltinToolGate)
    assert gate.check(CalculatorTool()) is None
    assert gate.check(_Mutating()) is not None


def test_build_builtin_gate_uses_the_passed_service():
    """Finding 2: ``build_builtin_gate(service)`` must actually wire the
    given service in -- proven by its kill switch taking effect."""
    svc = _FakeService(kill=True)
    gate = build_builtin_gate(svc)
    reason = gate.check(CalculatorTool())
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
    gate.begin_turn()
    gate.stamp("calculator", "approve_once")
    reason = gate.check(CalculatorTool())
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
    reason = gate.check(CalculatorTool())
    assert reason is not None and "off" in reason.lower()


# --- task-628: nested sub-agent stamp scoping -------------------------------


def test_stamp_scope_restores_stamps_after_a_nested_run():
    """A child run's begin_turn() must not clobber the parent's stamps.

    `spawn_subagent` runs the child's whole loop INLINE on the parent's
    call stack, and the child invokes the SAME shared review hook, whose
    first act is `begin_turn()`. Without a scope the parent's verdicts for
    this turn are wiped before its own remaining same-batch tool calls are
    dispatched. Mirrors `MCPToolProvider.stamp_scope`.
    """
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")

    with gate.stamp_scope():
        # Stand in for the child's own turn against the shared gate.
        gate.begin_turn()
        gate.stamp("other_tool", "deny")
        assert gate.check(_Mutating()) is not None  # child wiped it in-scope

    # Parent's verdict is back, and the child's is gone (restore, not merge).
    assert gate.check(_Mutating()) is None
    assert gate._stamps == {"write_thing": "approve_once"}


def test_stamp_scope_restores_even_when_the_nested_run_raises():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")

    try:
        with gate.stamp_scope():
            gate.begin_turn()
            raise RuntimeError("child blew up")
    except RuntimeError:
        pass

    assert gate.check(_Mutating()) is None


def test_stamp_scope_is_reentrant_for_nested_scopes():
    gate = BuiltinToolGate(_FakeService())
    gate.begin_turn()
    gate.stamp("write_thing", "approve_once")
    with gate.stamp_scope():
        gate.begin_turn()
        gate.stamp("write_thing", "deny")
        with gate.stamp_scope():
            gate.begin_turn()
        assert gate._stamps == {"write_thing": "deny"}
    assert gate._stamps == {"write_thing": "approve_once"}


def test_network_tag_floors_inherited_allow_to_ask():
    """Egress is the exfiltration leg of a prompt-injection chain.

    An untagged read-only fetch would resolve to the built-in allow floor
    and execute silently, so `network` joins HIGH_RISK_TAGS.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
    from tldw_chatbook.MCP.permission_store import resolve_builtin_state

    state = resolve_builtin_state({}, tool_ref(_Networked()))

    assert state.state == "ask"
    assert state.risk_floored is True
