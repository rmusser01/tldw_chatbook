import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
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
