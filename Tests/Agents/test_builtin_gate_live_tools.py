"""TASK-545 P2: the approval path with REAL ported tools.

P1's gate was exercised only by a synthetic `_Mutating` fixture, because no
shipped tool declared risk tags. These tests drive the same machinery with
the actual registered tools -- the first coverage that the feature works
for a user rather than for a test double.
"""

import threading

import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

from Tests.Agents.test_builtin_tool_gate import _FakeService


@pytest.fixture
def tools_config(monkeypatch):
    """Drive the [tools] gates (mirrors test_builtin_file_tools.py)."""
    values = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


@pytest.fixture
def all_gates_on(tools_config):
    for key in ("write_file_enabled", "create_note_enabled", "read_file_enabled"):
        tools_config[key] = True
    return tools_config


def _provider(service):
    return BuiltinToolProvider(gate=BuiltinToolGate(service=service))


def test_a_mutating_tool_is_refused_without_approval(all_gates_on):
    """The core of the phase: write_file cannot run unprompted."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:write_file", {"file_path": "x", "content": "y"})
    assert result.ok is False
    assert "approval" in result.error


def test_a_reads_tool_is_refused_without_approval(all_gates_on):
    """Closes the live gap: enabling read_file no longer means silent reads."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:read_file", {"file_path": "x"})
    assert result.ok is False
    assert "approval" in result.error


def test_an_untagged_tool_still_runs_unprompted(all_gates_on):
    """Regression guard: calculator must not start prompting."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:calculator", {"expression": "1+1"})
    assert result.ok is True


def test_a_stamped_permit_lets_a_mutating_tool_run(all_gates_on, tmp_path, monkeypatch):
    """The approval round trip: a per-turn permit reaches execution."""
    service = _FakeService()
    gate = BuiltinToolGate(service=service)
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    target = tmp_path / "out.txt"
    import tldw_chatbook.Tools.file_operation_tools as fot

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    result = provider.invoke(
        "builtin:write_file", {"file_path": "out.txt", "content": "hello"}
    )

    assert result.ok is True, result.error
    assert target.read_text(encoding="utf-8") == "hello"


def test_a_resolved_deny_beats_a_permitting_stamp(all_gates_on):
    """The property Qodo caught in P1, now with a real tool: `Off` is
    absolute. Built-ins have no catalog filtering, so invoke() is the only
    barrier -- a stamp must never shadow it."""
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_TOOL_SERVER_KEY: {
                        "tools": {"write_file": {"state": "deny"}}
                    }
                }
            }
        }
    }
    gate = BuiltinToolGate(service=_FakeService(payload=payload))
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    result = provider.invoke("builtin:write_file", {"file_path": "x", "content": "y"})
    assert result.ok is False
    assert "Off" in result.error


def test_a_refusal_is_a_result_never_an_exception(all_gates_on):
    """The pure loop must never see an exception from tool invocation."""
    provider = _provider(_FakeService())
    for tool_id, args in (
        ("builtin:write_file", {"file_path": "x", "content": "y"}),
        ("builtin:create_note", {"title": "t", "content": "c"}),
        ("builtin:read_file", {"file_path": "x"}),
    ):
        result = provider.invoke(tool_id, args)
        assert result.ok is False
        assert isinstance(result.error, str) and result.error


@pytest.mark.parametrize(
    "gate_key,tool_name,args",
    [
        ("create_note_enabled", "create_note", {"title": "t", "content": "c"}),
        ("update_note_enabled", "update_note", {"note_id": "n1", "title": "t2"}),
    ],
)
def test_note_tool_executes_on_a_worker_thread(
    tools_config, monkeypatch, gate_key, tool_name, args
):
    """The agent service invokes tools off the main thread, and these note
    tools have never run there. `asyncio.run` inside invoke() requires no
    running loop on that thread."""
    import tldw_chatbook.Tools.note_management_tools as nmt

    tools_config[gate_key] = True
    seen = {}

    class _FakeNotes:
        def __init__(self, **kwargs):
            pass

        def _record(self, user_id):
            seen["thread"] = threading.current_thread().name
            seen["user_id"] = user_id

        def add_note(self, user_id, title, content):
            self._record(user_id)
            return "note-1"

        def get_note_by_id(self, user_id, note_id):
            self._record(user_id)
            return {"id": note_id, "version": 1}

        def update_note(self, user_id, note_id, update_data, expected_version):
            self._record(user_id)
            return True

    monkeypatch.setattr(nmt, "NotesInteropService", _FakeNotes)
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")

    gate = BuiltinToolGate(service=_FakeService())
    gate.begin_turn()
    gate.stamp(tool_name, "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    box = {}

    def run():
        box["result"] = provider.invoke(f"builtin:{tool_name}", args)

    worker = threading.Thread(target=run, name="tool-worker")
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "tool invocation hung on the worker thread"
    assert box["result"].ok is True, box["result"].error
    assert seen["thread"] == "tool-worker"
    assert seen["user_id"] == "alice"


def test_a_parents_approval_survives_a_nested_sub_agent_run(all_gates_on):
    """task-628's stamp_scope, now carrying a REAL tool.

    A spawned sub-agent shares the parent's gate instance and review
    closure, and the hook's first act is begin_turn() -- which clears every
    stamp. Without the scope, a parent's approved write_file becomes
    unusable the moment a sub-agent runs. P1/task-628 proved this with a
    synthetic tool; this is the first coverage with a tool a user can
    actually reach.
    """
    gate = BuiltinToolGate(service=_FakeService())
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    with gate.stamp_scope():
        # Stand in for the child run: it clears the turn and records its own
        # verdicts on the SAME gate.
        gate.begin_turn()
        gate.stamp("read_file", "deny")

    # The parent's approval must be back: the resolved verdict for write_file
    # is checked directly (not inferred from invoke()'s outcome, which can
    # also fail closed on sandbox containment for unrelated reasons).
    assert gate.check(provider.tool_for("write_file")) is None, (
        "parent's stamp was clobbered by the nested run"
    )
    # The child's verdict must not leak to the parent's stamp table.
    assert "read_file" not in gate._stamps, "child's verdict leaked to the parent"
