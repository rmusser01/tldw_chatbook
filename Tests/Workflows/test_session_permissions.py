"""Fresh authority and exact-effect approval for session workflows."""

import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Tools.note_management_tools import CreateNoteTool


def test_missing_strict_authority_is_not_a_default_grant(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    gate = BuiltinToolGate(SimpleNamespace(permission_store=store))
    decision = gate.check_detailed(
        CreateNoteTool(), "workflow-test", strict=True, allow_session_approvals=False
    )
    assert decision.refusal_code == "unavailable"
    assert decision.refusal is not None
    assert not store.path.exists()


@pytest.fixture
def workflow_permissions(tmp_path, monkeypatch):
    from tldw_chatbook.Workflows.session_permissions import WorkflowPermissions

    store = MCPPermissionStore(tmp_path / "permissions.json")
    store.save(store.load())

    def forbidden(*args, **kwargs):
        pytest.fail("Workflow permission checks must not repair or read session grants")

    monkeypatch.setattr(store, "load", forbidden)
    monkeypatch.setattr(store, "save", forbidden)
    service = SimpleNamespace(
        permission_store=store,
        is_session_approved=forbidden,
        get_kill_switch=forbidden,
    )
    gate = BuiltinToolGate(service)
    return WorkflowPermissions(gate), gate, store


def test_effect_payload_is_private_and_frozen():
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    effect = EffectRequest("run", "step", "note", '{"content":"private-canary"}')
    assert "private-canary" not in repr(effect)
    with pytest.raises(FrozenInstanceError):
        effect.payload_json = "changed"
    with pytest.raises(TypeError):
        EffectRequest("run", "step", "note", "{}", approved_once=True)


@pytest.mark.parametrize(
    "kind,tool_name",
    [
        ("file", "workflow_read_file"),
        ("model", "workflow_local_model"),
        ("note", "create_note"),
    ],
)
def test_effects_use_existing_builtin_names_and_risk_floors(
    workflow_permissions, kind, tool_name
):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, _, store = workflow_permissions
    effect = EffectRequest("run", "step", kind, "{}")
    assert permissions.check(effect).refusal_code == "approval_required"
    payload = json.loads(store.path.read_text())
    payload["profiles"]["default"]["servers"] = {
        "agent:builtin": {"tools": {tool_name: {"state": "allow"}}}
    }
    store.path.write_text(json.dumps(payload))
    assert permissions.check(effect).refusal is None


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"run_id": "another-run"},
        {"step_id": "another-step"},
        {"payload_json": '{"destination":"changed"}'},
        {"kind": "model"},
    ],
)
def test_approval_does_not_survive_or_transfer_to_another_effect(
    workflow_permissions, change
):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, _, _ = workflow_permissions
    effect = EffectRequest("run", "step", "note", '{"destination":"original"}')
    approved = permissions.check(effect, approved_once=True)
    assert approved.refusal is None
    assert approved.approval_decision == "approved"
    assert (
        permissions.check(replace(effect, **change)).refusal_code == "approval_required"
    )
    # The coordinator may recheck its still-owned physical operation explicitly.
    assert permissions.check(effect, approved_once=True).refusal is None
    assert permissions.check(effect).refusal_code == "approval_required"


@pytest.mark.parametrize("revocation", ["deny", "kill", "missing", "corrupt"])
def test_fresh_revocation_beats_the_approved_effect(workflow_permissions, revocation):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, _, store = workflow_permissions
    effect = EffectRequest("run", "step", "note", "{}")
    assert permissions.check(effect, approved_once=True).refusal is None
    payload = json.loads(store.path.read_text())
    if revocation == "deny":
        payload["profiles"]["default"]["servers"] = {
            "agent:builtin": {"tools": {"create_note": {"state": "deny"}}}
        }
        store.path.write_text(json.dumps(payload))
        expected = "denied"
    elif revocation == "kill":
        payload["kill_switch"] = True
        store.path.write_text(json.dumps(payload))
        expected = "kill_switch"
    elif revocation == "missing":
        store.path.unlink()
        expected = "unavailable"
    else:
        store.path.write_text("{")
        expected = "unavailable"
    assert permissions.check(effect, approved_once=True).refusal_code == expected


def test_prior_stamps_are_cleared_and_payload_digest_distinguishes_keys(
    workflow_permissions, monkeypatch
):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, gate, _ = workflow_permissions
    begin_turn = gate.begin_turn
    keys = []

    def remember(key):
        keys.append(key)
        begin_turn(key)

    monkeypatch.setattr(gate, "begin_turn", remember)
    effect = EffectRequest("run", "step", "note", "private-canary")
    assert permissions.check(effect, approved_once=True).refusal is None
    key = keys[0]
    assert keys == [key, key]
    assert key.startswith("run:step:") and "private-canary" not in key
    assert gate.stamped(key, "create_note") is None
    gate.stamp(key, "create_note", "always_allow")
    assert permissions.check(effect).refusal_code == "approval_required"
    assert (
        permissions.check(replace(effect, payload_json="different")).refusal_code
        == "approval_required"
    )
    assert keys[-1] != key


def test_check_exception_clears_the_effect_stamp(workflow_permissions, monkeypatch):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, gate, _ = workflow_permissions
    effect = EffectRequest("run", "step", "note", "{}")
    check = gate.check_detailed
    keys = []

    def fail(tool, key, **kwargs):
        keys.append(key)
        assert gate.stamped(key, tool.name) == "approve_once"
        raise RuntimeError("check failed")

    monkeypatch.setattr(gate, "check_detailed", fail)
    with pytest.raises(RuntimeError, match="check failed"):
        permissions.check(effect, approved_once=True)
    assert gate.stamped(keys[0], "create_note") is None
    monkeypatch.setattr(gate, "check_detailed", check)
    assert permissions.check(effect).refusal_code == "approval_required"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "model"])
async def test_file_and_model_descriptors_cannot_execute(
    workflow_permissions, monkeypatch, kind
):
    from tldw_chatbook.Workflows.session_permissions import EffectRequest

    permissions, gate, _ = workflow_permissions
    tools = []
    check = gate.check_detailed

    def capture(tool, *args, **kwargs):
        tools.append(tool)
        return check(tool, *args, **kwargs)

    monkeypatch.setattr(gate, "check_detailed", capture)
    result = permissions.check(EffectRequest("run", "step", kind, "{}"))
    assert result.refusal_code == "approval_required"
    with pytest.raises(NotImplementedError):
        await tools[0].execute()
