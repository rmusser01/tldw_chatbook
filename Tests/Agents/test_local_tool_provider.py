from pathlib import Path

import pytest

from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
    LocalToolProvider,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

ALLOW = EffectiveToolState(state="allow", origin="tool_override")
ASK = EffectiveToolState(state="ask", origin="global_default")
DENY = EffectiveToolState(state="deny", origin="tool_override")


def make_provider(state=ALLOW, kill=False, **kwargs):
    return LocalToolProvider(
        workspace_root=Path(kwargs.pop("root", ".")).resolve() if "root" in kwargs else Path("."),
        resolve_state=lambda hub: state,
        kill_switch=lambda: kill,
        **kwargs,
    )


def test_catalog_lists_fs_list_with_local_ids(tmp_path):
    p = make_provider(root=tmp_path)
    entries = p.list_catalog()
    assert [e.id for e in entries] == ["local:fs_list"]
    assert entries[0].name == "fs_list" and entries[0].source == "local"
    schema = p.load_schema("local:fs_list")
    assert schema.parameters["required"] == ["path"]


def test_invoke_happy_path(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    p = make_provider(root=tmp_path)
    r = p.invoke("local:fs_list", {"path": "."})
    assert r.ok and "hello.txt" in r.content


def test_invoke_unknown_tool(tmp_path):
    r = make_provider(root=tmp_path).invoke("local:nope", {})
    assert not r.ok and "Unknown local tool" in r.error


def test_kill_switch_refuses(tmp_path):
    r = make_provider(root=tmp_path, kill=True).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


def test_deny_state_refuses(tmp_path):
    r = make_provider(state=DENY, root=tmp_path).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_ask_without_stamp_or_callback_fails_closed(tmp_path):
    r = make_provider(state=ASK, root=tmp_path).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_ask_with_approve_once_stamp_executes(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    assert p.invoke("local:fs_list", {"path": "."}).ok


def test_stamps_replace_not_merge(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    p.apply_batch_decisions({})  # next turn cleared first
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_pending_gate_for_ask_returns_pending_call(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    gate = p.pending_gate_for("fs_list", {"path": "."})
    assert gate is not None
    assert gate.server_key == "local:__local__" and gate.tool_name == "fs_list"
    assert gate.reason == "ask"
    assert p.pending_gate_for("unknown", {}) is None


def test_stamp_scope_isolates_nested_run(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    with p.stamp_scope():
        assert not p.invoke("local:fs_list", {"path": "."}).ok  # child: no stamps
    assert p.invoke("local:fs_list", {"path": "."}).ok  # parent stamps restored


def test_execution_error_becomes_result_string(tmp_path):
    r = make_provider(root=tmp_path).invoke("local:fs_list", {"path": "../escape"})
    assert not r.ok and "outside the workspace root" in r.error


# -- session approvals + persistence seams (Task 5) ---------------------------


def test_session_approval_skips_gate_and_executes(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    p = make_provider(
        state=ASK, root=tmp_path, is_session_approved=lambda hub: True
    )
    assert p.pending_gate_for("fs_list", {"path": "."}) is None
    assert p.invoke("local:fs_list", {"path": "."}).ok  # no stamp, no callback


def test_approve_session_stamp_persists(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    persisted = []
    p = make_provider(
        state=ASK,
        root=tmp_path,
        persist_approval=lambda hub, decision: persisted.append((hub.name, decision)),
    )
    p.apply_batch_decisions({"fs_list": "approve_session"})
    assert p.invoke("local:fs_list", {"path": "."}).ok
    assert persisted == [("fs_list", "approve_session")]


def test_always_allow_stamp_persists(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    persisted = []
    p = make_provider(
        state=ASK,
        root=tmp_path,
        persist_approval=lambda hub, decision: persisted.append((hub.name, decision)),
    )
    p.apply_batch_decisions({"fs_list": "always_allow"})
    assert p.invoke("local:fs_list", {"path": "."}).ok
    assert persisted == [("fs_list", "always_allow")]


def test_approve_once_stamp_does_not_persist(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    persisted = []
    p = make_provider(
        state=ASK,
        root=tmp_path,
        persist_approval=lambda hub, decision: persisted.append((hub.name, decision)),
    )
    p.apply_batch_decisions({"fs_list": "approve_once"})
    assert p.invoke("local:fs_list", {"path": "."}).ok
    assert persisted == []


def test_callback_approve_session_persists(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    persisted = []
    p = make_provider(
        state=ASK,
        root=tmp_path,
        approval_callback=lambda pending: {"fs_list": "approve_session"},
        persist_approval=lambda hub, decision: persisted.append((hub.name, decision)),
    )
    assert p.invoke("local:fs_list", {"path": "."}).ok
    assert persisted == [("fs_list", "approve_session")]


def test_persist_failure_does_not_block_execution(tmp_path):
    (tmp_path / "a.txt").write_text("a")

    def boom(hub, decision):
        raise RuntimeError("store write failed")

    p = make_provider(state=ASK, root=tmp_path, persist_approval=boom)
    p.apply_batch_decisions({"fs_list": "always_allow"})
    assert p.invoke("local:fs_list", {"path": "."}).ok


def test_session_approval_read_failure_is_not_approved(tmp_path):
    def boom(hub):
        raise RuntimeError("store read failed")

    p = make_provider(state=ASK, root=tmp_path, is_session_approved=boom)
    # read failure -> still gated, and invoke still fails closed without a stamp
    assert p.pending_gate_for("fs_list", {"path": "."}) is not None
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL
