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
    kwargs.setdefault("resolve_state", lambda hub: state)
    kwargs.setdefault("kill_switch", lambda: kill)
    return LocalToolProvider(
        workspace_root=Path(kwargs.pop("root", ".")).resolve() if "root" in kwargs else Path("."),
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


# -- fail-closed hardening: verdicts, guarded callables, real args ------------


def test_unrecognized_callback_decision_fails_closed(tmp_path):
    """A garbage decision string must refuse, never fall through to execution."""
    p = make_provider(
        state=ASK,
        root=tmp_path,
        approval_callback=lambda pending: {"fs_list": "yolo"},
    )
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_callback_returning_none_fails_closed(tmp_path):
    p = make_provider(
        state=ASK, root=tmp_path, approval_callback=lambda pending: None
    )
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_callback_raise_fails_closed(tmp_path):
    def boom(pending):
        raise RuntimeError("ui gone")

    p = make_provider(state=ASK, root=tmp_path, approval_callback=boom)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_resolve_state_raise_fails_closed_everywhere(tmp_path):
    def boom(hub):
        raise RuntimeError("store gone")

    p = make_provider(root=tmp_path, resolve_state=boom)
    # pending_gate_for: fail closed to "let invoke handle it" (never raises)
    assert p.pending_gate_for("fs_list", {"path": "."}) is None
    # invoke: refuses rather than raising onto the worker thread
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_kill_switch_read_failure_fails_closed(tmp_path):
    def boom():
        raise RuntimeError("store gone")

    p = make_provider(root=tmp_path, kill_switch=boom)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


def test_callback_receives_real_arguments(tmp_path):
    seen = []

    def callback(pending):
        seen.extend(pending)
        return {"fs_list": "approve_once"}

    p = make_provider(state=ASK, root=tmp_path, approval_callback=callback)
    (tmp_path / "sub").mkdir()
    assert p.invoke("local:fs_list", {"path": "sub"}).ok
    assert len(seen) == 1
    assert seen[0].arguments == {"path": "sub"}  # the approval card shows real args


# -- _fit_result + misc minors ------------------------------------------------


def _big_provider(text, tmp_path):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolSpec

    return LocalToolProvider(
        workspace_root=tmp_path,
        specs=[LocalToolSpec(name="big", description="big", parameters={}, handler=lambda args: text)],
        resolve_state=lambda hub: ALLOW,
    )


def test_fit_result_truncates_oversize(tmp_path):
    p = _big_provider("x" * 40_000, tmp_path)
    r = p.invoke("local:big", {})
    assert r.ok
    assert r.content.endswith("\n… [truncated]")
    assert len(r.content.encode("utf-8")) <= 32 * 1024 + len("\n… [truncated]".encode("utf-8"))
    assert r.content.startswith("x" * 100)


def test_fit_result_multibyte_boundary(tmp_path):
    # 32767 ASCII bytes + one 2-byte codepoint straddling the 32 KiB cut
    p = _big_provider("a" * 32767 + "é" + "tail", tmp_path)
    r = p.invoke("local:big", {})
    assert r.ok  # no UnicodeDecodeError across the boundary
    assert r.content == "a" * 32767 + "\n… [truncated]"


def test_load_schema_without_colon_raises_key_error_not_index_error(tmp_path):
    p = make_provider(root=tmp_path)
    with pytest.raises(KeyError):
        p.load_schema("nocolon")


def test_empty_exception_message_becomes_nonempty_error(tmp_path):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolSpec

    def boom(args):
        raise ValueError()

    p = LocalToolProvider(
        workspace_root=tmp_path,
        specs=[LocalToolSpec(name="boom", description="b", parameters={}, handler=boom)],
        resolve_state=lambda hub: ALLOW,
    )
    r = p.invoke("local:boom", {})
    assert not r.ok and r.error and "ValueError" in r.error


def test_pending_gate_for_accepts_prefixed_and_bare_names(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    bare = p.pending_gate_for("fs_list", {"path": "."})
    prefixed = p.pending_gate_for("local:fs_list", {"path": "."})
    assert bare is not None and prefixed is not None
    assert bare.llm_name == prefixed.llm_name == "fs_list"
    assert p.pending_gate_for("local:unknown", {}) is None
