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


def test_catalog_lists_default_specs_with_local_ids(tmp_path):
    p = make_provider(root=tmp_path)
    entries = p.list_catalog()
    assert [e.id for e in entries] == [
        "local:fs_list", "local:fs_read", "local:fs_write", "local:fs_edit",
        "local:fs_glob", "local:fs_grep", "local:web_fetch", "local:web_search",
    ]
    assert entries[0].name == "fs_list" and entries[0].source == "local"
    schema = p.load_schema("local:fs_list")
    assert schema.parameters["required"] == ["path"]


def test_catalog_lists_fs_read_with_paging_params(tmp_path):
    p = make_provider(root=tmp_path)
    entry = next(e for e in p.list_catalog() if e.id == "local:fs_read")
    assert entry.name == "fs_read" and entry.source == "local"
    schema = p.load_schema("local:fs_read")
    assert schema.parameters["required"] == ["path"]
    props = schema.parameters["properties"]
    assert props["path"]["type"] == "string"
    assert props["offset"]["type"] == "integer"
    assert props["limit"]["type"] == "integer"
    assert p.hub_tool_for("fs_read").tags == ()  # read-only: no risk tags


def test_fs_write_spec_carries_mutates_tag(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:fs_write")
    assert sorted(schema.parameters["required"]) == ["content", "path"]
    assert p.hub_tool_for("fs_write").tags == ("mutates",)


def test_fs_edit_spec_carries_mutates_tag(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:fs_edit")
    assert sorted(schema.parameters["required"]) == ["new_string", "old_string", "path"]
    props = schema.parameters["properties"]
    assert props["replace_all"]["type"] == "boolean"
    assert props["replace_all"]["default"] is False
    assert p.hub_tool_for("fs_edit").tags == ("mutates",)


def test_fs_glob_spec_read_only(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:fs_glob")
    assert schema.parameters["required"] == ["pattern"]
    assert "max_results" in schema.parameters["properties"]
    assert p.hub_tool_for("fs_glob").tags == ()  # read-only: no risk tags


def test_fs_grep_spec_read_only_with_mode_enum(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:fs_grep")
    assert schema.parameters["required"] == ["pattern"]
    props = schema.parameters["properties"]
    assert props["mode"]["enum"] == ["content", "files", "count"]
    assert props["mode"]["default"] == "content"
    assert "max_results" in props
    assert p.hub_tool_for("fs_grep").tags == ()  # read-only: no risk tags


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


# -- audit recording seam (record_decision) ------------------------------------
#
# MCP parity (mcp_tool_provider.py): `record_tool_decision` is called ONLY for
# decisions that never executed -- "denied" (kill switch, deny state, no
# callback, deny/unrecognized verdict) and "denied-timeout" (timeout verdict).
# Successful executions are recorded service-side by execute_hub_tool, which
# the local provider has no analogue for, so this seam records refusals only.


def _recording_provider(tmp_path, **kwargs):
    recorded = []
    p = make_provider(
        root=tmp_path,
        record_decision=lambda hub, decision: recorded.append((hub, decision)),
        **kwargs,
    )
    return p, recorded


def test_deny_state_records_denied(tmp_path):
    p, recorded = _recording_provider(tmp_path, state=DENY)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL
    assert [(h.name, d) for h, d in recorded] == [("fs_list", "denied")]
    assert recorded[0][0].server_key == "local:__local__"


def test_kill_switch_records_denied(tmp_path):
    p, recorded = _recording_provider(tmp_path, kill=True)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL
    assert [(h.name, d) for h, d in recorded] == [("fs_list", "denied")]


def test_timeout_stamp_records_denied_timeout(tmp_path):
    p, recorded = _recording_provider(tmp_path, state=ASK)
    p.apply_batch_decisions({"fs_list": "timeout"})
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL
    assert [(h.name, d) for h, d in recorded] == [("fs_list", "denied-timeout")]


def test_ask_without_callback_records_denied_timeout(tmp_path):
    # no_callback fails closed to the timeout refusal (pinned copy, spec §3.3),
    # so the recorded decision matches the refusal the model actually saw.
    p, recorded = _recording_provider(tmp_path, state=ASK)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL
    assert [(h.name, d) for h, d in recorded] == [("fs_list", "denied-timeout")]


def test_deny_stamp_records_denied(tmp_path):
    p, recorded = _recording_provider(tmp_path, state=ASK)
    p.apply_batch_decisions({"fs_list": "deny"})
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL
    assert [(h.name, d) for h, d in recorded] == [("fs_list", "denied")]


def test_allow_execution_records_nothing(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    p, recorded = _recording_provider(tmp_path)
    assert p.invoke("local:fs_list", {"path": "."}).ok
    assert recorded == []


def test_unknown_tool_records_nothing(tmp_path):
    p, recorded = _recording_provider(tmp_path)
    r = p.invoke("local:nope", {})
    assert not r.ok and "Unknown local tool" in r.error
    assert recorded == []


def test_record_decision_none_means_no_recording(tmp_path):
    # Seam is optional; refusal paths must work unchanged without it.
    p = make_provider(state=DENY, root=tmp_path)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_record_decision_raise_does_not_break_invoke(tmp_path):
    def boom(hub, decision):
        raise RuntimeError("audit store down")

    p = make_provider(state=DENY, root=tmp_path, record_decision=boom)
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


# -- web_fetch / web_search specs (phase 3a) ------------------------------------


def test_web_fetch_spec_schema(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:web_fetch")
    assert schema.parameters["required"] == ["url"]
    props = schema.parameters["properties"]
    assert props["url"]["type"] == "string"
    assert props["max_bytes"]["type"] == "integer"
    assert "max_bytes" not in schema.parameters["required"]
    # network-classed: default ask comes from the global permission default,
    # so no risk tags.
    assert p.hub_tool_for("web_fetch").tags == ()


def test_web_search_spec_schema(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:web_search")
    assert schema.parameters["required"] == ["query"]
    props = schema.parameters["properties"]
    assert props["query"]["type"] == "string"
    assert "duckduckgo" in props["search_engine"]["enum"]
    assert props["result_count"]["type"] == "integer"
    assert p.hub_tool_for("web_search").tags == ()


def _fake_search_payload(count, snippet_len=50):
    return {
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "snippet": f"snippet {i} " + ("x" * snippet_len),
            }
            for i in range(1, count + 1)
        ]
    }


def test_web_search_handler_wires_legacy_defaults_and_bounds_results(tmp_path, monkeypatch):
    seen = {}

    def fake_perform_websearch(**kwargs):
        seen.update(kwargs)
        return _fake_search_payload(count=3, snippet_len=10_000)

    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch",
        fake_perform_websearch,
    )
    p = make_provider(root=tmp_path)
    r = p.invoke("local:web_search", {"query": "python"})
    assert r.ok
    # legacy Tools/web_search_tool.py config-default wiring, passed through
    assert seen["search_engine"] == "duckduckgo"
    assert seen["search_query"] == "python"
    assert seen["content_country"] == "US"
    assert seen["search_lang"] == "en"
    assert seen["output_lang"] == "en"
    assert seen["result_count"] == 5
    assert seen["safesearch"] == "moderate"
    # each result block bounded to ~4 KiB
    blocks = [b for b in r.content.split("\n\n") if b.strip()]
    assert len(blocks) == 3
    for block in blocks:
        assert len(block) <= 4 * 1024 + len("… [truncated]")
    assert "… [truncated]" in r.content


def test_web_search_handler_enforces_total_cap(tmp_path, monkeypatch):
    # 10 results x ~4 KiB each would exceed the total cap without bounding.
    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch",
        lambda **kwargs: _fake_search_payload(count=10, snippet_len=10_000),
    )
    p = make_provider(root=tmp_path)
    r = p.invoke("local:web_search", {"query": "python", "result_count": 10})
    assert r.ok
    assert "omitted" in r.content
    assert len(r.content.encode("utf-8")) < 32 * 1024  # provider fit never triggers


def test_web_search_backend_error_becomes_result_string(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch", boom
    )
    p = make_provider(root=tmp_path)
    r = p.invoke("local:web_search", {"query": "python"})
    # legacy contract: backend failure is a result string, not an exception.
    assert r.ok
    assert "backend exploded" in r.content
