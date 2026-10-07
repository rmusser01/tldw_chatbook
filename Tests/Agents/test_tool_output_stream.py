"""Live snapshots stay bounded, scoped, and outside the durable runtime trace."""

import threading

from tldw_chatbook.Agents.tool_output import current_tool_output_sink, tool_output_scope


def test_pending_snapshot_flushes_while_execution_is_held_and_closes_cleanly():
    seen = []
    arrived = threading.Event()
    with tool_output_scope(lambda text: (seen.append(text), arrived.set())):
        sink = current_tool_output_sink()
        sink("stdout", "first")
        assert arrived.wait(1)
        arrived.clear()
        sink("stdout", " second")
        assert arrived.wait(1), "pending output must refresh without another write"
        assert seen[-1] == "stdout\nfirst second"
        for _ in range(30):
            sink("stderr", "x" * 1000)
    assert len(seen[-1]) <= 16000
    assert "truncated" in seen[-1]
    count = len(seen)
    sink("stdout", "late")
    assert len(seen) == count
    assert current_tool_output_sink() is None


def test_broken_display_observer_cannot_break_a_tool():
    def broken(_text):
        raise RuntimeError("secret output")

    with tool_output_scope(broken):
        current_tool_output_sink()("stdout", "secret")


def test_runtime_partial_output_is_projected_but_never_traced_or_sent_to_model():
    from dataclasses import replace

    from Tests.Agents.test_agent_runtime_review_hook import (
        CALC,
        CFG,
        _native_turn,
        make_deps,
    )
    from tldw_chatbook.Agents.agent_models import (
        ModelTurn,
        ToolCall,
        ToolRecordProjection,
        ToolResult,
    )
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    seen, traced, legacy, records = [], [], [], []
    histories = []

    def invoke(_call):
        current_tool_output_sink()("stdout", "SECRET_PARTIAL")
        return ToolResult(ok=True, content="FINAL")

    deps = make_deps(
        [_native_turn([ToolCall("calculator", {}, "call")]), ModelTurn(text="done")],
        invoke=invoke,
    )
    original = deps.call_model

    def model(messages, schemas):
        histories.append(str(messages))
        return original(messages, schemas)

    deps = replace(
        deps,
        call_model=model,
        on_tool_activity=seen.append,
        on_trace_step=traced.append,
        on_step=legacy.append,
        on_record=records.append,
        has_tool_record_projection=lambda call: True,
        project_tool_record=lambda audience, call, result: ToolRecordProjection(
            arguments={},
            content="SAFE_PARTIAL"
            if result and "SECRET_PARTIAL" in result.content
            else (result.content if result else ""),
        ),
    )
    run_agent_loop(CFG, [], [CALC], deps)
    output = [step for step in seen if step.kind == "tool_output"]
    assert len(output) == 1 and output[0].result == "SAFE_PARTIAL"
    assert output[0].source_step_index is not None
    assert all(step.kind != "tool_output" for step in traced + legacy)
    assert "PARTIAL" not in str(records) + str(histories)


def test_service_worker_inherits_only_output_observer_and_late_text_is_ignored():
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import _call_with_timeout

    release = threading.Event()
    late = threading.Event()
    seen = []

    def invoke():
        sink = current_tool_output_sink()
        sink("stdout", "before")
        release.wait(2)
        sink("stdout", "after")
        late.set()
        return ToolResult(ok=True, content="final")

    try:
        with tool_output_scope(seen.append):
            result = _call_with_timeout(invoke, 0.1, "held")
            assert result.outcome == "timeout"
        count = len(seen)
        release.set()
        assert late.wait(1)
        assert len(seen) == count and "after" not in str(seen)
    finally:
        release.set()


def test_blocked_observer_cannot_delay_execution_scope_close():
    from concurrent.futures import ThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()

    def observe(text):
        entered.set()
        release.wait(3)

    def execute():
        with tool_output_scope(observe):
            current_tool_output_sink()("stdout", "held display")
            assert entered.wait(2), "hold the observer before closing the scope"

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(execute)
        try:
            assert entered.wait(2)
            assert running.result(timeout=0.5) is None
        finally:
            release.set()


def test_dispatcher_start_failure_keeps_the_authorized_tool_body_running(monkeypatch):
    def unavailable(self):
        raise RuntimeError("cannot start a display thread")

    monkeypatch.setattr(threading.Thread, "start", unavailable)
    executed = []
    with tool_output_scope(lambda text: None):
        assert current_tool_output_sink() is None
        executed.append("ran")
    assert executed == ["ran"]
