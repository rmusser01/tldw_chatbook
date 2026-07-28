# Tests/Agents/test_run_log_on_record.py
"""on_record captures FULL fidelity at both loop call sites."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

from Tests.Agents.test_agent_runtime import make_deps


def collect():
    seen = []
    return seen, lambda kind, payload: seen.append((kind, payload))


def run(turns, *, invoke=None, budget=None):
    seen, hook = collect()
    deps = make_deps(turns, invoke=invoke)
    deps.on_record = hook
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=budget or RunBudget(max_steps=8, max_model_turns=8),
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    return seen, outcome


def test_model_record_carries_full_text_not_the_200_char_summary():
    long_text = "z" * 5000
    seen, _ = run([ModelTurn(text=long_text)])
    model_records = [p for kind, p in seen if kind == "model"]
    assert model_records and model_records[0]["content"] == long_text


def test_tool_result_record_carries_content_before_truncation():
    big = "q" * 40_000
    turns = [
        ModelTurn(
            text="",
            tool_calls=(ToolCall(name="calculator", args={}, call_id="c1"),),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(
        turns,
        invoke=lambda c: ToolResult(ok=True, content=big),
        budget=RunBudget(max_steps=8, max_model_turns=8, max_tool_result_chars=100),
    )
    results = [p for kind, p in seen if kind == "tool_result"]
    assert results and results[0]["content"] == big
    assert results[0]["call_id"] == "c1"


def test_tool_call_record_carries_full_args():
    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1+1"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(turns)
    calls = [p for kind, p in seen if kind == "tool_call"]
    assert calls and "1+1" in calls[0]["content"]


def test_runtime_tool_results_are_captured_too():
    # find_tools never reaches deps.invoke_tool -- a service-side wrapper
    # would have missed it entirely.
    turns = [
        ModelTurn(
            text="",
            tool_calls=(ToolCall(name="find_tools", args={"query": "x"}, call_id="c1"),),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(turns)
    tools = [p["tool"] for kind, p in seen if kind == "tool_result"]
    assert "find_tools" in tools


def test_refused_tool_call_still_emits_tool_call_and_tool_result_records():
    # Pins the placement of both _emit_record calls at the dispatch site:
    # they must sit at the `for call in calls:` body level, OUTSIDE the
    # `if verdict != "proceed": ... else: ...` pair, so a review_tool_calls
    # refusal (the same seam MCP approval refusals ride) is captured too.
    # A mutation that nests both calls one level deeper -- into the `else:`
    # branch that only runs on a "proceed" verdict -- silently stops
    # logging every refused call, and no other test in this suite catches
    # that: this one must fail if that happens.
    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1+1"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, hook = collect()
    deps = make_deps(turns)
    deps.on_record = hook
    deps.review_tool_calls = lambda calls: {"calculator": "blocked by policy"}
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=8, max_model_turns=8),
    )
    run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)

    tool_calls = [p for kind, p in seen if kind == "tool_call"]
    tool_results = [p for kind, p in seen if kind == "tool_result"]

    assert tool_calls, "refused call must still emit a tool_call record"
    assert tool_calls[0]["tool"] == "calculator"
    assert tool_calls[0]["call_id"] == "c1"

    assert tool_results, "refused call must still emit a tool_result record"
    refused = tool_results[0]
    assert refused["status"] == "refused"
    assert refused["content"] == "blocked by policy"
    assert refused["tool"] == "calculator"
    assert refused["call_id"] == "c1"


def test_successful_dispatched_call_is_still_status_ok():
    # Regression guard for the fix below: a genuine success must not
    # collapse to "error" just because the status computation now looks at
    # `result.ok` instead of the verdict alone.
    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1+1"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(turns, invoke=lambda c: ToolResult(ok=True, content="2"))
    results = [p for kind, p in seen if kind == "tool_result"]
    assert results and results[0]["status"] == "ok"


def test_failing_tool_call_emits_and_is_searchable_as_error_status():
    """tool_catalog.py documents `status` as filterable by "ok or error", but
    the loop only ever wrote "ok" (any "proceed" verdict, even a dispatch
    that actually failed -- see `content = ... f"ERROR: {result.error}"` in
    agent_runtime.py) or "refused" -- "error" was never reachable, so an
    agent filtering for its own failed calls got zero hits forever. This
    pins both ends: the loop must emit status="error" for a genuine dispatch
    failure, and search_records(..., status="error") must actually find it.
    """
    from tldw_chatbook.Agents.run_log_format import RunLogRecord
    from tldw_chatbook.Agents.run_log_search import search_records

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1/0"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    seen, _ = run(
        turns, invoke=lambda c: ToolResult(ok=False, error="division by zero")
    )
    results = [p for kind, p in seen if kind == "tool_result"]
    assert results, "a dispatched failure must still emit a tool_result record"
    failed = results[0]
    assert failed["status"] == "error"
    assert failed["content"] == "ERROR: division by zero"

    record = RunLogRecord(
        number=1,
        run_id="r",
        kind="primary",
        type="tool_result",
        ts="t",
        content=failed["content"],
        tool=failed["tool"],
        status=failed["status"],
        call_id=failed["call_id"],
    )
    hits = search_records([record], status="error")
    assert [r.number for r in hits] == [1]


def test_failing_hook_never_aborts_the_run():
    def boom(kind, payload):
        raise RuntimeError("log is on fire")

    deps = make_deps([ModelTurn(text="fine")])
    deps.on_record = boom
    config = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert outcome.final_text == "fine"


def test_absent_hook_is_a_no_op():
    deps = make_deps([ModelTurn(text="fine")])
    config = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert outcome.final_text == "fine"


# -- F5 (Qodo #5, PR #1066 review): tool_call must be durable BEFORE dispatch


def test_tool_call_record_exists_even_when_the_tool_raises():
    """A crash mid-dispatch must not erase the record that the call was
    ever attempted -- that is the whole point of a crash-durable log. The
    old placement (both _emit_record calls at the content-assembly point,
    AFTER dispatch) meant a tool that raised left NO durable record at all.
    This drives a tool whose invoke_tool raises, confirms the exception
    still propagates (this test does not change that -- only capture
    placement), and confirms the tool_call record was ALREADY written
    (present in `seen`, standing in for "already flushed to disk") before
    that exception ever happened.
    """

    def boom(call):
        raise RuntimeError("the tool call hung and the worker was killed")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "1+1"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
    ]
    seen, hook = collect()
    deps = make_deps(turns, invoke=boom)
    deps.on_record = hook
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=8, max_model_turns=8),
    )
    with pytest.raises(RuntimeError, match="worker was killed"):
        run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)

    tool_calls = [p for kind, p in seen if kind == "tool_call"]
    assert tool_calls, "the tool_call record must exist despite the crash"
    assert tool_calls[0]["tool"] == "calculator"
    assert tool_calls[0]["call_id"] == "c1"
    assert "1+1" in tool_calls[0]["content"]
    # The tool never returned, so no tool_result record can exist -- this
    # confirms the ordering (tool_call written, THEN dispatch attempted),
    # not merely that a tool_call record exists somewhere.
    assert not [p for kind, p in seen if kind == "tool_result"]


def test_tool_call_record_exists_even_when_the_tool_never_returns():
    """Same durability property, phrased for the "hangs forever" case named
    in the finding rather than "raises": a tool whose invoke_tool blocks
    indefinitely (simulated here as never being called at all, since the
    loop itself is synchronous and a real hang cannot be driven in a unit
    test) must still have left its tool_call record. This is really the
    same assertion as the "raises" test from the log's point of view: the
    record exists before dispatch resolves, by whatever means it resolves.
    """
    seen, hook = collect()

    def never_returns(call):
        raise TimeoutError("simulated: this call never returned")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name="calculator", args={"expr": "2+2"}, call_id="c9"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
    ]
    deps = make_deps(turns, invoke=never_returns)
    deps.on_record = hook
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=8, max_model_turns=8),
    )
    with pytest.raises(TimeoutError):
        run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)

    tool_calls = [p for kind, p in seen if kind == "tool_call"]
    assert tool_calls and tool_calls[0]["call_id"] == "c9"


def test_spawn_tool_call_record_is_emitted_before_the_spawn_dispatch_runs():
    """Pure-loop counterpart of the service-level ordering test in
    test_run_log_service_wiring.py (which drives a REAL nested spawn through
    RunLogWriter). Here `deps.spawn` itself appends records through the SAME
    shared hook while it runs -- simulating a child's own on_record calls
    happening DURING the parent's spawn dispatch, exactly as a real
    spawn_subagent does (it runs the child's entire loop inline, BEFORE the
    parent's own spawn_subagent tool_call/tool_result pair around it
    finishes assembling). Before the F5 fix, the parent's tool_call record
    for spawn_subagent was emitted AFTER `deps.spawn()` returned, so it
    would have appeared AFTER these simulated child records below --
    backwards. After the fix, it must appear FIRST.
    """
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME

    seen, hook = collect()

    def fake_spawn(task):
        # Simulate a child writing its own records mid-dispatch, exactly
        # as a real nested _run_one does through the shared writer.
        hook("model", {"content": "child turn 1", "tool": "", "status": "", "call_id": ""})
        hook(
            "tool_call",
            {"content": "{}", "tool": "child_tool", "status": "", "call_id": ""},
        )
        return ToolResult(ok=True, content="child done")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=SPAWN_TOOL_NAME, args={"task": "go investigate"}, call_id="s1"
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="parent done"),
    ]
    deps = make_deps(turns, spawn=fake_spawn)
    deps.on_record = hook
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=8, max_model_turns=8, max_subagents=1),
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert outcome.final_text == "parent done"

    kinds_in_order = [kind for kind, _ in seen]
    parent_spawn_call_index = next(
        i
        for i, (kind, p) in enumerate(seen)
        if kind == "tool_call" and p.get("tool") == SPAWN_TOOL_NAME
    )
    child_first_index = next(
        i for i, (kind, p) in enumerate(seen) if p.get("tool") == "child_tool"
    )
    assert parent_spawn_call_index < child_first_index, (
        "the parent's spawn_subagent tool_call record must be written "
        "BEFORE the child's own records, not after -- got order "
        f"{kinds_in_order}"
    )
