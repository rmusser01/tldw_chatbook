# Tests/Agents/test_run_log_on_record.py
"""on_record captures FULL fidelity at both loop call sites."""

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
