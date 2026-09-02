"""The loop retries a transient model failure instead of discarding the run.

TASK-25901 integration half. The unit tests in `test_model_retry.py` prove the
classifier and the backoff; these prove the loop actually uses them, and that a
terminal error still ends the run on the first attempt.

Nothing here sleeps for real: `LoopDeps.sleep` is injected and records what it
was asked to wait for.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import (
    RUN_DONE,
    LoopDeps,
    run_agent_loop,
)
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatRateLimitError,
)


def _run(script, *, budget=None, clock=None, config=None):
    """Drive the loop with a scripted call_model that may raise."""
    remaining = list(script)
    slept: list[float] = []

    def call_model(messages, active_schemas):
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="x"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=clock or (lambda: 0.0),
        sleep=slept.append,
    )
    cfg = config or AgentConfig(
        model="m",
        system_prompt="s",
        budget=budget or RunBudget(),
    )
    outcome = run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps)
    return outcome, slept, remaining


def test_a_transient_failure_is_retried_and_the_run_completes():
    """AC#1: one 429 must not discard the run."""
    outcome, slept, remaining = _run(
        [ChatRateLimitError("slow down"), ModelTurn(text="Tokyo.")]
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Tokyo."
    # Total time, not call count: the backoff sleep is sliced into <=0.5s
    # chunks so a Stop during backoff is honoured promptly.
    assert sum(slept) > 0, "should have backed off before the retry"
    assert not remaining


def test_a_terminal_failure_is_not_retried():
    """AC#3: retrying an auth failure spends money to reach the same place."""
    with pytest.raises(ChatAuthenticationError):
        _run([ChatAuthenticationError("bad key"), ModelTurn(text="unreachable")])


def test_retries_are_bounded():
    """AC#2: a provider that is down stays down; give up and say so."""
    budget = RunBudget(max_model_retries=2)
    with pytest.raises(ChatRateLimitError):
        _run([ChatRateLimitError("slow")] * 6, budget=budget)


def test_each_retry_is_visible_in_the_trace():
    """AC#4: a silently-retried run hides a failing provider."""
    seen = []

    remaining = [ChatRateLimitError("slow"), ModelTurn(text="ok")]

    def call_model(messages, active_schemas):
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="x"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        sleep=lambda s: None,
        on_trace_step=seen.append,
    )
    cfg = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps)

    summaries = " | ".join(str(s.summary) for s in seen)
    assert "retry" in summaries.lower(), f"no retry step in trace: {summaries}"


def test_retry_never_runs_past_the_wall_budget():
    """AC#5: the backoff must not extend a run beyond its own deadline."""
    now = {"t": 0.0}

    def clock():
        return now["t"]

    def advancing_sleep(seconds):
        now["t"] += seconds

    remaining = [ChatRateLimitError("slow")] * 8

    def call_model(messages, active_schemas):
        raise remaining.pop(0)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="x"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=clock,
        sleep=advancing_sleep,
    )
    budget = RunBudget(max_wall_seconds=5.0, max_model_retries=10)
    cfg = AgentConfig(model="m", system_prompt="s", budget=budget)

    with pytest.raises(ChatRateLimitError):
        run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps)

    assert now["t"] <= 5.0, (
        f"backoff slept until {now['t']}s, past the 5s wall budget"
    )


def test_no_retry_configured_reproduces_the_old_behaviour():
    """max_model_retries=0 must raise on the first failure, as before."""
    budget = RunBudget(max_model_retries=0)
    with pytest.raises(ChatRateLimitError):
        _run([ChatRateLimitError("slow"), ModelTurn(text="unreachable")], budget=budget)
