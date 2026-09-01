"""A run that hits its budget should end with a summary, not a dead stop.

TASK-26001. The four budget-exhaustion branches returned RUN_STUCK with a bare
error step -- the user got nothing usable from work already done. Two additions:
a one-time warning to the model as the budget approaches (attached cache-safely
to the newest tool result, never as a synthetic user turn), and one final
tools-stripped model call at exhaustion whose output becomes the run's
final_text while the status stays RUN_STUCK (an exhausted run must remain
distinguishable from success -- AC#6).
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import (
    RUN_DONE,
    RUN_STUCK,
    LoopDeps,
    run_agent_loop,
)


def _fence(name="calculator", args=None):
    body = json.dumps({"name": name, "arguments": args or {}})
    return f"```tool_call\n{body}\n```"


def _drive(script, budget, *, clock=None):
    """Drive the loop; capture every (messages, active) call_model received."""
    remaining = list(script)
    seen = []

    def call_model(messages, active_schemas):
        seen.append(([dict(m) for m in messages], tuple(active_schemas)))
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="tool-ok"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=clock or (lambda: 0.0),
        sleep=lambda s: None,
    )
    cfg = AgentConfig(
        model="m",
        system_prompt="s",
        provider="openai",
        allowed_tools=("calculator",),
        budget=budget,
    )
    outcome = run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps)
    return outcome, seen


def test_exhaustion_produces_a_wrapup_summary():
    """AC#3: one final tools-stripped call, its text as the result."""
    outcome, seen = _drive(
        [
            ModelTurn(text=_fence()),
            ModelTurn(text=_fence()),
            ModelTurn(text="Summary of what was done."),
        ],
        RunBudget(max_model_turns=2),
    )

    assert outcome.status == RUN_STUCK, "an exhausted run must not read as done"
    assert outcome.final_text == "Summary of what was done."
    wrap_messages, wrap_active = seen[-1]
    assert wrap_active == (), "the wrap-up call must carry no tool schemas"


def test_the_exhaustion_step_is_still_recorded():
    """AC#6: the honest error step survives alongside the summary."""
    outcome, _ = _drive(
        [
            ModelTurn(text=_fence()),
            ModelTurn(text=_fence()),
            ModelTurn(text="Summary."),
        ],
        RunBudget(max_model_turns=2),
    )

    summaries = [str(s.summary) for s in outcome.steps]
    assert any("model-turn budget exhausted" in s for s in summaries)


def test_a_failed_wrapup_still_terminates_honestly():
    """AC#5: the summary is a bonus, never a new failure mode."""
    outcome, _ = _drive(
        [
            ModelTurn(text=_fence()),
            ModelTurn(text=_fence()),
            RuntimeError("wrap-up call exploded"),
        ],
        RunBudget(max_model_turns=2),
    )

    assert outcome.status == RUN_STUCK
    assert not outcome.final_text


def test_tool_calls_in_the_wrapup_response_are_ignored():
    """AC#4: the wrap-up cannot loop or spawn tools."""
    outcome, seen = _drive(
        [
            ModelTurn(text=_fence()),
            ModelTurn(text=_fence()),
            ModelTurn(text=f"Ignore this: {_fence()}"),
        ],
        RunBudget(max_model_turns=2),
    )

    assert outcome.status == RUN_STUCK
    # exactly 3 calls: two real turns + one wrap-up; no dispatch of the fence
    assert len(seen) == 3


def test_wall_exhaustion_also_gets_a_wrapup():
    now = {"t": 0.0}

    def clock():
        now["t"] += 3.0
        return now["t"]

    outcome, seen = _drive(
        [ModelTurn(text=_fence()), ModelTurn(text="Summary.")],
        RunBudget(max_wall_seconds=5.0),
        clock=clock,
    )

    assert outcome.status == RUN_STUCK
    assert outcome.final_text == "Summary."


def test_warning_attaches_to_the_newest_tool_result_once():
    """AC#1/#2: told once, no synthetic user turn, cache prefix intact."""
    # args vary per call: three identical calls would trip the (correct)
    # cycle detector and mask what this test is about.
    outcome, seen = _drive(
        [
            ModelTurn(text=_fence(args={"expression": "1+1"})),
            ModelTurn(text=_fence(args={"expression": "2+2"})),
            ModelTurn(text=_fence(args={"expression": "3+3"})),
            ModelTurn(text="done before exhaustion"),
        ],
        RunBudget(max_model_turns=8, max_steps=50, budget_warning_fraction=0.25),
    )

    assert outcome.status == RUN_DONE
    # find the first call whose history carries the notice
    noticed = [
        (i, msgs) for i, (msgs, _a) in enumerate(seen)
        if any("budget notice" in str(m.get("content", "")) for m in msgs)
    ]
    assert noticed, "the warning never reached the model"
    first_index, first_msgs = noticed[0]
    # attached to the newest tool result, not a new message
    prev_msgs, _ = seen[first_index - 1]
    assert len(first_msgs) == len(prev_msgs) + 2, (
        "the notice must ride an existing message, not add one beyond the "
        "normal assistant+tool-result growth"
    )
    carriers = [m for m in first_msgs if "budget notice" in str(m.get("content", ""))]
    assert len(carriers) == 1
    # the carrier is the NEWEST message -- the tool result just appended
    assert first_msgs.index(carriers[0]) == len(first_msgs) - 1
    assert carriers[0]["role"] in ("tool", "user")
    # cache-prefix property (AC#2): every message that already existed at the
    # previous call is byte-identical -- only the tail changed
    assert first_msgs[: len(prev_msgs)] == prev_msgs


def test_warning_is_delivered_at_most_once():
    outcome, seen = _drive(
        [
            ModelTurn(text=_fence(args={"expression": "1+1"})),
            ModelTurn(text=_fence(args={"expression": "2+2"})),
            ModelTurn(text=_fence(args={"expression": "3+3"})),
            ModelTurn(text=_fence(args={"expression": "4+4"})),
            ModelTurn(text="done"),
        ],
        RunBudget(max_model_turns=10, max_steps=50, budget_warning_fraction=0.2),
    )

    final_msgs, _ = seen[-1]
    notices = sum(
        str(m.get("content", "")).count("budget notice") for m in final_msgs
    )
    assert notices == 1, f"warning delivered {notices} times"


def test_no_warning_below_the_fraction():
    outcome, seen = _drive(
        [ModelTurn(text=_fence()), ModelTurn(text="done")],
        RunBudget(max_model_turns=100, max_steps=50, budget_warning_fraction=0.9),
    )

    assert outcome.status == RUN_DONE
    assert not any(
        "budget notice" in str(m.get("content", ""))
        for msgs, _ in seen
        for m in msgs
    )
