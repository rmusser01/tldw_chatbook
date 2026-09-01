"""An empty model turn is not a finished answer.

TASK-26002. `run_agent_loop` returned RUN_DONE for any turn with no tool calls,
without checking it produced anything -- so a provider returning empty output
looked to the user exactly like the agent deciding it was finished. A named grep
for empty_response / blank response across Agents/ returned zero.

A single empty is a blip and is retried. Two consecutive empties from the same
provider and model are deterministic: something is wrong with the request or the
model, and asking again spends money to reach the same place.

The case that must NOT trip: a turn with tool calls and no text. That is the
normal shape of a model deciding to call a tool.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import (
    RUN_DONE,
    RUN_STUCK,
    LoopDeps,
    run_agent_loop,
)


def _run(script, *, provider="openai", model="gpt-x"):
    remaining = list(script)

    def call_model(messages, active_schemas):
        return remaining.pop(0)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="tool-ok"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        sleep=lambda s: None,
    )
    cfg = AgentConfig(
        model=model,
        system_prompt="s",
        provider=provider,
        allowed_tools=("calculator",),
        budget=RunBudget(),
    )
    outcome = run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps)
    return outcome, remaining


def test_one_empty_then_a_real_answer_completes():
    """AC#3: a single blank is a blip, not a verdict."""
    outcome, remaining = _run([ModelTurn(text=""), ModelTurn(text="Tokyo.")])

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Tokyo."
    assert not remaining


def test_two_consecutive_empties_stop_the_run():
    """AC#2: deterministic emptiness must not be retried forever."""
    outcome, _ = _run([ModelTurn(text="")] * 6)

    assert outcome.status == RUN_STUCK


def test_the_terminal_message_names_the_provider_and_model():
    """AC#4: the user has to know what to go and fix."""
    outcome, _ = _run(
        [ModelTurn(text="")] * 4, provider="anthropic", model="claude-x"
    )

    summaries = " | ".join(str(s.summary) for s in outcome.steps)
    assert "anthropic" in summaries
    assert "claude-x" in summaries


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t\n  "])
def test_whitespace_only_counts_as_empty(blank):
    outcome, _ = _run([ModelTurn(text=blank)] * 4)

    assert outcome.status == RUN_STUCK


def test_a_tool_call_with_no_text_is_not_empty():
    """AC#5: the normal shape of a model calling a tool must not trip this."""
    outcome, _ = _run(
        [
            ModelTurn(
                text="",
                tool_calls=(ToolCall(name="calculator", args={}, call_id="c1"),),
            ),
            ModelTurn(text="It is 42."),
        ]
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "It is 42."


def test_the_empty_counter_resets_on_intervening_content():
    """Two empties SEPARATED by content are not 'consecutive'.

    The intervening turn has to be a TOOL CALL, not text: a turn with text and
    no calls is a finished answer and ends the run, so it could never sit in
    the middle of a sequence. (The first version of this test used text and was
    simply wrong about the loop's shape.)
    """
    outcome, remaining = _run(
        [
            ModelTurn(text=""),
            ModelTurn(
                text="",
                tool_calls=(ToolCall(name="calculator", args={}, call_id="c1"),),
            ),
            ModelTurn(text=""),
            ModelTurn(text="Tokyo."),
        ]
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Tokyo."
    assert not remaining


def test_a_normal_answer_is_unaffected():
    outcome, _ = _run([ModelTurn(text="Tokyo.")])

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Tokyo."
