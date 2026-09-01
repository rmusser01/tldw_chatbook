"""Redirect: cut off the current model response, keep everything else.

TASK-26000. Stop was terminal -- correcting a running agent discarded every
tool result in the turn and the correction became a fresh turn that redid the
work. Redirect aborts only the in-flight model request, keeps completed tool
results, preserves the partial text the user watched stream, appends the
correction as a plain user-authored message, and re-runs the SAME turn --
inside the loop, because the sticky fallback switch lives in loop locals and a
re-entry would silently restart on a failed primary provider.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    STEERING_SOURCE_REDIRECT,
    STEERING_SOURCE_USER,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import (
    RUN_CANCELLED,
    RUN_DONE,
    LoopDeps,
    run_agent_loop,
)
from tldw_chatbook.Chat.Chat_Deps import ChatRateLimitError


def _fence(name="calculator", args=None):
    body = json.dumps({"name": name, "arguments": args or {}})
    return f"```tool_call\n{body}\n```"


class _Mailbox:
    """A fake of the service mailbox: drain + redirect-presence probe."""

    def __init__(self):
        self.entries = []

    def post(self, source, text):
        self.entries.append((source, text))

    def drain(self):
        drained, self.entries = self.entries, []
        return drained

    def has_redirect(self):
        return any(s == STEERING_SOURCE_REDIRECT for s, _ in self.entries)


def _deps(call_model, mailbox, *, cancel=None, invoked=None, fallback=None):
    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoked or (lambda c: ToolResult(ok=True, content="tool-ok")),
        spawn=lambda t: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=cancel or (lambda: False),
        clock=lambda: 0.0,
        drain_mailbox=mailbox.drain,
        sleep=lambda s: None,
        fallback=fallback,
        has_pending_redirect=mailbox.has_redirect,
    )


def _cfg(**kw):
    kw.setdefault("model", "m")
    kw.setdefault("system_prompt", "s")
    kw.setdefault("provider", "openai")
    kw.setdefault("allowed_tools", ("calculator",))
    kw.setdefault("budget", RunBudget(max_steps=50))
    return AgentConfig(**kw)


def test_redirect_mid_stream_keeps_tool_results_and_rewrites_the_turn():
    """AC#1/#2/#3/#6a: the core behaviour, in order."""
    mailbox = _Mailbox()
    histories = []
    remaining = [
        ModelTurn(text=_fence(args={"expression": "6*7"})),
        # the aborted call: partial narration only (transport returned early)
        ModelTurn(text="Looking at the JSON parser, the"),
        ModelTurn(text="Right — the YAML parser."),
    ]

    def call_model(messages, active):
        histories.append([dict(m) for m in messages])
        if len(histories) == 2:
            # user redirects while this call streams; the abort returns the
            # partial as a normal turn
            mailbox.post(STEERING_SOURCE_REDIRECT, "No — the YAML parser")
        return remaining.pop(0)

    outcome = run_agent_loop(
        _cfg(), [{"role": "user", "content": "analyze"}], [], _deps(call_model, mailbox)
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Right — the YAML parser."
    third_view = histories[2]
    contents = [str(m.get("content", "")) for m in third_view]
    tool_i = next(i for i, c in enumerate(contents) if "Tool result" in c)
    partial_i = next(
        i for i, c in enumerate(contents) if "Looking at the JSON parser" in c
    )
    correction_i = next(
        i for i, c in enumerate(contents) if "No — the YAML parser" in c
    )
    assert tool_i < partial_i < correction_i, (
        f"order wrong: tool={tool_i} partial={partial_i} correction={correction_i}"
    )
    assert third_view[partial_i]["role"] == "assistant"
    assert third_view[correction_i]["role"] == "user"
    assert "[Steering" not in contents[correction_i], (
        "a redirect correction is a plain user message, not wrapped steering"
    )


def test_redirect_after_a_fallback_switch_stays_on_the_fallback_provider():
    """THE enforcer: a loop re-entry implementation restarts on the primary
    and fails this test."""
    from tldw_chatbook.Agents.fallback_chain import (
        FallbackCandidate,
        FallbackRuntime,
    )

    mailbox = _Mailbox()
    events = []

    def primary(messages, active, cont=None):
        events.append("primary")
        raise ChatRateLimitError("dead")

    fallback_turns = [
        ModelTurn(text="partial from fallback"),
        ModelTurn(text="corrected answer"),
    ]

    def build(provider):
        def call(messages, active, cont=None):
            events.append(f"candidate:{provider}")
            if len([e for e in events if e.startswith("candidate")]) == 1:
                mailbox.post(STEERING_SOURCE_REDIRECT, "actually do Y")
            return fallback_turns.pop(0)

        return call

    runtime = FallbackRuntime(
        candidates=(FallbackCandidate(provider="groq", native=True, ready=True),),
        build=build,
    )
    outcome = run_agent_loop(
        _cfg(budget=RunBudget(max_model_retries=0, max_steps=50)),
        [{"role": "user", "content": "go"}],
        [],
        _deps(primary, mailbox, fallback=runtime),
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "corrected answer"
    assert events.count("primary") == 1, (
        f"redirect re-ran the failed primary: {events}"
    )


def test_redirect_does_not_redeliver_the_budget_warning():
    """The warning-delivered flag must survive a redirect."""
    mailbox = _Mailbox()
    histories = []
    remaining = [
        ModelTurn(text=_fence(args={"expression": "1"})),
        ModelTurn(text=_fence(args={"expression": "2"})),
        ModelTurn(text="partial before redirect"),
        ModelTurn(text="done"),
    ]

    def call_model(messages, active):
        histories.append([dict(m) for m in messages])
        if len(histories) == 3:
            mailbox.post(STEERING_SOURCE_REDIRECT, "change course")
        return remaining.pop(0)

    outcome = run_agent_loop(
        _cfg(
            budget=RunBudget(
                max_model_turns=20, max_steps=50, budget_warning_fraction=0.1
            )
        ),
        [{"role": "user", "content": "go"}],
        [],
        _deps(call_model, mailbox),
    )

    assert outcome.status == RUN_DONE
    final_view = histories[-1]
    notices = sum(
        str(m.get("content", "")).count("budget notice") for m in final_view
    )
    assert notices <= 1, f"warning delivered {notices} times after redirect"


def test_redirect_during_tool_execution_degrades_to_plain_steering():
    """AC#4/#6b: no model call in flight -> the text rides the next drain,
    rendered PLAIN (it is a user reply, not injected guidance)."""
    mailbox = _Mailbox()
    histories = []

    def invoked(call):
        # redirect lands while the tool is running
        mailbox.post(STEERING_SOURCE_REDIRECT, "prefer the YAML file")
        return ToolResult(ok=True, content="tool-ok")

    remaining = [
        ModelTurn(text=_fence(args={"expression": "6*7"})),
        ModelTurn(text="used the YAML file"),
    ]

    def call_model(messages, active):
        histories.append([dict(m) for m in messages])
        return remaining.pop(0)

    outcome = run_agent_loop(
        _cfg(),
        [{"role": "user", "content": "go"}],
        [],
        _deps(call_model, mailbox, invoked=invoked),
    )

    assert outcome.status == RUN_DONE
    second_view = histories[1]
    contents = [str(m.get("content", "")) for m in second_view]
    tool_i = next(i for i, c in enumerate(contents) if "Tool result" in c)
    corr_i = next(i for i, c in enumerate(contents) if "prefer the YAML file" in c)
    assert tool_i < corr_i, "pairing must complete before the correction lands"
    assert second_view[corr_i]["role"] == "user"
    assert "[Steering" not in contents[corr_i]


def test_a_complete_fence_inside_the_partial_is_stripped_not_executed():
    """AC#6c-adjacent: an aborted call's partial can contain a whole fence;
    executing it would run a tool the user just cancelled."""
    mailbox = _Mailbox()
    executed = []
    remaining = [
        ModelTurn(text="about to compute\n" + _fence(args={"expression": "9*9"})),
        ModelTurn(text="redirected answer"),
    ]
    histories = []

    def call_model(messages, active):
        histories.append([dict(m) for m in messages])
        if len(histories) == 1:
            mailbox.post(STEERING_SOURCE_REDIRECT, "don't compute, explain")
        return remaining.pop(0)

    outcome = run_agent_loop(
        _cfg(),
        [{"role": "user", "content": "go"}],
        [],
        _deps(
            call_model,
            mailbox,
            invoked=lambda c: executed.append(c.name) or ToolResult(ok=True, content="81"),
        ),
    )

    assert outcome.status == RUN_DONE
    assert executed == [], "the cancelled turn's tool call was executed anyway"
    second_view = histories[1]
    joined = " | ".join(str(m.get("content", "")) for m in second_view)
    assert "about to compute" in joined, "visible partial text must survive"
    assert "```tool_call" not in joined, "the fence must be stripped from context"


def test_plain_stop_still_beats_a_pending_redirect():
    """AC#5: Stop is terminal, byte-identical, and wins."""
    mailbox = _Mailbox()
    mailbox.post(STEERING_SOURCE_REDIRECT, "too late")
    cancelled = {"flag": False}

    def call_model(messages, active):
        cancelled["flag"] = True
        return ModelTurn(text="partial")

    outcome = run_agent_loop(
        _cfg(),
        [{"role": "user", "content": "go"}],
        [],
        _deps(call_model, mailbox, cancel=lambda: cancelled["flag"]),
    )

    assert outcome.status == RUN_CANCELLED


def test_no_redirect_probe_means_byte_identical_behaviour():
    mailbox = _Mailbox()
    deps = _deps(lambda m, a: ModelTurn(text="fine"), mailbox)
    deps = LoopDeps(
        **{
            f.name: getattr(deps, f.name)
            for f in __import__("dataclasses").fields(LoopDeps)
            if f.name != "has_pending_redirect"
        }
    )

    outcome = run_agent_loop(
        _cfg(), [{"role": "user", "content": "hi"}], [], deps
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "fine"
