"""Steering the PRIMARY run mid-flight.

TASK-25903. Every piece of steering machinery existed -- the protocol-coherent
drain point, `format_steering_message`, the 4000-char cap, the steering bar --
but was wired for fleet children only: `drain_mailbox` was None for a primary
BY DESIGN, so a user's correction could only queue for the next turn, and the
run they were watching finished wrong first.

The service now keeps a mailbox per live primary run. Delivery reuses the
existing drain seam untouched, so steered text can never split a native
tool_calls/role:"tool" pair (AC#2) -- that property is the drain point's, and
these tests prove the primary path actually reaches it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    MAX_STEERING_CHARS,
    STEERING_SOURCE_USER,
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
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry


def _service(**kwargs):
    return AgentService(
        db=SimpleNamespace(), registry=ToolCatalogRegistry(), **kwargs
    )


# --- the service-side mailbox -----------------------------------------------


def test_steering_an_unknown_run_is_refused_honestly():
    """AC#5: no silent drop, a reason comes back."""
    service = _service()

    refusal = service.steer_primary("no-such-run", "go left")

    assert refusal is not None
    assert "not running" in refusal.lower()


def test_steering_a_finished_run_is_refused():
    service = _service()
    drain = service._register_primary_mailbox("run-1")
    service._unregister_primary_mailbox("run-1")

    refusal = service.steer_primary("run-1", "too late")

    assert refusal is not None


def test_empty_text_is_refused():
    """AC#6: the same validation children get."""
    service = _service()
    service._register_primary_mailbox("run-1")

    assert service.steer_primary("run-1", "   ") is not None


def test_over_cap_text_is_refused_naming_the_cap():
    service = _service()
    service._register_primary_mailbox("run-1")

    refusal = service.steer_primary("run-1", "x" * (MAX_STEERING_CHARS + 1))

    assert refusal is not None
    assert str(MAX_STEERING_CHARS) in refusal


def test_accepted_text_is_drained_as_user_sourced():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    assert service.steer_primary("run-1", "check the tests first") is None

    assert drain() == [(STEERING_SOURCE_USER, "check the tests first")]
    assert drain() == [], "a drain consumes the mailbox"


def test_registration_surfaces_a_ready_callback():
    """The Console learns how to steer THIS run without knowing run ids."""
    hooks = []
    service = _service(on_primary_steer_ready=hooks.append)

    service._register_primary_mailbox("run-9")

    assert len(hooks) == 1
    assert hooks[0]("do it differently") is None          # accepted
    service._unregister_primary_mailbox("run-9")
    assert hooks[0]("again?") is not None                  # honest refusal


# --- through the real loop ---------------------------------------------------


def test_steered_text_reaches_the_transcript_as_user_authored():
    """AC#2/#3: the primary drain rides the existing protocol-coherent seam."""
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    seen_histories = []
    remaining = [
        ModelTurn(
            text='```tool_call\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n```'
        ),
        ModelTurn(text="Adjusted."),
    ]

    def call_model(messages, active_schemas):
        seen_histories.append([dict(m) for m in messages])
        if len(seen_histories) == 1:
            # the user steers while the first turn's tool is running
            assert service.steer_primary("run-1", "actually compute 2+2") is None
        return remaining.pop(0)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(ok=True, content="2"),
        spawn=lambda t: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        drain_mailbox=drain,
        sleep=lambda s: None,
    )
    cfg = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(),
    )
    outcome = run_agent_loop(cfg, [{"role": "user", "content": "compute"}], [], deps)

    assert outcome.status == RUN_DONE
    second_view = seen_histories[1]
    steering = [
        m
        for m in second_view
        if m.get("role") == "user" and "actually compute 2+2" in str(m.get("content"))
    ]
    assert steering, "the steered text never reached the model"
    # the drain point appends AFTER the batch's results -- pairing intact
    tool_result_index = max(
        i for i, m in enumerate(second_view)
        if "Tool result" in str(m.get("content", ""))
    )
    assert second_view.index(steering[0]) > tool_result_index
