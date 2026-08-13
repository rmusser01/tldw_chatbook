"""Durable provider-continuation barriers in the pure agent loop."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_ERROR,
    STEP_TOOL_RESULT,
    AgentConfig,
    ContinuationEventContext,
    FinalContinuation,
    ModelTurn,
    ToolBatchReady,
    ToolCall,
    ToolCallExecuting,
    ToolCallFinished,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)


CALCULATOR = ToolSchema(
    id="builtin:calculator",
    name="calculator",
    description="math",
    parameters={"type": "object"},
)
CONFIG = AgentConfig(
    model="test-model",
    system_prompt="system",
    allowed_tools=("calculator",),
)


def _checkpoint(
    *calls: ContinuationCall,
    revision: int = 1,
    state: str = "active",
    assistant_content: str = "",
    reasoning: tuple[str, ...] = ("private",),
) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=revision,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state=state,  # type: ignore[arg-type]
        rounds=(
            ContinuationRound(
                assistant_content=assistant_content,
                reasoning_blocks=reasoning,
                calls=tuple(calls),
            ),
        ),
    )


def _pending_call(
    call_id: str = "call-1",
    *,
    name: str = "calculator",
    args: dict[str, object] | None = None,
) -> ContinuationCall:
    return ContinuationCall(
        call_id=call_id,
        name=name,
        arguments=json.dumps(args or {"expression": "2+2"}, separators=(",", ":")),
        state="pending",
    )


def _native_turn(
    calls: tuple[ToolCall, ...],
    checkpoint: ProviderContinuationCheckpoint,
) -> ModelTurn:
    raw_calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.args, separators=(",", ":")),
            },
        }
        for call in calls
    ]
    return ModelTurn(
        tool_calls=calls,
        assistant_message={"role": "assistant", "content": "", "tool_calls": raw_calls},
        provider_continuation=checkpoint,
    )


def _deps(
    turns: list[ModelTurn],
    *,
    order: list[str],
    persist,
    invoke,
    review=None,
    context: ContinuationEventContext | None = None,
    cancel=lambda: False,
) -> LoopDeps:
    script = iter(turns)

    def call_model(messages, active_schemas):
        order.append("model")
        return next(script)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True, content="spawned"),
        find_tools=lambda query: [],
        load_schemas=lambda ids: [],
        should_cancel=cancel,
        clock=lambda: 0.0,
        review_tool_calls=review,
        on_step=lambda step: order.append(f"step:{step.kind}"),
        continuation_context=context
        or ContinuationEventContext(
            owner_message_id="assistant-owner",
            run_id="run-1",
            agent_kind="primary",
            durability="persistent",
        ),
        persist_provider_continuation=persist,
    )


def test_cycle_4a_barriers_order_batch_review_execute_finish_history_and_model() -> (
    None
):
    order: list[str] = []
    call = ToolCall(name="calculator", args={"expression": "2+2"}, call_id="call-1")
    checkpoint = _checkpoint(_pending_call())
    final_checkpoint = _checkpoint(
        ContinuationCall(
            call_id="call-1",
            name="calculator",
            arguments='{"expression":"2+2"}',
            state="completed",
            result=ContinuationResult("4"),
        ),
        revision=4,
        state="complete",
    )
    events = []

    def persist(event) -> None:
        events.append(event)
        order.append(type(event).__name__)

    def review(batch):
        order.append("review")
        assert batch == [call]
        return {}

    def invoke(actual):
        order.append("invoke")
        assert actual == call
        return ToolResult(ok=True, content="4")

    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "2+2?"}],
        [CALCULATOR],
        _deps(
            [
                _native_turn((call,), checkpoint),
                ModelTurn(text="4", provider_continuation=final_checkpoint),
            ],
            order=order,
            persist=persist,
            invoke=invoke,
            review=review,
        ),
    )

    assert outcome.status == RUN_DONE
    assert order.index("ToolBatchReady") < order.index("review")
    assert order.index("review") < order.index("ToolCallExecuting")
    assert order.index("ToolCallExecuting") < order.index("invoke")
    assert order.index("invoke") < order.index("ToolCallFinished")
    assert order.index("ToolCallFinished") < order.index(f"step:{STEP_TOOL_RESULT}")
    assert order.index(f"step:{STEP_TOOL_RESULT}") < order.index("model", 1)
    assert [type(event) for event in events] == [
        ToolBatchReady,
        ToolCallExecuting,
        ToolCallFinished,
        FinalContinuation,
    ]
    assert events[0].expected_checkpoint_revision is None
    assert events[1].expected_checkpoint_revision == 1
    assert events[2].expected_checkpoint_revision == 2
    assert events[2].target_state == "completed"
    assert events[2].result == ContinuationResult("4")


def test_cycle_4a_persistence_failure_stops_before_side_effect_or_next_model() -> None:
    order: list[str] = []
    call = ToolCall(name="calculator", args={"expression": "2+2"}, call_id="call-1")
    checkpoint = _checkpoint(_pending_call())

    def persist(event) -> None:
        order.append(type(event).__name__)
        if isinstance(event, ToolCallExecuting):
            raise RuntimeError("PRIVATE-CHECKPOINT-CANARY")

    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "2+2?"}],
        [CALCULATOR],
        _deps(
            [_native_turn((call,), checkpoint), ModelTurn(text="must not run")],
            order=order,
            persist=persist,
            invoke=lambda actual: (
                order.append("invoke") or ToolResult(ok=True, content="4")
            ),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert order == [
        "model",
        "step:model",
        "ToolBatchReady",
        "ToolCallExecuting",
        "step:error",
    ]
    assert outcome.steps[-1].kind == "error"
    assert "PRIVATE-CHECKPOINT-CANARY" not in outcome.steps[-1].summary


def test_model_turn_continuation_defaults_to_none() -> None:
    assert ModelTurn(text="unchanged").provider_continuation is None


def test_closed_event_types_are_frozen_and_context_has_no_metadata_bag() -> None:
    context = ContinuationEventContext(
        owner_message_id="owner",
        run_id="run",
        agent_kind="fleet",
        durability="ephemeral",
    )
    assert set(vars(context)) == {
        "owner_message_id",
        "run_id",
        "agent_kind",
        "durability",
    }
    assert FinalContinuation.__dataclass_params__.frozen is True


def test_cycle_4b_two_call_batch_persists_once_and_transitions_independently() -> None:
    order: list[str] = []
    calls = (
        ToolCall(name="calculator", args={"expression": "1+1"}, call_id="a"),
        ToolCall(name="calculator", args={"expression": "2+2"}, call_id="b"),
    )
    checkpoint = _checkpoint(
        _pending_call("a", args={"expression": "1+1"}),
        _pending_call("b", args={"expression": "2+2"}),
    )
    final = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=6,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("private",),
                calls=(
                    ContinuationCall(
                        "a",
                        "calculator",
                        '{"expression":"1+1"}',
                        "completed",
                        ContinuationResult("result:a"),
                    ),
                    ContinuationCall(
                        "b",
                        "calculator",
                        '{"expression":"2+2"}',
                        "completed",
                        ContinuationResult("result:b"),
                    ),
                ),
            ),
        ),
    )
    events = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [
                _native_turn(calls, checkpoint),
                ModelTurn(text="done", provider_continuation=final),
            ],
            order=order,
            persist=events.append,
            invoke=lambda call: ToolResult(ok=True, content=f"result:{call.call_id}"),
        ),
    )

    assert outcome.status == RUN_DONE
    assert [type(event) for event in events].count(ToolBatchReady) == 1
    assert [
        (type(event), event.expected_checkpoint_revision) for event in events[:-1]
    ] == [
        (ToolBatchReady, None),
        (ToolCallExecuting, 1),
        (ToolCallFinished, 2),
        (ToolCallExecuting, 3),
        (ToolCallFinished, 4),
    ]


def test_cycle_4b_cancel_after_first_call_leaves_second_pending() -> None:
    order: list[str] = []
    calls = (
        ToolCall(name="calculator", args={"expression": "1+1"}, call_id="a"),
        ToolCall(name="calculator", args={"expression": "2+2"}, call_id="b"),
    )
    events = []
    checks = iter([False, False, True])
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [
                _native_turn(
                    calls,
                    _checkpoint(
                        _pending_call("a", args={"expression": "1+1"}),
                        _pending_call("b", args={"expression": "2+2"}),
                    ),
                )
            ],
            order=order,
            persist=events.append,
            invoke=lambda call: ToolResult(ok=True, content=f"result:{call.call_id}"),
            cancel=lambda: next(checks, True),
        ),
    )

    assert outcome.status == "cancelled"
    assert [getattr(event, "call_id", None) for event in events] == [None, "a", "a"]


def test_cycle_4b_duplicate_call_ids_fail_before_event_or_dispatch() -> None:
    order: list[str] = []
    duplicate = (
        ToolCall(name="calculator", args={"expression": "1"}, call_id="same"),
        ToolCall(name="calculator", args={"expression": "2"}, call_id="same"),
    )
    events = []
    invoked = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [
                _native_turn(
                    duplicate,
                    _checkpoint(
                        _pending_call("a", args={"expression": "1"}),
                        _pending_call("b", args={"expression": "2"}),
                    ),
                )
            ],
            order=order,
            persist=events.append,
            invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert events == []
    assert invoked == []


@pytest.mark.parametrize("agent_kind", ["primary", "subagent", "fleet"])
def test_cycle_4b_context_owner_is_exact_for_every_agent_kind(agent_kind: str) -> None:
    order: list[str] = []
    call = ToolCall(name="calculator", args={"expression": "2+2"}, call_id="a")
    events = []
    context = ContinuationEventContext(
        owner_message_id=f"owner-{agent_kind}",
        run_id=f"run-{agent_kind}",
        agent_kind=agent_kind,  # type: ignore[arg-type]
        durability="persistent",
    )
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [_native_turn((call,), _checkpoint(_pending_call("a")))],
            order=order,
            persist=events.append,
            invoke=lambda actual: ToolResult(ok=True, content="4"),
            context=context,
            cancel=lambda: len(events) >= 3,
        ),
    )

    assert outcome.status == "cancelled"
    assert events
    assert all(event.context == context for event in events)


def test_cycle_4b_persistent_missing_owner_stops_but_ephemeral_is_non_resumable() -> (
    None
):
    call = ToolCall(name="calculator", args={"expression": "2+2"}, call_id="a")
    checkpoint = _checkpoint(_pending_call("a"))

    for durability, expected_status in (
        ("persistent", RUN_ERROR),
        ("ephemeral", "cancelled"),
    ):
        order: list[str] = []
        invoked = []
        events = []
        context = ContinuationEventContext(
            owner_message_id=None,
            run_id=f"run-{durability}",
            agent_kind="subagent",
            durability=durability,  # type: ignore[arg-type]
        )
        outcome = run_agent_loop(
            CONFIG,
            [{"role": "user", "content": "go"}],
            [CALCULATOR],
            _deps(
                [_native_turn((call,), checkpoint)],
                order=order,
                persist=events.append,
                invoke=lambda actual: (
                    invoked.append(actual) or ToolResult(ok=True, content="4")
                ),
                context=context,
                cancel=lambda: len(events) >= 3,
            ),
        )

        assert outcome.status == expected_status
        if durability == "persistent":
            assert events == [] and invoked == []
        else:
            assert invoked == [call]
            assert any("non-resumable" in step.summary for step in outcome.steps)


def test_cycle_4c_restore_without_resume_is_paused_and_runs_nothing() -> None:
    order: list[str] = []
    invoked = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="must not run")],
            order=order,
            persist=lambda event: None,
            invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
        ),
        restore_provider_continuation=_checkpoint(_pending_call()),
    )

    assert outcome.status == "stuck"
    assert order == ["step:error"]
    assert invoked == []
    assert "explicit resume" in outcome.steps[-1].summary


def test_cycle_4c_restore_replays_terminal_results_and_never_invokes_them() -> None:
    order: list[str] = []
    invoked = []
    seen_history = []
    checkpoint = _checkpoint(
        ContinuationCall(
            "done",
            "calculator",
            '{"expression":"2+2"}',
            "completed",
            ContinuationResult("exact completed"),
        ),
        ContinuationCall(
            "failed",
            "calculator",
            '{"expression":"bad"}',
            "failed",
            ContinuationResult("exact failure"),
        ),
        revision=3,
    )

    def call_model(messages, active):
        seen_history.extend(messages)
        return ModelTurn(text="final")

    deps = _deps(
        [],
        order=order,
        persist=lambda event: None,
        invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
    )
    deps.call_model = call_model
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        deps,
        restore_provider_continuation=checkpoint,
        resume_provider_continuation=True,
    )

    assert outcome.status == RUN_ERROR  # missing required final checkpoint
    assert invoked == []
    assert [
        (message["tool_call_id"], message["content"])
        for message in seen_history
        if message.get("role") == "tool"
    ] == [("done", "exact completed"), ("failed", "exact failure")]


def test_cycle_4c_executing_restore_is_ambiguous_and_blocked() -> None:
    order: list[str] = []
    invoked = []
    executing = _checkpoint(
        ContinuationCall(
            "call-1",
            "calculator",
            '{"expression":"2+2"}',
            "executing",
        ),
        revision=2,
    )
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="must not run")],
            order=order,
            persist=lambda event: None,
            invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
        ),
        restore_provider_continuation=executing,
        resume_provider_continuation=True,
    )

    assert outcome.status == "stuck"
    assert order == ["step:error"]
    assert invoked == []
    assert "ambiguous" in outcome.steps[-1].summary


def test_cycle_4c_pending_resume_requires_fresh_review_then_barrier() -> None:
    order: list[str] = []
    events = []
    invoked = []
    checkpoint = _checkpoint(_pending_call())

    def review(batch):
        order.append("review")
        assert [call.call_id for call in batch] == ["call-1"]
        return {"call-1": "fresh refusal"}

    def persist(event):
        events.append(event)
        order.append(type(event).__name__)

    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="must not run")],
            order=order,
            persist=persist,
            invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
            review=review,
            cancel=lambda: len(events) >= 2,
        ),
        restore_provider_continuation=checkpoint,
        resume_provider_continuation=True,
    )

    assert outcome.status == "cancelled"
    assert invoked == []
    assert order.index("review") < order.index("ToolCallExecuting")
    assert [type(event) for event in events] == [
        ToolCallExecuting,
        ToolCallFinished,
    ]
    assert events[-1].target_state == "failed"
    assert events[-1].result == ContinuationResult("fresh refusal")


def _kimi_checkpoint(
    *,
    revision: int,
    state: str,
    rounds: tuple[ContinuationRound, ...],
) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=revision,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state=state,  # type: ignore[arg-type]
        rounds=rounds,
    )


def test_cycle_4d_tool_free_k3_first_create_uses_none_before_done() -> None:
    order: list[str] = []
    checkpoint = _kimi_checkpoint(
        revision=1,
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="visible answer",
                reasoning_blocks=("private",),
                calls=(),
            ),
        ),
    )
    events = []

    def persist(event):
        events.append(event)
        order.append(type(event).__name__)

    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="visible answer", provider_continuation=checkpoint)],
            order=order,
            persist=persist,
            invoke=lambda call: pytest.fail("no tool expected"),
        ),
    )

    assert outcome.status == RUN_DONE
    assert events == [
        FinalContinuation(
            context=events[0].context,
            checkpoint=checkpoint,
            expected_checkpoint_revision=None,
            assistant_content="visible answer",
        )
    ]
    assert order[-1] == "FinalContinuation"


@pytest.mark.parametrize(
    "checkpoint",
    [
        _checkpoint(
            ContinuationCall(
                "call-1",
                "calculator",
                '{"expression":"2+2"}',
                "completed",
                ContinuationResult("4"),
            ),
            revision=1,
            state="complete",
        ),
        _kimi_checkpoint(
            revision=2,
            state="complete",
            rounds=(
                ContinuationRound(
                    assistant_content="answer",
                    reasoning_blocks=("private",),
                    calls=(),
                ),
            ),
        ),
    ],
)
def test_cycle_4d_invalid_first_none_creation_is_rejected(checkpoint) -> None:
    order: list[str] = []
    events = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="answer", provider_continuation=checkpoint)],
            order=order,
            persist=events.append,
            invoke=lambda call: pytest.fail("no tool expected"),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert events == []


def test_cycle_4d_post_tool_k3_final_uses_exact_current_revision() -> None:
    order: list[str] = []
    call = ToolCall(name="calculator", args={"expression": "2+2"}, call_id="a")
    tool_round_pending = ContinuationRound(
        assistant_content="",
        reasoning_blocks=("tool reasoning",),
        calls=(ContinuationCall("a", "calculator", '{"expression":"2+2"}', "pending"),),
    )
    tool_round_complete = ContinuationRound(
        assistant_content="",
        reasoning_blocks=("tool reasoning",),
        calls=(
            ContinuationCall(
                "a",
                "calculator",
                '{"expression":"2+2"}',
                "completed",
                ContinuationResult("4"),
            ),
        ),
    )
    initial = _kimi_checkpoint(revision=1, state="active", rounds=(tool_round_pending,))
    final = _kimi_checkpoint(
        revision=4,
        state="complete",
        rounds=(
            tool_round_complete,
            ContinuationRound(
                assistant_content="visible answer",
                reasoning_blocks=("final reasoning",),
                calls=(),
            ),
        ),
    )
    events = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [
                _native_turn((call,), initial),
                ModelTurn(text="visible answer", provider_continuation=final),
            ],
            order=order,
            persist=events.append,
            invoke=lambda actual: ToolResult(ok=True, content="4"),
        ),
    )

    assert outcome.status == RUN_DONE
    final_event = events[-1]
    assert isinstance(final_event, FinalContinuation)
    assert final_event.expected_checkpoint_revision == 3


def test_cycle_4d_raised_final_persistence_stops_safely() -> None:
    checkpoint = _kimi_checkpoint(
        revision=1,
        state="complete",
        rounds=(ContinuationRound("answer", ("private",), ()),),
    )

    def persist(event) -> None:
        raise RuntimeError("PRIVATE-CANARY")

    order: list[str] = []
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        _deps(
            [ModelTurn(text="answer", provider_continuation=checkpoint)],
            order=order,
            persist=persist,
            invoke=lambda call: pytest.fail("no tool expected"),
        ),
    )
    assert outcome.status == RUN_ERROR
    assert "PRIVATE-CANARY" not in outcome.steps[-1].summary
