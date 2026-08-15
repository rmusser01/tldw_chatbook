"""Durable provider-continuation barriers in the pure agent loop."""

from __future__ import annotations

import json
from dataclasses import replace

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
    RunBudget,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
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
        arguments=json.dumps(
            {"expression": "2+2"} if args is None else args,
            separators=(",", ":"),
        ),
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
    on_record=None,
    expand=None,
) -> LoopDeps:
    script = iter(turns)

    def call_model(messages, active_schemas):
        order.append("model")
        return next(script)

    deps = LoopDeps(
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
        on_record=on_record,
    )
    if expand is not None:
        deps.expand_provider_continuation = expand
    return deps


@pytest.mark.parametrize("turn_kind", ["native", "fence"])
def test_barriers_precede_all_observable_runtime_hooks(turn_kind: str) -> None:
    order: list[str] = []
    events = []
    raw = '{"expression":"2+2"}'
    call = ToolCall(
        name="calculator",
        args={"expression": "2+2"},
        call_id="call-1" if turn_kind == "native" else "",
        raw_arguments=raw,
    )
    fence_text = (
        '```tool_call\n{"name":"calculator","arguments":'
        '{"expression":"2+2"},"call_id":"fence-1"}\n```'
    )
    checkpoint = _checkpoint(
        ContinuationCall(call.call_id or "fence-1", "calculator", raw, "pending"),
        assistant_content=fence_text if turn_kind == "fence" else "",
    )
    if turn_kind == "native":
        turn = _native_turn((call,), checkpoint)
    else:
        call = replace(call, call_id="fence-1")
        turn = ModelTurn(
            text=fence_text,
            provider_continuation=checkpoint,
        )

    def persist(event) -> None:
        events.append(event)
        order.append(type(event).__name__)

    deps = _deps(
        [turn],
        order=order,
        persist=persist,
        invoke=lambda actual: (
            order.append("invoke") or ToolResult(ok=True, content="4")
        ),
        review=lambda batch: order.append("review") or {},
        cancel=lambda: len(events) >= 3,
        on_record=lambda kind, payload: order.append(f"record:{kind}"),
    )
    outcome = run_agent_loop(CONFIG, [], [CALCULATOR], deps)

    assert outcome.status == "cancelled", outcome.steps
    assert order.index("ToolBatchReady") < order.index("step:model")
    assert order.index("ToolBatchReady") < order.index("record:model")
    assert order.index("ToolBatchReady") < order.index("review")
    assert order.index("ToolCallExecuting") < order.index("record:tool_call")
    assert order.index("ToolCallExecuting") < order.index("step:tool_call")
    assert order.index("ToolCallExecuting") < order.index("invoke")


def test_executing_failure_emits_no_later_step_record_or_dispatch() -> None:
    order: list[str] = []
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )

    def persist(event) -> None:
        order.append(type(event).__name__)
        if isinstance(event, ToolCallExecuting):
            raise RuntimeError("private")

    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        _deps(
            [_native_turn((call,), _checkpoint(_pending_call()))],
            order=order,
            persist=persist,
            invoke=lambda actual: order.append("invoke") or ToolResult(ok=True),
            review=lambda batch: order.append("review") or {},
            on_record=lambda kind, payload: order.append(f"record:{kind}"),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert order[-2:] == ["ToolCallExecuting", "step:error"]
    assert "record:tool_call" not in order
    assert "invoke" not in order


def test_finished_failure_emits_no_result_record_step_history_or_next_model() -> None:
    order: list[str] = []
    calls = 0
    private_result = "PRIVATE-RESULT-CANARY"
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )

    def persist(event) -> None:
        order.append(type(event).__name__)
        if isinstance(event, ToolCallFinished):
            raise RuntimeError("private persistence failure")

    def call_model(messages, active):
        nonlocal calls
        calls += 1
        order.append("model")
        return _native_turn((call,), _checkpoint(_pending_call()))

    deps = _deps(
        [],
        order=order,
        persist=persist,
        invoke=lambda actual: (
            order.append("invoke") or ToolResult(ok=True, content=private_result)
        ),
        on_record=lambda kind, payload: order.append(
            f"record:{kind}:{payload.get('content', '')}"
        ),
    )
    deps.call_model = call_model
    outcome = run_agent_loop(CONFIG, [], [CALCULATOR], deps)

    assert outcome.status == RUN_ERROR
    assert calls == 1
    assert any(item == "invoke" for item in order)
    assert not any(item.startswith("record:tool_result") for item in order)
    assert not any(step.kind == STEP_TOOL_RESULT for step in outcome.steps)
    assert private_result not in repr(outcome)


@pytest.mark.parametrize(
    ("tool_name", "args", "dependency"),
    [
        ("spawn_subagent", {"task": "work"}, "spawn"),
        ("wait_agents", {}, "wait_agents"),
        ("check_agents", {}, "check_agents"),
        ("find_tools", {"query": "x"}, "find_tools"),
        ("load_tools", {"ids": []}, "load_schemas"),
        ("skill_file", {}, "read_skill_file"),
        ("install_skill", {}, "install_skill"),
        ("run_skill_script", {}, "run_skill_script"),
        ("search_run_log", {}, "search_run_log"),
        ("run_log_stats", {}, "run_log_stats"),
        ("run_log_slice", {}, "run_log_slice"),
        ("generic_tool", {}, "invoke_tool"),
    ],
    ids=[
        "spawn-subagent",
        "wait-agents",
        "check-agents",
        "find-tools",
        "load-schemas",
        "read-skill-file",
        "install-skill",
        "run-skill-script",
        "search-run-log",
        "run-log-stats",
        "run-log-slice",
        "generic-tool",
    ],
)
def test_common_executing_barrier_dominates_every_dispatch_branch(
    tool_name: str, args: dict, dependency: str
) -> None:
    order: list[str] = []
    events = []
    raw = json.dumps(args, separators=(",", ":"))
    call = ToolCall(tool_name, args, "call-1", raw)
    config = replace(CONFIG, allowed_tools=("calculator", "spawn_subagent"))
    deps = _deps(
        [_native_turn((call,), _checkpoint(_pending_call(name=tool_name, args=args)))],
        order=order,
        persist=lambda event: (
            events.append(event) or order.append(type(event).__name__)
        ),
        invoke=lambda actual: (
            order.append("dispatch") or ToolResult(ok=True, content="ok")
        ),
        cancel=lambda: len(events) >= 3,
    )

    def dispatch_result(*values, **keywords):
        order.append("dispatch")
        return ToolResult(ok=True, content="ok")

    if dependency == "find_tools":
        deps.find_tools = lambda query: order.append("dispatch") or []
    elif dependency == "load_schemas":
        deps.load_schemas = lambda ids: order.append("dispatch") or []
    else:
        setattr(deps, dependency, dispatch_result)

    outcome = run_agent_loop(config, [], [CALCULATOR], deps)

    assert outcome.status == "cancelled", outcome.steps
    assert order.index("ToolCallExecuting") < order.index("dispatch")


@pytest.mark.parametrize(
    ("budget_cap", "raw_size", "expected_cap"),
    [(16000, 16001, 16000), (11, 100, 11), (0, 16001, 16000)],
)
def test_finished_result_is_one_total_bounded_provider_string(
    budget_cap: int, raw_size: int, expected_cap: int
) -> None:
    order: list[str] = []
    events = []
    histories: list[list[dict]] = []
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
    initial = _checkpoint(_pending_call())

    def persist(event) -> None:
        events.append(event)

    def call_model(messages, active):
        histories.append(list(messages))
        if len(histories) == 1:
            return _native_turn((call,), initial)
        finished = next(
            event for event in events if isinstance(event, ToolCallFinished)
        )
        return ModelTurn(
            text="done",
            provider_continuation=_checkpoint(
                ContinuationCall(
                    "call-1",
                    "calculator",
                    '{"expression":"2+2"}',
                    "completed",
                    finished.result,
                ),
                revision=4,
                state="complete",
            ),
        )

    config = replace(CONFIG, budget=RunBudget(max_tool_result_chars=budget_cap))
    deps = _deps(
        [],
        order=order,
        persist=persist,
        invoke=lambda actual: ToolResult(ok=True, content="x" * raw_size),
    )
    deps.call_model = call_model
    outcome = run_agent_loop(config, [], [CALCULATOR], deps)

    assert outcome.status == RUN_DONE
    finished = next(event for event in events if isinstance(event, ToolCallFinished))
    result_step = next(step for step in outcome.steps if step.kind == STEP_TOOL_RESULT)
    history_result = histories[1][-1]["content"]
    assert len(finished.result.value) == expected_cap
    assert finished.result.value == history_result
    assert result_step.result == finished.result.value[:2000]
    assert len(result_step.result) <= 2000


def test_cycle_4a_barriers_order_batch_review_execute_finish_history_and_model() -> (
    None
):
    order: list[str] = []
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
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
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
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
        "ToolBatchReady",
        "step:model",
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
        ToolCall("calculator", {"expression": "1+1"}, "a", '{"expression":"1+1"}'),
        ToolCall("calculator", {"expression": "2+2"}, "b", '{"expression":"2+2"}'),
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
        ToolCall("calculator", {"expression": "1+1"}, "a", '{"expression":"1+1"}'),
        ToolCall("calculator", {"expression": "2+2"}, "b", '{"expression":"2+2"}'),
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
        ToolCall("calculator", {"expression": "1"}, "same", '{"expression":"1"}'),
        ToolCall("calculator", {"expression": "2"}, "same", '{"expression":"2"}'),
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


@pytest.mark.parametrize(
    ("runtime_raw", "runtime_args", "canonical_raw"),
    [
        ('{"value":1}', {"value": 1}, '{"value":1.0}'),
        ('{"value":1}', {"value": 1}, '{"value":true}'),
        ('{"a":1,"b":2}', {"a": 1, "b": 2}, '{ "b": 2, "a": 1 }'),
        ("", {"value": 1}, '{"value":1}'),
    ],
)
def test_continuation_call_arguments_require_exact_raw_bytes(
    runtime_raw: str, runtime_args: dict, canonical_raw: str
) -> None:
    events = []
    invoked = []
    call = ToolCall("calculator", runtime_args, "call-1", runtime_raw)
    checkpoint = _checkpoint(
        ContinuationCall("call-1", "calculator", canonical_raw, "pending")
    )
    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        _deps(
            [_native_turn((call,), checkpoint)],
            order=[],
            persist=events.append,
            invoke=lambda actual: invoked.append(actual) or ToolResult(ok=True),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert events == invoked == []


@pytest.mark.parametrize("mutation", ["id", "name", "order"])
def test_continuation_call_identity_and_order_must_match(mutation: str) -> None:
    raw_a = '{"expression":"1+1"}'
    raw_b = '{"expression":"2+2"}'
    calls = (
        ToolCall("calculator", {"expression": "1+1"}, "a", raw_a),
        ToolCall("calculator", {"expression": "2+2"}, "b", raw_b),
    )
    canonical = [
        ContinuationCall("a", "calculator", raw_a, "pending"),
        ContinuationCall("b", "calculator", raw_b, "pending"),
    ]
    if mutation == "id":
        calls = (replace(calls[0], call_id="wrong"), calls[1])
    elif mutation == "name":
        calls = (replace(calls[0], name="different"), calls[1])
    else:
        calls = tuple(reversed(calls))
    events = []
    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        _deps(
            [_native_turn(calls, _checkpoint(*canonical))],
            order=[],
            persist=events.append,
            invoke=lambda actual: pytest.fail("mismatch must not invoke"),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert events == []


@pytest.mark.parametrize("agent_kind", ["primary", "subagent", "fleet"])
def test_cycle_4b_context_owner_is_exact_for_every_agent_kind(agent_kind: str) -> None:
    order: list[str] = []
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
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
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
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
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
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
        expand=lambda actual: [
            {"role": "tool", "tool_call_id": call.call_id, "content": call.result.value}
            for round_ in actual.rounds
            for call in round_.calls
        ],
    )
    deps.call_model = call_model
    outcome = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [CALCULATOR],
        deps,
        restore_provider_continuation=checkpoint,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
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
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
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
            cancel=lambda: len(events) >= 1,
            expand=lambda actual: [],
        ),
        restore_provider_continuation=checkpoint,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
        resume_provider_continuation=True,
    )

    assert outcome.status == "cancelled"
    assert invoked == []
    assert order.index("review") < order.index("ToolCallFinished")
    assert [type(event) for event in events] == [ToolCallFinished]
    assert events[-1].expected_checkpoint_revision == 1
    assert events[-1].target_state == "failed"
    assert events[-1].result == ContinuationResult("fresh refusal")


def test_refusal_finished_failure_leaves_pending_without_executing_or_observability() -> (
    None
):
    order: list[str] = []
    history = []
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )

    def persist(event) -> None:
        order.append(type(event).__name__)
        if isinstance(event, ToolCallFinished):
            raise RuntimeError("PRIVATE-REFUSAL-CANARY")

    outcome = run_agent_loop(
        CONFIG,
        history,
        [CALCULATOR],
        _deps(
            [_native_turn((call,), _checkpoint(_pending_call()))],
            order=order,
            persist=persist,
            invoke=lambda actual: order.append("invoke") or ToolResult(ok=True),
            review=lambda batch: {"call-1": "denied"},
            on_record=lambda kind, payload: order.append(f"record:{kind}"),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert "ToolCallExecuting" not in order
    assert "invoke" not in order
    assert "record:tool_call" not in order
    assert "record:tool_result" not in order
    assert "step:tool_call" not in order
    assert "step:tool_result" not in order
    assert not any(row.get("role") == "tool" for row in history)


@pytest.mark.parametrize("restored", [False, True])
def test_continuation_review_exception_fails_closed_without_logging_or_dispatch(
    capfd, restored: bool
) -> None:
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
    events = []
    invoked = []

    def review(batch):
        raise RuntimeError("PRIVATE-REVIEW-CANARY")

    checkpoint = _checkpoint(_pending_call())
    deps = _deps(
        [] if restored else [_native_turn((call,), checkpoint)],
        order=[],
        persist=events.append,
        invoke=lambda actual: invoked.append(actual) or ToolResult(ok=True),
        review=review,
        expand=(lambda actual: []) if restored else None,
    )
    kwargs = (
        {
            "restore_provider_continuation": checkpoint,
            "restore_provider_target": ContinuationRestoreTarget(
                "deepseek",
                "deepseek-v4-flash",
                "responses",
                "https://api.deepseek.com/v1",
            ),
            "resume_provider_continuation": True,
        }
        if restored
        else {}
    )
    outcome = run_agent_loop(CONFIG, [], [CALCULATOR], deps, **kwargs)

    assert outcome.status == RUN_ERROR
    assert [type(event) for event in events] == ([] if restored else [ToolBatchReady])
    assert invoked == []
    captured = capfd.readouterr()
    assert "PRIVATE-REVIEW-CANARY" not in captured.out + captured.err
    assert "PRIVATE-REVIEW-CANARY" not in repr(outcome)


@pytest.mark.parametrize("contradiction", ["turn", "assistant"])
def test_tool_batch_requires_exact_assistant_content_association(
    contradiction: str,
) -> None:
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
    checkpoint = _checkpoint(_pending_call(), assistant_content="canonical")
    turn = _native_turn((call,), checkpoint)
    if contradiction == "turn":
        turn = replace(turn, text="different")
    else:
        turn = replace(
            turn,
            text="canonical",
            assistant_message={**turn.assistant_message, "content": "different"},
        )
    events = []
    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        _deps(
            [turn],
            order=[],
            persist=events.append,
            invoke=lambda actual: pytest.fail("mismatch must not invoke"),
        ),
    )

    assert outcome.status == RUN_ERROR
    assert events == []


@pytest.mark.parametrize("field", ["provider", "model", "protocol", "api_base_url"])
def test_restore_target_mismatch_stops_before_translation_or_model(field: str) -> None:
    order: list[str] = []
    target = ContinuationRestoreTarget(
        "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
    )
    target = replace(target, **{field: f"wrong-{field}"})
    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        _deps(
            [],
            order=order,
            persist=lambda event: None,
            invoke=lambda call: order.append("invoke") or ToolResult(ok=True),
            expand=lambda checkpoint: order.append("expand") or [],
            on_record=lambda kind, payload: order.append(f"record:{kind}"),
        ),
        restore_provider_continuation=_checkpoint(_pending_call()),
        restore_provider_target=target,
        resume_provider_continuation=True,
    )

    assert outcome.status == RUN_ERROR
    assert order == ["step:error"]


def test_restore_forwards_only_injected_history_rows_byte_exact() -> None:
    order: list[str] = []
    rows = [{"role": "opaque-provider-row", "private": {"exact": [1, True]}}]
    seen = []
    checkpoint = _checkpoint(
        ContinuationCall(
            "done",
            "calculator",
            '{"expression":"2+2"}',
            "completed",
            ContinuationResult("4"),
        ),
        revision=3,
    )
    final = replace(checkpoint, checkpoint_revision=4, state="complete")

    def call_model(messages, active):
        seen.extend(messages)
        return ModelTurn(text="done", provider_continuation=final)

    deps = _deps(
        [],
        order=order,
        persist=lambda event: None,
        invoke=lambda call: pytest.fail("terminal call must not execute"),
        expand=lambda actual: list(rows),
    )
    deps.call_model = call_model
    outcome = run_agent_loop(
        CONFIG,
        [],
        [CALCULATOR],
        deps,
        restore_provider_continuation=checkpoint,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
        resume_provider_continuation=True,
    )

    assert outcome.status == RUN_DONE
    assert seen == rows


def test_restore_without_target_or_translator_fails_before_model() -> None:
    for target in (
        None,
        ContinuationRestoreTarget(
            "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
        ),
    ):
        order: list[str] = []
        outcome = run_agent_loop(
            CONFIG,
            [],
            [CALCULATOR],
            _deps(
                [],
                order=order,
                persist=lambda event: None,
                invoke=lambda call: order.append("invoke") or ToolResult(ok=True),
            ),
            restore_provider_continuation=_checkpoint(_pending_call()),
            restore_provider_target=target,
            resume_provider_continuation=True,
        )
        assert outcome.status == RUN_ERROR
        assert order == ["step:error"]


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
    assert order.index("FinalContinuation") < order.index("step:model")


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
    call = ToolCall("calculator", {"expression": "2+2"}, "a", '{"expression":"2+2"}')
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
