"""Instruction deferrals must settle native checkpoints before another response."""

from __future__ import annotations

import json

import pytest

from Tests.Agents.test_agent_runtime_preparation import _receipt, _row
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_ERROR,
    AgentConfig,
    ContinuationEventContext,
    ModelTurn,
    ToolBatchReady,
    ToolCall,
    ToolCallExecuting,
    ToolCallFinished,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import (
    LoopDeps,
    ToolBatchPreparation,
    run_agent_loop,
)
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatStream
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedPresetFinishPolicy,
    HostedProviderResolution,
    HostedProviderStream,
)
from tldw_chatbook.provider_registry import CUSTOM_HOSTED

READ = ToolSchema(
    id="builtin:code_review_read",
    name="code_review_read",
    description="Read captured review evidence.",
    parameters={"type": "object"},
)
RESOLUTION = HostedProviderResolution(
    provider="custom-hosted",
    model="offline-qwen",
    api_key="",
    base_url="http://offline.invalid/v1",
    timeout=1,
    retries=0,
    retry_delay=0,
    streaming=True,
)


def _response(ids, checkpoint):
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": READ.name, "arguments": '{"view":"source"}'},
        }
        for call_id in ids
    ]
    delta = (
        {"tool_calls": [dict(call, index=i) for i, call in enumerate(calls)]}
        if calls
        else {"content": "done"}
    )
    frames = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    ]
    records = iter(
        [SSERecord(None, json.dumps(frame)) for frame in frames]
        + [SSERecord(None, "[DONE]")]
    )
    stream = HostedProviderStream(
        HostedChatStream(
            records,
            finish_policy=HostedPresetFinishPolicy(CUSTOM_HOSTED),
            tolerant_top_level_extras=True,
        ),
        record=CUSTOM_HOSTED,
        resolution=RESOLUTION,
        provider_continuations=(checkpoint,) if checkpoint else (),
    )
    list(stream)
    turn = stream.terminal_turn
    return ModelTurn(
        text=turn.text,
        assistant_message=turn.assistant_message,
        tool_calls=tuple(
            ToolCall(
                READ.name,
                {"view": "source"},
                call["id"],
                raw_arguments=call["function"]["arguments"],
            )
            for call in calls
        ),
        provider_continuation=stream.provider_continuation,
    )


def _journey(*, batch_size=4, fail_at=None, fail_ack=False, reissue=True):
    deferred = tuple(f"deferred-{i}" for i in range(batch_size))
    replies = iter([deferred, ("fresh-read",), ()] if reissue else [deferred, ()])
    requests, events, acknowledged, executed, reviewed, errors = [], [], [], [], [], []
    refused = []

    def call_model(messages, _active, checkpoint):
        requests.append(([dict(row) for row in messages], checkpoint))
        try:
            return _response(next(replies), checkpoint)
        except Exception as exc:
            errors.append(type(exc).__name__)
            raise

    def persist(event):
        if (
            isinstance(event, ToolCallFinished)
            and event.call_id == fail_at
            and not fail_ack
        ):
            refused.append((event.call_id, "persist"))
            raise OSError("offline persistence refusal")
        events.append(event)

    def acknowledge(original, persisted, model_result):
        if (
            isinstance(original, ToolCallFinished)
            and original.call_id == fail_at
            and fail_ack
        ):
            refused.append((original.call_id, "ack"))
            raise OSError("offline acknowledgment refusal")
        acknowledged.append((original, persisted, model_result))

    outcome = run_agent_loop(
        AgentConfig(model="offline-qwen", system_prompt="", allowed_tools=(READ.name,)),
        [{"role": "user", "content": "Inspect captured source."}],
        [READ],
        LoopDeps(
            call_model=lambda *_: pytest.fail("must use the checkpoint-aware request"),
            call_model_with_continuation=call_model,
            prepare_tool_calls=lambda calls: (
                ToolBatchPreparation(
                    "retry_with_context", (_row("nested"),), _receipt("nested")
                )
                if calls[0].call_id in deferred
                else ToolBatchPreparation("proceed")
            ),
            review_tool_calls=lambda calls: (
                reviewed.extend(c.call_id for c in calls) or {}
            ),
            invoke_tool=lambda call: (
                executed.append(call.call_id) or ToolResult(ok=True, content="source")
            ),
            spawn=lambda *_: pytest.fail("no child launch in this regression"),
            find_tools=lambda _: [],
            load_schemas=lambda _: [],
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            continuation_context=ContinuationEventContext(
                owner_message_id="owner",
                run_id="run",
                agent_kind="primary",
                durability="persistent",
            ),
            persist_provider_continuation=persist,
            acknowledge_provider_continuation=acknowledge,
        ),
    )
    return outcome, requests, events, acknowledged, executed, reviewed, errors, refused


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("reissue", [False, True])
def test_deferred_native_batch_is_terminal_before_next_hosted_response(
    batch_size, reissue
):
    outcome, requests, events, acknowledged, executed, reviewed, errors, refused = (
        _journey(batch_size=batch_size, reissue=reissue)
    )
    assert outcome.status == RUN_DONE, (errors, outcome.steps)
    assert refused == []
    assert executed == reviewed == (["fresh-read"] if reissue else [])
    checkpoint = requests[1][1]
    assert [call.state for call in checkpoint.rounds[-1].calls] == [
        "failed"
    ] * batch_size
    stubs = requests[1][0][-batch_size - 1 : -1]
    assert [row["content"] for row in stubs] == [
        call.result.value for call in checkpoint.rounds[-1].calls
    ]
    deferred_events = [
        event
        for event in events
        if isinstance(event, ToolCallFinished) and event.call_id.startswith("deferred-")
    ]
    assert len(deferred_events) == batch_size
    assert all(event.target_state == "failed" for event in deferred_events)
    assert not any(
        isinstance(event, ToolCallExecuting) and event.call_id.startswith("deferred-")
        for event in events
    )
    assert [
        item[2]
        for item in acknowledged
        if isinstance(item[0], ToolCallFinished)
        and item[0].call_id.startswith("deferred-")
    ] == [row["content"] for row in stubs]
    assert isinstance(events[0], ToolBatchReady)


@pytest.mark.parametrize("fail_at", ["deferred-0", "deferred-1"])
@pytest.mark.parametrize("fail_ack", [False, True])
def test_deferral_persistence_failure_prevents_next_request_and_any_dispatch(
    fail_at, fail_ack
):
    outcome, requests, events, _ack, executed, reviewed, _errors, refused = _journey(
        fail_at=fail_at, fail_ack=fail_ack
    )
    assert outcome.status == RUN_ERROR
    assert len(requests) == 1
    assert executed == reviewed == []
    assert isinstance(events[0], ToolBatchReady)
    assert refused == [(fail_at, "ack" if fail_ack else "persist")]
