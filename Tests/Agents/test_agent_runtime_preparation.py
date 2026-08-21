"""Whole-batch project-instruction preparation in the pure agent loop."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    AgentConfig,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import (
    LoopDeps,
    ToolBatchPreparation,
    run_agent_loop,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    PROJECT_INSTRUCTION_ROW_KEY,
    InstructionDeliveryReceipt,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY


SCHEMA = ToolSchema(
    id="builtin:one",
    name="one",
    description="one",
    parameters={"type": "object"},
)
CONFIG = AgentConfig(model="m", system_prompt="s", allowed_tools=("one", "two"))


def _receipt(*row_keys: str) -> InstructionDeliveryReceipt:
    return InstructionDeliveryReceipt(
        receipt_id="pir-1",
        chain_id="primary",
        through_revision=1,
        source_digests=(),
        outcome_keys=(),
        row_keys=row_keys,
    )


def _row(key: str, body: str = "nested instructions") -> dict:
    return {
        "role": "user",
        "content": body,
        EPHEMERAL_ORIGIN_KEY: "project_instructions",
        PROJECT_INSTRUCTION_ROW_KEY: key,
    }


def _native_turn(calls: list[ToolCall]) -> ModelTurn:
    return ModelTurn(
        text="",
        tool_calls=tuple(calls),
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": "{}"},
                }
                for call in calls
            ],
        },
    )


def _deps(turns: list[ModelTurn], **overrides) -> LoopDeps:
    script = iter(turns)
    values = {
        "call_model": lambda messages, active: next(script),
        "invoke_tool": lambda call: ToolResult(ok=True, content=f"ran:{call.name}"),
        "spawn": lambda task: ToolResult(ok=True, content="child"),
        "find_tools": lambda query: [],
        "load_schemas": lambda ids: [],
        "should_cancel": lambda: False,
        "clock": lambda: 0.0,
    }
    values.update(overrides)
    return LoopDeps(**values)


def test_tool_batch_preparation_is_frozen_and_accepts_only_exact_contract():
    proceed = ToolBatchPreparation("proceed")
    assert proceed.ephemeral_rows == () and proceed.delivery_receipt is None
    with pytest.raises(FrozenInstanceError):
        proceed.status = "retry_with_context"  # type: ignore[misc]

    row = _row("row-1")
    retry = ToolBatchPreparation("retry_with_context", (row,), _receipt("row-1"))
    assert retry.ephemeral_rows == (row,)

    with pytest.raises(ValueError):
        ToolBatchPreparation("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ToolBatchPreparation("proceed", (row,), None)
    with pytest.raises(ValueError):
        ToolBatchPreparation("retry_with_context")
    with pytest.raises(ValueError):
        ToolBatchPreparation("retry_with_context", (_row("wrong"),), _receipt("row-1"))
    with pytest.raises(ValueError):
        ToolBatchPreparation(
            "retry_with_context",
            ({**row, EPHEMERAL_ORIGIN_KEY: "rag"},),
            _receipt("row-1"),
        )


def test_preparation_captures_full_batch_once_before_review_and_dispatch():
    calls = [
        ToolCall("one", {"path": "a"}, "call-1"),
        ToolCall("two", {"path": "b"}, "call-2"),
    ]
    events: list[object] = []

    class PayloadState:
        def capture(self, messages, active, captured_calls):
            events.append(
                (
                    "capture",
                    [dict(row) for row in messages],
                    tuple(active),
                    list(captured_calls),
                )
            )

    def prepare(batch):
        events.append(("prepare", list(batch)))
        return ToolBatchPreparation("proceed")

    def review(batch):
        events.append(("review", list(batch)))
        return {}

    def invoke(call):
        events.append(("invoke", call.name))
        return ToolResult(ok=True, content="ok")

    initial = [{"role": "user", "content": "go"}]
    out = run_agent_loop(
        CONFIG,
        initial,
        [SCHEMA],
        _deps(
            [_native_turn(calls), ModelTurn(text="done")],
            prepare_tool_calls=prepare,
            project_instruction_payload_state=PayloadState(),
            review_tool_calls=review,
            invoke_tool=invoke,
        ),
    )

    assert out.status == RUN_DONE
    assert [event[0] for event in events] == [
        "capture",
        "prepare",
        "review",
        "invoke",
        "invoke",
    ]
    capture = events[0]
    assert capture[1][0] == initial[0]
    assert capture[1][-1]["role"] == "assistant"
    assert capture[2] == (SCHEMA,)
    assert capture[3] == calls
    assert events[1][1] == calls


def test_retry_appends_fixed_stubs_then_separate_context_and_skips_dispatch():
    calls = [ToolCall("one", {}, "a"), ToolCall("two", {}, "b")]
    rows = (_row("row-1"),)
    seen_requests: list[list[dict]] = []
    turns = iter([_native_turn(calls), ModelTurn(text="done")])

    def call_model(messages, active):
        seen_requests.append([dict(row) for row in messages])
        return next(turns)

    reviewed: list = []
    invoked: list = []
    out = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [SCHEMA],
        _deps(
            [],
            call_model=call_model,
            prepare_tool_calls=lambda batch: ToolBatchPreparation(
                "retry_with_context", rows, _receipt("row-1")
            ),
            review_tool_calls=lambda batch: reviewed.append(batch) or {},
            invoke_tool=lambda call: invoked.append(call) or ToolResult(ok=True),
        ),
    )

    assert out.status == RUN_DONE
    assert reviewed == [] and invoked == []
    deferred = seen_requests[1][-3:]
    assert [(row["tool_call_id"], row["name"]) for row in deferred[:2]] == [
        ("a", "one"),
        ("b", "two"),
    ]
    assert all(
        "Deferred because project instructions were loaded" in row["content"]
        for row in deferred[:2]
    )
    assert deferred[2] == rows[0]


def test_preparation_failure_is_sanitized_and_fails_open_to_review():
    call = ToolCall("one", {}, "a")
    warnings: list[tuple] = []
    reviewed: list[list[ToolCall]] = []
    invoked: list[str] = []

    def prepare(_batch):
        raise RuntimeError("SECRET-INSTRUCTION-BODY /private/root/AGENTS.md")

    out = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [SCHEMA],
        _deps(
            [_native_turn([call]), ModelTurn(text="done")],
            prepare_tool_calls=prepare,
            on_ephemeral_runtime_warning=lambda code, names, count: warnings.append(
                (code, names, count)
            ),
            review_tool_calls=lambda batch: reviewed.append(list(batch)) or {},
            invoke_tool=lambda tool_call: (
                invoked.append(tool_call.name) or ToolResult(ok=True, content="ok")
            ),
        ),
    )

    assert out.status == RUN_DONE
    assert warnings == [("project_instruction_preparation_failed", ("one",), 1)]
    assert reviewed == [[call]] and invoked == ["one"]
    assert "SECRET" not in repr(warnings)


def test_warning_callback_failure_is_swallowed_and_review_still_runs():
    call = ToolCall("one", {}, "a")
    reviewed: list = []

    def prepare(_batch):
        raise RuntimeError("SECRET")

    def warning(*_args):
        raise RuntimeError("SECOND-SECRET")

    out = run_agent_loop(
        CONFIG,
        [{"role": "user", "content": "go"}],
        [SCHEMA],
        _deps(
            [_native_turn([call]), ModelTurn(text="done")],
            prepare_tool_calls=prepare,
            on_ephemeral_runtime_warning=warning,
            review_tool_calls=lambda batch: reviewed.append(list(batch)) or {},
        ),
    )
    assert out.status == RUN_DONE and reviewed == [[call]]
