"""Focused runtime coverage for ADR-154's per-invocation denial breaker."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolLoadSelection,
    ToolResult,
    ToolReviewDecision,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.Chat.console_chat_controller import (
    ApprovalDecisions,
    build_virtual_cli_review_hook,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def run_batch(results, *, limit=3, review=None):
    calls = [
        ToolCall(name=f"tool_{i}", args={}, call_id=f"c{i}")
        for i in range(len(results))
    ]
    raw = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.args)},
        }
        for call in calls
    ]
    script = [
        ModelTurn(
            tool_calls=tuple(calls),
            assistant_message={
                "role": "assistant",
                "content": "partial plan",
                "tool_calls": raw,
            },
        ),
        ModelTurn(text="done"),
    ]
    invoked, records, model_calls = [], [], []

    def model(messages, schemas):
        model_calls.append(list(messages))
        return script.pop(0)

    def invoke(call):
        invoked.append(call.call_id)
        return results[int(call.call_id[1:])]

    deps = LoopDeps(
        call_model=model,
        invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda ids, messages, call: ToolLoadSelection(),
        should_cancel=lambda: False,
        clock=lambda: 0,
        review_tool_calls=review,
        on_record=lambda kind, payload: records.append((kind, payload)),
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=tuple(call.name for call in calls),
        budget=RunBudget(max_steps=100, denial_circuit_breaker_limit=limit),
    )
    return (
        run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps),
        invoked,
        records,
        model_calls,
    )


def run_turns(turns, results, *, limit=3, review=None):
    script = list(turns)
    invoked, records, model_calls = [], [], []

    def model(messages, schemas):
        model_calls.append(list(messages))
        return script.pop(0)

    def invoke(call):
        invoked.append(call.call_id or call.name)
        return results.pop(0)

    deps = LoopDeps(
        call_model=model,
        invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda ids, messages, call: ToolLoadSelection(),
        should_cancel=lambda: False,
        clock=lambda: 0,
        review_tool_calls=review,
        on_record=lambda kind, payload: records.append((kind, payload)),
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=tuple(f"tool_{index}" for index in range(10)),
        budget=RunBudget(max_steps=100, denial_circuit_breaker_limit=limit),
        native_tools=False,
    )
    return (
        run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps),
        invoked,
        records,
        model_calls,
    )


def test_completed_batch_reports_observed_count_and_keeps_every_reply():
    denied = ToolResult.blocked("no", approval_decision="denied")
    out, invoked, records, model_calls = run_batch([denied] * 4)
    assert out.status == "stuck" and out.denial_count == 4
    assert invoked == ["c0", "c1", "c2", "c3"]
    assert len(model_calls) == 1
    assert [
        row["tool_call_id"] for row in out.final_messages if row["role"] == "tool"
    ] == invoked
    assert len([record for record in records if record[0] == "error"]) == 1


def test_approved_tail_resets_completed_batch_streak():
    denied = ToolResult.blocked("no", approval_decision="denied")
    out, _, _, model_calls = run_batch(
        [denied] * 3
        + [
            ToolResult(
                ok=False,
                error="execution failed",
                approval_decision="approved",
            )
        ]
    )
    assert out.status == "done" and out.denial_count == 0
    assert len(model_calls) == 2


@pytest.mark.parametrize("limit", [0, "0"])
def test_zero_disables_breaker(limit):
    denied = ToolResult.blocked("no", approval_decision="denied")
    out, _, _, model_calls = run_batch([denied] * 4, limit=limit)
    assert out.status == "done"
    assert len(model_calls) == 2


def test_successful_denial_looking_result_resets_streak():
    denied = ToolResult.blocked("no", approval_decision="denied")
    out, _, _, _ = run_batch(
        [
            denied,
            denied,
            ToolResult(ok=True, content="denied", approval_decision="denied"),
        ]
    )
    assert out.status == "done"


def test_structured_review_denial_counts_but_legacy_string_does_not():
    results = [ToolResult(ok=True)] * 3
    structured = lambda calls: {
        call.call_id: ToolReviewDecision("no", "denied") for call in calls
    }
    out, invoked, _, _ = run_batch(results, review=structured)
    assert out.status == "stuck" and out.denial_count == 3 and invoked == []

    out, invoked, _, _ = run_batch(
        results, review=lambda calls: {c.call_id: "no" for c in calls}
    )
    assert out.status == "done" and invoked == []


def test_actual_invocation_fact_overrides_earlier_review_approval():
    denied = ToolResult.blocked("later authority refusal", approval_decision="denied")
    review = lambda calls: {
        call.call_id: ToolReviewDecision("proceed", "approved") for call in calls
    }
    out, invoked, _, _ = run_batch([denied] * 3, review=review)
    assert out.status == "stuck" and invoked == ["c0", "c1", "c2"]


def test_streak_crosses_model_turns_and_non_denial_resets_it():
    denied = ToolResult.blocked("no", approval_decision="denied")
    turns = [
        ModelTurn(
            tool_calls=(ToolCall(f"tool_{index}", {}, f"c{index}"),),
            assistant_message={"role": "assistant", "tool_calls": []},
        )
        for index in range(5)
    ] + [ModelTurn(text="done")]
    out, _, _, calls = run_turns(
        turns,
        [denied, denied, ToolResult(ok=False, error="ordinary"), denied, denied],
    )
    assert out.status == "done" and len(calls) == 6

    out, _, _, calls = run_turns(turns[:3], [denied, denied, denied])
    assert out.status == "stuck" and out.denial_count == 3 and len(calls) == 3


def test_same_name_distinct_call_ids_select_independent_structured_reviews():
    results = [ToolResult(ok=True), ToolResult(ok=True), ToolResult(ok=True)]
    decisions = {
        "c0": ToolReviewDecision("no", "denied"),
        "c1": ToolReviewDecision("proceed", "approved"),
        "c2": ToolReviewDecision("no", "denied"),
    }
    calls = [ToolCall("same", {"index": index}, f"c{index}") for index in range(3)]
    raw = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.args)},
        }
        for call in calls
    ]
    out, invoked, _, model_calls = run_turns(
        [
            ModelTurn(
                tool_calls=tuple(calls),
                assistant_message={"role": "assistant", "tool_calls": raw},
            ),
            ModelTurn(text="done"),
        ],
        results,
        review=lambda batch: decisions,
    )
    assert out.status == "done" and invoked == ["c1"] and len(model_calls) == 2


@pytest.mark.parametrize(
    "tail,expected", [(None, "stuck"), (ToolResult(ok=True), "done")]
)
def test_fence_protocol_settles_history_and_applies_tail_reset(tail, expected):
    denied = ToolResult.blocked("no", approval_decision="denied")
    result_rows = [denied, denied, denied]
    turns = [
        ModelTurn(
            text=(
                "```tool_call\n"
                + json.dumps({"name": f"tool_{index}", "arguments": {}})
                + "\n```"
            )
        )
        for index in range(3)
    ]
    if tail is not None:
        result_rows.append(tail)
        turns.append(
            ModelTurn(text='```tool_call\n{"name":"tool_3","arguments":{}}\n```')
        )
    turns.append(ModelTurn(text="done"))
    out, _, _, model_calls = run_turns(
        turns, result_rows, limit=4 if tail is not None else 3
    )
    assert out.status == expected
    assert len([row for row in out.final_messages if row["role"] == "user"]) >= 4
    assert len(model_calls) == (3 if expected == "stuck" else 5)


@pytest.mark.parametrize("decisions,expected", [("deny", "stuck"), (None, "done")])
def test_virtual_cli_final_invocation_facts_control_breaker(
    tmp_path, decisions, expected
):
    for index in range(3):
        (tmp_path / f"{index}.txt").write_text(str(index), encoding="utf-8")
    provider = VirtualCliProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    calls = [
        ToolCall(
            "virtual_cli",
            {"command": "cat", "argv": [f"{index}.txt"]},
            f"c{index}",
        )
        for index in range(3)
    ]
    raw = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.args)},
        }
        for call in calls
    ]
    turns = [
        ModelTurn(
            tool_calls=tuple(calls),
            assistant_message={"role": "assistant", "tool_calls": raw},
        ),
        ModelTurn(text="done"),
    ]
    review_hook = build_virtual_cli_review_hook(
        provider,
        lambda rows: (
            {row.call_id: decisions for row in rows}
            if decisions
            else _unresolved_denials(rows)
        ),
    )
    model_calls = []

    def model(messages, schemas):
        model_calls.append(messages)
        return turns.pop(0)

    def invoke(call):
        with use_run_id("run"), use_tool_call_id(call.call_id):
            return provider.invoke(call.name, call.args)

    deps = LoopDeps(
        call_model=model,
        invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda ids, messages, call: ToolLoadSelection(),
        should_cancel=lambda: False,
        clock=lambda: 0,
        review_tool_calls=lambda batch: review_hook(batch, "run"),
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("virtual_cli",),
        budget=RunBudget(max_steps=100),
    )

    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)

    assert outcome.status == expected
    assert len(model_calls) == (1 if expected == "stuck" else 2)


def _unresolved_denials(rows):
    decisions = ApprovalDecisions({row.call_id: "deny" for row in rows})
    decisions.unresolved_keys = frozenset(decisions)
    return decisions


def test_virtual_cli_later_root_refusal_resets_earlier_review_denial(tmp_path):
    provider = VirtualCliProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    calls = [
        ToolCall("virtual_cli", {"command": "ls", "argv": [str(index)]}, f"c{index}")
        for index in range(3)
    ]
    raw = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.args)},
        }
        for call in calls
    ]
    turns = [
        ModelTurn(
            tool_calls=tuple(calls),
            assistant_message={"role": "assistant", "tool_calls": raw},
        ),
        ModelTurn(text="done"),
    ]
    hook = build_virtual_cli_review_hook(
        provider, lambda rows: {row.call_id: "deny" for row in rows}
    )

    def invoke(call):
        provider._authority_is_valid = lambda authority: False
        with use_run_id("run"), use_tool_call_id(call.call_id):
            return provider.invoke(call.name, call.args)

    deps = LoopDeps(
        call_model=lambda messages, schemas: turns.pop(0),
        invoke_tool=invoke,
        spawn=lambda task: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda ids, messages, call: ToolLoadSelection(),
        should_cancel=lambda: False,
        clock=lambda: 0,
        review_tool_calls=lambda batch: hook(batch, "run"),
    )
    outcome = run_agent_loop(
        AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("virtual_cli",),
            budget=RunBudget(max_steps=100),
        ),
        [],
        [],
        deps,
    )
    assert outcome.status == "done" and outcome.denial_count == 0
