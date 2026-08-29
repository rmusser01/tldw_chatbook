from __future__ import annotations

import pytest

from tldw_chatbook.Chat import console_context_compaction as compaction
from tldw_chatbook.Chat.console_prepared_request import (
    IDLE_REQUEST_SENTINEL,
    PreparedConsoleRequest,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)


def _message(
    message_id: str,
    role: str,
    *,
    version: int | None = 1,
    parent_message_id: str | None = None,
    status: str = "complete",
    deleted: bool = False,
    provider_visible: bool = True,
    tool_calls: tuple[dict[str, object], ...] = (),
    tool_call_id: str | None = None,
) -> compaction.DurableMessageSnapshot:
    return compaction.DurableMessageSnapshot(
        message_id=message_id,
        version=version,
        role=role,
        content=f"content for {message_id}",
        parent_message_id=parent_message_id,
        status=status,
        deleted=deleted,
        provider_visible=provider_visible,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


@pytest.mark.parametrize(
    ("roles", "expected_units"),
    [
        (
            (
                ("system", "system"),
                ("greeting", "assistant"),
                ("u1", "user"),
                ("a1", "assistant"),
                ("u2", "user"),
                ("a2", "assistant"),
            ),
            (("u1", "a1"), ("u2", "a2")),
        ),
        (
            (("u1", "user"),),
            (),
        ),
        (
            (("u1", "user"), ("result1", "tool"), ("a1", "assistant")),
            (),
        ),
        (
            (("u1", "user"), ("call1", "assistant"), ("result1", "tool")),
            (),
        ),
        (
            (
                ("u1", "user"),
                ("a1", "assistant"),
                ("u2", "user"),
            ),
            (("u1", "a1"),),
        ),
        (
            (
                ("u1", "user"),
                ("a1", "assistant"),
                ("u2", "user"),
                ("result2", "tool"),
                ("a2", "assistant"),
                ("u3", "user"),
                ("a3", "assistant"),
            ),
            (("u1", "a1"),),
        ),
    ],
)
def test_complete_durable_units_accept_only_closed_user_led_exchanges(
    roles: tuple[tuple[str, str], ...],
    expected_units: tuple[tuple[str, ...], ...],
) -> None:
    helper = getattr(compaction, "complete_durable_units", None)
    assert callable(helper), "complete_durable_units must own the shared predicate"
    messages = tuple(_message(message_id, role) for message_id, role in roles)

    units = helper(messages)

    assert tuple(
        tuple(message.message_id for message in unit.messages) for unit in units
    ) == expected_units


@pytest.mark.parametrize(
    ("changed_index", "changes"),
    [
        (0, {"version": None}),
        (0, {"version": 0}),
        (0, {"deleted": True}),
        (0, {"provider_visible": False}),
        (1, {"status": "generating"}),
        (1, {"status": "streaming"}),
        (1, {"status": "stopped"}),
        (1, {"status": "failed"}),
        (1, {"deleted": True}),
        (1, {"provider_visible": False}),
    ],
)
def test_complete_durable_units_rejects_unavailable_or_nonterminal_rows(
    changed_index: int,
    changes: dict[str, object],
) -> None:
    facts = [
        {
            "message_id": "u1",
            "role": "user",
            "parent_message_id": None,
        },
        {
            "message_id": "a1",
            "role": "assistant",
            "parent_message_id": "u1",
        },
    ]
    facts[changed_index].update(changes)
    messages = tuple(_message(**fact) for fact in facts)

    assert compaction.complete_durable_units(messages) == ()


def test_complete_durable_units_rejects_incomplete_tool_results() -> None:
    messages = (
        _message("u1", "user"),
        _message("call1", "assistant"),
        _message("result1", "tool", status="generating"),
        _message("a1", "assistant"),
    )

    assert compaction.complete_durable_units(messages) == ()


@pytest.mark.parametrize(
    ("rows", "expected_ids"),
    [
        (
            (
                {"message_id": "u1", "role": "user"},
                {
                    "message_id": "call1",
                    "role": "assistant",
                    "tool_calls": (
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                    ),
                },
                {"message_id": "a1", "role": "assistant"},
            ),
            (),
        ),
        (
            (
                {"message_id": "u1", "role": "user"},
                {
                    "message_id": "call1",
                    "role": "assistant",
                    "tool_calls": (
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                    ),
                },
                {
                    "message_id": "result1",
                    "role": "tool",
                    "tool_call_id": "call-A",
                },
                {
                    "message_id": "result2",
                    "role": "tool",
                    "tool_call_id": "call-A",
                },
                {"message_id": "a1", "role": "assistant"},
            ),
            (),
        ),
        (
            (
                {"message_id": "u1", "role": "user"},
                {
                    "message_id": "call1",
                    "role": "assistant",
                    "tool_calls": (
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                    ),
                },
                {
                    "message_id": "result1",
                    "role": "tool",
                    "tool_call_id": "call-B",
                },
                {"message_id": "a1", "role": "assistant"},
            ),
            (),
        ),
        (
            (
                {"message_id": "u1", "role": "user"},
                {
                    "message_id": "call1",
                    "role": "assistant",
                    "tool_calls": (
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                        {
                            "id": "call-B",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": "{}"},
                        },
                    ),
                },
                {
                    "message_id": "result1",
                    "role": "tool",
                    "tool_call_id": "call-A",
                },
                {"message_id": "a1", "role": "assistant"},
            ),
            (),
        ),
        (
            (
                {"message_id": "u1", "role": "user"},
                {
                    "message_id": "call1",
                    "role": "assistant",
                    "tool_calls": (
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                        {
                            "id": "call-B",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": "{}"},
                        },
                    ),
                },
                {
                    "message_id": "result2",
                    "role": "tool",
                    "tool_call_id": "call-B",
                },
                {
                    "message_id": "result1",
                    "role": "tool",
                    "tool_call_id": "call-A",
                },
                {"message_id": "a1", "role": "assistant"},
            ),
            ("u1", "call1", "result2", "result1", "a1"),
        ),
    ],
)
def test_complete_durable_units_requires_exact_tool_call_result_matching(
    rows: tuple[dict[str, object], ...],
    expected_ids: tuple[str, ...],
) -> None:
    messages = tuple(_message(**row) for row in rows)

    units = compaction.complete_durable_units(messages)

    actual_ids = tuple(
        message.message_id for unit in units for message in unit.messages
    )
    assert actual_ids == expected_ids


def _long_tool_exchange_messages() -> tuple[compaction.DurableMessageSnapshot, ...]:
    return (
        _message("greeting", "assistant"),
        compaction.DurableMessageSnapshot(
            "u1",
            1,
            "user",
            "question " * 40,
        ),
        compaction.DurableMessageSnapshot(
            "call1",
            1,
            "assistant",
            "calling lookup",
            tool_calls=(
                {
                    "id": "call-A",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"weather"}',
                    },
                },
            ),
        ),
        compaction.DurableMessageSnapshot(
            "result1",
            1,
            "tool",
            "result " * 30,
            tool_call_id="call-A",
        ),
        compaction.DurableMessageSnapshot(
            "a1",
            1,
            "assistant",
            "answer " * 40,
        ),
    )


def _long_exchange_messages() -> tuple[compaction.DurableMessageSnapshot, ...]:
    rows = [_message("greeting", "assistant")]
    for index in range(1, 4):
        rows.extend(
            (
                compaction.DurableMessageSnapshot(
                    f"u{index}",
                    1,
                    "user",
                    f"question-{index} " + "question " * 40,
                ),
                compaction.DurableMessageSnapshot(
                    f"a{index}",
                    1,
                    "assistant",
                    f"answer-{index} " + "answer " * 40,
                ),
            )
        )
    return tuple(rows)


def _count(messages: list[dict], _model: str) -> int:
    return sum(len(str(message.get("content", "")).split()) + 1 for message in messages)


def _prepare(
    semantic: PreparedConsoleRequest,
    *,
    response_tokens: int = 40,
    window: int = 2_000,
):
    return prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(
            context_window_tokens=window,
            requested_response_tokens=response_tokens,
        ),
        count_fn=_count,
        apply_safety_window=False,
    )


def test_manual_plan_preserves_durable_tool_envelopes_in_both_inputs() -> None:
    result = compaction.plan_manual_range(
        messages=_long_tool_exchange_messages(),
        selected_prompt_message_id="u1",
        current_leaf_message_id="a1",
        system_messages=(),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="short memory",
        prepare_projection=_prepare,
        prepare_auxiliary=lambda messages, cap: _prepare(
            PreparedConsoleRequest(active_request=messages),
            response_tokens=cap,
        ),
    )

    assert result.reason is None
    assert result.plan is not None
    raw_rows = result.plan.before_projection.semantic.compactable[0].messages
    assert thaw_json(raw_rows[1]["tool_calls"]) == [
        {
            "id": "call-A",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query":"weather"}',
            },
        }
    ]
    assert raw_rows[2]["tool_call_id"] == "call-A"
    auxiliary_data = result.plan.auxiliary_messages[1]["content"]
    assert '"tool_calls":[{"function":' in auxiliary_data
    assert '"tool_call_id":"call-A"' in auxiliary_data


@pytest.mark.parametrize(
    (
        "planner_name",
        "kwargs",
        "coverage",
        "selected_ids",
        "retained_ids",
        "start",
        "boundary",
    ),
    [
        (
            "plan_manual_prefix",
            {"selected_prompt_message_id": "u3"},
            "prefix",
            ("u1", "a1", "u2", "a2"),
            ("u3", "a3"),
            "u1",
            "a2",
        ),
        (
            "plan_manual_range",
            {
                "selected_prompt_message_id": "u2",
                "current_leaf_message_id": "a3",
            },
            "range",
            ("u2", "a2", "u3", "a3"),
            ("u1", "a1"),
            "u2",
            "a3",
        ),
    ],
)
def test_manual_planners_select_exact_units_and_canonical_idle_projections(
    planner_name: str,
    kwargs: dict[str, str],
    coverage: str,
    selected_ids: tuple[str, ...],
    retained_ids: tuple[str, ...],
    start: str,
    boundary: str,
) -> None:
    planner = getattr(compaction, planner_name, None)
    assert callable(planner), f"{planner_name} must be defined"
    result = planner(
        messages=_long_exchange_messages(),
        system_messages=(
            {"role": "system", "content": "system contract"},
            {"role": "system", "content": "identity contract"},
        ),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="short memory",
        prepare_projection=_prepare,
        prepare_auxiliary=lambda messages, cap: _prepare(
            PreparedConsoleRequest(active_request=messages),
            response_tokens=cap,
        ),
        **kwargs,
    )

    assert result.reason is None
    assert result.plan is not None
    plan = result.plan
    assert plan.coverage_kind.value == coverage
    assert tuple(
        row.message_id for unit in plan.selected_units for row in unit.messages
    ) == selected_ids
    assert tuple(
        row.message_id for unit in plan.retained_units for row in unit.messages
    ) == retained_ids
    assert plan.selection_anchor_message_id == kwargs["selected_prompt_message_id"]
    assert plan.start_message_id == start
    assert plan.boundary_message_id == boundary
    assert plan.before_projection.semantic.system == plan.after_projection.semantic.system
    assert plan.before_projection.semantic.memory == ()
    assert plan.after_projection.semantic.memory
    assert plan.before_projection.semantic.active_request == (IDLE_REQUEST_SENTINEL,)
    assert plan.after_projection.semantic.active_request == (IDLE_REQUEST_SENTINEL,)
    assert plan.before_projection.dropped_units == 0
    assert plan.after_projection.dropped_units == 0
    # 6 system/identity + 4 greeting + 252 raw-unit + 3 idle-sentinel tokens.
    assert plan.before_tokens == 265
    assert plan.after_tokens < plan.before_tokens
    assert "prior_generated_memory_json=" not in plan.auxiliary_messages[1]["content"]
    assert tuple(message["role"] for message in plan.auxiliary_messages) == (
        "system",
        "user",
    )
    with pytest.raises(TypeError):
        plan.auxiliary_messages[0]["content"] = "mutated"
    assert "question-" not in repr(plan.provenance)
    assert "short memory" not in repr(plan.provenance)


def test_manual_plan_refuses_oversized_auxiliary_input_in_one_attempt() -> None:
    preparations = 0

    def prepare_oversized(messages, cap):
        nonlocal preparations
        preparations += 1
        return _prepare(
            PreparedConsoleRequest(active_request=messages),
            response_tokens=cap,
            window=620,
        )

    result = compaction.plan_manual_range(
        messages=_long_exchange_messages(),
        selected_prompt_message_id="u2",
        current_leaf_message_id="a3",
        system_messages=({"role": "system", "content": "system contract"},),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="short memory",
        prepare_projection=_prepare,
        prepare_auxiliary=prepare_oversized,
    )

    assert preparations == 1
    assert result.plan is None
    assert result.reason == "manual_auxiliary_input_too_large"


def test_manual_plan_rejects_non_improving_canonical_idle_projection() -> None:
    result = compaction.plan_manual_prefix(
        messages=_long_exchange_messages(),
        selected_prompt_message_id="u2",
        system_messages=({"role": "system", "content": "system contract"},),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="replacement " * 160,
        prepare_projection=_prepare,
        prepare_auxiliary=lambda messages, cap: _prepare(
            PreparedConsoleRequest(active_request=messages),
            response_tokens=cap,
        ),
    )

    assert result.plan is None
    assert result.reason == "manual_memory_did_not_make_progress"


def test_manual_prefix_refuses_an_incomplete_current_leaf_without_a_call() -> None:
    result = compaction.plan_manual_prefix(
        messages=_long_exchange_messages()[:-1],
        selected_prompt_message_id="u2",
        system_messages=(),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="short",
        prepare_projection=_prepare,
        prepare_auxiliary=lambda _messages, _cap: pytest.fail(
            "an incomplete manual span must not prepare an auxiliary call"
        ),
    )

    assert result.plan is None
    assert result.reason == "incomplete_current_leaf"


@pytest.mark.parametrize(
    ("planner", "kwargs", "reason"),
    [
        (
            compaction.plan_manual_prefix,
            {"selected_prompt_message_id": "missing"},
            "invalid_selection_anchor",
        ),
        (
            compaction.plan_manual_prefix,
            {"selected_prompt_message_id": "a2"},
            "invalid_selection_anchor",
        ),
        (
            compaction.plan_manual_range,
            {
                "selected_prompt_message_id": "u2",
                "current_leaf_message_id": "a2",
            },
            "incomplete_or_invalid_range_end",
        ),
    ],
)
def test_manual_plans_refuse_invalid_anchors_without_auxiliary_preparation(
    planner,
    kwargs: dict[str, str],
    reason: str,
) -> None:
    result = planner(
        messages=_long_exchange_messages(),
        system_messages=(),
        prompt=compaction.CompactionPromptSnapshot("Preserve decisions."),
        requested_output_cap=40,
        candidate_memory="short",
        prepare_projection=_prepare,
        prepare_auxiliary=lambda _messages, _cap: pytest.fail(
            "invalid anchors must not prepare an auxiliary call"
        ),
        **kwargs,
    )

    assert result.plan is None
    assert result.reason == reason
