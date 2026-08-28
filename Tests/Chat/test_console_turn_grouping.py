"""Tests for pure Console Assistant-turn ownership and visual order."""

from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleMessageStatus,
    RawCliPresentation,
)
from tldw_chatbook.Chat.console_turn_grouping import (
    ConsoleAssistantTurn,
    ConsoleThinkingActivityRef,
    ConsoleTranscriptUnit,
    group_console_transcript_messages,
    ordered_assistant_activities,
    visual_messages,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
)


def _message(
    role: ConsoleMessageRole,
    content: str,
    *,
    message_id: str,
    status: ConsoleMessageStatus = "complete",
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=role,
        content=content,
        id=message_id,
        status=status,
    )


def _ids(messages: Iterable[ConsoleChatMessage]) -> list[str]:
    return [message.id for message in messages]


def test_groups_contiguous_tools_under_the_preceding_assistant() -> None:
    user_before = _message(ConsoleMessageRole.USER, "Question", message_id="u1")
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    first_tool = _message(ConsoleMessageRole.TOOL, "Read", message_id="t1")
    second_tool = _message(ConsoleMessageRole.TOOL, "Write", message_id="t2")
    user_after = _message(ConsoleMessageRole.USER, "Next", message_id="u2")

    units = group_console_transcript_messages(
        [user_before, assistant, first_tool, second_tool, user_after]
    )

    assert units == (
        ConsoleTranscriptUnit.for_standalone(user_before),
        ConsoleTranscriptUnit.for_assistant_turn(
            ConsoleAssistantTurn(assistant, (first_tool, second_tool))
        ),
        ConsoleTranscriptUnit.for_standalone(user_after),
    )


def test_live_raw_cli_marker_is_standalone_after_the_pending_assistant() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    ordinary_tool = _message(
        ConsoleMessageRole.TOOL,
        "Read",
        message_id="t1",
    )
    raw_marker = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="Raw command running",
        id="raw1",
        raw_cli_presentation=RawCliPresentation(
            invocation_id="raw-invocation",
            caller="user",
            lifecycle_state="running",
            command="printf ok",
            shell="/bin/zsh",
            cwd="/private/tmp",
            started_at_monotonic=1.0,
            elapsed_seconds=0.1,
            exit_code=None,
            truncated=False,
            cleanup_proven=None,
        ),
    )

    units = group_console_transcript_messages([assistant, ordinary_tool, raw_marker])

    assert units == (
        ConsoleTranscriptUnit.for_assistant_turn(
            ConsoleAssistantTurn(assistant, (ordinary_tool,))
        ),
        ConsoleTranscriptUnit.for_standalone(raw_marker),
    )
    assert _ids(visual_messages(units)) == ["t1", "a1", "raw1"]


def test_leading_tool_is_a_standalone_orphan() -> None:
    orphan = _message(ConsoleMessageRole.TOOL, "Orphan", message_id="t1")
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")

    units = group_console_transcript_messages([orphan, assistant])

    assert units[0].standalone is orphan
    assert units[1].assistant_turn == ConsoleAssistantTurn(assistant)


def test_system_message_closes_the_assistant_turn() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    first_tool = _message(ConsoleMessageRole.TOOL, "Owned", message_id="t1")
    system = _message(ConsoleMessageRole.SYSTEM, "Notice", message_id="s1")
    orphan = _message(ConsoleMessageRole.TOOL, "Orphan", message_id="t2")

    units = group_console_transcript_messages([assistant, first_tool, system, orphan])

    assert units[0].assistant_turn == ConsoleAssistantTurn(assistant, (first_tool,))
    assert units[1].standalone is system
    assert units[2].standalone is orphan


def test_new_assistant_closes_the_previous_turn() -> None:
    first = _message(ConsoleMessageRole.ASSISTANT, "First", message_id="a1")
    second = _message(ConsoleMessageRole.ASSISTANT, "Second", message_id="a2")
    tool = _message(ConsoleMessageRole.TOOL, "Second's tool", message_id="t1")

    units = group_console_transcript_messages([first, second, tool])

    assert units == (
        ConsoleTranscriptUnit.for_assistant_turn(ConsoleAssistantTurn(first)),
        ConsoleTranscriptUnit.for_assistant_turn(ConsoleAssistantTurn(second, (tool,))),
    )


def test_groups_two_assistant_turns_without_cross_ownership() -> None:
    first = _message(ConsoleMessageRole.ASSISTANT, "First", message_id="a1")
    first_tool = _message(ConsoleMessageRole.TOOL, "First tool", message_id="t1")
    user = _message(ConsoleMessageRole.USER, "Again", message_id="u1")
    second = _message(ConsoleMessageRole.ASSISTANT, "Second", message_id="a2")
    second_tool = _message(ConsoleMessageRole.TOOL, "Second tool", message_id="t2")

    units = group_console_transcript_messages(
        (first, first_tool, user, second, second_tool)
    )

    assert units[0].assistant_turn == ConsoleAssistantTurn(first, (first_tool,))
    assert units[1].standalone is user
    assert units[2].assistant_turn == ConsoleAssistantTurn(second, (second_tool,))


@pytest.mark.parametrize(
    ("content", "status"),
    [
        ("", "complete"),
        ("Partial answer", "streaming"),
        ("Failed answer", "failed"),
    ],
)
def test_assistant_content_and_status_do_not_affect_tool_ownership(
    content: str,
    status: ConsoleMessageStatus,
) -> None:
    assistant = _message(
        ConsoleMessageRole.ASSISTANT,
        content,
        message_id="a1",
        status=status,
    )
    tool = _message(ConsoleMessageRole.TOOL, "Activity", message_id="t1")

    units = group_console_transcript_messages([assistant, tool])

    assert units[0].assistant_turn == ConsoleAssistantTurn(assistant, (tool,))


def test_assistant_without_tools_still_forms_a_turn() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")

    units = group_console_transcript_messages([assistant])

    assert units == (
        ConsoleTranscriptUnit.for_assistant_turn(ConsoleAssistantTurn(assistant)),
    )


def test_consecutive_orphan_tools_remain_separate_standalone_units() -> None:
    first = _message(ConsoleMessageRole.TOOL, "First", message_id="t1")
    second = _message(ConsoleMessageRole.TOOL, "Second", message_id="t2")

    units = group_console_transcript_messages([first, second])

    assert tuple(unit.standalone for unit in units) == (first, second)


def test_only_supplied_active_path_messages_can_appear() -> None:
    user = _message(ConsoleMessageRole.USER, "Question", message_id="u1")
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Active", message_id="a1")
    active_tool = _message(ConsoleMessageRole.TOOL, "Active tool", message_id="t1")
    off_branch_tool = _message(
        ConsoleMessageRole.TOOL,
        "Off-branch tool",
        message_id="t-off",
    )

    units = group_console_transcript_messages(
        message for message in (user, assistant, active_tool)
    )

    assert _ids(visual_messages(units)) == ["u1", "t1", "a1"]
    assert off_branch_tool.id not in _ids(visual_messages(units))


def test_grouping_preserves_message_identity_and_does_not_mutate_input() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    tool = _message(ConsoleMessageRole.TOOL, "Activity", message_id="t1")
    messages = [assistant, tool]
    original_order = tuple(messages)
    original_values = tuple(message.__dict__.copy() for message in messages)

    units = group_console_transcript_messages(messages)

    turn = units[0].assistant_turn
    assert turn is not None
    assert turn.assistant is assistant
    assert turn.activities[0] is tool
    assert tuple(messages) == original_order
    assert tuple(message.__dict__ for message in messages) == original_values


def test_visual_order_places_activities_before_answer_but_ownership_is_causal() -> None:
    user = _message(ConsoleMessageRole.USER, "Question", message_id="u1")
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    first_tool = _message(ConsoleMessageRole.TOOL, "Read", message_id="t1")
    second_tool = _message(ConsoleMessageRole.TOOL, "Write", message_id="t2")
    turn = ConsoleAssistantTurn(assistant, (first_tool, second_tool))
    units = (
        ConsoleTranscriptUnit.for_standalone(user),
        ConsoleTranscriptUnit.for_assistant_turn(turn),
    )

    visual = tuple(visual_messages(units))

    assert visual == (user, first_tool, second_tool, assistant)
    assert turn.owned_message_ids == ("a1", "t1", "t2")
    assert visual[0] is user
    assert visual[1] is first_tool
    assert visual[2] is second_tool
    assert visual[3] is assistant


def test_grouping_accepts_a_single_pass_generator() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    tool = _message(ConsoleMessageRole.TOOL, "Activity", message_id="t1")
    consumed: list[str] = []

    def source() -> Iterator[ConsoleChatMessage]:
        for message in (assistant, tool):
            consumed.append(message.id)
            yield message

    units = group_console_transcript_messages(source())

    assert consumed == ["a1", "t1"]
    assert units[0].assistant_turn == ConsoleAssistantTurn(assistant, (tool,))


def test_assistant_turn_rejects_non_assistant_owner() -> None:
    user = _message(ConsoleMessageRole.USER, "Question", message_id="u1")

    with pytest.raises(ValueError, match="ASSISTANT"):
        ConsoleAssistantTurn(user)


def test_assistant_turn_rejects_non_tool_activity() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    user = _message(ConsoleMessageRole.USER, "Question", message_id="u1")

    with pytest.raises(ValueError, match="TOOL"):
        ConsoleAssistantTurn(assistant, (user,))


@pytest.mark.parametrize("include_both", [False, True])
def test_transcript_unit_requires_exactly_one_payload(include_both: bool) -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    turn = ConsoleAssistantTurn(assistant)

    with pytest.raises(ValueError, match="exactly one"):
        ConsoleTranscriptUnit(
            standalone=assistant if include_both else None,
            assistant_turn=turn if include_both else None,
        )


def test_projection_value_objects_are_frozen() -> None:
    assistant = _message(ConsoleMessageRole.ASSISTANT, "Answer", message_id="a1")
    turn = ConsoleAssistantTurn(assistant)
    unit = ConsoleTranscriptUnit.for_assistant_turn(turn)

    with pytest.raises(FrozenInstanceError):
        turn.activities = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        unit.assistant_turn = None  # type: ignore[misc]


def _thinking_block(block_id: str, round_ordinal: int) -> DisplayableThinkingBlock:
    return DisplayableThinkingBlock(
        block_id=block_id,
        round_ordinal=round_ordinal,
        provider="local_vllm",
        model="thinking-model",
        protocol="openai_chat",
        source_format="start_anchored_think",
        status="complete",
        text=f"thinking round {round_ordinal}",
    )


def _activity(
    message_id: str,
    kind: str,
    label: str,
    *,
    round_ordinal: int | None = None,
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=label,
        id=message_id,
        activity_presentation=ConsoleActivityPresentation(
            kind,
            label,
            "done",  # type: ignore[arg-type]
        ),
        activity_round_ordinal=round_ordinal,
    )


def _ordered_ids(
    activities: tuple[ConsoleChatMessage | ConsoleThinkingActivityRef, ...],
) -> list[str]:
    return [
        activity.activity_id
        if isinstance(activity, ConsoleThinkingActivityRef)
        else activity.id
        for activity in activities
    ]


def test_ordered_activities_place_each_block_before_first_activity_in_round() -> None:
    assistant = _message(
        ConsoleMessageRole.ASSISTANT,
        "Answer",
        message_id="a1",
    )
    assistant.thinking = ThinkingEnvelope(
        (_thinking_block("block-0", 0), _thinking_block("block-1", 1))
    )
    planning_0 = _activity("planning-0", "thinking", "Thinking", round_ordinal=0)
    tool_0 = _activity("tool-0", "tool", "fs_read", round_ordinal=0)
    planning_1 = _activity("planning-1", "thinking", "Thinking", round_ordinal=1)
    tool_1 = _activity("tool-1", "tool", "fs_write", round_ordinal=1)

    ordered = ordered_assistant_activities(
        ConsoleAssistantTurn(
            assistant,
            (planning_0, tool_0, planning_1, tool_1),
        ),
    )

    assert [
        activity.block_id
        if isinstance(activity, ConsoleThinkingActivityRef)
        else activity.id
        for activity in ordered
    ] == [
        "block-0",
        "planning-0",
        "tool-0",
        "block-1",
        "planning-1",
        "tool-1",
    ]


def test_ordered_activities_never_infer_rounds_from_activity_positions() -> None:
    assistant = _message(
        ConsoleMessageRole.ASSISTANT,
        "Answer",
        message_id="a1",
    )
    assistant.thinking = ThinkingEnvelope(
        (_thinking_block("block-0", 0), _thinking_block("block-1", 1))
    )
    legacy_marker = _activity("planning-0", "thinking", "Thinking")
    tool = _activity("tool-0", "tool", "fs_read")

    ordered = ordered_assistant_activities(
        ConsoleAssistantTurn(assistant, (legacy_marker, tool)),
    )

    assert [
        activity.block_id
        if isinstance(activity, ConsoleThinkingActivityRef)
        else activity.id
        for activity in ordered
    ] == ["block-0", "block-1", "planning-0", "tool-0"]


def test_ordered_activities_without_evidence_preserve_tool_identity_and_order() -> None:
    assistant = _message(
        ConsoleMessageRole.ASSISTANT,
        "Answer",
        message_id="a1",
    )
    first = _activity("tool-0", "tool", "fs_read")
    second = _activity("tool-1", "tool", "fs_write")

    ordered = ordered_assistant_activities(
        ConsoleAssistantTurn(assistant, (first, second)),
    )

    assert ordered == (first, second)
    assert _ordered_ids(ordered) == ["tool-0", "tool-1"]


def test_explicit_round_ownership_handles_multirow_skips_and_trailing_rows() -> None:
    assistant = _message(
        ConsoleMessageRole.ASSISTANT,
        "Answer",
        message_id="a1",
    )
    assistant.thinking = ThinkingEnvelope(
        (
            _thinking_block("block-0", 0),
            _thinking_block("block-2", 2),
            _thinking_block("block-3", 3),
        )
    )
    round_0_first = _activity("round-0-first", "tool", "fs_read", round_ordinal=0)
    round_0_second = _activity("round-0-second", "tool", "fs_grep", round_ordinal=0)
    round_1_without_thinking = _activity("round-1", "tool", "fs_list", round_ordinal=1)
    round_2 = _activity("round-2", "tool", "fs_write", round_ordinal=2)
    post_run = _activity("post-run", "changes", "Changes")

    ordered = ordered_assistant_activities(
        ConsoleAssistantTurn(
            assistant,
            (
                round_0_first,
                round_0_second,
                round_1_without_thinking,
                round_2,
                post_run,
            ),
        ),
    )

    assert [
        activity.block_id
        if isinstance(activity, ConsoleThinkingActivityRef)
        else activity.id
        for activity in ordered
    ] == [
        "block-0",
        "round-0-first",
        "round-0-second",
        "round-1",
        "block-2",
        "round-2",
        "block-3",
        "post-run",
    ]
