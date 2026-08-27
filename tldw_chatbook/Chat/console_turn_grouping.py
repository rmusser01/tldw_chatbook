"""Pure ownership and visual-order projections for Console Assistant turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator
from uuid import NAMESPACE_URL, uuid5

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityStatus,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleThinkingActivityRef,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)


_THINKING_STATUS: dict[str, ConsoleActivityStatus] = {
    "complete": "done",
    "stopped": "stopped",
    "failed": "failed",
}


def thinking_activity_id(
    *,
    session_id: str,
    assistant_message_id: str,
    generation_id: str,
    block_id: str,
) -> str:
    """Return a deterministic UI-safe identity without exposing owner input."""
    identity = json.dumps(
        (session_id, assistant_message_id, generation_id, block_id),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"thinking-{uuid5(NAMESPACE_URL, identity).hex}"


def project_thinking_activities(
    *,
    session_id: str,
    assistant: ConsoleChatMessage,
    generation_id: str,
    live_block_id: str | None = None,
) -> tuple[ConsoleThinkingActivityRef, ...]:
    """Project only supported envelope evidence into trusted activity refs."""
    if assistant.role is not ConsoleMessageRole.ASSISTANT:
        raise ValueError("Thinking activities require an ASSISTANT owner")
    envelope = assistant.thinking
    if not isinstance(envelope, ThinkingEnvelope):
        return ()

    refs: list[ConsoleThinkingActivityRef] = []
    for block in envelope.blocks:
        if isinstance(block, ProprietaryThinkingBlock):
            status: ConsoleActivityStatus = "unavailable"
        elif isinstance(block, DisplayableThinkingBlock):
            status = (
                "live"
                if block.block_id == live_block_id
                else _THINKING_STATUS[block.status]
            )
        else:  # ThinkingEnvelope rejects this, but presentation still fails closed.
            continue
        refs.append(
            ConsoleThinkingActivityRef(
                activity_id=thinking_activity_id(
                    session_id=session_id,
                    assistant_message_id=assistant.id,
                    generation_id=generation_id,
                    block_id=block.block_id,
                ),
                assistant_message_id=assistant.id,
                block_id=block.block_id,
                label="Thinking",
                status=status,
            )
        )
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ConsoleAssistantTurn:
    """An Assistant message and its immediately following tool activities."""

    assistant: ConsoleChatMessage
    activities: tuple[ConsoleChatMessage, ...] = ()

    def __post_init__(self) -> None:
        """Validate the causal roles and freeze the activity collection."""
        activities = tuple(self.activities)
        object.__setattr__(self, "activities", activities)
        if self.assistant.role != ConsoleMessageRole.ASSISTANT:
            raise ValueError("ConsoleAssistantTurn owner must have role ASSISTANT")
        if any(activity.role != ConsoleMessageRole.TOOL for activity in activities):
            raise ValueError("ConsoleAssistantTurn activities must have role TOOL")

    @property
    def owned_message_ids(self) -> tuple[str, ...]:
        """Return causal membership: Assistant id, then activity ids."""
        return (self.assistant.id, *(activity.id for activity in self.activities))


@dataclass(frozen=True, slots=True)
class ConsoleTranscriptUnit:
    """One standalone message or one grouped Assistant turn."""

    standalone: ConsoleChatMessage | None = None
    assistant_turn: ConsoleAssistantTurn | None = None

    def __post_init__(self) -> None:
        """Require exactly one transcript-unit payload."""
        if (self.standalone is None) == (self.assistant_turn is None):
            raise ValueError(
                "ConsoleTranscriptUnit requires exactly one of standalone or "
                "assistant_turn"
            )

    @classmethod
    def for_standalone(cls, message: ConsoleChatMessage) -> "ConsoleTranscriptUnit":
        """Create a unit containing one ungrouped message."""
        return cls(standalone=message)

    @classmethod
    def for_assistant_turn(
        cls,
        turn: ConsoleAssistantTurn,
    ) -> "ConsoleTranscriptUnit":
        """Create a unit containing one Assistant turn."""
        return cls(assistant_turn=turn)


def ordered_assistant_activities(
    turn: ConsoleAssistantTurn,
    *,
    session_id: str,
    generation_id: str,
    live_block_id: str | None = None,
) -> tuple[ConsoleChatMessage | ConsoleThinkingActivityRef, ...]:
    """Merge thinking refs using only explicit model-round ownership.

    A block precedes the first activity owned by its round. Unanchored blocks
    are kept together before wholly unowned activity rows, or immediately
    before trailing unowned rows after the last explicitly owned round.
    """
    refs = project_thinking_activities(
        session_id=session_id,
        assistant=turn.assistant,
        generation_id=generation_id,
        live_block_id=live_block_id,
    )
    if not refs:
        return turn.activities

    blocks = turn.assistant.thinking
    assert isinstance(blocks, ThinkingEnvelope)
    ordered: list[ConsoleChatMessage | ConsoleThinkingActivityRef] = []
    emitted_rounds: set[int] = set()
    remaining_owned = sum(
        activity.activity_round_ordinal is not None for activity in turn.activities
    )

    def append_through(round_ordinal: int | None = None) -> None:
        for block, ref in zip(blocks.blocks, refs):
            if block.round_ordinal in emitted_rounds or (
                round_ordinal is not None and block.round_ordinal > round_ordinal
            ):
                continue
            ordered.append(ref)
            emitted_rounds.add(block.round_ordinal)

    for activity in turn.activities:
        round_ordinal = activity.activity_round_ordinal
        if round_ordinal is not None:
            append_through(round_ordinal)
            remaining_owned -= 1
        elif remaining_owned == 0:
            append_through()
        ordered.append(activity)

    append_through()
    return tuple(ordered)


def group_console_transcript_messages(
    messages: Iterable[ConsoleChatMessage],
) -> tuple[ConsoleTranscriptUnit, ...]:
    """Group each Assistant with only its immediately following tool messages.

    Args:
        messages: Active-path messages in causal store order.

    Returns:
        Immutable transcript units in the supplied order. Message objects are
        retained by identity and are not mutated.
    """
    units: list[ConsoleTranscriptUnit] = []
    assistant: ConsoleChatMessage | None = None
    activities: list[ConsoleChatMessage] = []

    for message in messages:
        if assistant is not None and message.role == ConsoleMessageRole.TOOL:
            activities.append(message)
            continue

        if assistant is not None:
            units.append(
                ConsoleTranscriptUnit.for_assistant_turn(
                    ConsoleAssistantTurn(assistant, tuple(activities))
                )
            )
            assistant = None
            activities = []

        if message.role == ConsoleMessageRole.ASSISTANT:
            assistant = message
        else:
            units.append(ConsoleTranscriptUnit.for_standalone(message))

    if assistant is not None:
        units.append(
            ConsoleTranscriptUnit.for_assistant_turn(
                ConsoleAssistantTurn(assistant, tuple(activities))
            )
        )

    return tuple(units)


def visual_messages(
    units: Iterable[ConsoleTranscriptUnit],
) -> Iterator[ConsoleChatMessage]:
    """Yield messages in navigation order, with activities before the answer.

    Args:
        units: Grouped transcript units in causal order.

    Returns:
        An iterator that yields each unit's activities before its Assistant
        answer while preserving standalone message order.
    """
    for unit in units:
        if unit.standalone is not None:
            yield unit.standalone
            continue

        turn = unit.assistant_turn
        assert turn is not None  # The dataclass invariant narrows the union here.
        yield from turn.activities
        yield turn.assistant
