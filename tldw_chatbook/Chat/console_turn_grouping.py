"""Pure ownership and visual-order projections for Console Assistant turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)


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
    """Yield messages in navigation order, with activities before the answer."""
    for unit in units:
        if unit.standalone is not None:
            yield unit.standalone
            continue

        turn = unit.assistant_turn
        assert turn is not None  # The dataclass invariant narrows the union here.
        yield from turn.activities
        yield turn.assistant
