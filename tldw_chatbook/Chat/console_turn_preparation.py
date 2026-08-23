"""Pure in-memory state for one admitted Console user turn."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping

from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext


_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z", re.ASCII)


class ConsoleTurnPreparationState(str, Enum):
    """Closed lifecycle states for one admitted Console turn."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTING = "committing"
    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    DISPATCHED = "dispatched"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SETTLED = "settled"


class ConsolePreparationPauseKind(str, Enum):
    """Closed pre-dispatch reasons that require an explicit user decision."""

    RETRIEVAL = "retrieval"
    PERSISTENCE = "persistence"
    DESTINATION_CHANGED = "destination_changed"


PAUSE_ACTIONS: Mapping[ConsolePreparationPauseKind, tuple[str, ...]] = (
    MappingProxyType(
        {
            ConsolePreparationPauseKind.RETRIEVAL: ("retry", "bypass", "cancel"),
            ConsolePreparationPauseKind.PERSISTENCE: ("retry", "cancel"),
            ConsolePreparationPauseKind.DESTINATION_CHANGED: ("retry", "cancel"),
        }
    )
)


@dataclass(frozen=True, slots=True)
class ConsoleTurnPreparation:
    """All preparation-owned inputs for one immediate or queued turn."""

    preparation_id: str
    attempt_id: str
    session_id: str
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    executed_draft: str
    execution_context: ConsoleTurnExecutionContext
    transient_user_message_id: str | None
    attachment_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    prefill_id: str | None
    queue_generation: int | None
    pre_send_title: str
    pre_send_conversation_id: str | None
    state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    one_shot_bypass: bool
    ephemeral: bool


@dataclass(frozen=True, slots=True)
class ConsolePreparationTransition:
    """One expected-state compare-and-set request for a preparation."""

    preparation_id: str
    expected_state: ConsoleTurnPreparationState
    new_state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    new_attempt_id: str | None


def initial_preparation_state(
    auto_retrieve: ConsoleAutoRetrieve,
) -> ConsoleTurnPreparationState:
    """Return the first state after authority and destination resolution.

    ``Never`` enters ``ready`` directly, so it neither runs nor displays a Library
    preparation stage. Automatic enters ``preparing``.
    """
    if auto_retrieve is ConsoleAutoRetrieve.NEVER:
        return ConsoleTurnPreparationState.READY
    if auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC:
        return ConsoleTurnPreparationState.PREPARING
    raise TypeError("auto_retrieve must be ConsoleAutoRetrieve")


def preparation_actions(preparation: ConsoleTurnPreparation) -> tuple[str, ...]:
    """Return the action data available in the current precommit state."""
    if preparation.state is ConsoleTurnPreparationState.PAUSED:
        if preparation.pause_kind is None:
            return ()
        return PAUSE_ACTIONS.get(preparation.pause_kind, ())
    if preparation.pause_kind is not None:
        return ()
    if preparation.state in {
        ConsoleTurnPreparationState.PREPARING,
        ConsoleTurnPreparationState.READY,
    }:
        return ("cancel",)
    return ()


def apply_preparation_transition(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> ConsoleTurnPreparation:
    """Apply one legal expected-state transition, otherwise return unchanged.

    Returning the same immutable object for a stale, racing, repeated, or
    malformed transition makes action delivery idempotent without granting a
    caller any state beyond the frozen transition matrix.
    """
    if (
        transition.preparation_id != preparation.preparation_id
        or transition.expected_state is not preparation.state
        or not _transition_is_legal(preparation, transition)
    ):
        return preparation

    attempt_id = transition.new_attempt_id or preparation.attempt_id
    bypass = preparation.one_shot_bypass or (
        preparation.state is ConsoleTurnPreparationState.PAUSED
        and preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        and transition.new_state is ConsoleTurnPreparationState.READY
    )
    return replace(
        preparation,
        attempt_id=attempt_id,
        state=transition.new_state,
        pause_kind=(
            transition.pause_kind
            if transition.new_state is ConsoleTurnPreparationState.PAUSED
            else None
        ),
        one_shot_bypass=bypass,
    )


def _transition_is_legal(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> bool:
    """Validate the closed transition matrix and its attempt/pause shape."""
    current = preparation.state
    new = transition.new_state
    pause = preparation.pause_kind

    if current is ConsoleTurnPreparationState.PAUSED and pause is None:
        return False
    if current is not ConsoleTurnPreparationState.PAUSED and pause is not None:
        return False
    if new is ConsoleTurnPreparationState.PAUSED:
        if transition.new_attempt_id is not None:
            return False
        if current is ConsoleTurnPreparationState.PREPARING:
            return transition.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        if current is ConsoleTurnPreparationState.COMMITTING:
            return transition.pause_kind in {
                ConsolePreparationPauseKind.PERSISTENCE,
                ConsolePreparationPauseKind.DESTINATION_CHANGED,
            }
        return False
    if transition.pause_kind is not None:
        return False

    if new is ConsoleTurnPreparationState.CANCELLED:
        return transition.new_attempt_id is None and current in {
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.PAUSED,
        }

    if current is ConsoleTurnPreparationState.PAUSED:
        if pause is ConsolePreparationPauseKind.RETRIEVAL:
            if new is ConsoleTurnPreparationState.PREPARING:
                return _has_new_attempt(preparation, transition)
            return (
                new is ConsoleTurnPreparationState.READY
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.PERSISTENCE:
            return (
                new is ConsoleTurnPreparationState.COMMITTING
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.DESTINATION_CHANGED:
            return (
                new is ConsoleTurnPreparationState.COMMITTING
                and transition.new_attempt_id is None
            )
        return False

    legal_without_new_attempt = {
        (
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
        ),
        (ConsoleTurnPreparationState.READY, ConsoleTurnPreparationState.COMMITTING),
        (
            ConsoleTurnPreparationState.COMMITTING,
            ConsoleTurnPreparationState.ACCEPTED,
        ),
        (
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
        ),
        (ConsoleTurnPreparationState.ACCEPTED, ConsoleTurnPreparationState.SETTLED),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            ConsoleTurnPreparationState.DISPATCHED,
        ),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            ConsoleTurnPreparationState.SETTLED,
        ),
        (
            ConsoleTurnPreparationState.DISPATCHED,
            ConsoleTurnPreparationState.SETTLED,
        ),
    }
    if (current, new) in legal_without_new_attempt:
        return transition.new_attempt_id is None
    if (
        current is ConsoleTurnPreparationState.DISPATCH_STARTED
        and new is ConsoleTurnPreparationState.DISPATCH_STARTED
    ):
        return _has_new_attempt(preparation, transition)
    return False


def _has_new_attempt(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> bool:
    """Return whether a retry supplies a distinct nonblank attempt identity."""
    value = transition.new_attempt_id
    return (
        type(value) is str
        and _IDENTIFIER_RE.fullmatch(value) is not None
        and value != preparation.attempt_id
    )
