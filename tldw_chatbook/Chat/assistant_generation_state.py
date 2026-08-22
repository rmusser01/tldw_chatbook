"""Pure state contracts for portable assistant-generation recovery."""

from __future__ import annotations

from enum import Enum


class AssistantGenerationState(str, Enum):
    """Closed lifecycle vocabulary persisted for assistant response owners."""

    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    CONTINUATION_ACTIVE = "continuation_active"
    COMPLETE = "complete"
    STOPPED = "stopped"
    FAILED = "failed"
    DISCARDED = "discarded"


_UNRESOLVED_IMPORTED_STATE_COPY = {
    AssistantGenerationState.ACCEPTED: (
        "Response accepted on another device; waiting for dispatch."
    ),
    AssistantGenerationState.DISPATCH_STARTED: (
        "Response delivery status is unknown on the source device."
    ),
}


def normalize_assistant_generation_state(
    *, role: object, raw_state: object, has_valid_active_continuation: bool
) -> AssistantGenerationState | None:
    """Return the effective state for an imported or persisted message row.

    A valid active ADR-063 continuation remains authoritative over historical
    NULL values and stale persisted assistant-generation states.
    """
    if str(role or "").lower() != "assistant":
        return None
    if has_valid_active_continuation:
        return AssistantGenerationState.CONTINUATION_ACTIVE
    if raw_state is None:
        return None
    return AssistantGenerationState(str(raw_state))


def unresolved_imported_generation_state_copy(state: object) -> str | None:
    """Return literal visible status copy for an unresolved imported owner."""
    try:
        normalized = (
            state
            if isinstance(state, AssistantGenerationState)
            else AssistantGenerationState(str(state))
        )
    except ValueError:
        return None
    return _UNRESOLVED_IMPORTED_STATE_COPY.get(normalized)
