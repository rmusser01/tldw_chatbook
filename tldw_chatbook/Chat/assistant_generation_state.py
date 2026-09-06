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


def assistant_state_allows_provider_history(
    *, state: object, has_valid_continuation: bool, content: object
) -> bool:
    """Return whether one assistant owner belongs in ordinary provider history.

    Valid active ADR-063 continuation owners are projected only through their
    private, provider-specific sidecar. Closed or unresolved blank owners never
    create an empty provider message.
    """
    if has_valid_continuation:
        return False
    rendered = content if isinstance(content, str) else str(content or "")
    if not rendered:
        return False
    try:
        normalized = (
            state
            if isinstance(state, AssistantGenerationState)
            else None
            if state is None
            else AssistantGenerationState(str(state))
        )
    except ValueError:
        return False
    return normalized in {
        None,
        AssistantGenerationState.COMPLETE,
        AssistantGenerationState.STOPPED,
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
    normalized = None if raw_state is None else AssistantGenerationState(str(raw_state))
    if has_valid_active_continuation:
        return AssistantGenerationState.CONTINUATION_ACTIVE
    return normalized


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


def render_exported_assistant_content(
    *, role: object, content: object, state: object
) -> str:
    """Render bounded literal copy for an otherwise-empty assistant owner."""
    rendered = content if isinstance(content, str) else str(content or "")
    if str(role or "").lower() != "assistant" or rendered:
        return rendered
    pending = unresolved_imported_generation_state_copy(state)
    if pending is not None:
        return pending
    try:
        normalized = (
            state
            if isinstance(state, AssistantGenerationState)
            else AssistantGenerationState(str(state))
        )
    except ValueError:
        return rendered
    if normalized is AssistantGenerationState.COMPLETE:
        return "No response was generated."
    if normalized is AssistantGenerationState.FAILED:
        return "Response failed."
    if normalized is AssistantGenerationState.DISCARDED:
        return "Response discarded."
    return rendered
