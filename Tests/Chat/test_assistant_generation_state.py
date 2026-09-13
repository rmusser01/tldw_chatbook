"""Contracts for portable assistant-generation state."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.assistant_generation_state import (
    AssistantGenerationState,
    normalize_assistant_generation_state,
    unresolved_imported_generation_state_copy,
)


def test_non_assistant_rows_have_no_generation_state():
    """Assigning generation state to a user row would corrupt message ownership."""
    assert (
        normalize_assistant_generation_state(
            role="user",
            raw_state="complete",
            has_valid_active_continuation=True,
        )
        is None
    )


@pytest.mark.parametrize("role", ["assistant", "ASSISTANT", "Assistant"])
def test_active_adr_063_continuation_overrides_null_or_stale_assistant_state(role):
    """A stale terminal state cannot hide a valid continuation recovery owner."""
    assert (
        normalize_assistant_generation_state(
            role=role,
            raw_state="complete",
            has_valid_active_continuation=True,
        )
        is AssistantGenerationState.CONTINUATION_ACTIVE
    )


def test_historical_assistant_null_state_remains_unresolved():
    """Turning NULL into a new state would rewrite historical meaning."""
    assert (
        normalize_assistant_generation_state(
            role="assistant",
            raw_state=None,
            has_valid_active_continuation=False,
        )
        is None
    )


def test_assistant_state_uses_the_closed_state_vocabulary():
    """An unknown imported state must not be silently treated as terminal."""
    assert (
        normalize_assistant_generation_state(
            role="assistant",
            raw_state="dispatch_started",
            has_valid_active_continuation=False,
        )
        is AssistantGenerationState.DISPATCH_STARTED
    )
    with pytest.raises(ValueError):
        normalize_assistant_generation_state(
            role="assistant",
            raw_state="mystery_state",
            has_valid_active_continuation=False,
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            AssistantGenerationState.ACCEPTED,
            "Response accepted on another device; waiting for dispatch.",
        ),
        (
            AssistantGenerationState.DISPATCH_STARTED,
            "Response delivery status is unknown on the source device.",
        ),
    ],
)
def test_unresolved_imported_states_use_literal_status_copy(state, expected):
    """Remote recovery owners need a visible bounded status instead of a blank row."""
    assert unresolved_imported_generation_state_copy(state) == expected
