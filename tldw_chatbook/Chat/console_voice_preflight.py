"""App-owned voice readiness identity and safe failure categories."""

from dataclasses import dataclass, field
from typing import Any


_FAILURE_CATEGORIES = frozenset(
    {"session_unavailable", "provider_unavailable", "unexpected", "stale"}
)


class VoicePreparationError(RuntimeError):
    """Carry only a validated app-owned category across the voice boundary."""

    def __init__(self, category: str) -> None:
        if type(category) is not str or category not in _FAILURE_CATEGORIES:
            raise ValueError("Invalid voice preparation category")
        self.category = category
        super().__init__(category)


def voice_failure_category(failure: BaseException) -> str:
    """Never interpret arbitrary provider exception text as an app category."""
    category = failure.category if type(failure) is VoicePreparationError else None
    return (
        category
        if type(category) is str and category in _FAILURE_CATEGORIES
        else "unexpected"
    )


def voice_failure_message(category: str, *, startup: bool = False) -> str:
    """Return fixed, actionable copy without including provider-owned values."""
    reason = {
        "session_unavailable": "The conversation is unavailable. Open a conversation and try again.",
        "provider_unavailable": "The selected provider is unavailable. Check its settings and connection, then try again.",
    }.get(
        category, "Provider preparation failed. Check provider settings and try again."
    )
    prefix = (
        "Hands-free could not start. "
        if startup
        else "No reply started; your transcript was preserved in the draft. "
    )
    return prefix + reason


@dataclass(frozen=True, slots=True)
class VoiceEntryStamp:
    """Ephemeral ownership fence; never persisted or exposed in the UI."""

    session: Any = field(repr=False)
    active_session_epoch: int
    settings_revision: int
    selection: Any = field(repr=False)
    config_generation: int | None = None
