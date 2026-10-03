"""A send refused before dispatch reads 'Not sent', not 'accepted'.

TASK-34100.5 AC#3 (entry-exit-handoff-03). A durable turn that the context
preflight refuses (it cannot fit the selected model) keeps its fail-closed
dispatch owner, so the user can still Retry or Discard it. That owner's
recovery panel read 'Response accepted; waiting for dispatch.' with an enabled
Retry -- a delay, not a block -- and Retry re-ran the same refusal. This
module presents such an owner as not sent, with Retry disabled until a setting
that could change the outcome does: the session's provider, model, endpoint
or reply limit, or the saved configuration (a context-window entry).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryState,
)

NOT_SENT_COPY = "Not sent — this message doesn't fit the selected model."
CHANGE_SETTING_HINT = "Change a setting above, then retry"
CHANGED_COPY = "Settings changed — Retry sends it again."


@dataclass(frozen=True, slots=True)
class PredispatchBlock:
    """One refused assistant turn and the settings it was refused under."""

    assistant_message_id: str
    fingerprint: tuple[Any, ...]


def settings_fingerprint(settings: Any) -> tuple[Any, ...]:
    """Return the settings that decide whether a request fits the model.

    Args:
        settings: The session's ``ConsoleSessionSettings`` (or ``None``).

    Returns:
        Provider, model, endpoint and reply limit, plus the runtime config
        generation (a saved context-window entry changes the window).
    """
    from tldw_chatbook.config import get_runtime_config_generation

    try:
        generation: Any = get_runtime_config_generation()
    except Exception:  # noqa: BLE001 -- an unreadable generation never re-enables
        generation = None
    return (
        getattr(settings, "provider", None),
        getattr(settings, "model", None),
        getattr(settings, "base_url", None),
        getattr(settings, "max_tokens", None),
        generation,
    )


def present_predispatch_block(
    recovery: ConsoleDispatchRecoveryState, *, settings_changed: bool
) -> ConsoleDispatchRecoveryState:
    """Return the recovery owner as a not-sent block.

    Args:
        recovery: The fail-closed dispatch owner of the refused turn.
        settings_changed: Whether a deciding setting changed since the refusal.

    Returns:
        The same owner with not-sent copy, no duplicate-request warning, and
        Retry disabled (with the hint) until ``settings_changed``.
    """
    actions = tuple(
        replace(action, enabled=False, disabled_reason=CHANGE_SETTING_HINT)
        if action.action_id is ConsoleDispatchRecoveryActionId.RETRY_RESPONSE
        and not settings_changed
        else action
        for action in recovery.actions
    )
    tail = CHANGED_COPY if settings_changed else f"{CHANGE_SETTING_HINT}."
    return replace(
        recovery,
        visible_copy=f"{NOT_SENT_COPY} {tail}",
        warning="",
        actions=actions,
    )
