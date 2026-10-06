"""A send refused before dispatch reads 'Not sent', not 'accepted'.

TASK-34100.5 AC#3 (entry-exit-handoff-03). A durable turn that the context
preflight refuses (it cannot fit the selected model) keeps its fail-closed
dispatch owner, so the user can still Retry or Discard it. That owner's
recovery panel read 'Response accepted; waiting for dispatch.' with an enabled
Retry -- a delay, not a block. Retry replays the accepted turn exactly as it
was accepted (its provider, model, reply limit and window are frozen), so it
can only meet the same refusal: live, Retry after switching model with Alt+M
failed again under the old model. This module presents such an owner as not
sent, with Retry disabled and the working path named instead: change the
setting, Discard, then Resend the message (Resend uses current settings).
Since TASK-34350 a composer send that cannot fit is refused before it is
committed, so this owner arises only for a send that skips that check (a
queued prompt, or one that keeps the composer as a buddy-conversation send
does).

Which turns were refused before dispatch is recorded here, by assistant
message id, not in the store (review round 1, F8: the store is over its size
budget). The card's presentation (``derive_dispatch_recovery_presentation``)
and the controller's Retry both read it, so Retry is refused for any caller,
not just on the card (F13). The record is in memory: after a restart the owner
presents as an ordinary accepted turn again, as before.

The overflow refusal itself is decided in ``ConsoleCompactionPreflight``
(``console_context_compaction``), which reaches the controller only through
``_context_overflow_alert`` and ``_block_context_preflight``. The controller
remembers each alert copy the first returns and notes a block when the second
receives one of them, so that module stays line-for-line as dev has it (it is
at its size budget).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryState,
)

NOT_SENT_COPY = (
    "Not sent — this message doesn't fit the selected model. Change a setting "
    "above, then Discard and Resend the message."
)
RETRY_DISABLED_REASON = (
    "Retry resends to the same model and limit — Discard, then Resend"
)


def present_predispatch_block(
    recovery: ConsoleDispatchRecoveryState,
) -> ConsoleDispatchRecoveryState:
    """Return the recovery owner as a not-sent block.

    Args:
        recovery: The fail-closed dispatch owner of the refused turn.

    Returns:
        The same owner with not-sent copy, no duplicate-request warning, and
        Retry disabled with the reason; Discard is unchanged.
    """
    actions = tuple(
        replace(action, enabled=False, disabled_reason=RETRY_DISABLED_REASON)
        if action.action_id is ConsoleDispatchRecoveryActionId.RETRY_RESPONSE
        else action
        for action in recovery.actions
    )
    return replace(recovery, visible_copy=NOT_SENT_COPY, warning="", actions=actions)


#: How many refused turns are remembered; the oldest is forgotten first, so
#: the record cannot grow with a long session (review round 1, F13).
MAX_NOTED_BLOCKS = 256
#: Assistant message ids (globally unique) of turns refused before dispatch.
_NOTED: "OrderedDict[str, None]" = OrderedDict()


def note_predispatch_block(message_id: str) -> None:
    """Record that ``message_id``'s turn was refused before dispatch."""
    _NOTED.pop(message_id, None)
    _NOTED[message_id] = None
    while len(_NOTED) > MAX_NOTED_BLOCKS:
        _NOTED.popitem(last=False)


def is_predispatch_block(message_id: str) -> bool:
    """Whether ``message_id``'s turn was refused before dispatch."""
    return message_id in _NOTED


#: Recent context-overflow alert copies, bounded like the block record.
_ALERTS: "OrderedDict[str, None]" = OrderedDict()


def remember_overflow_alert(alert: str | None) -> str | None:
    """Record a context-overflow alert's copy and return it unchanged."""
    if alert:
        _ALERTS.pop(alert, None)
        _ALERTS[alert] = None
        while len(_ALERTS) > MAX_NOTED_BLOCKS:
            _ALERTS.popitem(last=False)
    return alert


def note_if_overflow_alert(message_id: str, copy: str) -> None:
    """Note ``message_id``'s turn as refused when ``copy`` is an overflow alert.

    Other preflight refusals (a compaction threshold, a changed character, a
    continuation conflict) keep the ordinary recovery card.
    """
    if copy in _ALERTS:
        note_predispatch_block(message_id)


def is_repeat_of_last_row(store: Any, session_id: str, copy: str) -> bool:
    """Whether the session already ends with this exact refusal row.

    A retried turn the preflight refuses again for the same reason must not
    append a second copy of the row it is already showing.
    """
    try:
        messages = store.messages_for_session(session_id)
    except Exception:  # noqa: BLE001 -- an unknown session appends as before
        return False
    for message in reversed(tuple(messages)):
        role = getattr(getattr(message, "role", None), "value", None)
        if role == "assistant" and getattr(message, "status", "") == "failed":
            continue  # the refused turn's own empty assistant row
        return role == "system" and getattr(message, "content", None) == copy
    return False


def presented_owner(
    recovery: ConsoleDispatchRecoveryState | None,
) -> ConsoleDispatchRecoveryState | None:
    """Return ``recovery`` as the user should see it.

    Args:
        recovery: The raw dispatch owner, or ``None``.

    Returns:
        The not-sent presentation when this exact turn was refused before
        dispatch, otherwise ``recovery`` unchanged.
    """
    if recovery is None or recovery.assistant_message_id not in _NOTED:
        return recovery
    return present_predispatch_block(recovery)


def claim_offered_action(
    store: Any,
    session_id: str,
    recovery: ConsoleDispatchRecoveryState,
    action_id: ConsoleDispatchRecoveryActionId,
) -> tuple[ConsoleDispatchRecoveryState | None, str]:
    """Claim ``action_id`` only when the presented owner offers it enabled.

    The controller's Retry goes through here, so a turn refused before
    dispatch is refused for every caller, not only on its card (F13).

    Args:
        store: The Console chat store that owns the recovery.
        session_id: The owner's session.
        recovery: The raw dispatch owner.
        action_id: The recovery action to claim.

    Returns:
        ``(claimed owner, "")``, or ``(None, why)`` where ``why`` is the
        presented action's disabled reason, else a generic refusal.
    """
    shown = presented_owner(recovery) or recovery
    action = next((a for a in shown.actions if a.action_id is action_id), None)
    claimed = (
        store.claim_dispatch_recovery_action(session_id, action_id)
        if action is not None and action.enabled
        else None
    )
    if claimed is not None:
        return claimed, ""
    if action is not None and action.disabled_reason:
        return None, action.disabled_reason
    return None, "That response recovery action is unavailable."
