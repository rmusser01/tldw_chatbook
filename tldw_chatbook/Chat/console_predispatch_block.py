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
"""

from __future__ import annotations

from dataclasses import replace

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
