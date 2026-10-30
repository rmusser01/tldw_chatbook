"""Stop a non-composer turn before commit at the Ask trigger (TASK-33621.4).

A composer send over the Ask compaction trigger is held before commit and
offered the hold card (TASK-34350). Continue, Regenerate and Retry cannot
show that card; on dev they committed an assistant row, marked it Failed and
explained afterwards. A deliberate policy hold never marks a turn Failed
(TASK-33621.4 AC#2), so these turns now stop before anything is committed,
with the numbers and the way forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tldw_chatbook.Chat.console_context_budget_copy import ask_trigger_refusal_copy
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode


async def refuse_at_ask_trigger(
    controller: Any,
    session_id: str,
    resolution: Any,
    provider_messages: Sequence[dict[str, Any]],
    action: str,
) -> Any | None:
    """Refuse this turn before commit when the Ask policy would hold it.

    Args:
        controller: The Console chat controller.
        session_id: The session the turn belongs to.
        resolution: The turn's resolved provider.
        provider_messages: The turn's assembled request.
        action: What did not happen, e.g. ``"continued"``.

    Returns:
        The blocked result to return, or None to go on.
    """

    session = next(
        (row for row in controller.store.sessions() if row.id == session_id), None
    )
    if (
        session is None
        or controller._compaction_mode_for(session) is not ContextCompactionMode.ASK
    ):
        return None
    hold, _alert = await controller._assess_context_compaction(
        session_id=session_id,
        resolution=resolution,
        provider_messages=list(provider_messages),
        uncommitted_user_message_id=None,
    )
    if hold is None:
        return None
    return controller._block(
        session_id,
        ask_trigger_refusal_copy(
            action=action,
            used_tokens=hold.used_tokens,
            budget_tokens=hold.budget_tokens,
            trigger_tokens=hold.trigger_tokens,
            estimated=hold.estimated,
        ),
    )
