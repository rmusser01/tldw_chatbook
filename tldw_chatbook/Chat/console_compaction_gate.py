"""Stop a non-composer turn before commit at the Ask trigger (TASK-33621.4).

A composer send over the Ask compaction trigger is held before commit and
offered the hold card (TASK-34350). Continue, Regenerate and Retry cannot
show that card; on dev they committed an assistant row, marked it Failed and
explained afterwards. A deliberate policy hold never marks a turn Failed
(TASK-33621.4 AC#2), so these turns now stop before anything is committed,
with the numbers and the way forward; Edit & resend does the same. A queued
follow-up and a Persona Buddy reply cannot show it either; they are refused
before commit by ``refuse_without_card``: the queued entry returns to the
head of the queue, and Buddy keeps its text. A send with no preparation (a
"$" or "/" draft while exchange capture is off) has no card to show and is
refused by ``refuse_unprepared``; the composer keeps its text, and the
unsent-turn shelf lists it for Restore. An agent wake pauses its chain before
acceptance (``pause_wake_at_ask_trigger``, TASK-33621.80).

The controller imports this module at each call site, not at module scope:
it is needed only once a turn is sent, and the UI-ready module census
(ADR-097) does not budget for it or for its copy module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_context_budget_copy import (
    ask_trigger_refusal_copy,
    wake_ask_pause_copy,
)
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution


async def refuse_at_ask_trigger(
    controller: Any,
    session_id: str,
    resolution: Any,
    provider_messages: Sequence[dict[str, Any]],
    action: str,
    *,
    turn_skill_bindings: tuple[str, ...] = (),
) -> Any | None:
    """Refuse this turn before commit when the Ask policy would hold it.

    Args:
        controller: The Console chat controller.
        session_id: The session the turn belongs to.
        resolution: The turn's resolved provider.
        provider_messages: The turn's assembled request.
        action: What did not happen, e.g. ``"continued"``.
        turn_skill_bindings: Exact admitted skill identities for this request.

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
        turn_skill_bindings=turn_skill_bindings,
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


def refuse_without_card(
    controller: Any,
    preparation: Any,
    hold: Any,
    *,
    origin: Any,
    queue_entry_id: str | None,
    preserve_composer: bool,
) -> Any | None:
    """Refuse a queued or Buddy send at the Ask trigger, before commit.

    Cancelling the preparation removes the send's transient echo. A queued
    send's result is not accepted, so the queue returns the entry to its head
    and pauses on "Start refused"; Buddy keeps its draft and shows the copy.

    Args:
        controller: The Console chat controller.
        preparation: The send's uncommitted preparation.
        hold: The Ask hold the probe found, or None below the trigger.
        origin: The send's submission origin.
        queue_entry_id: The claimed queue entry, for a queued send.
        preserve_composer: True for a Buddy reply: the Console draft stays.

    Returns:
        The refusal to return, or None to go on.
    """

    if hold is None:
        return None
    session_id = preparation.session_id
    draft = controller.store.session_draft(session_id)
    controller._abandon_preparation(preparation.preparation_id)
    # Cancelling a manual-origin preparation puts its text in the composer;
    # a Buddy reply's text stays in Buddy, so give the composer its own back.
    if preserve_composer and controller.store.session_draft(session_id) != draft:
        controller.store.set_session_draft(session_id, draft)
    return _refuse_send(controller, session_id, hold, origin, queue_entry_id)


def refuse_unprepared(
    controller: Any,
    session_id: str,
    hold: Any,
    *,
    echoed_user: Any,
    origin: Any,
    queue_entry_id: str | None,
) -> Any | None:
    """Refuse a send that has no preparation at the Ask trigger, before commit.

    Without exchange capture a "$" or "/" draft gets no preparation, so it has
    no hold card. Its echo is dropped, the way cancelling a preparation drops
    it; the composer keeps its text, and the runtime lists the refused send
    as an unsent turn whose Restore puts it back once the composer is clear.

    Args:
        controller: The Console chat controller.
        session_id: The session the send belongs to.
        hold: The Ask hold the probe found, or None below the trigger.
        echoed_user: The send's uncommitted echo row, if any.
        origin: The send's submission origin.
        queue_entry_id: The claimed queue entry, for a queued send.

    Returns:
        The refusal to return, or None to go on.
    """

    if hold is None:
        return None
    if echoed_user is not None:
        controller.store.delete_message(echoed_user.id)
    return _refuse_send(controller, session_id, hold, origin, queue_entry_id)


async def pause_wake_at_ask_trigger(
    controller: Any,
    session: Any,
    authorization: Any,
    resolution: Any,
    turn_context: Any,
    notice: str,
) -> Any | None:
    """Pause an agent wake before acceptance when Ask would hold it.

    TASK-33621.80: a wake has no hold card and nobody waiting on it, so it
    neither compacts nor sends over the trigger on its own (ADR-052 Ask).
    Its chain pauses with its results kept (ADR-134) until the user decides:
    a compaction lifts the pause, as does leaving Ask; a message the user
    sends is their own turn. Nothing is committed and no turn is Failed.

    Args:
        controller: The Console chat controller.
        session: The session the wake would run in.
        authorization: The wake's live, not yet accepted, authority.
        resolution: The wake's resolved provider.
        turn_context: The wake's frozen turn configuration.
        notice: The wake notice the request would end with.

    Returns:
        The blocked result to return, or None to go on (below the trigger,
        a probe that could not run, or an authority that lapsed meanwhile,
        which acceptance then refuses as it refuses any lapsed wake).
    """

    if not isinstance(resolution, ConsoleProviderResolution):
        return None
    try:
        if controller._compaction_mode_for(session) is not ContextCompactionMode.ASK:
            return None
        # The request a wake sends: the chat's history, then its notice as a
        # trailing user-role entry (see the wake branch of the submit path).
        messages = [
            *controller._provider_messages_for_session(
                session.id, annotate_ids=True, turn_context=turn_context
            ),
            {"role": ConsoleMessageRole.USER.value, "content": notice},
        ]
    except Exception as exc:  # noqa: BLE001 - fail open, as the composer probe does
        logger.warning(
            "Wake compaction hold probe unavailable; exception_type={}",
            type(exc).__name__,
        )
        return None
    hold, _alert = await controller._assess_context_compaction(
        session_id=session.id,
        resolution=resolution,
        provider_messages=messages,
        uncommitted_user_message_id=None,
    )
    if hold is None:
        return None
    if not await controller._fleet_wake.pause_at_ask_trigger(authorization):
        return None
    return controller._block(
        session.id,
        wake_ask_pause_copy(
            used_tokens=hold.used_tokens,
            budget_tokens=hold.budget_tokens,
            trigger_tokens=hold.trigger_tokens,
            estimated=hold.estimated,
        ),
    )


def _refuse_send(
    controller: Any,
    session_id: str,
    hold: Any,
    origin: Any,
    queue_entry_id: str | None,
) -> Any:
    """Block the send with the refusal copy and keep its queue identity."""

    blocked = controller._block(
        session_id,
        ask_trigger_refusal_copy(
            action=(
                "sent from the queue"
                if origin is ConsoleSubmissionOrigin.QUEUED
                else "sent"
            ),
            used_tokens=hold.used_tokens,
            budget_tokens=hold.budget_tokens,
            trigger_tokens=hold.trigger_tokens,
            estimated=hold.estimated,
        ),
    )
    return replace(
        blocked, session_id=session_id, origin=origin, queue_entry_id=queue_entry_id
    )
