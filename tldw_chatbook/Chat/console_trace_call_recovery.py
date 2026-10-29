"""Trace-blocked send recovery that is not an unknown delivery (TASK-33621.6).

A send paused on the "Trace capture blocked" card never sent the call that
failed: its trace reservation failed before the provider adapter. These
helpers keep that fact when the user acts on the card, instead of
escalating to the unknown-delivery panel ("Retry anyway may send a
duplicate request") that is meant for a request which may have reached the
provider without a known outcome.

Loaded on first use, never at UI-ready (ADR-097 boot ratchet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tldw_chatbook.Chat.console_chat_models import ConsoleRunState
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleChatController,
        ConsoleSubmitResult,
    )


@dataclass(frozen=True, slots=True)
class TraceCallProviderContact:
    """Whether a paused send reached the provider, for the card (TASK-33621.5).

    Attributes:
        contacted: "yes" when a call of this send got a provider outcome,
            "no" when none crossed the dispatch checkpoint, else "unknown".
        rejection: The provider's last definitive HTTP refusal, if any (a
            ``TraceProviderRejection``: provider, status_code, meaning).
    """

    contacted: Literal["yes", "no", "unknown"]
    rejection: object | None = None


def provider_contact_for_pause(
    gateway: object, signals: object, *, dispatch_started: bool
) -> TraceCallProviderContact:
    """Derive the card's provider-contacted fact for a send that just paused.

    Args:
        gateway: The provider gateway that ran the send.
        signals: The send's stream signals.
        dispatch_started: Whether a call of this send crossed the durable
            dispatch checkpoint, which immediately precedes adapter entry.

    Returns:
        The recorded fact; never "yes" without a provider outcome.
    """
    if not dispatch_started:
        return TraceCallProviderContact("no")
    contact_for = getattr(gateway, "trace_provider_contact", None)
    contact = contact_for(signals) if callable(contact_for) else None
    if contact is None or not contact[0]:
        return TraceCallProviderContact("unknown")
    return TraceCallProviderContact("yes", contact[1])


#: The card's status line when Capture On cannot be retried.
UNCONFIRMED_RESERVATION_COPY = (
    "Trace capture could not be confirmed for this send, so it was not "
    "retried. Send without capture, or Cancel send."
)


def repause_unconfirmed_trace_call(
    controller: ConsoleChatController,
    preparation_id: str,
) -> ConsoleSubmitResult | None:
    """Put a Capture-On retry whose reservation is unknown back on its card.

    Args:
        controller: The Console chat controller resuming the send.
        preparation_id: The send Retry capture resumed.

    Returns:
        The refusal the card shows, or ``None`` when the send is no longer
        the accepted, unsent preparation (the caller then fails closed).
    """
    current = controller._preparation_by_id(preparation_id)
    if current is None or current.state is not ConsoleTurnPreparationState.ACCEPTED:
        return None
    paused = controller.store.compare_and_set_preparation(
        current.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation_id,
            expected_state=ConsoleTurnPreparationState.ACCEPTED,
            new_state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_CALL,
            new_attempt_id=None,
        ),
    )
    if paused is None:
        return None
    controller._set_run_state(
        ConsoleRunState.blocked(UNCONFIRMED_RESERVATION_COPY),
        session_id=current.session_id,
    )
    return controller._prepared_action_refusal(paused, UNCONFIRMED_RESERVATION_COPY)


__all__ = [
    "UNCONFIRMED_RESERVATION_COPY",
    "TraceCallProviderContact",
    "provider_contact_for_pause",
    "repause_unconfirmed_trace_call",
]
