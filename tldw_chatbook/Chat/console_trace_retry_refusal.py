"""Keep a refused Retry capture at its trace pause (TASK-34100.3 review G3V2-03).

When a Console send's trace capture fails before any call boundary exists (the
planner refused the request, as on the third send of a saved Kimi chat), the
controller pauses the send at the "Trace capture blocked" card and records the
failure itself in place of a boundary. Retry capture then has nothing it can
verify: ``_verify_trace_preparation_recovery`` needs a boundary owned by this
send, so it refuses again. The controller used to treat that refusal as an
unknown delivery outcome and showed the "delivery status is unknown ... Retry
anyway may send a duplicate request" card with Send blocked, although neither
attempt had built a call, let alone reached the provider.

This module returns that send to the same trace pause instead. It acts only on
exactly that recorded shape, so a reservation whose dispatch outcome really is
unknown still goes to response recovery.

The controller module is pinned near its size budget, so the decision lives
here and the controller calls it from its one refusal branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
    pause_for_trace_call_failure,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids an import cycle
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleChatController,
        ConsoleSubmitResult,
    )

#: What the card's status line says after the refused retry.
REFUSED_RETRY_COPY = (
    "Trace capture was refused again; the provider was not contacted. "
    "Send without capture, or cancel the send."
)


def repause_refused_trace_retry(
    controller: ConsoleChatController,
    preparation_id: str,
    continuation: Any,
) -> ConsoleSubmitResult | None:
    """Return the send to its trace pause when a retry was refused unbuilt.

    Args:
        controller: The Console controller that owns the paused send.
        preparation_id: The send's preparation, just moved from PAUSED to
            ACCEPTED by the Retry capture action.
        continuation: The send's durable postcommit continuation.

    Returns:
        The paused submit result, or None when the recorded pause evidence is
        anything other than "no call boundary was built" (the caller then
        keeps its unknown-delivery handling).
    """
    recorded = controller._trace_call_boundaries_by_preparation.get(preparation_id)
    if (
        not isinstance(recorded, TraceCallPersistenceError)
        or recorded.boundary is not None
        or continuation.origin is not ConsoleSubmissionOrigin.MANUAL
    ):
        return None
    current = controller._preparation_by_id(preparation_id)
    if current is None or current.state not in {
        ConsoleTurnPreparationState.ACCEPTED,
        ConsoleTurnPreparationState.DISPATCH_STARTED,
    }:
        return None
    paused_shape = pause_for_trace_call_failure(current, recorded)
    if paused_shape.state is not ConsoleTurnPreparationState.PAUSED:
        return None  # not a legal pause from here; leave it to the caller
    paused = controller.store.compare_and_set_preparation(
        current.session_id,
        ConsolePreparationTransition(
            preparation_id=current.preparation_id,
            expected_state=current.state,
            new_state=paused_shape.state,
            pause_kind=paused_shape.pause_kind,
            new_attempt_id=None,
        ),
    )
    if paused is None:
        return None
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult

    session_id = continuation.session_id
    commit = continuation.commit
    controller._set_run_state(
        ConsoleRunState.blocked(REFUSED_RETRY_COPY), session_id=session_id
    )
    return ConsoleSubmitResult(
        True,
        True,
        REFUSED_RETRY_COPY,
        session_id=session_id,
        user_message_id=commit.user_message_id,
        assistant_message_id=commit.assistant_message_id,
        terminal_status=ConsoleRunStatus.BLOCKED,
        origin=continuation.origin,
        queue_entry_id=continuation.queue_entry_id,
        committed_context_epoch=continuation.committed_context_epoch,
        preparation_id=preparation_id,
        # Mirrors the original pause: the send's checkpoint transition is
        # complete, though no call of this send reached the provider.
        provider_started=True,
    )
