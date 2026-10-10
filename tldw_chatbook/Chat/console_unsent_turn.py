"""Release the paused Library preparation an unsent turn leaves behind.

TASK-33621.20. When an Automatic Library search times out or fails, the send
pauses twice over: the store keeps a PAUSED (retrieval) preparation, and the
runtime keeps the unsent turn on the shelf with Restore and Discard. Those two
shelf actions used to act on the runtime's copy only. The paused preparation
stayed, and the store refuses every later send in a conversation that has
one: "Last send is blocked; resolve it first." -- after Discard, after a
policy switch to Never, for good.

The runtime now records which preparation an unsent turn paused, and calls
this when that turn leaves the shelf. Restore and Discard decide the draft:
cancelling the preparation would otherwise write the sent text back into the
session's draft, so the draft the shelf action left is put back.

The other way round, the paused send's card (Retry / Send once without
Library / Cancel) settles the send, and its shelf copy is then dropped.

Imported lazily: it runs only when a paused turn is released or settled.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)


def release_unsent_turn_preparation(
    controller: Any, session_id: str, preparation_id: str
) -> bool:
    """Cancel one exact retrieval-paused preparation, keeping the draft as is.

    Args:
        controller: The Console chat controller that owns the preparation.
        session_id: The unsent turn's session.
        preparation_id: The preparation the turn's send paused.

    Returns:
        True when that preparation was paused for retrieval and is now
        cancelled; False when it is gone, was resumed, or paused otherwise.
    """
    store = controller.store
    if not library_paused(store.preparation_for_session(session_id), preparation_id):
        return False
    draft = store.session_draft(session_id)
    controller.cancel_library_preparation(preparation_id)
    if store.session_draft(session_id) != draft:
        store.set_session_draft(session_id, draft)
    current = store.preparation_for_session(session_id)
    return current is None or current.preparation_id != preparation_id


def library_paused(preparation: Any, preparation_id: str) -> bool:
    """Whether ``preparation`` is exactly this send, paused for retrieval.

    Args:
        preparation: The session's current preparation, if any.
        preparation_id: The paused send's preparation id.

    Returns:
        True only for that preparation, PAUSED with ``RETRIEVAL``; False for
        anything else, including a projection that carries no state.
    """
    return (
        getattr(preparation, "preparation_id", None) == preparation_id
        and getattr(preparation, "state", None) is ConsoleTurnPreparationState.PAUSED
        and getattr(preparation, "pause_kind", None)
        is ConsolePreparationPauseKind.RETRIEVAL
    )


def settle_unsent_turn(runtime: Any, preparation_id: str) -> int:
    """Drop the shelf copy of a paused send its card has settled.

    Retry and Send once without Library send the paused turn; Cancel puts it
    back in the composer. Either way the unsent-turn shelf must stop offering
    Restore/Discard for it, or the same message could be restored twice.

    Args:
        runtime: The Console runtime owning the unsent-turn shelf, or None.
        preparation_id: The settled send's preparation id.

    Returns:
        How many shelf entries were dropped.
    """
    forget = getattr(runtime, "forget_turn_recoveries_for_preparation", None)
    return int(forget(preparation_id)) if callable(forget) else 0
