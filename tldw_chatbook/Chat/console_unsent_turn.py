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

Imported lazily by the runtime: it runs only when a paused turn is released.
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
    preparation = store.preparation_for_session(session_id)
    if (
        preparation is None
        or preparation.preparation_id != preparation_id
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
        or preparation.pause_kind is not ConsolePreparationPauseKind.RETRIEVAL
    ):
        return False
    draft = store.session_draft(session_id)
    controller.cancel_library_preparation(preparation_id)
    if store.session_draft(session_id) != draft:
        store.set_session_draft(session_id, draft)
    current = store.preparation_for_session(session_id)
    return current is None or current.preparation_id != preparation_id
