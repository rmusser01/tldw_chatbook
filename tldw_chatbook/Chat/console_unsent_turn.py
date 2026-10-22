"""Keep a paused send's two surfaces -- shelf and card -- from disagreeing.

TASK-33621.20. When an Automatic Library search times out, fails or is
stopped, the send pauses twice over: the store keeps a PAUSED preparation, and
the runtime keeps the unsent turn on the shelf with Restore and Discard. The
transcript card offers Retry / Send once without Library / Cancel for the same
send.

Shelf to card: Restore and Discard used to act on the runtime's copy only.
The paused preparation stayed, and the store refused every later send in the
conversation ("Last send is blocked; resolve it first.") -- after Discard,
after a policy switch to Never, for good. They now release the exact paused
preparation. Cancelling it writes the sent text back into the stored draft, so
the draft the shelf action left is put back.

Card to shelf (pre-merge review): while a card action sends the turn, the
shelf must not offer it -- Restore would send it twice, and Discard would not
stop it. The card holds the turn off the shelf for the action, keeps holding
it if the send pauses again (the card covers every re-pause), drops it once
the send is accepted or Cancel has put it back in the composer with its
attachments, and offers it again otherwise.

Imported lazily: it runs only when a paused turn is released or acted on.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Chat.console_turn_preparation import (
    SHELF_RELEASABLE_PAUSES,
    ConsoleTurnPreparationState,
)


def shelf_releasable(preparation: Any, preparation_id: str) -> bool:
    """Whether ``preparation`` is exactly this send, in a pause the shelf ends.

    Args:
        preparation: The session's current preparation, if any.
        preparation_id: The paused send's preparation id.

    Returns:
        True only for that preparation, PAUSED for retrieval, a destination
        that is not ready, or persistence; False for anything else, including
        a projection that carries no state.
    """
    return (
        getattr(preparation, "preparation_id", None) == preparation_id
        and getattr(preparation, "state", None) is ConsoleTurnPreparationState.PAUSED
        and getattr(preparation, "pause_kind", None) in SHELF_RELEASABLE_PAUSES
    )


def release_unsent_turn_preparation(
    controller: Any, session_id: str, preparation_id: str
) -> bool:
    """Cancel one exact shelf-releasable paused preparation, keeping the draft.

    Args:
        controller: The Console chat controller that owns the preparation.
        session_id: The unsent turn's session.
        preparation_id: The preparation the turn's send paused.

    Returns:
        True when that preparation was paused in a shelf-releasable pause and
        is now cancelled; False when it is gone, live, or paused otherwise.
    """
    store = controller.store
    if not shelf_releasable(store.preparation_for_session(session_id), preparation_id):
        return False
    draft = store.session_draft(session_id)
    try:
        controller.cancel_library_preparation(preparation_id)
    finally:
        # Cancelling restores the sent text as the draft; a discarded (or
        # already restored) turn must not come back that way -- not even when
        # the cancel then fails, which propagates so the shelf keeps the turn.
        if store.session_draft(session_id) != draft:
            store.set_session_draft(session_id, draft)
    current = store.preparation_for_session(session_id)
    return current is None or current.preparation_id != preparation_id


def own_draft_takes_it_back(store: Any, session_id: str, sent_text: str) -> bool:
    """Whether a cancelled send may go back into its own (unshown) draft.

    TASK-33621.20: a card Cancel that finishes after a tab switch returns the
    message to its own conversation's stored draft instead of the composer.
    Like the composer refill, that must never replace a follow-up typed there
    meanwhile; the store's cancel keeps such a draft, so it no longer matches.

    Args:
        store: The Console chat store.
        session_id: The cancelled send's session.
        sent_text: The text the send carried.

    Returns:
        True when that draft is empty or is the sent text put back.
    """
    try:
        draft = store.session_draft(session_id).strip()
    except KeyError:  # the conversation is gone
        return False
    return draft in {"", sent_text.strip()}


def card_action_started(runtime: Any, preparation_id: str) -> None:
    """Take the paused send's unsent turn off the shelf while its card acts.

    Args:
        runtime: The Console runtime owning the unsent-turn shelf, or None.
        preparation_id: The paused send's preparation id.
    """
    hold = getattr(runtime, "hold_turn_recoveries_for_preparation", None)
    if callable(hold):
        hold(preparation_id)


def card_action_finished(
    runtime: Any,
    controller: Any,
    preparation_id: str,
    *,
    accepted: bool,
    return_to_composer: bool,
    interrupted: bool = False,
) -> Any:
    """Settle the shelf after a card action on a paused send.

    Args:
        runtime: The Console runtime owning the unsent-turn shelf, or None.
        controller: The Console chat controller.
        preparation_id: The paused send's preparation id.
        accepted: The action sent the turn.
        return_to_composer: The action was a Cancel that took effect and the
            composer is empty -- or, after a tab switch, the turn's own draft
            is free (`own_draft_takes_it_back`) -- so the turn goes back there.
        interrupted: The action raised or was cancelled -- opening the
            Console terminal removes the card mid-action -- so no card is
            known to cover a re-pause and the shelf offers the turn again.

    Returns:
        The unsent turn put back as the draft (its ``draft`` is the text to
        load), or None.
    """
    if runtime is None:
        return None
    if accepted:
        runtime.forget_turn_recoveries_for_preparation(preparation_id)
        return None
    store = controller.store
    if not interrupted and shelf_releasable(
        store.preparation_by_id(preparation_id), preparation_id
    ):
        return None  # paused again: the card covers it, the shelf stays clear
    if return_to_composer:
        return runtime.return_turn_recovery_to_draft(preparation_id)
    runtime.unhold_turn_recoveries_for_preparation(preparation_id)
    return None
