"""Land on the Console chat a provider recovery left for Settings (TASK-33008.1).

``open_provider_setup`` (``model_switcher.py``) stages a
``ConsoleProviderReturnIntent`` that names the active chat, and Settings
comes back with a ``ConsoleProviderReturnTarget``. This claims that exact
revision and returns to the chat: the composer takes focus, or the Get started
card's primary action while the card still blocks.

A NEEDS SETUP row in Switch model also names its provider and model in the
intent. Its return then reopens Switch model on them, saved or not
(TASK-33008.2); a return that does not land reopens nothing.

It applies nothing. A Providers & Models save sets the new-chat default, and
D1's pristine refresh moves an untouched chat onto it in the sync below; a chat
that holds work keeps its own settings (ADR-095).

A return the user does not take must not stay armed. Once Settings has the
screen, ``open_provider_setup`` leaves the staged ``(revision, intent)`` on
the Console. If the Console gets the screen back with no return (the user
walked back by hand), it discards exactly that revision, so Settings stops
offering a return to a chat the user is no longer coming from.

ADR-097: ``ChatScreen`` imports this only when a return arrives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..Navigation.conversation_settings_navigation import (
    ConsoleProviderReturnIntent,
    ConsoleProviderReturnTarget,
)
from ..Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffClaim,
    PendingHandoffStore,
)

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen

STALE_RETURN_COPY = "This return from Settings is no longer current. Nothing changed."
MISSING_CHAT_COPY = "The chat that opened Settings is closed. Nothing changed."


def capture_provider_return(screen: ChatScreen, context: Mapping[str, object]) -> bool:
    """Keep a Console return from Settings until the Console owns the screen.

    Args:
        screen: The Console screen Settings navigated to.
        context: The navigation context Settings sent.

    Returns:
        Whether the context was a Console return; False leaves it to the caller.
    """
    target = ConsoleProviderReturnTarget.from_context(context)
    if target is None:
        return False
    screen._pending_console_provider_return = target
    if screen.is_mounted:
        screen.call_after_refresh(consume_provider_return, screen)
    return True


def consume_provider_return(screen: ChatScreen) -> None:
    """Claim the kept return's exact revision and land on its chat.

    A superseded, settled or mismatched handoff shows a notice and applies
    nothing (AC#6); so does a chat that has closed. Until the Console owns the
    screen the return is kept, and the mount and resume consumers call again.

    When Settings sent no return, what is kept is the ``(revision, intent)``
    this Console staged: the user came back by hand, so that exact revision
    is discarded. A newer request of either kind is not touched.

    Args:
        screen: The Console screen holding the return.
    """
    target = getattr(screen, "_pending_console_provider_return", None)
    handoffs = getattr(screen.app_instance, "pending_handoffs", None)
    if (
        target is None
        or not isinstance(handoffs, PendingHandoffStore)
        or not screen.is_mounted
        or not screen._owns_console_screen_stack()
    ):
        return
    screen._pending_console_provider_return = None
    if not isinstance(target, ConsoleProviderReturnTarget):
        handoffs.discard_pending_exact(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, *target
        )
        return
    claim = handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    if (
        claim is None
        or claim.revision != target.return_revision
        or type(claim.value) is not ConsoleProviderReturnIntent
        # The same chat, and the same pair for Switch model to reopen on.
        or claim.value.return_target(claim.revision, target.outcome) != target
    ):
        if claim is not None:
            handoffs.release(claim)  # Another owner's newer request stays.
        screen._notify_conversation_settings_return(STALE_RETURN_COPY)
        return
    screen.run_worker(
        _land(screen, handoffs, claim),
        exclusive=False,
        group="console-provider-return",
        exit_on_error=False,  # A failed landing must not end the app.
    )


async def _land(
    screen: ChatScreen,
    handoffs: PendingHandoffStore,
    claim: HandoffClaim[ConsoleProviderReturnIntent],
) -> None:
    """Switch to the chat, sync it, and focus where the user continues.

    A NEEDS SETUP row's return then reopens Switch model over that focus, so
    Esc there leaves it on the composer, or on the card while it blocks. Not
    if the Console lost the screen during the sync: the list would open over
    whatever took it.
    """
    try:
        store = screen._ensure_console_chat_store()
        if store.active_session_id != claim.value.session_id:
            try:
                store.switch_session(claim.value.session_id)
            except KeyError:
                screen._notify_conversation_settings_return(MISSING_CHAT_COPY)
                return
        # The sync reads the config Settings just saved, so the focus below
        # sees whether the card still blocks.
        await screen._sync_native_console_chat_ui()
        if not screen._focus_console_setup_modal_if_blocking():
            screen._focus_console_composer_if_needed(force=True)
        if claim.value.switcher_provider and screen._owns_console_screen_stack():
            from .model_switcher import reopen_model_switcher

            reopen_model_switcher(
                screen, claim.value.switcher_provider, claim.value.switcher_model
            )
    finally:
        handoffs.acknowledge(claim)  # One landing; nothing is left to replay.
