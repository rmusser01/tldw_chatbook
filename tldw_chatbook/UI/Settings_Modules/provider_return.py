"""Copy for the Providers & Models return continuation (TASK-33008.1).

Settings is opened with a way back from two places: the Chat settings modal's
Configure credential, and a Console provider recovery. The navigation target
says which (its ``session_id`` names the Console chat), and this module words
the return for either. The modal's strings are unchanged.

The chat's title is read from the live Console store when the copy is drawn;
it is never part of the handoff or of a navigation context.

``settings_screen`` imports this inside the functions that need it (ADR-097).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..Navigation.conversation_settings_navigation import (
    ConversationSettingsReturnOutcome as Outcome,
)

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen

#: A longer chat title is cut so the copy it sits in stays on one line.
CHAT_TITLE_LIMIT = 24
INACTIVE_COPY = "Provider settings handoff is inactive."
#: outcome -> (what was saved, what the save does not prove).
_SAVED_COPY = {
    Outcome.CREDENTIAL_SAVED: (
        "Credential saved.",
        "provider acceptance is not yet verified.",
    ),
    Outcome.PROVIDER_SETTINGS_SAVED: (
        "Provider settings saved.",
        "generation is not yet verified.",
    ),
}


def fitted_chat_title(title: str) -> str:
    """A chat's title as Settings copy names it: stripped, never blank, cut.

    Args:
        title: The Console session's title.

    Returns:
        The title, "untitled" when blank, cut to ``CHAT_TITLE_LIMIT`` characters.
    """
    title = title.strip() or "untitled"
    if len(title) > CHAT_TITLE_LIMIT:
        title = title[: CHAT_TITLE_LIMIT - 1].rstrip() + "…"
    return title


def _console_chat_title(screen: SettingsScreen, session_id: str) -> str | None:
    """The open Console chat's title, or None when that chat is gone."""
    from ...Chat.console_chat_store import ConsoleChatStore

    store = screen._custom_endpoints_store()
    if not isinstance(store, ConsoleChatStore):
        return None
    session = next((s for s in store.sessions() if s.id == session_id), None)
    return None if session is None else fitted_chat_title(session.title)


def continuation_copy(screen: SettingsScreen) -> str:
    """The status above the return action after a save.

    Args:
        screen: The Settings screen holding the return target and its outcome.

    Returns:
        What was saved, where the return goes and what is still unverified.
        A Console return names the chat while that chat is open.
    """
    saved = _SAVED_COPY.get(screen._provider_return_outcome)
    if saved is None:
        return INACTIVE_COPY
    target = screen._provider_return_target
    if target is None or target.session_id is None:
        return f"{saved[0]} Return to Chat settings to check readiness; {saved[1]}"
    title = _console_chat_title(screen, target.session_id)
    chat = f" to continue in “{title}”" if title else ""
    return f"{saved[0]} Return to Console{chat}; {saved[1]}"
