"""The Providers & Models return continuation: when it shows, what it says.

Settings is opened with a way back from two places: the Chat settings modal's
Configure credential, and a Console provider recovery (TASK-33008.1). The
navigation target says which (its ``session_id`` names the Console chat).

Chat settings' return shows after a save, with its strings unchanged. A
Console recovery's return also shows on arrival, because a recovery that
saves nothing (a key test) must not leave the user to find the chat by hand.
A Console return the user did not take lapses: the Console takes it back when
it is reached by hand, and the next Settings visit drops it.

The chat's title is read from the live Console store when the copy is drawn;
it is never part of the handoff or of a navigation context.

``settings_screen`` imports this inside the functions that need it (ADR-097).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..Navigation.conversation_settings_navigation import (
    ConversationSettingsReturnOutcome as Outcome,
)
from ..Navigation.pending_handoff_store import HandoffChannel, PendingHandoffStore
from ..Screens.settings_config_models import SettingsCategoryId

if TYPE_CHECKING:
    from ..Navigation.conversation_settings_navigation import (
        ProviderSettingsNavigationTarget,
    )
    from ..Screens.settings_screen import SettingsScreen

#: A longer chat title is cut so the copy it sits in stays on one line.
CHAT_TITLE_LIMIT = 24
INACTIVE_COPY = "Provider settings handoff is inactive."
#: A Console return offered with nothing saved: going back loses nothing.
ARRIVAL_COPY = "No unsaved changes."
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


def return_outcome(screen: SettingsScreen) -> Outcome | None:
    """What the return action sends now, or None while it is not offered.

    After a save it is that save's outcome, for either kind. A Console
    recovery's return is also offered with nothing saved, so a recovery that
    needs no save can still go back in one action. It is withdrawn while
    unsaved edits or a provider conflict need an answer first: Return without
    saving and the conflict panel own those.

    Args:
        screen: The Settings screen holding the return target and its outcome.

    Returns:
        The save's outcome; ``WITHOUT_SAVING`` for a Console return with
        nothing to save; None for no return, or Chat settings' before a save.
    """
    outcome = screen._provider_return_outcome
    target = screen._provider_return_target
    if (
        outcome is not None
        or target is None
        or target.session_id is None
        or screen._provider_navigation_conflict_target is not None
        or screen._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS)
    ):
        return outcome
    return Outcome.WITHOUT_SAVING


def route_focus(screen: SettingsScreen, selector: str | None) -> str | None:
    """Where focus goes when Settings opens on a provider route.

    The field the route names, while it can take focus. Two routes have no
    such field: one that names none (a cloud Retry), and one whose field is
    disabled (the API key while a Claude subscription signs in). Both arrive
    on the return when it is offered, so one Enter goes back. A disabled field
    with no return offered gives way to Sign in with, the control that says
    why the key is not asked for.

    A registry-locked provider's key is disabled too, but it has neither
    landing: no return can name it, and Sign in with is Anthropic's alone and
    disabled for it. Its focus stays where it was, as before.

    Args:
        screen: The Settings screen holding the return target and its outcome.
        selector: The named field's selector, or None when the route names none.

    Returns:
        The selector to focus, or None to leave focus where it is.
    """
    disabled = bool(selector) and any(w.disabled for w in screen.query(selector))
    if selector and not disabled:
        return selector
    if return_outcome(screen) is not None:
        return "#settings-provider-return"
    return "#settings-provider-auth-source" if disabled else None


def can_return_without_saving(screen: SettingsScreen) -> bool:
    """Whether Return without saving is offered.

    Only before a save, and never beside the conflict panel. Chat settings'
    return needs a labelled edit to its own provider, as before. A Console
    recovery's is offered for every unsaved Providers & Models edit (another
    provider picked, a snapshot preference), so whenever an unsaved edit makes
    ``return_outcome`` withdraw the return, this one takes over: its dialog
    discards the whole category.

    Args:
        screen: The Settings screen holding the return target and its outcome.

    Returns:
        True while Return without saving shows.
    """
    target = screen._provider_return_target
    if (
        target is None
        or screen._provider_return_outcome is not None
        or screen._provider_navigation_conflict_target is not None
    ):
        return False
    if target.session_id is None:
        return screen._provider_same_target_has_draft()
    return screen._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS)


def console_return_lapsed(
    screen: SettingsScreen, target: ProviderSettingsNavigationTarget
) -> bool:
    """Whether a Console return kept from an earlier visit lost its handoff.

    The Console takes an unused return back when the user reaches it by hand,
    and a newer request replaces it. Kept, the return would name a chat the
    user is no longer coming from. Chat settings' return is not checked.

    Args:
        screen: The Settings screen restoring its kept return.
        target: The kept return target.

    Returns:
        True for a Console return whose exact revision is no longer pending.
    """
    if target.session_id is None:
        return False
    handoffs = getattr(screen.app_instance, "pending_handoffs", None)
    return not (
        isinstance(handoffs, PendingHandoffStore)
        and handoffs.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, target.return_revision
        )
        == "pending"
    )


def continuation_copy(screen: SettingsScreen) -> str:
    """The status above the return action.

    Args:
        screen: The Settings screen holding the return target and its outcome.

    Returns:
        After a save: what was saved, where the return goes and what is still
        unverified. For a Console return with nothing saved: that nothing is
        unsaved and where the return goes. A Console return names the chat
        while that chat is open.
    """
    target = screen._provider_return_target
    saved = _SAVED_COPY.get(screen._provider_return_outcome)
    if target is None or target.session_id is None:
        if saved is None:
            return INACTIVE_COPY
        return f"{saved[0]} Return to Chat settings to check readiness; {saved[1]}"
    if saved is None and return_outcome(screen) is None:
        return INACTIVE_COPY
    title = _console_chat_title(screen, target.session_id)
    chat = f" to continue in “{title}”" if title else ""
    if saved is None:
        return f"{ARRIVAL_COPY} Return to Console{chat}."
    return f"{saved[0]} Return to Console{chat}; {saved[1]}"
