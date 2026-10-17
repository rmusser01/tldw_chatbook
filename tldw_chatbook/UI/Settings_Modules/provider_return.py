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

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...Chat.provider_readiness import provider_config_key
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
#: Where focus goes with no field and no return: the first shown and enabled.
_SIGN_IN_WITH = "#settings-provider-auth-source"
_EDIT_ENDPOINT = "#settings-provider-edit-custom-endpoint"
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
#: The Key check word for a cloud key the provider's listing accepted.
VERIFIED_PREFIX = "Ready · verified"
#: A Console return's credential line once the Key check reads verified.
KEY_ACCEPTED_COPY = "the provider accepted the key."


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


def routed_model(screen: SettingsScreen, values: Mapping[str, object]) -> str:
    """The model shown beside a routed provider before any edit.

    The route's own model when it names or pins one (Chat settings pins even
    none), else Settings' own. One route had no right answer in that rule: a route
    that names no model to a provider other than the one Settings holds.
    Settings' model is another provider's there, and the first edit pinned it
    beside the routed provider, so a saved key made "<provider> · <another
    provider's model>" the default for new chats.

    Such a route is resolved here, once, to the routed provider's configured
    model or none, as choosing that provider in the Provider control does. It
    is then held as the route's own model: the page's echo guards compare
    what a mounted field posts with the routed model, and any other shown
    model would be staged as an edit the user never made.

    Args:
        screen: The Settings screen a provider route is in effect on.
        values: Settings' own provider values (no draft exists yet).

    Returns:
        The model id, or "" for none.
    """
    provider = str(screen._navigation_provider or "")
    held = str(values.get("provider") or "")
    if (
        not screen._navigation_model_is_explicit
        and not screen._navigation_model
        and provider_config_key(provider) != provider_config_key(held)
    ):
        screen._navigation_model = screen._provider_model_default(provider)
        screen._navigation_model_is_explicit = True
    model = str(screen._navigation_model or "").strip()
    if not model and not screen._navigation_model_is_explicit:
        model = str(values.get("model") or "").strip()
    return model


def route_focus(screen: SettingsScreen, selector: str | None) -> str | None:
    """Where focus goes when Settings opens on a provider route.

    The field the route names, while it can take focus. Two routes have no
    such field: one that names none (a cloud Retry), and one whose field is
    disabled (the API key while a Claude subscription signs in; the API key,
    Endpoint or Default model of a named custom endpoint, which Settings
    locks). The Default model field is disabled through the picker it sits
    in, which is why ``is_disabled`` is read. Both arrive on the
    return when it is offered, so one Enter goes back. A Console return that
    an unsaved edit has withdrawn (one kept from an earlier visit) gives its
    landing to Return without saving, which asks before it discards. Chat
    settings' return is not offered on arrival and has no landing to give.

    With no return offered, focus goes to the control that acts instead, the
    first of these that is shown and enabled. Sign in with, which says why no
    key is asked for, stands in for a disabled field only: it is enabled on
    every Anthropic route, and a route that names no field has nothing to do
    with it. Then Edit this endpoint in Custom endpoints, where a custom
    endpoint's key, Endpoint and models are changed. No return can name a
    custom endpoint, so that button is where its routes land.

    A custom endpoint id whose entry is gone has neither control, because the
    Edit button is hidden for it. It lands on the Provider control, where
    another provider is chosen: the search box, not the Select behind it,
    which sits in a hidden row and would take focus unseen. Any other route
    with no field and no return (a one-way route to a built-in provider)
    leaves focus where it is.

    Args:
        screen: The Settings screen holding the return target and its outcome.
        selector: The named field's selector, or None when the route names none.

    Returns:
        The selector to focus, or None to leave focus where it is.
    """
    disabled = bool(selector) and any(w.is_disabled for w in screen.query(selector))
    if selector and not disabled:
        return selector
    if return_outcome(screen) is not None:
        return "#settings-provider-return"
    if (
        can_return_without_saving(screen)
        and screen._provider_return_target.session_id is not None
    ):
        return "#settings-provider-return-without-save"
    for landing in (_SIGN_IN_WITH, _EDIT_ENDPOINT) if disabled else (_EDIT_ENDPOINT,):
        # display, not focusable: the Edit button is hidden through display.
        if any(w.display and not w.disabled for w in screen.query(landing)):
            return landing
    if screen._provider_is_registry_id(screen._provider_widget_value()):
        return "#settings-provider-search"
    return None


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


def discard_question(screen: SettingsScreen) -> str:
    """The question Return without saving asks before it discards.

    Discard changes also goes back, so the question names where to.

    Args:
        screen: The Settings screen holding the return target.

    Returns:
        The confirmation's message.
    """
    return (
        "Discard all unsaved changes to Providers & Models and return to "
        f"{screen._provider_return_destination()}?"
    )


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


def _new_chats_copy(screen: SettingsScreen) -> str:
    """Which pair new chats use, after a save that leaves them on this page's
    provider; else "".

    The owner's decision of 2026-10-17 (PR #3144, D5): a save of a provider's
    key, Env var, endpoint or model also makes it, with the model the page
    shows, the default for new chats (``build_provider_setup_mutation``),
    while Switch model, reopened by the return, says "this chat" of its own
    pick. The save is not changed; its line says what it did. A save that
    writes no connection (a model default of another provider) leaves new
    chats where they were, and says nothing of them.
    """
    from ...Chat.console_session_settings import blank_console_session_settings
    from .providers_models_card import provider_model_pair  # the card imports this

    chat = blank_console_session_settings(screen._app_config_mapping())
    page = provider_config_key(screen._provider_widget_value())
    if provider_config_key(chat.provider) != page:
        return ""
    return f" New chats use {provider_model_pair(screen, chat.provider, chat.model)}."


def continuation_copy(screen: SettingsScreen) -> str:
    """The status above the return action.

    Args:
        screen: The Settings screen holding the return target and its outcome.

    Returns:
        After a save: what was saved, for a Console return which pair new
        chats now use (``_new_chats_copy``), where the return goes and what
        is still unverified; for a Console return, once the Key check of the
        recovery's provider reads verified, that the provider accepted the
        key. For a Console return with nothing
        saved: that nothing is unsaved and where the return goes. A Console
        return names the chat while that chat is open.
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
    unverified = saved[1]
    if screen._provider_return_outcome is Outcome.CREDENTIAL_SAVED:
        # Lazy: the card module imports this one.
        from .providers_models_card import key_check_verdict

        # A key check after the save (t) answers what the save could not, for
        # the recovery's provider only: the check reads the form's provider.
        if key_check_verdict(screen).startswith(VERIFIED_PREFIX) and (
            provider_config_key(screen._provider_widget_value()) == target.provider
        ):
            unverified = KEY_ACCEPTED_COPY
    return f"{saved[0]}{_new_chats_copy(screen)} Return to Console{chat}; {unverified}"
