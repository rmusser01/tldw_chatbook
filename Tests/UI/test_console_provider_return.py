"""TASK-33008.1: a Console provider recovery returns to the chat it left.

The unit tests drive the Console's consumer (``Console_Modules/
provider_return.py``) and the Settings offer and copy (``Settings_Modules/
provider_return.py``) with the real handoff store and the real Console chat
store. The private-profile tests run the real app: card -> Settings -> key ->
save -> Enter, the saved model an untouched chat comes back on, Return without
saving in every unsaved state, the return offered on arrival, where focus
arrives when the route names no field or a disabled one, where it goes after
Keep editing, a return that lapses when the user walks back by hand, and edits
made after the save.

TASK-33008.2: a NEEDS SETUP row in Switch model leaves through the same route,
and its return reopens Switch model on that row's provider. The unit tests
drive the consumer's reopen; the private-profile tests run the row -> Settings
-> Switch model round trip on the real app, saved and unsaved, and from the
Get started card. One more runs the real Settings screen on a routed provider
that is not the saved default, from that row and from Chat settings: what the
page shows while its key is typed, and what the save writes.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app as _build_production_app
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.custom_endpoint_registry import canonical_custom_endpoint_id
from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
from tldw_chatbook.LLM_Calls import anthropic_subscription
from tldw_chatbook.LLM_Provider_Catalog import (
    openai_compatible_model_discovery as discovery_module,
)
from tldw_chatbook.UI.Console_Modules import model_switcher
from tldw_chatbook.UI.Console_Modules.model_switcher import open_provider_setup
from tldw_chatbook.UI.Console_Modules.provider_return import (
    MISSING_CHAT_COPY,
    STALE_RETURN_COPY,
    capture_provider_return,
    consume_provider_return,
)
from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
    ConsoleProviderReturnIntent,
    ConversationSettingsReturnIntent,
    ConversationSettingsReturnOutcome,
    ProviderSettingsNavigationTarget,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Settings_Modules.provider_return import (
    INACTIVE_COPY,
    can_return_without_saving,
    console_return_lapsed,
    continuation_copy,
    fitted_chat_title,
    return_outcome,
    route_focus,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker

CHANNEL = HandoffChannel.CONVERSATION_SETTINGS_RETURN
SAVED = ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
UNSAVED = ConversationSettingsReturnOutcome.WITHOUT_SAVING


def _console(store: ConsoleChatStore, *, blocking: bool = False, owns: bool = True):
    """A Console stand-in over the real stores, recording what a return does."""
    events: list[object] = []
    workers: list[object] = []
    worker_options: list[dict[str, object]] = []

    async def sync() -> None:
        events.append("sync")

    def run_worker(coroutine, **options) -> None:
        workers.append(coroutine)
        worker_options.append(options)

    def focus_card() -> bool:
        if blocking:
            events.append("card")
        return blocking

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(pending_handoffs=PendingHandoffStore()),
        is_mounted=True,
        _pending_console_provider_return=None,
        _conversation_settings_return_restore_in_progress=True,
        _owns_console_screen_stack=lambda: owns,
        _ensure_console_chat_store=lambda: store,
        _sync_native_console_chat_ui=sync,
        _focus_console_setup_modal_if_blocking=focus_card,
        _console_setup_modal_blocking=lambda: blocking,
        _provider_readiness_app_config=dict,
        _focus_console_composer_if_needed=lambda *, force=False: events.append(
            ("composer", force)
        ),
        _notify_conversation_settings_return=lambda message: events.append(
            ("notice", message)
        ),
        call_after_refresh=lambda callback, *args: events.append("scheduled"),
        run_worker=run_worker,
        worker_options=worker_options,
    )
    return screen, events, workers


def _stage_return(
    screen, session_id: str, *switcher_pair: str | None, outcome=SAVED
) -> dict[str, object]:
    """Stage a recovery's return as the Console does; give Settings' answer.

    ``switcher_pair`` is the NEEDS SETUP row's provider and model, when a row
    in Switch model is what left for Settings (TASK-33008.2).
    """
    intent = ConsoleProviderReturnIntent(session_id, *switcher_pair)
    revision = screen.app_instance.pending_handoffs.stage(CHANNEL, intent)
    return intent.return_target(revision, outcome).to_context()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking", [False, True])
async def test_return_lands_on_the_originating_chat(blocking: bool) -> None:
    """AC#3 (R39): switch to the chat, sync, then focus the composer -- or the
    card's primary action while the card still blocks; the handoff settles."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    other = store.create_session(title="Other")
    assert store.active_session_id == other.id
    screen, events, workers = _console(store, blocking=blocking)
    context = _stage_return(screen, origin.id)

    assert capture_provider_return(screen, context) is True
    assert events == ["scheduled"]
    consume_provider_return(screen)
    consume_provider_return(screen)  # A second consumer finds nothing kept.
    (landing,) = workers
    await landing

    assert store.active_session_id == origin.id
    assert events[1:] == ["sync", "card" if blocking else ("composer", True)]
    handoffs = screen.app_instance.pending_handoffs
    assert handoffs.exact_revision_status(CHANNEL, context["return_revision"]) == (
        "settled"
    )
    assert screen._pending_console_provider_return is None


@pytest.mark.asyncio
async def test_return_to_a_closed_chat_stays_put_with_a_notice() -> None:
    """AC#6: the chat is gone, so the Console keeps its chat and says so."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    other = store.create_session(title="Other")
    screen, events, workers = _console(store)
    context = _stage_return(screen, origin.id)
    store.close_session(origin.id)
    assert store.active_session_id == other.id

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)
    (landing,) = workers
    await landing

    assert store.active_session_id == other.id
    assert events[1:] == [("notice", MISSING_CHAT_COPY)]
    assert not screen.app_instance.pending_handoffs.has_pending(CHANNEL)


@pytest.mark.asyncio
async def test_a_failed_landing_settles_its_claim_and_never_ends_the_app() -> None:
    """A landing that raises must not exit the app just after the user saved
    a key: the worker is started with exit_on_error off, as the rest of
    Console_Modules does, and the claim still settles."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    screen, _events, workers = _console(store)

    async def failing_sync() -> None:
        raise RuntimeError("sync failed")

    screen._sync_native_console_chat_ui = failing_sync
    context = _stage_return(screen, origin.id)

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)
    (landing,) = workers
    assert screen.worker_options[0]["exit_on_error"] is False
    with pytest.raises(RuntimeError, match="sync failed"):
        await landing

    handoffs = screen.app_instance.pending_handoffs
    assert handoffs.exact_revision_status(CHANNEL, context["return_revision"]) == (
        "settled"
    )


@pytest.mark.parametrize("newer", ["console", "modal", "settled", "other-chat"])
def test_a_stale_return_applies_nothing_and_says_so(newer: str) -> None:
    """AC#5/AC#6: a superseded, settled or mismatched return changes nothing,
    and a newer request of either kind stays pending for its own owner."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    other = store.create_session(title="Other")
    screen, events, workers = _console(store)
    handoffs = screen.app_instance.pending_handoffs
    context = _stage_return(screen, origin.id)
    if newer == "console":
        handoffs.stage(CHANNEL, ConsoleProviderReturnIntent(other.id))
    elif newer == "modal":
        handoffs.stage(
            CHANNEL, ConversationSettingsReturnIntent(origin.id, 1, "model", None)
        )
    elif newer == "settled":
        handoffs.acknowledge(handoffs.claim(CHANNEL))
    else:  # The right revision, but Settings names another chat.
        context["session_id"] = other.id

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)

    assert workers == []
    assert events[1:] == [("notice", STALE_RETURN_COPY)]
    assert store.active_session_id == other.id
    assert handoffs.has_pending(CHANNEL) is (newer != "settled")


def test_return_is_kept_until_the_console_owns_the_screen() -> None:
    """R39: a warm Console gets the context before it is back on top, so the
    mount and resume consumers (through ChatScreen's own hook) claim it."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    screen, _events, workers = _console(store, owns=False)
    context = _stage_return(screen, origin.id)
    modal = ConversationSettingsReturnIntent(origin.id, 1, "model", None)

    # The screen's own entry point routes a recovery's return here, and
    # leaves the modal's return and anything else alone.
    ChatScreen.apply_navigation_context(screen, context)
    kept = screen._pending_console_provider_return
    assert kept is not None and kept.session_id == origin.id
    assert not capture_provider_return(
        screen, modal.return_target(1, SAVED).to_context()
    )
    assert not capture_provider_return(screen, {"session_id": origin.id})

    ChatScreen._consume_pending_conversation_settings_return(screen)
    assert workers == []
    assert screen._pending_console_provider_return is kept
    assert screen.app_instance.pending_handoffs.has_pending(CHANNEL)

    screen._owns_console_screen_stack = lambda: True
    ChatScreen._consume_pending_conversation_settings_return(screen)
    assert len(workers) == 1
    workers[0].close()


@pytest.mark.parametrize("newer", [None, "console", "modal"])
def test_an_unused_return_is_taken_back_when_the_console_is_reached_by_hand(
    newer: str | None,
) -> None:
    """A return the user did not take must not stay armed: once Settings has
    had the screen and the Console gets it back with no return, the Console
    discards exactly the revision it staged. A newer request stays."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    screen, events, workers = _console(store)
    posted: list[NavigateToScreen] = []
    screen.post_message = posted.append
    handoffs = screen.app_instance.pending_handoffs

    open_provider_setup(screen, "openai", "gpt-5", field="api_key")
    (navigation,) = posted
    revision = navigation.screen_context["return_revision"]
    # The Console still has the screen: a consumer that runs now (the mount
    # hedge) must leave the return it has only just staged.
    ChatScreen._consume_pending_conversation_settings_return(screen)
    assert handoffs.exact_revision_status(CHANNEL, revision) == "pending"

    navigation.commit_target_ownership()  # Settings has the screen.
    screen._owns_console_screen_stack = lambda: False
    ChatScreen._consume_pending_conversation_settings_return(screen)
    assert handoffs.exact_revision_status(CHANNEL, revision) == "pending"
    if newer == "console":
        handoffs.stage(CHANNEL, ConsoleProviderReturnIntent(origin.id))
    elif newer == "modal":
        handoffs.stage(
            CHANNEL, ConversationSettingsReturnIntent(origin.id, 1, "model", None)
        )

    screen._owns_console_screen_stack = lambda: True  # Back, with no return.
    ChatScreen._consume_pending_conversation_settings_return(screen)

    assert screen._pending_console_provider_return is None
    assert handoffs.has_pending(CHANNEL) is (newer is not None)
    assert handoffs.exact_revision_status(CHANNEL, revision) == (
        "settled" if newer is None else "superseded"
    )
    assert workers == [] and events == []  # Nothing lands, nothing is said.


# --- TASK-33008.2: a NEEDS SETUP row's return reopens Switch model ------------


def _record_switch_model(monkeypatch, screen, events) -> list[dict[str, object]]:
    """Record how the return reopens Switch model, in place of pushing it."""
    opened: list[dict[str, object]] = []

    def open_switcher(opened_for, **options) -> None:
        assert opened_for is screen
        events.append("switch-model")
        opened.append(options)

    monkeypatch.setattr(model_switcher, "_open_switcher", open_switcher)
    return opened


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, blocking, model",
    [
        (SAVED, False, "gpt-5"),
        (UNSAVED, False, "gpt-5"),
        (UNSAVED, True, "gpt-5"),
        (SAVED, False, None),
    ],
    ids=["saved", "without-saving", "card-still-blocks", "provider-lists-no-model"],
)
async def test_a_needs_setup_rows_return_reopens_switch_model_on_its_provider(
    monkeypatch, outcome, blocking: bool, model: str | None
) -> None:
    """AC#2, AC#3 (R44-R46): the return lands as any recovery's does, then
    Switch model reopens with the provider's name in Find and the row's own
    pair to highlight, in the mode the chat is in now (the card's while it
    still blocks). It reopens whether or not anything was saved. Focus is put
    on the composer or the card first, so Esc from the switcher leaves it
    there."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    store.create_session(title="Other")
    screen, events, workers = _console(store, blocking=blocking)
    opened = _record_switch_model(monkeypatch, screen, events)
    context = _stage_return(screen, origin.id, "openai", model, outcome=outcome)

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)
    (landing,) = workers
    await landing

    assert store.active_session_id == origin.id
    assert events[1:] == [
        "sync",
        "card" if blocking else ("composer", True),
        "switch-model",
    ]
    assert opened == [
        {
            "query": "OpenAI ",
            "card": blocking,
            "highlight": ("openai", model) if model else None,
            # Without a pair too: the highlight must not leave the provider.
            "reopened": "openai",
        }
    ]
    assert not screen.app_instance.pending_handoffs.has_pending(CHANNEL)


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", ["chat-closed", "superseded", "another-pair"])
async def test_a_return_that_does_not_land_reopens_nothing(monkeypatch, lost) -> None:
    """R46: Switch model does not reopen when the chat has closed, when a
    newer request replaced this one, or when Settings' answer names another
    pair than the one that was staged (it is stale, like another chat's)."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    store.create_session(title="Other")
    screen, events, workers = _console(store)
    opened = _record_switch_model(monkeypatch, screen, events)
    context = _stage_return(screen, origin.id, "openai", "gpt-5")
    if lost == "chat-closed":
        store.close_session(origin.id)
    elif lost == "superseded":
        screen.app_instance.pending_handoffs.stage(
            CHANNEL, ConsoleProviderReturnIntent(origin.id, "openai", "gpt-5")
        )
    else:
        context["switcher_model"] = "gpt-4.1"

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)
    for landing in workers:
        await landing

    assert opened == []
    assert events[1:] == [
        ("notice", MISSING_CHAT_COPY if lost == "chat-closed" else STALE_RETURN_COPY)
    ]
    assert bool(workers) is (lost == "chat-closed")


@pytest.mark.asyncio
async def test_a_return_reopens_nothing_over_a_screen_that_took_the_consoles_place(
    monkeypatch,
) -> None:
    """The landing awaits the Console's sync. If another screen has the stack
    by the time it ends, Switch model would open over that screen, so it is
    not opened; the handoff still settles."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    screen, events, workers = _console(store)
    opened = _record_switch_model(monkeypatch, screen, events)
    context = _stage_return(screen, origin.id, "openai", "gpt-5")

    async def sync_while_the_user_leaves() -> None:
        events.append("sync")
        screen._owns_console_screen_stack = lambda: False

    screen._sync_native_console_chat_ui = sync_while_the_user_leaves

    assert capture_provider_return(screen, context) is True
    consume_provider_return(screen)
    (landing,) = workers
    await landing

    assert opened == []
    assert "switch-model" not in events
    assert not screen.app_instance.pending_handoffs.has_pending(CHANNEL)


def test_a_recoverys_return_names_no_pair_and_reopens_nothing(monkeypatch) -> None:
    """R43: only a NEEDS SETUP row asks for the reopen. The card's action, a
    blocked chat's and a cloud Retry stage a return that names the chat alone,
    and land on the composer as before."""
    store = ConsoleChatStore()
    origin = store.create_session(title="Origin")
    screen, _events, _workers = _console(store)
    posted: list[NavigateToScreen] = []
    screen.post_message = posted.append

    open_provider_setup(screen, "openai", "gpt-5", field="api_key")

    claim = screen.app_instance.pending_handoffs.claim(CHANNEL)
    assert claim is not None
    assert claim.value == ConsoleProviderReturnIntent(origin.id)
    assert (claim.value.switcher_provider, claim.value.switcher_model) == (None, None)


def _settings(
    target,
    outcome,
    store=None,
    *,
    dirty=False,
    conflict=None,
    own_draft=False,
    key_check="Ready · not tested",
):
    """What the Settings offer and copy read from their screen.

    ``dirty`` is any unsaved Providers & Models edit; ``own_draft`` is the
    narrower case of a labelled edit to the target's own provider.
    ``key_check`` is the Key check row's word, read from the last check's
    Readiness line as the real screen keeps it.
    """
    return SimpleNamespace(
        _provider_return_target=target,
        _provider_return_outcome=outcome,
        _provider_navigation_conflict_target=conflict,
        _category_has_unsaved_changes=lambda _category: dirty or own_draft,
        _provider_same_target_has_draft=lambda: own_draft and conflict is None,
        _custom_endpoints_store=lambda: store,
        _provider_test_result=f"Readiness  {key_check}\nKey  saved in config",
    )


def test_a_console_return_is_offered_before_any_save_and_the_modals_is_not() -> None:
    """Parent AC#3: a recovery that saves nothing (a key test) still has its
    return, so the Console kind is offered on arrival and names the chat. An
    unsaved edit or a provider conflict withdraws it, because Return without
    saving and the conflict panel answer those. Chat settings' return shows
    after a save only, as before (AC#7).

    A Console recovery is never left with no return: every unsaved state that
    withdraws Return to Console offers Return without saving, including the
    two with no labelled edit to the target's provider (another provider
    picked in the Provider row; only a snapshot preference edited). Chat
    settings' Return without saving still needs that edit."""
    store = ConsoleChatStore()
    chat = store.create_session(title="Chat 1")
    console = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", 2, chat.id
    )
    modal = replace(console, session_id=None)
    arrival = "No unsaved changes. Return to Console to continue in “Chat 1”."

    on_arrival = _settings(console, None, store)
    assert return_outcome(on_arrival) is UNSAVED
    assert continuation_copy(on_arrival) == arrival
    assert not can_return_without_saving(on_arrival)
    for unsaved in (
        _settings(console, None, store, own_draft=True),  # the key, typed
        # No labelled edit to the target's provider: another provider picked,
        # or only a snapshot preference edited (both driven on the real app).
        _settings(console, None, store, dirty=True),
    ):
        assert return_outcome(unsaved) is None
        assert can_return_without_saving(unsaved)
    assert can_return_without_saving(_settings(modal, None, store, own_draft=True))
    for no_question in (
        _settings(modal, None, store, dirty=True),  # as before: its own draft only
        _settings(console, None, store, dirty=True, conflict=console),
        _settings(console, SAVED, store, dirty=True),
        _settings(modal, SAVED, store, own_draft=True),  # after a save: neither kind
        _settings(None, None, store, dirty=True),
    ):
        assert not can_return_without_saving(no_question)
    for withdrawn in (
        _settings(console, None, store, dirty=True),
        _settings(console, None, store, conflict=console),
        _settings(modal, None, store),
        _settings(None, None, store),
    ):
        assert return_outcome(withdrawn) is None
        assert continuation_copy(withdrawn) == INACTIVE_COPY
    # A save's outcome is what the return sends, whatever is unsaved since.
    assert return_outcome(_settings(console, SAVED, store, dirty=True)) is SAVED
    assert return_outcome(_settings(modal, SAVED, store)) is SAVED
    assert return_outcome(_settings(modal, SAVED, store, own_draft=True)) is SAVED
    store.close_session(chat.id)
    assert continuation_copy(on_arrival) == "No unsaved changes. Return to Console."


def test_a_kept_console_return_lapses_with_its_handoff() -> None:
    """A Console return kept across Settings visits is dropped once its exact
    revision is no longer pending (taken back, used or superseded); Chat
    settings' return is never checked."""
    handoffs = PendingHandoffStore()
    screen = SimpleNamespace(app_instance=SimpleNamespace(pending_handoffs=handoffs))
    intent = ConsoleProviderReturnIntent("session-1")
    revision = handoffs.stage(CHANNEL, intent)
    console = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", revision, "session-1"
    )
    modal = replace(console, session_id=None)

    assert console_return_lapsed(screen, console) is False
    newer = handoffs.stage(CHANNEL, intent)
    assert console_return_lapsed(screen, console) is True  # superseded
    assert console_return_lapsed(screen, replace(console, return_revision=newer)) is (
        False
    )
    assert handoffs.discard_pending_exact(CHANNEL, newer, intent)
    assert console_return_lapsed(screen, replace(console, return_revision=newer))
    assert console_return_lapsed(screen, modal) is False
    no_store = SimpleNamespace(app_instance=SimpleNamespace())
    assert console_return_lapsed(no_store, console) is True
    assert console_return_lapsed(no_store, modal) is False


def test_a_route_with_no_field_to_focus_lands_on_the_first_control_that_acts() -> None:
    """Where focus goes when the route names a disabled field, or none: the
    return when it is offered, otherwise the first control that can act.

    For a disabled field that is Sign in with, then Edit this endpoint in
    Custom endpoints. The rule reads no field name, so a custom endpoint's
    Endpoint lands as its API key does, and so does its Default model, which
    is disabled through the picker it sits in (the real app's case is
    ``test_a_custom_endpoint_row_with_no_field_to_focus_arrives_where_it_can_act``).
    A route that names no field skips
    Sign in with, which is enabled on every Anthropic route and would take a
    one-way route that has nothing to do with it. The Edit button is hidden
    through display, which Textual's focusable does not read: a custom
    endpoint id whose entry is gone lands on the Provider control instead.

    A Console return that an unsaved edit has withdrawn (one kept from an
    earlier visit) gives its landing to Return without saving. Chat settings'
    return is not offered on arrival, so it has no landing to give."""
    field = "#settings-provider-endpoint-value"
    sign_in = "#settings-provider-auth-source"
    edit = "#settings-provider-edit-custom-endpoint"
    picker = "#settings-provider-search"

    def landing(
        provider="custom-ep:lab",
        *,
        named=True,
        field_disabled=True,
        edit_shown=True,
        outcome=None,
        target=None,
        **unsaved,
    ):
        widgets = {
            # is_disabled: its own disabled, or that of a control it sits in.
            field: SimpleNamespace(is_disabled=field_disabled),
            sign_in: SimpleNamespace(display=True, disabled=provider != "anthropic"),
            edit: SimpleNamespace(display=edit_shown, disabled=False),
        }
        screen = _settings(target, outcome, **unsaved)
        screen.query = lambda selector: [widgets[selector]]
        screen._provider_widget_value = lambda: provider
        screen._provider_is_registry_id = SettingsScreen._provider_is_registry_id
        return route_focus(screen, field if named else None)

    assert landing(field_disabled=False) == field
    # A disabled field: Sign in with first, then the Edit button.
    assert landing("anthropic") == sign_in
    assert landing("anthropic", edit_shown=False) == sign_in
    assert landing() == edit
    assert landing("openai", edit_shown=False) is None
    # No field named: the Edit button, and never Sign in with.
    assert landing(named=False) == edit
    assert landing("anthropic", named=False, edit_shown=False) is None
    assert landing("openai", named=False, edit_shown=False) is None
    # The entry is gone, in either spelling of its id: another provider is chosen.
    for gone in ("custom-ep:gone", "custom_ep:gone"):
        assert landing(gone, edit_shown=False) == picker
        assert landing(gone, named=False, edit_shown=False) == picker
    # A return that is offered comes first; after a save it is, for either kind.
    assert landing(outcome=SAVED) == "#settings-provider-return"
    assert landing(named=False, outcome=SAVED) == "#settings-provider-return"
    # Withdrawn by a kept edit, the Console's return hands its landing on.
    console = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", None, 2, "session-1"
    )
    modal = replace(console, field="api_key", session_id=None)
    without_saving = "#settings-provider-return-without-save"
    kept = {"target": console, "own_draft": True, "edit_shown": False}
    assert landing("openai", named=False, **kept) == without_saving
    assert landing("anthropic", **kept) == without_saving
    assert landing("openai", field_disabled=False, **kept) == field
    # Not beside the conflict panel, which shows no Return without saving.
    assert landing("openai", named=False, conflict=console, **kept) is None
    assert landing("anthropic", target=modal, own_draft=True) == sign_in


def test_settings_copy_names_the_console_chat_and_keeps_the_modal_copy() -> None:
    """AC#2/AC#7 (R38): a Console return names its chat by title, read from
    the live store; the Chat settings modal's two strings are unchanged."""
    store = ConsoleChatStore()
    chat = store.create_session(title="  Refactor plan for the gateway rewrite ")
    console = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", 2, chat.id
    )
    modal = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", 2
    )
    other = ConversationSettingsReturnOutcome.PROVIDER_SETTINGS_SAVED

    assert continuation_copy(_settings(console, SAVED, store)) == (
        "Credential saved. Return to Console to continue in "
        "“Refactor plan for the g…”; provider acceptance is not yet verified."
    )
    assert continuation_copy(_settings(console, other, store)) == (
        "Provider settings saved. Return to Console to continue in "
        "“Refactor plan for the g…”; generation is not yet verified."
    )
    # Capture finding 1: a key check after the save (t) that the provider
    # accepted ends the line's doubt, for a saved key only. Chat settings'
    # line and a local server's "reachable" (it may ignore the key) keep it.
    verified = "Ready · verified 21:24"
    assert continuation_copy(_settings(console, SAVED, store, key_check=verified)) == (
        "Credential saved. Return to Console to continue in "
        "“Refactor plan for the g…”; the provider accepted the key."
    )
    for unchanged in (
        _settings(console, SAVED, store, key_check="Ready · reachable 21:24"),
        _settings(console, other, store, key_check=verified),
        _settings(modal, SAVED, key_check=verified),
    ):
        assert continuation_copy(unchanged).endswith("is not yet verified.")
    store.close_session(chat.id)
    assert continuation_copy(_settings(console, SAVED, store)) == (
        "Credential saved. Return to Console; provider acceptance is not yet verified."
    )
    assert continuation_copy(_settings(modal, SAVED)) == (
        "Credential saved. Return to Chat settings to check readiness; "
        "provider acceptance is not yet verified."
    )
    assert continuation_copy(_settings(modal, other)) == (
        "Provider settings saved. Return to Chat settings to check "
        "readiness; generation is not yet verified."
    )
    assert continuation_copy(_settings(modal, None, store)) == INACTIVE_COPY
    assert fitted_chat_title("   ") == "untitled"
    for target, destination in ((console, "Console"), (modal, "Chat settings")):
        screen = SimpleNamespace(_provider_return_target=target)
        assert SettingsScreen._provider_return_destination(screen) == destination
    assert (
        SettingsScreen._provider_return_destination(
            SimpleNamespace(_provider_return_target=None)
        )
        == "Chat settings"
    )


# --- the real app: card -> Settings -> key -> save -> Enter ------------------

_KEY = "p8returnkey0123456789"


async def _until(pilot, condition, what: str, *, attempts: int = 240) -> None:
    for _ in range(attempts):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Timed out waiting for {what}")


async def _recover_from_the_card(app, pilot):
    """Press the Get started card's action; arrive in Settings at the key."""
    await _until(
        pilot,
        lambda: (
            isinstance(app._navigation_outgoing_screen(), ChatScreen)
            and getattr(app.focused, "id", None) == "console-setup-modal-action"
        ),
        "the Get started card's action to hold focus",
    )
    console = app._navigation_outgoing_screen()
    store = console._ensure_console_chat_store()
    # Read, never ensure: with no active chat the real route goes one way.
    assert store.active_session_id is not None
    session = next(s for s in store.sessions() if s.id == store.active_session_id)
    await pilot.press("enter")
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, SettingsScreen)
            and getattr(app.focused, "id", None) == "settings-provider-api-key"
        ),
        "Settings to open with focus in the API key field",
    )
    settings = app.screen
    target = settings._provider_return_target
    assert target is not None
    assert (target.provider, target.model, target.field) == (
        "openai",
        "gpt-5",
        "api_key",
    )
    assert target.session_id == session.id
    assert settings._provider_widget_value() == "openai"
    # ADR-012: the handoff names the chat and nothing else.
    claim = app.pending_handoffs.claim(CHANNEL)
    assert claim is not None and claim.value == ConsoleProviderReturnIntent(session.id)
    app.pending_handoffs.release(claim)
    return console, session, settings


def _fully_on_screen(screen, widget) -> bool:
    """Whether the whole widget is painted: no scroll or Tab is needed."""
    placed = screen._compositor.visible_widgets.get(widget)
    return placed is not None and placed[1].contains_region(placed[0])


def _editing_the_endpoint(app, settings) -> bool:
    """Whether the endpoint editor's URL field holds focus and is painted.

    The field sits in the Custom endpoints disclosure, which is closed until
    the Edit button opens it: the id alone would also pass on a field nobody
    sees."""
    on_url = getattr(app.focused, "id", None) == "settings-cep-edit-url"
    return on_url and _fully_on_screen(settings, app.focused)


async def _keep_editing(app, pilot, settings) -> None:
    """Enter on Return without saving, then Enter on the confirmation's own
    default, Keep editing. Focus comes back to the action it left, with the
    edit unsaved and the handoff pending, so the next Enter asks again.

    The action is disabled while the confirmation is open, which takes its
    focus away. Not given back, focus fell to the nav bar's Home button."""
    action = "settings-provider-return-without-save"
    assert app.focused.id == action
    await pilot.press("enter")
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, ConfirmationDialog)
            and getattr(app.focused, "id", None) == "cancel-button"
        ),
        "the discard confirmation, with focus on Keep editing",
    )
    # It names where Discard goes: this Console chat, or Chat settings.
    destination = settings._provider_return_destination()
    assert app.screen.message.endswith(f"and return to {destination}?")
    await pilot.press("enter")
    await _until(
        pilot,
        lambda: app.screen is settings and getattr(app.focused, "id", None) == action,
        "Settings again, with focus back on Return without saving",
    )
    assert _fully_on_screen(settings, app.focused)
    assert settings._provider_same_target_has_draft()
    assert app.pending_handoffs.has_pending(CHANNEL)


def _cloud_app(monkeypatch, **saved):
    """The real app on its scratch profile: OpenAI is the default, keyless.

    ``saved`` lays config sections over that, such as a saved key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_production_app(configured_default="chat")
    assert config_module.save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "chat_defaults": {"provider": "openai", "model": "gpt-5"},
            "api_settings.openai": {"api_key": ""},
            **saved,
        }
    )
    app.app_config = config_module.load_settings(force_reload=True)
    app.providers_models = {"openai": ["gpt-5"]}
    return app


def _saved_config() -> dict[str, object]:
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    # The scratch profile, never the user's own config.
    assert config_path.is_relative_to(Path(os.environ["TLDW_TEST_CONFIG_ROOT"]))
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _saved_openai_key() -> object:
    return _saved_config().get("api_settings", {}).get("openai", {}).get("api_key")


@pytest.mark.asyncio
@private_profile_test
async def test_recovery_round_trip_returns_to_the_chat_on_one_enter(
    monkeypatch, request
):
    """AC#9 (R41): the real store, the real Settings screen and the real
    router on a scratch config. Card -> Settings -> key -> Esc, s -> Enter
    ends in the same chat, composer focused, ready to send."""
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        # The return is offered on arrival and names the chat, and focus
        # stays in the field the blocker named (AC#1, AC#2).
        continuation = settings.query_one("#settings-provider-return-continuation")
        status = settings.query_one(
            "#settings-provider-return-continuation-status", Static
        )
        assert continuation.display is True
        assert str(status.renderable) == (
            f"No unsaved changes. Return to Console to continue in “{session.title}”."
        )
        assert app.focused.id == "settings-provider-api-key"
        # AC#2's "shows", at 211x44: the status and the action are painted
        # as Settings opens, with focus still at the top of the card.
        await pilot.pause()
        action = settings.query_one("#settings-provider-return", Button)
        assert _fully_on_screen(settings, status)
        assert _fully_on_screen(settings, action)
        # The helper can say no: a closed disclosure's row is not painted.
        assert not _fully_on_screen(
            settings, settings.query_one("#settings-snapshot-enabled", Checkbox)
        )

        # Another chat is active by the time Settings returns, so only the
        # Console's consumer can put the user back on the chat they left.
        store = console._ensure_console_chat_store()
        other = store.create_session(title="Other")
        assert store.active_session_id == other.id != session.id

        await pilot.press(*_KEY)
        # An unsaved key withdraws it: Return without saving asks first.
        await _until(
            pilot, lambda: not continuation.display, "the unsaved key to hide it"
        )
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        assert str(app.focused.label) == "Return to Console"
        assert str(status.renderable) == (
            f"Credential saved. Return to Console to continue in “{session.title}”; "
            "provider acceptance is not yet verified."
        )
        assert _saved_openai_key() == _KEY

        await pilot.press("enter")
        composer = None

        def back_in_the_composer() -> bool:
            nonlocal composer
            if app.screen is not console or app.current_tab != "chat":
                return False
            composer = console.query_one("#console-native-composer")
            return console._is_descendant_or_self(app.focused, composer)

        await _until(pilot, back_in_the_composer, "the composer to hold focus")
        assert store.active_session_id == session.id
        _settings_now, readiness = console._active_console_settings_readiness()
        assert readiness.operability == "ready_to_send"
        assert not console._console_setup_modal_blocking()
        assert not app.pending_handoffs.has_pending(CHANNEL)
        assert settings._provider_return_target is None


@pytest.mark.asyncio
@private_profile_test
async def test_a_key_check_after_the_save_updates_the_return_line(
    monkeypatch, request
):
    """Capture finding 1 (03e): after s, t lists models with the saved key.
    Key check then read "Ready · verified" while the line above Return to
    Console still said the key was not verified. The check redraws the line,
    and focus stays on the return. Only the listing is stubbed."""
    listed: list[httpx.Request] = []

    def listing(request: httpx.Request) -> httpx.Response:
        listed.append(request)
        return httpx.Response(200, json={"data": [{"id": "gpt-5"}]})

    monkeypatch.setattr(
        discovery_module,
        "build_httpx_async_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(listing), **kwargs
        ),
    )
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        _console, session, settings = await _recover_from_the_card(app, pilot)
        status = settings.query_one(
            "#settings-provider-return-continuation-status", Static
        )
        await pilot.press(*_KEY)
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        line = f"Credential saved. Return to Console to continue in “{session.title}”; "
        assert str(status.renderable) == (
            f"{line}provider acceptance is not yet verified."
        )

        await pilot.press("t")
        verdict = settings.query_one("#settings-provider-readiness", Static)
        await _until(
            pilot,
            lambda: str(verdict.renderable).startswith("Ready · verified"),
            "the key check to read verified",
        )
        assert len(listed) == 1 and listed[0].url.path.endswith("/models")
        assert str(status.renderable) == f"{line}the provider accepted the key."
        assert app.focused.id == "settings-provider-return"


@pytest.mark.asyncio
@pytest.mark.parametrize("holds_work", [False, True], ids=["untouched", "holds-work"])
@private_profile_test
async def test_the_returning_chat_follows_a_newly_saved_model_only_while_untouched(
    monkeypatch, request, holds_work: bool
):
    """AC#3, second sentence (ADR-095, D1), on the return path: the visit
    saves another default model as well as the key. An untouched chat comes
    back on the saved provider · model; a chat that holds work keeps its own.
    The return adds no convergence code, so this is the Console's own sync."""
    app = _cloud_app(monkeypatch)
    app.providers_models = {"openai": ["gpt-5", "gpt-4.1"]}

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        store = console._ensure_console_chat_store()
        if holds_work:
            store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")

        await pilot.press(*_KEY)
        settings.query_one("#settings-model-value", Input).value = "gpt-4.1"
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        saved = _saved_config()["chat_defaults"]
        assert (saved["provider"], saved["model"]) == ("openai", "gpt-4.1")

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and console._is_descendant_or_self(
                    app.focused, console.query_one("#console-native-composer")
                )
            ),
            "the composer to hold focus",
        )
        assert store.active_session_id == session.id
        chat, readiness = console._active_console_settings_readiness()
        assert (chat.provider, chat.model) == (
            "openai",
            "gpt-5" if holds_work else "gpt-4.1",
        )
        assert readiness.operability == "ready_to_send"


@pytest.mark.asyncio
@private_profile_test
async def test_recovery_return_without_saving_lands_on_the_same_chat_unchanged(
    monkeypatch, request
):
    """AC#4: Return without saving asks before discarding, then lands on the
    same chat, still blocked, with nothing written. R38: both return labels
    say Console. R34: the Console kind survives a fresh Settings screen."""
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        await pilot.press(*_KEY)
        await pilot.pause()
        without_saving = settings.query_one(
            "#settings-provider-return-without-save", Button
        )
        assert without_saving.display is True
        assert str(without_saving.label) == "Return without saving"
        # Capture finding 10b (03c): a key typed on this visit is not an
        # "existing" change; the summary names it plainly.
        summary = settings.query_one(
            "#settings-provider-existing-changes-summary", Static
        )
        assert summary.display is True
        assert str(summary.renderable) == "Unsaved changes: API key."

        # R38: both returns that name a destination say Console, and follow
        # the target when it changes kind with no recompose.
        target = settings._provider_return_target
        for kind, label in (
            (replace(target, session_id=None), "Return to Chat settings"),
            (target, "Return to Console"),
        ):
            settings._provider_return_target = kind
            settings._update_provider_return_widgets()
            for selector in (
                "#settings-provider-return",
                "#settings-provider-conflict-return",
            ):
                assert str(settings.query_one(selector, Button).label) == label

        fresh = SettingsScreen(app)
        fresh.restore_state(settings.save_state())
        assert fresh._provider_return_target == settings._provider_return_target
        assert fresh._provider_return_destination() == "Console"

        without_saving.focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the discard confirmation",
        )
        # Capture finding 9 (04b): the question says Discard also goes back.
        assert (app.screen.title, app.screen.message) == (
            "Return without saving",
            (
                "Discard all unsaved changes to Providers & Models and return to "
                "Console?"
            ),
        )
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console's card to hold focus again, its return claimed",
        )
        assert console._ensure_console_chat_store().active_session_id == session.id
        assert console._console_setup_modal_blocking()
        assert not _saved_openai_key()
        assert not app.pending_handoffs.has_pending(CHANNEL)


@pytest.mark.asyncio
@private_profile_test
async def test_every_unsaved_recovery_state_offers_return_without_saving(
    monkeypatch, request
):
    """A Console recovery is never left with no return. Two unsaved states
    carry no labelled edit to the target's provider, so Chat settings' rule
    would hide both returns until a save or a revert: only a snapshot
    preference edited, and another provider picked in the Provider row. Each
    withdraws Return to Console and offers Return without saving, which asks,
    discards the whole category and lands on the same chat.

    A save of only the snapshot preference is not the recovery's save: it
    sets no outcome, moves no focus and leaves the way back as it was before
    that edit, so the next edit asks first again."""
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)

        def offered() -> tuple[bool, bool]:
            """(Return to Console shows, Return without saving shows)."""
            return (
                settings.query_one("#settings-provider-return-continuation").display,
                settings.query_one("#settings-provider-return-without-save").display,
            )

        assert offered() == (True, False)
        status = settings.query_one(
            "#settings-provider-return-continuation-status", Static
        )
        on_arrival = str(status.renderable)
        assert on_arrival == (
            f"No unsaved changes. Return to Console to continue in “{session.title}”."
        )
        snapshots = settings.query_one("#settings-snapshot-enabled", Checkbox)
        assert not snapshots.disabled
        snapshot_saved = snapshots.value
        snapshots.toggle()
        await _until(
            pilot, lambda: offered() == (False, True), "a snapshot edit to ask first"
        )
        assert not settings._provider_same_target_has_draft()

        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.pause()
        focus_before_the_save = app.focused
        await pilot.press("s")
        await _until(
            pilot,
            lambda: (
                not settings._snapshot_preferences_saving
                and not settings._snapshot_preferences_dirty()
                and offered() == (True, False)
            ),
            "the snapshot preference's save to land",
        )
        await pilot.pause()  # A focus move would be scheduled after the refresh.
        assert _saved_config()["llamacpp_snapshots"]["enabled"] is (not snapshot_saved)
        assert settings._provider_return_outcome is None
        assert str(status.renderable) == on_arrival
        assert app.focused is focus_before_the_save
        assert getattr(app.focused, "id", None) != "settings-provider-return"
        snapshots.toggle()
        await _until(
            pilot, lambda: offered() == (False, True), "the next edit to ask first"
        )
        snapshots.toggle()
        await _until(
            pilot, lambda: offered() == (True, False), "the undone edit to stop asking"
        )

        settings.query_one("#settings-provider-value", Select).value = "anthropic"
        await _until(
            pilot, lambda: offered() == (False, True), "another provider to ask first"
        )
        assert not settings._provider_same_target_has_draft()

        settings.query_one("#settings-provider-return-without-save").focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the discard confirmation",
        )
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console's card to hold focus again, its return claimed",
        )
        assert console._ensure_console_chat_store().active_session_id == session.id
        assert _saved_config()["chat_defaults"]["provider"] == "openai"
        assert not _saved_openai_key()
        assert not app.pending_handoffs.has_pending(CHANNEL)


_OWN_MODEL = "own-model-x"
#: Each provider's own model, laid in the scratch config: the shipped template
#: is no part of these tests, so a model id it refreshes or adds breaks none.
#: "" is none (a missing key would load the template's value).
_SAVED_MODELS = {
    "api_settings.cerebras": {"model": ""},
    "api_settings.deepseek": {"model": _OWN_MODEL},
}
#: route -> (provider, the model Settings shows for a route that names none).
_ROUTES_THAT_NAME_NO_MODEL = {
    # The default provider: Settings' own model stays (TASK-33008.1).
    "default-provider": ("openai", "gpt-5"),
    # Another provider with no list and no model of its own saved, as 24 of
    # the 60 shipped providers are: none, never the default provider's.
    "lists-no-models": ("cerebras", ""),
    # Another provider that lists nothing here and has a model of its own
    # saved: that model.
    "its-own-saved-model": ("deepseek", _OWN_MODEL),
}


async def _route_with_no_model(app, pilot, provider: str, **route):
    """From the Get started card, open Settings on ``provider`` as a NEEDS
    SETUP row that names no model does; arrive in its API key field."""
    await _until(
        pilot,
        lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
        "the Get started card's action to hold focus",
    )
    console = app._navigation_outgoing_screen()
    open_provider_setup(console, provider, **route)
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, SettingsScreen)
            and getattr(app.focused, "id", None) == "settings-provider-api-key"
        ),
        "Settings to open with focus in the API key field",
    )
    target = app.screen._provider_return_target
    assert target is not None and target.model is None
    assert target.pins_model is False
    assert app.screen._provider_widget_value() == provider
    return console, app.screen


async def _save_the_typed_key(app, pilot) -> None:
    await pilot.press(*_KEY)
    await pilot.press("escape")  # leave the field, so s is Save
    await pilot.press("s")
    await _until(
        pilot,
        lambda: getattr(app.focused, "id", None) == "settings-provider-return",
        "focus on the Return to Console action after the save",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(_ROUTES_THAT_NAME_NO_MODEL))
@private_profile_test
async def test_a_route_that_names_no_model_never_borrows_another_providers_model(
    monkeypatch, request, route: str
):
    """Controller ruling (c) of 2026-10-11. Rewritten on purpose from
    ``test_a_route_that_names_no_model_keeps_settings_own_model``, which
    pinned Settings' own model beside every provider routed without one.

    A NEEDS SETUP row of a provider that lists no models routes no model.
    Settings kept the saved default's model in the field, the first edit
    pinned it, and the save wrote "<provider> · <the old default's model>" as
    the default for new chats. A route to the default provider still keeps
    Settings' own model. A route to any other provider shows that provider's
    saved model or none, as choosing it in the Provider control does, and the
    save writes what the page shows."""
    provider, shown = _ROUTES_THAT_NAME_NO_MODEL[route]
    for name in ("CEREBRAS_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    app = _cloud_app(monkeypatch, **_SAVED_MODELS)

    async with app.run_test(size=(211, 44)) as pilot:
        _console, settings = await _route_with_no_model(app, pilot, provider)
        model_field = settings.query_one("#settings-model-value", Input)
        assert model_field.value == shown
        # Showing it is no edit: the return is offered as on any arrival.
        assert not settings._category_has_unsaved_changes(
            settings._active_category_id()
        )
        assert settings.query_one("#settings-provider-return-continuation").display

        await _save_the_typed_key(app, pilot)
        saved = _saved_config()
        assert saved["api_settings"][provider]["api_key"] == _KEY
        defaults = saved["chat_defaults"]
        assert (defaults["provider"], defaults["model"]) == (provider, shown)
        assert model_field.value == shown


def _painted_pairs(switcher) -> str:
    pairs = switcher.query_one("#console-popover-pairs").region
    painted = [strip.text for strip in switcher._compositor.render_strips()]
    return "\n".join(painted[pairs.y : pairs.bottom])


def _reopened_on_its_setup_row(app, provider: str) -> bool:
    """Switch model is open with ``provider``'s model-less NEEDS SETUP row
    highlighted."""
    if not isinstance(app.screen, ConsoleModelPopover):
        return False
    row = app.screen.highlighted_row()
    return row is not None and (row.kind, row.provider, row.model) == (
        "setup",
        provider,
        None,
    )


@pytest.mark.asyncio
@private_profile_test
async def test_a_provider_saved_with_no_model_is_finished_on_the_reopened_list(
    monkeypatch, request
):
    """Controller ruling (c), the rest of the trip, on the real app. Cerebras
    lists no models and has none saved here, so its NEEDS SETUP row routes none
    and the saved default for new chats is "Cerebras" with no model. The
    untouched chat follows it, so the Get started card still blocks, and
    Switch model reopens over it on "Cerebras ". Its row is still under NEEDS
    SETUP, now for its model, and READY PROVIDERS opens with the line that
    says what to type. The id typed after the name applies to the chat and,
    with no working new-chat default, becomes that too."""
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    app = _cloud_app(monkeypatch, **_SAVED_MODELS)

    async with app.run_test(size=(211, 44)) as pilot:
        console, _settings = await _route_with_no_model(
            app, pilot, "cerebras", reopen_switcher=True
        )
        await _save_the_typed_key(app, pilot)
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: _reopened_on_its_setup_row(app, "cerebras"),
            "Switch model on Cerebras's own row",
        )
        reopened = app.screen
        _chat, readiness = console._active_console_settings_readiness()
        assert _chat_pair(console) == ("cerebras", None)
        assert (readiness.blocker, readiness.recovery_action) == (
            "model_missing",
            "select_model",
        )
        assert reopened.query_one("#console-popover-find", Input).value == "Cerebras "
        assert reopened._scope_copy == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        assert (
            "Cerebras · no models reported · type its name and an id"
            in _painted_pairs(reopened)
        )

        await pilot.press(*"llama-x")
        await _until(
            pilot,
            lambda: (
                (row := reopened.highlighted_row()) is not None
                and (row.kind, row.provider, row.model)
                == ("typed", "cerebras", "llama-x")
            ),
            "the typed id's row to be the highlight",
        )
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and not console._console_setup_modal_blocking()
                and _saved_config()["chat_defaults"].get("model") == "llama-x"
            ),
            "the pick to unlock the card and be saved as the new-chat default",
        )
        assert _chat_pair(console) == ("cerebras", "llama-x")
        assert _saved_config()["chat_defaults"]["provider"] == "cerebras"
        _chat, readiness = console._active_console_settings_readiness()
        assert readiness.operability == "ready_to_send"


@pytest.mark.asyncio
@private_profile_test
async def test_a_reopened_list_sets_the_new_chat_default_while_none_works(
    monkeypatch, request
):
    """Controller ruling of 2026-10-12 (final fix wave 1, item 2, Option A).

    The chat holds work, so no card blocks and the list reopens in its plain
    mode. Cerebras's row named no model, so its saved key left "Cerebras"
    with no model as the default for new chats, which cannot send. The pick
    applied to this chat only, and the next new chat started blocked. While
    no new-chat default works, the reopened list says the pick is also that
    default, and it is."""
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    app = _switcher_app(monkeypatch, **_SAVED_MODELS)

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-native-composer",
            "the composer to hold focus",
        )
        console = app._navigation_outgoing_screen()
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id, role=ConsoleMessageRole.USER, content="hi"
        )
        open_provider_setup(console, "cerebras", reopen_switcher=True)
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == "settings-provider-api-key"
            ),
            "Settings to open with focus in Cerebras's API key field",
        )
        await _save_the_typed_key(app, pilot)
        defaults = _saved_config()["chat_defaults"]
        assert (defaults["provider"], defaults["model"]) == ("cerebras", "")
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: _reopened_on_its_setup_row(app, "cerebras"),
            "Switch model on Cerebras's own row",
        )
        reopened = app.screen
        assert not console._console_setup_modal_blocking()
        assert _chat_pair(console) == ("openai", "gpt-5")
        assert reopened._scope_copy == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS

        await pilot.press(*"llama-x")
        await _until(
            pilot,
            lambda: (
                (row := reopened.highlighted_row()) is not None
                and (row.kind, row.provider, row.model)
                == ("typed", "cerebras", "llama-x")
            ),
            "the typed id's row to be the highlight",
        )
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and _saved_config()["chat_defaults"].get("model") == "llama-x"
            ),
            "the pick to be saved as the new-chat default",
        )
        assert _chat_pair(console) == ("cerebras", "llama-x")
        assert _saved_config()["chat_defaults"]["provider"] == "cerebras"


@pytest.mark.asyncio
@private_profile_test
async def test_a_no_model_rows_enter_opens_settings_in_the_default_model_field(
    monkeypatch, request
):
    """Final fix wave 2, on the real app. After ruling (c) a provider saved
    with a key and no model comes back with its NEEDS SETUP row highlighted,
    now for its model. Enter there named no field, so Settings opened on
    Return to Console, and Enter on that reopened the list on the same row:
    two keys that changed nothing. The row's blocker is its model, so the
    route names the Default model field and focus starts in it. Focus is no
    edit: the return is still offered."""
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    app = _cloud_app(monkeypatch, **_SAVED_MODELS)

    async with app.run_test(size=(211, 44)) as pilot:
        await _route_with_no_model(app, pilot, "cerebras", reopen_switcher=True)
        await _save_the_typed_key(app, pilot)
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: _reopened_on_its_setup_row(app, "cerebras"),
            "Switch model on Cerebras's own row",
        )

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and (target := app.screen._provider_return_target) is not None
                and target.field != "api_key"  # the first visit's is gone
            ),
            "Settings to open again on the row's own route",
        )
        settings = app.screen
        target = settings._provider_return_target
        assert (target.provider, target.model, target.field) == (
            "cerebras",
            None,
            "model",
        )
        picker = settings.query_one("#settings-model-picker", ModelSearchPicker)
        field = picker.query_one("#model-search-picker-input", Input)
        await _until(
            pilot,
            lambda: app.focused is field,
            "focus in the Default model field",
        )
        assert not settings._category_has_unsaved_changes(
            settings._active_category_id()
        )
        assert settings.query_one("#settings-provider-return-continuation").display


@pytest.mark.asyncio
@private_profile_test
async def test_a_model_saved_in_the_default_model_field_is_the_reopened_highlight(
    monkeypatch, request
):
    """Final fix wave 3, item 3, on the real app. Cerebras has a key and no
    model, and the chat holds work. Enter on its row opens Settings in the
    Default model field; a Custom ID saved there makes Cerebras ready on that
    model. The row named no model, so the reopened list had no pair: it said
    "no models reported", highlighted nothing, and the chat kept its old pair
    until the id was typed again. It reopens on the model just saved, and
    Enter applies it."""
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    app = _switcher_app(
        monkeypatch, **{"api_settings.cerebras": {"model": "", "api_key": _KEY}}
    )

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-native-composer",
            "the composer to hold focus",
        )
        console = app._navigation_outgoing_screen()
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id, role=ConsoleMessageRole.USER, content="hi"
        )
        open_provider_setup(console, "cerebras", reopen_switcher=True)
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and isinstance(app.focused, Input)
                and app.focused.id == "model-search-picker-input"
            ),
            "Settings to open with focus in Cerebras's Default model field",
        )
        settings = app.screen
        picker = settings.query_one("#settings-model-picker", ModelSearchPicker)
        await pilot.press("tab")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "model-search-picker-custom",
            "focus on Custom ID",
        )
        await pilot.press("enter")
        await _until(pilot, lambda: picker.custom_mode, "Custom ID's own field")
        await pilot.press(*"llama-x")
        await pilot.press("escape")
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        assert _saved_config()["api_settings"]["cerebras"]["model"] == "llama-x"

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, ConsoleModelPopover)
                and (row := app.screen.highlighted_row()) is not None
                and (row.kind, row.provider, row.model)
                == ("pair", "cerebras", "llama-x")
            ),
            "Switch model on the model just saved",
        )
        assert _chat_pair(console) == ("openai", "gpt-5")
        await pilot.press("enter")
        await _until(pilot, lambda: app.screen is console, "the pick to apply")
        assert _chat_pair(console) == ("cerebras", "llama-x")


@pytest.mark.asyncio
@private_profile_test
async def test_a_recovery_that_saves_nothing_returns_from_the_arrival_action(
    monkeypatch, request
):
    """Parent AC#3: a recovery that needs no save (a cloud Retry's key test)
    still goes back in one action. The arrival return asks nothing, writes
    nothing and lands on the chat that opened Settings."""
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        assert settings.query_one("#settings-provider-return-continuation").display
        # Keyboard only: Tab walks from the named field on to the return.
        tabs = 0
        while getattr(app.focused, "id", None) != "settings-provider-return":
            assert tabs < 12, f"Tab never reached the return; at {app.focused!r}"
            await pilot.press("tab")
            tabs += 1
        assert str(app.focused.label) == "Return to Console"

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console's card to hold focus again, its return claimed",
        )
        assert console._ensure_console_chat_store().active_session_id == session.id
        assert not _saved_openai_key()
        assert not app.pending_handoffs.has_pending(CHANNEL)
        assert settings._provider_return_target is None


@pytest.mark.asyncio
@pytest.mark.parametrize("keyed", [False, True], ids=["first-run-card", "saved-key"])
@private_profile_test
async def test_a_route_that_names_no_field_arrives_on_the_return(
    monkeypatch, request, keyed: bool
):
    """A cloud Retry names no field: the user tests the key with t and goes
    back. With no field to put focus in, the offered return takes it, and t
    leaves it there, so after the test one Enter returns (parent AC#3).

    Back in the Console focus is where the chat continues: the Get started
    card's action while it blocks (no key), the composer once a key is saved.
    Only the saved key can be tested, through a stubbed transport."""
    listed: list[httpx.Request] = []

    def listing(request: httpx.Request) -> httpx.Response:
        listed.append(request)
        return httpx.Response(200, json={"data": [{"id": "gpt-5"}]})

    monkeypatch.setattr(
        discovery_module,
        "build_httpx_async_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(listing), **kwargs
        ),
    )
    app = _cloud_app(
        monkeypatch, **({"api_settings.openai": {"api_key": _KEY}} if keyed else {})
    )
    resting = "console-native-composer" if keyed else "console-setup-modal-action"

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == resting,
            f"{resting} to hold focus",
        )
        console = app._navigation_outgoing_screen()
        session_id = console._ensure_console_chat_store().active_session_id
        open_provider_setup(console, "openai", "gpt-5", field="")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == "settings-provider-return"
            ),
            "Settings to open with focus on Return to Console",
        )
        settings = app.screen
        assert settings._provider_return_target.field is None

        if keyed:
            result = settings.query_one("#settings-provider-test-result", Static)
            await pilot.press("t")
            # A bounded poll: the screen's other workers outlive the listing.
            await _until(
                pilot,
                lambda: "model listing reached" in str(result.renderable),
                "the key test to report its listing",
            )
            assert len(listed) == 1 and listed[0].url.path.endswith("/models")
            assert app.focused.id == "settings-provider-return"
            # Settings shows and hides the return through its container.
            assert settings.query_one("#settings-provider-return-continuation").display
        assert bool(listed) is keyed

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == resting
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            f"the Console's {resting} to hold focus again, its return claimed",
        )
        assert console._ensure_console_chat_store().active_session_id == session_id
        assert not app.pending_handoffs.has_pending(CHANNEL)


_SUBSCRIPTION_CHAT = {
    "chat_defaults": {"provider": "anthropic", "model": "claude-sonnet-5"},
    "api_settings.anthropic": {"auth_source": "claude_subscription"},
}
_LAB_KEY_ENV = "P8_RETURN_LAB_KEY"  # Never set: the endpoint's key is missing.
_CUSTOM_ENDPOINT_CHAT = {
    "chat_defaults": {"provider": "custom-ep:lab", "model": "lab-model"},
    "custom_endpoints.lab": {
        "display_name": "Lab",
        "family": "openai_compatible",
        "base_url": "http://127.0.0.1:9/v1",
        "models": ["lab-model"],
        "api_key_env": _LAB_KEY_ENV,
    },
}
#: route -> (saved config, the card's action, the control focus arrives on).
_ROUTES_TO_A_DISABLED_FIELD = {
    "console": (_SUBSCRIPTION_CHAT, "Review credentials", "settings-provider-return"),
    "chat-settings": (
        _SUBSCRIPTION_CHAT,
        "Review credentials",
        "settings-provider-auth-source",
    ),
    "custom-endpoint": (
        _CUSTOM_ENDPOINT_CHAT,
        "Set up provider",
        "settings-provider-edit-custom-endpoint",
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(_ROUTES_TO_A_DISABLED_FIELD))
@private_profile_test
async def test_a_route_whose_named_field_is_disabled_arrives_where_it_can_act(
    monkeypatch, request, route: str
):
    """A Claude subscription whose credential is missing is fixed outside the
    app (log in with Claude Code), so this recovery saves nothing either. The
    card's Review credentials names the API key, and Settings disables that
    field while the subscription signs in. A field that cannot take focus is
    no field (AC#1): the offered return takes it and one Enter goes back.

    Chat settings' Configure credential names the same field and offers no
    return before a save, so there focus goes to Sign in with, the control
    that says why no key is asked for.

    A chat on a named custom endpoint whose key is missing names the API key
    too. Settings locks that provider's key and Endpoint, Sign in with is
    Anthropic's alone, and no return can name a custom endpoint (R35). Focus
    goes to Edit this endpoint in Custom endpoints, where the locked fields
    are changed, and Enter there opens the editor."""
    saved, action, arrives_on = _ROUTES_TO_A_DISABLED_FIELD[route]
    from_console = route == "console"
    # Stubbed, so the user's own Claude Code credential is never read.
    monkeypatch.setattr(
        anthropic_subscription,
        "subscription_credential_status",
        lambda **_kwargs: "missing",
    )
    monkeypatch.delenv(_LAB_KEY_ENV, raising=False)
    app = _cloud_app(monkeypatch, **saved)
    if saved is _SUBSCRIPTION_CHAT:
        app.providers_models = {"anthropic": ["claude-sonnet-5"]}

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
        )
        assert str(app.focused.label) == action
        console = app._navigation_outgoing_screen()
        session_id = console._ensure_console_chat_store().active_session_id

        if route == "chat-settings":
            assert await console._open_console_settings() is True
            await _until(
                pilot,
                lambda: bool(
                    app.screen.query("#console-settings-configure-credential")
                ),
                "Chat settings to offer Configure credential",
            )
            app.screen.query_one("#console-settings-configure-credential").focus()
            await pilot.pause()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == arrives_on
            ),
            f"Settings to open with focus on {arrives_on}",
        )
        settings = app.screen
        target = settings._provider_return_target
        assert settings.query_one("#settings-provider-api-key", Input).disabled
        continuation = settings.query_one("#settings-provider-return-continuation")
        assert continuation.display is from_console
        if route == "custom-endpoint":
            # R35: no return can name a custom-ep: id, so this route goes one way.
            assert target is None and not app.pending_handoffs.has_pending(CHANNEL)
            assert settings._provider_widget_value() == "custom-ep:lab"
            assert settings.query_one("#settings-provider-auth-source").disabled
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: _editing_the_endpoint(app, settings),
                "the Custom endpoints editor to open on the endpoint's URL, painted",
            )
            return
        assert target is not None and target.field == "api_key"
        assert (target.session_id == session_id) is from_console
        if not from_console:
            assert target.session_id is None  # Chat settings' kind: after a save.
            return
        assert str(app.focused.label) == "Return to Console"

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console's card to hold focus again, its return claimed",
        )
        assert console._ensure_console_chat_store().active_session_id == session_id
        assert not app.pending_handoffs.has_pending(CHANNEL)


#: route -> (saved config, the field an earlier visit leaves edited, the edit).
_ROUTES_ONTO_A_KEPT_EDIT = {
    "retry": ({}, "#settings-provider-api-key", _KEY),
    "review-credentials": (
        _SUBSCRIPTION_CHAT,
        "#settings-provider-endpoint-value",
        "x",
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(_ROUTES_ONTO_A_KEPT_EDIT))
@private_profile_test
async def test_a_route_with_no_field_to_focus_arrives_on_return_without_saving_over_a_kept_edit(
    monkeypatch, request, route: str
):
    """A recovery can arrive on an unsaved edit kept from an earlier visit:
    a field edited for the same provider, then Settings left by the nav bar.
    That edit withdraws Return to Console, so Return without saving, which
    takes its place, takes its focus too, for a route that names no field (a
    cloud Retry) and for one whose field is disabled (Review credentials).
    Enter asks before it discards. Keep editing, the default answer, gives
    focus back to the action; Discard changes goes back with nothing written."""
    saved, edited, typed = _ROUTES_ONTO_A_KEPT_EDIT[route]
    # Stubbed, so the user's own Claude Code credential is never read.
    monkeypatch.setattr(
        anthropic_subscription,
        "subscription_credential_status",
        lambda **_kwargs: "missing",
    )
    app = _cloud_app(monkeypatch, **saved)
    if saved is _SUBSCRIPTION_CHAT:
        app.providers_models = {"anthropic": ["claude-sonnet-5"]}

    async with app.run_test(size=(211, 44)) as pilot:

        async def recover() -> ChatScreen:
            await _until(
                pilot,
                lambda: (
                    getattr(app.focused, "id", None) == "console-setup-modal-action"
                ),
                "the Get started card's action to hold focus",
            )
            console = app._navigation_outgoing_screen()
            if route == "retry":  # The card's Retry needs a failed key check.
                open_provider_setup(console, "openai", "gpt-5", field="")
            else:
                await pilot.press("enter")
            return console

        console = await recover()
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == "settings-provider-return"
            ),
            "Settings to open with focus on Return to Console",
        )
        settings = app.screen
        settings.query_one(edited, Input).focus()
        await pilot.pause()
        assert _fully_on_screen(settings, app.focused), app.focused
        await pilot.press(*typed)
        await _until(
            pilot,
            settings._provider_same_target_has_draft,
            "the typed edit to be unsaved",
        )
        settings.post_message(NavigateToScreen(TAB_CHAT))  # as the nav bar does
        await _until(
            pilot,
            lambda: (
                app.screen is console and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console to take its unused return back",
        )
        saved_before = _saved_config()

        assert await recover() is console
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and app.screen is not settings
                and getattr(app.focused, "id", None)
                == "settings-provider-return-without-save"
            ),
            "a fresh Settings screen with focus on Return without saving",
        )
        fresh = app.screen
        assert fresh._provider_same_target_has_draft()
        # The route still has no field to focus: none named, or a disabled one.
        named = fresh._provider_return_target.field
        assert named is None or (
            named == "api_key"
            and fresh.query_one("#settings-provider-api-key", Input).disabled
        )
        assert not fresh.query_one("#settings-provider-return-continuation").display
        assert _fully_on_screen(fresh, app.focused)

        # Enter, Enter is the confirmation's default answer; a third Enter
        # must ask again, not leave Settings by the nav bar.
        await _keep_editing(app, pilot, fresh)
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConfirmationDialog),
            "the discard confirmation",
        )
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
                and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console's card to hold focus again, its return claimed",
        )
        assert _saved_config() == saved_before


@pytest.mark.asyncio
@private_profile_test
async def test_keep_editing_gives_focus_back_on_chat_settings_return_too(
    monkeypatch, request
):
    """Return without saving is one handler for both kinds. Chat settings'
    Configure credential, with an unsaved edit to its provider, gets the same
    answer to Keep editing: focus back on the action, nothing discarded, the
    modal's own handoff still pending."""
    monkeypatch.setattr(
        anthropic_subscription,
        "subscription_credential_status",
        lambda **_kwargs: "missing",
    )
    app = _cloud_app(monkeypatch, **_SUBSCRIPTION_CHAT)
    app.providers_models = {"anthropic": ["claude-sonnet-5"]}

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
        )
        console = app._navigation_outgoing_screen()
        assert await console._open_console_settings() is True
        await _until(
            pilot,
            lambda: bool(app.screen.query("#console-settings-configure-credential")),
            "Chat settings to offer Configure credential",
        )
        app.screen.query_one("#console-settings-configure-credential").focus()
        await pilot.pause()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == "settings-provider-auth-source"
            ),
            "Settings to open with focus on Sign in with",
        )
        settings = app.screen
        assert settings._provider_return_target.session_id is None
        settings.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()
        await pilot.press("x")
        await _until(
            pilot,
            settings._provider_same_target_has_draft,
            "the typed edit to be unsaved",
        )
        settings.query_one("#settings-provider-return-without-save", Button).focus()
        await _until(
            pilot,
            lambda: (
                getattr(app.focused, "id", None)
                == "settings-provider-return-without-save"
            ),
            "focus on Return without saving",
        )

        await _keep_editing(app, pilot, settings)


#: row -> (saved config, the card's action, the control focus arrives on).
_CUSTOM_ENDPOINT_ROWS = {
    "no-models": (
        {
            "chat_defaults": {"provider": "custom-ep:lab", "model": ""},
            "custom_endpoints.lab": {
                "display_name": "Lab",
                "family": "openai_compatible",
                "base_url": "http://127.0.0.1:9/v1",
                "models": [],
            },
        },
        "Choose model",
        "settings-provider-edit-custom-endpoint",
    ),
    "entry-gone": (
        {"chat_defaults": {"provider": "custom-ep:gone", "model": ""}},
        "Choose provider",
        "settings-provider-search",
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("row", list(_CUSTOM_ENDPOINT_ROWS))
@private_profile_test
async def test_a_custom_endpoint_row_with_no_field_to_focus_arrives_where_it_can_act(
    monkeypatch, request, row: str
):
    """Renamed on purpose from
    ``test_a_custom_endpoint_row_that_names_no_field_arrives_where_it_can_act``:
    a row whose blocker is its model now names the Default model field, which
    Settings locks for a custom endpoint. The landings are unchanged.

    A NEEDS SETUP row in Switch model opens Settings too, and for a custom
    endpoint it has no field to focus: nothing in Providers & Models fixes an
    endpoint that lists no models, or one whose entry was deleted. No return
    can name a custom endpoint either (R35), so nothing took focus and it
    stayed on the nav bar's Home button, where one more Enter opened Home.

    With its entry in place focus goes to Edit this endpoint in Custom
    endpoints, and Enter there opens the editor. With the entry gone that
    button is hidden, and focus goes to the Provider control: Enter there does
    nothing until its list is open. The deleted id is the saved default here,
    so the route ends when another provider is chosen and saved.

    Each landing must be painted, the editor's URL field too. Focus also lands
    on a widget inside a hidden row (Textual's focusable does not read an
    ancestor's display), and the Select that backs the Provider control is
    one: its id took focus, Enter opened a list nobody saw, and Down then
    Enter picked a provider blind."""
    saved, action, arrives_on = _CUSTOM_ENDPOINT_ROWS[row]
    provider = saved["chat_defaults"]["provider"]
    app = _cloud_app(monkeypatch, **saved)

    def in_find() -> bool:
        """Switch model has the screen and its Find box takes the keys."""
        return (
            isinstance(app.screen, ConsoleModelPopover)
            and getattr(app.focused, "id", None) == "console-popover-find"
        )

    def on_the_endpoints_row() -> bool:
        if not in_find():
            return False
        highlighted = app.screen.highlighted_row()
        # The row may spell the id custom_ep:<slug>; both name one endpoint.
        return (
            highlighted is not None
            and highlighted.kind == "setup"
            and canonical_custom_endpoint_id(highlighted.provider) == provider
        )

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
        )
        assert str(app.focused.label) == action
        await pilot.press("enter")
        if row == "entry-gone":
            # Find brings the id's own row under the highlight.
            await _until(pilot, in_find, "Switch model to open on Find")
            await pilot.press(*"gone")
        await _until(
            pilot, on_the_endpoints_row, "the endpoint's NEEDS SETUP row to be the one"
        )
        assert app.screen.highlighted_row().note == "Enter: open Settings"

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == arrives_on
            ),
            f"Settings to open with focus on {arrives_on}",
        )
        settings = app.screen
        # R35: no return can name a custom-ep: id, so this route goes one way.
        assert settings._provider_return_target is None
        assert not app.pending_handoffs.has_pending(CHANNEL)
        assert not settings.query_one("#settings-provider-return-continuation").display
        assert canonical_custom_endpoint_id(settings._provider_widget_value()) == (
            provider
        )
        edit = settings.query_one("#settings-provider-edit-custom-endpoint", Button)
        assert edit.display is (row == "no-models")
        await _until(
            pilot,
            lambda: _fully_on_screen(settings, app.focused),
            f"{arrives_on} to be painted while it holds focus",
        )

        await pilot.press("enter")
        if row == "no-models":
            await _until(
                pilot,
                lambda: _editing_the_endpoint(app, settings),
                "the Custom endpoints editor to open on the endpoint's URL, painted",
            )
            return
        # Enter chooses only from the open list: no Home, and nothing edited.
        await pilot.pause()
        assert app.screen is settings and app.focused.id == arrives_on
        category = settings._active_category_id()
        assert not settings._category_has_unsaved_changes(category)
        providers = settings.query_one("#settings-provider-picker")
        assert not providers.display
        await pilot.press("down")
        await _until(pilot, lambda: providers.display, "the Provider list to open")
        # The route ends where another provider is chosen, not at the open list.
        await pilot.press(*"OpenAI")
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                settings._provider_widget_value() == "openai"
                and settings._category_has_unsaved_changes(category)
            ),
            "OpenAI to be the chosen provider, as an unsaved edit",
        )
        await pilot.press("escape")  # leave the control, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: (
                _saved_config()["chat_defaults"]["provider"] == "openai"
                and not settings._category_has_unsaved_changes(category)
            ),
            "the saved default to be OpenAI, with nothing left unsaved",
        )


@pytest.mark.asyncio
@private_profile_test
async def test_an_unused_return_lapses_when_the_user_walks_back_by_hand(
    monkeypatch, request
):
    """A return that was not taken does not stay armed. Walking back to the
    Console by hand takes the handoff back, and the next Settings visit no
    longer holds a return, so a later save cannot offer one to the old chat.

    The same walk measures that "leaving keeps the draft" has no snapshot
    exception: Settings keeps an unsaved snapshot preference when it is left
    by the nav bar (TASK-33007, abd77af7c9), and the next visit shows it,
    still unsaved."""
    app = _cloud_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, _session, settings = await _recover_from_the_card(app, pilot)
        snapshots = settings.query_one("#settings-snapshot-enabled", Checkbox)
        snapshot_saved = snapshots.value
        snapshots.toggle()
        await _until(
            pilot, settings._snapshot_preferences_dirty, "an unsaved snapshot edit"
        )

        settings.post_message(NavigateToScreen(TAB_CHAT))  # as the nav bar does
        await _until(
            pilot,
            lambda: (
                app.screen is console and not app.pending_handoffs.has_pending(CHANNEL)
            ),
            "the Console to take its unused return back",
        )
        assert console._pending_console_provider_return is None

        console.post_message(NavigateToScreen(TAB_SETTINGS))
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and app.screen is not settings
                and bool(app.screen.query("#settings-provider-return-continuation"))
                and bool(app.screen.query("#settings-provider-value SelectOverlay"))
                and bool(app.screen.query("#settings-snapshot-enabled"))
                and app.screen._snapshot_preferences_loaded is not None
            ),
            "a fresh Settings screen on Providers & Models",
        )
        await pilot.pause()  # Let the fresh screen finish mounting before exit.
        assert app.screen._provider_return_target is None
        assert not app.screen.query_one(
            "#settings-provider-return-continuation"
        ).display
        # Kept by Settings: the old screen's edit is back, still unsaved, so
        # the category carries its "*".
        fresh_snapshots = app.screen.query_one("#settings-snapshot-enabled", Checkbox)
        assert fresh_snapshots.value is not snapshot_saved
        assert app.screen._category_has_unsaved_changes(
            app.screen._active_category_id()
        )


@pytest.mark.asyncio
@private_profile_test
async def test_a_field_edited_after_the_save_waits_in_settings_as_an_unsaved_edit(
    monkeypatch, request
):
    """The guide's after-save sentences, on the real app.

    A later save of only a snapshot preference leaves the way back as the
    recovery's save left it: same outcome, same status, focus not moved.
    Edits made after the save keep Return to Console, and the return neither
    asks about them nor saves them. A field edit (another key) waits in
    Settings as an unsaved edit, so the next recovery, for another provider,
    opens on the conflict panel; a snapshot preference waits there too, kept
    by Settings (TASK-33007, abd77af7c9). Saving the waiting edit from the panel is not that
    recovery's save: the page moves on to its provider as on arrival, with
    focus in the named field, not on the button, and the next edit asks first
    again."""
    app = _cloud_app(monkeypatch)
    later_key = _KEY[::-1]

    def offered(screen) -> tuple[bool, bool]:
        """(Return to Console shows, Return without saving shows)."""
        return (
            screen.query_one("#settings-provider-return-continuation").display,
            screen.query_one("#settings-provider-return-without-save").display,
        )

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        await pilot.press(*_KEY)
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        assert _saved_openai_key() == _KEY

        # A save of only a snapshot preference leaves the way back as it was
        # before that edit, here as the recovery's own save left it.
        saved_status = settings.query_one(
            "#settings-provider-return-continuation-status", Static
        )
        after_the_save = str(saved_status.renderable)
        snapshots = settings.query_one("#settings-snapshot-enabled", Checkbox)
        snapshots.toggle()
        await _until(
            pilot, settings._snapshot_preferences_dirty, "an unsaved snapshot edit"
        )
        assert offered(settings) == (True, False)
        await pilot.press("s")
        await _until(
            pilot,
            lambda: (
                not settings._snapshot_preferences_saving
                and not settings._snapshot_preferences_dirty()
            ),
            "the snapshot preference's save to land",
        )
        await pilot.pause()
        assert settings._provider_return_outcome is SAVED
        assert offered(settings) == (True, False)
        assert str(saved_status.renderable) == after_the_save
        assert app.focused.id == "settings-provider-return"
        saved = _saved_config()
        assert saved["llamacpp_snapshots"]["enabled"] is snapshots.value

        settings.query_one("#settings-provider-api-key", Input).focus()
        await pilot.press(*later_key)
        snapshot_saved = snapshots.value
        snapshots.toggle()
        await _until(
            pilot,
            lambda: (
                settings._provider_same_target_has_draft()
                and settings._snapshot_preferences_dirty()
            ),
            "both later edits to be unsaved",
        )
        assert offered(settings) == (True, False)
        settings.query_one("#settings-provider-return", Button).focus()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and console._is_descendant_or_self(
                    app.focused, console.query_one("#console-native-composer")
                )
            ),
            "the Console's composer, with no question asked",
        )
        assert _saved_config() == saved  # Neither later edit was written.

        open_provider_setup(console, "anthropic", field="api_key")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and app.screen is not settings
                and bool(app.screen.query("#settings-provider-value SelectOverlay"))
                and bool(app.screen.query("#settings-snapshot-enabled"))
                and app.screen._snapshot_preferences_loaded is not None
                and app.screen.query_one(
                    "#settings-provider-navigation-conflict"
                ).display
            ),
            "a fresh Settings screen on the conflict panel",
        )
        fresh = app.screen
        assert fresh._provider_draft_value("api_key") == later_key
        assert fresh._provider_widget_value() == "openai"
        assert offered(fresh) == (False, False)  # The panel owns the answer.
        fresh_snapshots = fresh.query_one("#settings-snapshot-enabled", Checkbox)
        # Kept by Settings: the old screen's snapshot edit is back, unsaved.
        assert fresh_snapshots.value is not snapshot_saved

        fresh.query_one("#settings-provider-conflict-review", Button).press()
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-api-key",
            "the review to put focus in the waiting edit's field",
        )
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: _saved_openai_key() == later_key and offered(fresh)[0],
            "the waiting edit's save to land",
        )
        assert fresh._provider_navigation_conflict_target is None
        assert fresh._provider_return_outcome is None
        assert fresh._provider_widget_value() == "anthropic"
        assert offered(fresh) == (True, False)
        status = fresh.query_one(
            "#settings-provider-return-continuation-status", Static
        )
        assert str(status.renderable) == (
            f"No unsaved changes. Return to Console to continue in “{session.title}”."
        )
        assert app.focused.id == "settings-provider-api-key"

        fresh_snapshots.toggle()
        await _until(
            pilot,
            lambda: offered(fresh) == (False, True),
            "the next edit to ask first again",
        )


# --- TASK-33008.2: the real app, NEEDS SETUP row -> Settings -> Switch model --

_GROQ_MODEL = "llama-3.3-70b-versatile"
#: Groq's second model: never Find's best match for "Groq ", so only the
#: highlight the return asks for puts the reopened list on it.
_GROQ_SECOND = "llama-3.1-8b-instant"


def _switcher_app(monkeypatch, **saved):
    """The real app: OpenAI is the default and has a key, so the chat can
    send. Groq lists two models and has no key, so Switch model puts it under
    NEEDS SETUP. No listing needs the network: both model lists are saved."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = _cloud_app(
        monkeypatch,
        **{
            "api_settings.openai": {"api_key": _KEY},
            "api_settings.groq": {"api_key": ""},
            **saved,
        },
    )
    app.providers_models = {"openai": ["gpt-5"], "groq": [_GROQ_MODEL, _GROQ_SECOND]}
    return app


def _groq_row(app, model: str = _GROQ_MODEL) -> tuple[str, str] | None:
    """The highlighted row's kind and readiness words, once Switch model has
    the screen with focus in Find and Groq's pair on ``model`` is the
    highlight."""
    switcher = app.screen
    if not isinstance(switcher, ConsoleModelPopover) or (
        getattr(app.focused, "id", None) != "console-popover-find"
    ):
        return None
    row = switcher.highlighted_row()
    if row is None or (row.provider, row.model) != ("groq", model):
        return None
    pairs = switcher.query_one("#console-popover-pairs").region
    painted = [strip.text for strip in switcher._compositor.render_strips()]
    line = next((text for text in painted[pairs.y : pairs.bottom] if model in text), "")
    words = next(
        (
            words
            for words in ("Not ready · no key", "Ready · not tested")
            if words in line
        ),
        "",
    )
    return row.kind, words


async def _leave_a_groq_row_for_settings(app, pilot, console):
    """Alt+M, find Groq's NEEDS SETUP rows, Down to the second, Enter:
    Settings opens on Groq's key, for that row's model."""
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    await pilot.press("alt+m")
    await _until(
        pilot,
        lambda: getattr(app.focused, "id", None) == "console-popover-find",
        "Switch model to open on Find",
    )
    first = app.screen
    await pilot.press(*"groq")
    await _until(
        pilot,
        lambda: _groq_row(app) == ("setup", "Not ready · no key"),
        "Groq's NEEDS SETUP row to be the highlight",
    )
    await pilot.press("down")
    await _until(
        pilot,
        lambda: _groq_row(app, _GROQ_SECOND) == ("setup", "Not ready · no key"),
        "Groq's second NEEDS SETUP row to be the highlight",
    )
    assert first.highlighted_row().note == "Enter: add key in Settings"

    await pilot.press("enter")
    await _until(
        pilot,
        lambda: (
            isinstance(app.screen, SettingsScreen)
            and getattr(app.focused, "id", None) == "settings-provider-api-key"
        ),
        "Settings to open with focus in Groq's API key field",
    )
    settings = app.screen
    target = settings._provider_return_target
    assert target is not None
    assert (target.provider, target.model, target.field, target.session_id) == (
        "groq",
        _GROQ_SECOND,
        "api_key",
        session_id,
    )
    assert settings._provider_widget_value() == "groq"
    # R43: the return names the chat and the row's pair, nothing else.
    claim = app.pending_handoffs.claim(CHANNEL)
    assert claim is not None
    assert claim.value == ConsoleProviderReturnIntent(session_id, "groq", _GROQ_SECOND)
    app.pending_handoffs.release(claim)
    return first, settings


def _chat_pair(console) -> tuple[str, str | None]:
    chat, _readiness = console._active_console_settings_readiness()
    return chat.provider, chat.model


@pytest.mark.asyncio
@pytest.mark.parametrize("saves", [True, False], ids=["saves-a-key", "without-saving"])
@private_profile_test
async def test_a_needs_setup_row_round_trip_reopens_switch_model_on_its_provider(
    monkeypatch, request, saves: bool
):
    """AC#1-AC#4 (R47): the real store, Settings and Switch model on a scratch
    config. Enter on Groq's NEEDS SETUP row opens Settings at its API key.

    With the key saved, Return to Console reopens Switch model with "Groq "
    in Find and the row's pair highlighted, now "Ready · not tested": the
    chat holds work, so it kept openai · gpt-5 through the save (ADR-095),
    and one Enter moves it to the pair.

    Return without saving reopens it the same way, unchanged: the row is
    still NEEDS SETUP, and Esc leaves the chat on its own pair with the
    composer in focus.

    The row is Groq's second: Find's best match for "Groq " is its first, so
    the reopened highlight is on the row's pair only because the return
    carries that pair all the way to the list."""
    app = _switcher_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-native-composer",
            "the composer to hold focus",
        )
        console = app._navigation_outgoing_screen()
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(session_id, role=ConsoleMessageRole.USER, content="hi")
        assert _chat_pair(console) == ("openai", "gpt-5")

        first, settings = await _leave_a_groq_row_for_settings(app, pilot, console)
        await pilot.press(*_KEY)  # the same text as OpenAI's saved key
        if saves:
            await pilot.press("escape")  # leave the field, so s is Save
            await pilot.press("s")
            await _until(
                pilot,
                lambda: getattr(app.focused, "id", None) == "settings-provider-return",
                "focus on the Return to Console action after the save",
            )
            # The key went to Groq's table; OpenAI's is as it was.
            assert _saved_config()["api_settings"]["groq"]["api_key"] == _KEY
            assert _saved_openai_key() == _KEY
            await pilot.press("enter")
        else:
            without_saving = settings.query_one(
                "#settings-provider-return-without-save", Button
            )
            await _until(pilot, lambda: without_saving.display, "the unsaved key")
            without_saving.focus()
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: isinstance(app.screen, ConfirmationDialog),
                "the discard confirmation",
            )
            app.screen.query_one("#confirm-button", Button).press()

        await _until(
            pilot,
            lambda: (
                _groq_row(app, _GROQ_SECOND)
                == (
                    ("pair", "Ready · not tested")
                    if saves
                    else ("setup", "Not ready · no key")
                )
            ),
            "Switch model to reopen on the row's own pair, with its readiness words",
        )
        reopened = app.screen
        assert reopened is not first
        assert reopened.query_one("#console-popover-find", Input).value == "Groq "
        assert reopened._reopened == "groq"  # the highlight stays on Groq's rows
        assert reopened._scope_copy == model_switcher.SCOPE_THIS_CHAT
        assert not app.pending_handoffs.has_pending(CHANNEL)
        assert store.active_session_id == session_id
        # The chat holds work: the save did not move it (ADR-095).
        assert _chat_pair(console) == ("openai", "gpt-5")

        await pilot.press("enter" if saves else "escape")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-native-composer"
                and (_chat_pair(console) == ("groq", _GROQ_SECOND)) is saves
            ),
            "the Console again, composer in focus, on the pair this path ends on",
        )
        chat, readiness = console._active_console_settings_readiness()
        if saves:
            assert (chat.provider, chat.model) == ("groq", _GROQ_SECOND)
            assert readiness.operability == "ready_to_send"
        else:
            assert (chat.provider, chat.model) == ("openai", "gpt-5")
            assert not _saved_config()["api_settings"]["groq"].get("api_key")
        assert store.active_session_id == session_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, model_first",
    [("needs-setup-row", False), ("chat-settings", False), ("needs-setup-row", True)],
    ids=["needs-setup-row", "chat-settings", "model-picked-first"],
)
@private_profile_test
async def test_a_routed_providers_key_is_typed_under_its_own_name(
    monkeypatch, request, route: str, model_first: bool
):
    """The controller's ruling of 2026-10-10 (Option A), on the real Settings
    screen: Groq is routed while OpenAI is the saved default, from a NEEDS
    SETUP row and from Chat settings' Configure credential.

    The first typed character made the Provider row read "OpenAI" on Groq's
    page, and Chat settings' route then offered no Return without saving.
    The edit was read against OpenAI's saved key, so typing that same text
    for Groq read as no change. A model picked on the page did the same to
    the row, and to any key typed after it. The row keeps the routed
    provider's name, the key reads "edited *" against Groq's own saved key
    (none), and the save writes Groq's table."""
    app = _switcher_app(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-native-composer",
            "the composer to hold focus",
        )
        console = app._navigation_outgoing_screen()
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(session_id, role=ConsoleMessageRole.USER, content="hi")
        if route == "needs-setup-row":
            _first, settings = await _leave_a_groq_row_for_settings(app, pilot, console)
            model = _GROQ_SECOND
        else:
            store.replace_session_settings(
                session_id, ConsoleSessionSettings(provider="groq", model=_GROQ_MODEL)
            )
            assert await console._open_console_settings() is True
            await _until(
                pilot,
                lambda: bool(
                    app.screen.query("#console-settings-configure-credential")
                ),
                "Chat settings to offer Configure credential",
            )
            app.screen.query_one("#console-settings-configure-credential").focus()
            await pilot.pause()
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: (
                    isinstance(app.screen, SettingsScreen)
                    and getattr(app.focused, "id", None) == "settings-provider-api-key"
                ),
                "Settings to open with focus in Groq's API key field",
            )
            settings = app.screen
            assert settings._provider_return_target.session_id is None
            model = _GROQ_MODEL
        provider_row = settings.query_one("#settings-provider-search", Input)
        key_word = settings.query_one("#settings-provider-key-status", Static)
        without_saving = settings.query_one(
            "#settings-provider-return-without-save", Button
        )

        def shown() -> tuple[str, str, bool]:
            return provider_row.value, str(key_word.renderable), without_saving.display

        assert shown()[0] == "Groq" and shown()[1] != "edited *"

        if model_first:
            # Groq's other model, chosen in the Default model picker.
            model = _GROQ_MODEL
            settings.query_one(ModelSearchPicker).post_message(
                ModelSearchPicker.ModelSelected(model)
            )
            await _until(
                pilot,
                lambda: without_saving.display,
                "the picked model to be an unsaved edit",
            )
            assert shown()[0] == "Groq"

        await pilot.press(_KEY[0])
        await _until(
            pilot,
            lambda: shown()[1] == "edited *",
            "the first typed character to be an unsaved edit",
        )
        assert shown() == ("Groq", "edited *", True)
        # The rest makes it the same text as OpenAI's saved key.
        assert _saved_openai_key() == _KEY
        await pilot.press(*_KEY[1:])
        await pilot.pause()
        assert settings.query_one("#settings-provider-api-key", Input).value == _KEY
        assert shown() == ("Groq", "edited *", True)
        assert settings._provider_same_target_has_draft()

        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the return action after the save",
        )
        saved = _saved_config()
        assert saved["api_settings"]["groq"]["api_key"] == _KEY
        assert _saved_openai_key() == _KEY
        defaults = saved["chat_defaults"]
        assert (defaults["provider"], defaults["model"]) == ("groq", model)
        assert provider_row.value == "Groq"


@pytest.mark.asyncio
@private_profile_test
async def test_a_card_opened_rows_return_reopens_switch_model_over_the_card(
    monkeypatch, request
):
    """AC#3 (R44, R46) on first run: the Get started card's Choose provider
    opens Switch model, and a NEEDS SETUP row there leaves for Settings. The
    arrival Return to Console saves nothing, so the card still blocks, and
    Alt+M stays shut under it. Switch model reopens all the same, in the
    card's mode: its scope line still says the pick also becomes the default
    for new chats. Esc puts focus back on the card's action."""
    app = _switcher_app(
        monkeypatch, chat_defaults={"provider": "custom-ep:gone", "model": ""}
    )

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
        )
        assert str(app.focused.label) == "Choose provider"
        console = app._navigation_outgoing_screen()
        session_id = console._ensure_console_chat_store().active_session_id
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-popover-find",
            "Switch model to open on Find",
        )
        first = app.screen
        assert first._scope_copy == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        await pilot.press(*"groq")
        await _until(
            pilot,
            lambda: _groq_row(app) == ("setup", "Not ready · no key"),
            "Groq's NEEDS SETUP row to be the highlight",
        )
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and getattr(app.focused, "id", None) == "settings-provider-api-key"
            ),
            "Settings to open with focus in Groq's API key field",
        )
        settings = app.screen
        settings.query_one("#settings-provider-return", Button).focus()
        await pilot.press("enter")

        await _until(
            pilot,
            lambda: _groq_row(app) == ("setup", "Not ready · no key"),
            "Switch model to reopen on Groq's row, still NEEDS SETUP",
        )
        reopened = app.screen
        assert reopened is not first
        assert reopened.query_one("#console-popover-find", Input).value == "Groq "
        assert reopened._scope_copy == model_switcher.SCOPE_THIS_CHAT_AND_NEW_CHATS
        assert console._console_setup_modal_blocking()
        assert not app.pending_handoffs.has_pending(CHANNEL)

        await pilot.press("escape")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
            ),
            "the card's action to hold focus again",
        )
        assert console._console_setup_modal_blocking()
        assert console._ensure_console_chat_store().active_session_id == session_id
        provider, model = _chat_pair(console)
        # The chat may spell the id custom_ep:<slug>; both name one endpoint.
        assert (canonical_custom_endpoint_id(provider), model) == (
            "custom-ep:gone",
            None,
        )
        assert not _saved_config()["api_settings"]["groq"].get("api_key")
