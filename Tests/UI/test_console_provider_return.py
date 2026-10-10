"""TASK-33008.1: a Console provider recovery returns to the chat it left.

The unit tests drive the Console's consumer (``Console_Modules/
provider_return.py``) and the Settings offer and copy (``Settings_Modules/
provider_return.py``) with the real handoff store and the real Console chat
store. The private-profile tests run the real app: card -> Settings -> key ->
save -> Enter, the saved model an untouched chat comes back on, Return without
saving in every unsaved state, the return offered on arrival, a return that
lapses when the user walks back by hand, and edits made after the save.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app as _build_production_app
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
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
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

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


def _stage_return(screen, session_id: str) -> dict[str, object]:
    """Stage a recovery's return as the Console does; give Settings' answer."""
    intent = ConsoleProviderReturnIntent(session_id)
    revision = screen.app_instance.pending_handoffs.stage(CHANNEL, intent)
    return intent.return_target(revision, SAVED).to_context()


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


def _settings(
    target, outcome, store=None, *, dirty=False, conflict=None, own_draft=False
):
    """What the Settings offer and copy read from their screen.

    ``dirty`` is any unsaved Providers & Models edit; ``own_draft`` is the
    narrower case of a labelled edit to the target's own provider.
    """
    return SimpleNamespace(
        _provider_return_target=target,
        _provider_return_outcome=outcome,
        _provider_navigation_conflict_target=conflict,
        _category_has_unsaved_changes=lambda _category: dirty or own_draft,
        _provider_same_target_has_draft=lambda: own_draft and conflict is None,
        _custom_endpoints_store=lambda: store,
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


def _cloud_app_without_a_key(monkeypatch):
    """The real app on its scratch profile: OpenAI is the default, keyless."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_production_app(configured_default="chat")
    assert config_module.save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "chat_defaults": {"provider": "openai", "model": "gpt-5"},
            "api_settings.openai": {"api_key": ""},
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
    app = _cloud_app_without_a_key(monkeypatch)

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
@pytest.mark.parametrize("holds_work", [False, True], ids=["untouched", "holds-work"])
@private_profile_test
async def test_the_returning_chat_follows_a_newly_saved_model_only_while_untouched(
    monkeypatch, request, holds_work: bool
):
    """AC#3, second sentence (ADR-095, D1), on the return path: the visit
    saves another default model as well as the key. An untouched chat comes
    back on the saved provider · model; a chat that holds work keeps its own.
    The return adds no convergence code, so this is the Console's own sync."""
    app = _cloud_app_without_a_key(monkeypatch)
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
    app = _cloud_app_without_a_key(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        console, session, settings = await _recover_from_the_card(app, pilot)
        await pilot.press(*_KEY)
        await pilot.pause()
        without_saving = settings.query_one(
            "#settings-provider-return-without-save", Button
        )
        assert without_saving.display is True
        assert str(without_saving.label) == "Return without saving"

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
        app.screen.query_one("#confirm-button", Button).press()
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
            ),
            "the Console's card to hold focus again",
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
    app = _cloud_app_without_a_key(monkeypatch)

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
            ),
            "the Console's card to hold focus again",
        )
        assert console._ensure_console_chat_store().active_session_id == session.id
        assert _saved_config()["chat_defaults"]["provider"] == "openai"
        assert not _saved_openai_key()
        assert not app.pending_handoffs.has_pending(CHANNEL)


@pytest.mark.asyncio
@private_profile_test
async def test_a_route_that_names_no_model_keeps_settings_own_model(
    monkeypatch, request
):
    """A NEEDS SETUP row of a provider that lists no models routes no model.
    The return-bearing route must then leave Settings' own model in the
    field, as the one-way route did; only a routed model replaces it."""
    app = _cloud_app_without_a_key(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
        )
        open_provider_setup(app._navigation_outgoing_screen(), "openai")
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
        assert app.screen.query_one("#settings-model-value", Input).value == "gpt-5"
        assert target.pins_model is False


@pytest.mark.asyncio
@private_profile_test
async def test_a_recovery_that_saves_nothing_returns_from_the_arrival_action(
    monkeypatch, request
):
    """Parent AC#3: a recovery that needs no save (a cloud Retry's key test)
    still goes back in one action. The arrival return asks nothing, writes
    nothing and lands on the chat that opened Settings."""
    app = _cloud_app_without_a_key(monkeypatch)

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
            ),
            "the Console's card to hold focus again",
        )
        assert console._ensure_console_chat_store().active_session_id == session.id
        assert not _saved_openai_key()
        assert not app.pending_handoffs.has_pending(CHANNEL)
        assert settings._provider_return_target is None


@pytest.mark.asyncio
@private_profile_test
async def test_a_route_that_names_no_field_arrives_on_the_return(monkeypatch, request):
    """A cloud Retry names no field: the user tests the key with t and goes
    back. With no field to put focus in, the offered return takes it, so
    after the test one Enter returns (parent AC#3)."""
    app = _cloud_app_without_a_key(monkeypatch)

    async with app.run_test(size=(211, 44)) as pilot:
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "console-setup-modal-action",
            "the Get started card's action to hold focus",
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
        assert app.screen._provider_return_target.field is None

        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                app.screen is console
                and getattr(app.focused, "id", None) == "console-setup-modal-action"
            ),
            "the Console's card to hold focus again",
        )
        assert console._ensure_console_chat_store().active_session_id == session_id
        assert not app.pending_handoffs.has_pending(CHANNEL)


@pytest.mark.asyncio
@private_profile_test
async def test_an_unused_return_lapses_when_the_user_walks_back_by_hand(
    monkeypatch, request
):
    """A return that was not taken does not stay armed. Walking back to the
    Console by hand takes the handoff back, and the next Settings visit no
    longer holds a return, so a later save cannot offer one to the old chat.

    The same walk measures the guide's snapshot exception to "leaving keeps
    the draft": the screen alone holds an unsaved snapshot preference, so
    leaving Settings by the nav bar drops it."""
    app = _cloud_app_without_a_key(monkeypatch)

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
        # Loaded from the file again: the old screen's edit went with it, and
        # with nothing else unsaved the category carries no "*".
        fresh_snapshots = app.screen.query_one("#settings-snapshot-enabled", Checkbox)
        assert fresh_snapshots.value is snapshot_saved
        assert not app.screen._category_has_unsaved_changes(
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
    opens on the conflict panel; a snapshot preference is kept by the screen
    only and is dropped. Saving the waiting edit from the panel is not that
    recovery's save: the page moves on to its provider as on arrival, with
    focus in the named field, not on the button, and the next edit asks first
    again."""
    app = _cloud_app_without_a_key(monkeypatch)
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
        # Loaded from the file again: the old screen's edit went with it.
        assert fresh_snapshots.value is snapshot_saved

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
