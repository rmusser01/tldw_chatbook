"""TASK-33008.1: a Console provider recovery returns to the chat it left.

The unit tests drive the Console's consumer (``Console_Modules/
provider_return.py``) and the Settings copy (``Settings_Modules/
provider_return.py``) with the real handoff store and the real Console chat
store. The two private-profile tests run the real app: card -> Settings ->
key -> save -> Enter, and Return without saving.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app as _build_production_app
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
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
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Settings_Modules.provider_return import (
    INACTIVE_COPY,
    continuation_copy,
    fitted_chat_title,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

CHANNEL = HandoffChannel.CONVERSATION_SETTINGS_RETURN
SAVED = ConversationSettingsReturnOutcome.CREDENTIAL_SAVED


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


def _settings(target, outcome, store=None):
    """What the Settings copy reads from its screen."""
    return SimpleNamespace(
        _provider_return_target=target,
        _provider_return_outcome=outcome,
        _custom_endpoints_store=lambda: store,
    )


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
    assert continuation_copy(_settings(console, None, store)) == INACTIVE_COPY
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


def _saved_openai_key() -> object:
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    # The scratch profile, never the user's own config.
    assert config_path.is_relative_to(Path(os.environ["TLDW_TEST_CONFIG_ROOT"]))
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return saved.get("api_settings", {}).get("openai", {}).get("api_key")


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
        # R37: nothing to return with until something is saved.
        assert settings.query_one("#settings-provider-return-continuation").display is (
            False
        )

        # Another chat is active by the time Settings returns, so only the
        # Console's consumer can put the user back on the chat they left.
        store = console._ensure_console_chat_store()
        other = store.create_session(title="Other")
        assert store.active_session_id == other.id != session.id

        await pilot.press(*_KEY)
        await pilot.press("escape")  # leave the field, so s is Save
        await pilot.press("s")
        await _until(
            pilot,
            lambda: getattr(app.focused, "id", None) == "settings-provider-return",
            "focus on the Return to Console action after the save",
        )
        assert str(app.focused.label) == "Return to Console"
        status = settings.query_one(
            "#settings-provider-return-continuation-status", Static
        )
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
