"""Exact ownership and live ports for Console roleplay durability."""

import asyncio
import weakref
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import ConsoleSettingsAction
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultMutationOutcome,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultSavePhase,
    RuntimeConfigPublicationResult,
)
from tldw_chatbook.UI.Console_Modules import settings_durability
from tldw_chatbook.UI.Console_Modules.settings_durability import (
    ConsoleSettingsDurabilityController,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


ROLEPLAY_STATE = {
    "_last_console_roleplay_refresh_key": None,
    "_console_roleplay_persistence_task": None,
    "_console_roleplay_writer_task": None,
    "_console_roleplay_active_plan": None,
    "_console_roleplay_pending_plan": None,
    "_console_roleplay_drain_scheduled": False,
    "_console_roleplay_tearing_down": False,
    "_console_roleplay_repair_generation": 0,
    "_console_roleplay_repair_inflight_generation": 0,
    "_console_roleplay_repair_plan": None,
}


def test_roleplay_methods_have_one_durability_owner_and_keep_dynamic_screen_hook():
    methods = (
        "_global_chat_display_name",
        "_apply_console_settings_result",
        "_refresh_console_roleplay_projections",
        "_drain_console_roleplay_persistence",
        "_await_console_roleplay_persistence_task",
        "_finish_console_roleplay_persistence_task",
        "_console_roleplay_unmount_timeout_seconds",
        "_publish_console_roleplay_repair_marker",
        "_teardown_console_roleplay_persistence",
        "_start_console_roleplay_persistence_drain",
        "_dispatch_active_console_roleplay_refresh",
        "_consume_pending_console_roleplay_repair",
    )
    for name in methods:
        assert name in ConsoleSettingsDurabilityController.__dict__
        if name != "_consume_pending_console_roleplay_repair":
            assert name not in ChatScreen.__dict__
    screen = ChatScreen(_build_test_app())
    calls = []
    screen._settings_durability = SimpleNamespace(
        _consume_pending_console_roleplay_repair=lambda: calls.append("current") or True
    )
    assert screen._consume_pending_console_roleplay_repair() is True
    assert calls == ["current"]


@pytest.mark.parametrize("name, default", ROLEPLAY_STATE.items())
def test_roleplay_state_is_owned_once_and_screen_assignment_remains_writable(
    name, default
):
    screen = ChatScreen(_build_test_app())
    owner = screen._settings_durability
    assert name not in vars(screen)
    assert getattr(owner, name) == default
    marker = object()
    setattr(screen, name, marker)
    assert getattr(owner, name) is marker
    replacement = object()
    setattr(owner, name, replacement)
    assert getattr(screen, name) is replacement


def test_durability_wiring_resolves_replaced_app_store_and_mounted_state(monkeypatch):
    screen = ChatScreen(_build_test_app())
    owner = screen._settings_durability
    framework = [object()]
    mounted = [False]
    current_store = [None]
    monkeypatch.setattr(ChatScreen, "app", property(lambda _self: framework[0]))
    monkeypatch.setattr(ChatScreen, "is_mounted", property(lambda _self: mounted[0]))
    monkeypatch.setattr(
        ChatScreen, "_console_chat_store", property(lambda _self: current_store[0])
    )

    def never_ensure():
        raise AssertionError("current-store inspection must not create a store")

    screen._ensure_console_chat_store = never_ensure
    screen.app_instance = SimpleNamespace(
        app_config={"chat_defaults": {"user_display_name": "First"}}
    )
    assert owner._global_chat_display_name() == "First"
    assert owner.app is framework[0]
    assert owner.app is not owner.app_instance
    assert owner.is_mounted is False
    assert owner._dispatch_active_console_roleplay_refresh() is False
    framework[0] = object()
    mounted[0] = True
    current_store[0] = SimpleNamespace(active_session_id=None)
    screen.app_instance = SimpleNamespace(
        app_config={"chat_defaults": {"user_display_name": "Second"}}
    )
    assert owner._global_chat_display_name() == "Second"
    assert owner.app is framework[0]
    assert owner.is_mounted is True
    assert owner._console_chat_store is current_store[0]
    assert owner._dispatch_active_console_roleplay_refresh() is False


def _owner(app, framework):
    def unused(*args, **kwargs):
        raise AssertionError("unrelated durability dependency entered")

    return ConsoleSettingsDurabilityController(
        app_instance_accessor=lambda: app,
        app_accessor=lambda: framework,
        is_mounted_accessor=lambda: False,
        current_console_chat_store_accessor=lambda: None,
        _ensure_console_chat_controller=unused,
        _ensure_console_chat_store=unused,
        _provider_readiness_app_config=unused,
        _sync_console_identity_surfaces=unused,
        _sync_console_settings_recovery_surfaces=unused,
        _sync_native_console_chat_ui=unused,
        run_worker=unused,
    )


def test_runtime_display_name_view_hook_resolves_replaced_durability_owner():
    screen = ChatScreen(_build_test_app())
    display_name = screen.console_view_hooks()["_global_user_display_name"]
    screen._settings_durability = SimpleNamespace(
        _global_chat_display_name=lambda: "Replacement owner"
    )
    assert display_name() == "Replacement owner"


def test_settings_apply_uses_replaced_provider_store_and_app_ports():
    screen = ChatScreen(_build_test_app())
    owner = screen._settings_durability
    calls = []
    previous = ConsoleSessionSettings(
        provider="openai", system_prompt="Keep system", pinned_prefill="Keep prefill"
    )
    store = SimpleNamespace(
        active_session_id=None,
        session_settings=lambda session_id: previous,
        replace_session_settings=lambda session_id, settings: calls.append(
            (session_id, settings)
        ),
    )
    screen._ensure_console_chat_store = lambda: store
    screen._provider_selection = SimpleNamespace(
        _provider_readiness_app_config=lambda: calls.append("current provider") or {}
    )
    notifications = []
    screen.app_instance = SimpleNamespace(
        notify=lambda text, **kwargs: notifications.append((text, kwargs))
    )
    owner._apply_console_settings_result(
        ConsoleSessionSettings(provider="openai", model="new-model"),
        origin_session_id="origin",
        origin_system_prompt="Keep system",
        origin_pinned_prefill="Keep prefill",
    )
    session_id, settings = calls[0]
    assert session_id == "origin"
    assert settings.model == "new-model"
    assert settings.source == "user"
    assert settings.system_prompt == "Keep system"
    assert settings.pinned_prefill == "Keep prefill"
    assert calls[1:] == ["current provider"]
    assert notifications == [("Console settings saved.", {"severity": "success"})]


@pytest.mark.asyncio
async def test_repair_callback_keeps_only_framework_app_and_resolves_current_screen(
    monkeypatch,
):
    app = SimpleNamespace(app_config={})
    calls = []
    framework = SimpleNamespace(
        screen=SimpleNamespace(
            _consume_pending_console_roleplay_repair=lambda: calls.append("departed")
        )
    )
    scheduled = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop,
        "call_later",
        lambda delay, callback, *args: scheduled.append((delay, callback, args)),
    )
    owner = _owner(app, framework)
    owner._publish_console_roleplay_repair_marker()
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    assert app._console_roleplay_repair_generation == 1
    assert app._console_roleplay_repair_global_name == "User"
    delay, callback, args = scheduled.pop()
    assert delay == 0.1
    assert args == (framework,)
    framework.screen = SimpleNamespace(
        _consume_pending_console_roleplay_repair=lambda: calls.append("current")
    )
    callback(*args)
    assert calls == ["current"]


@pytest.mark.asyncio
async def test_empty_durability_teardown_does_not_create_shared_store():
    owner = _owner(SimpleNamespace(), object())
    await owner._teardown_console_roleplay_persistence()
    assert owner._console_roleplay_tearing_down is True
    assert owner._console_roleplay_persistence_task is None


def _default_recovery_case(monkeypatch, phase):
    """Keep real admission/state owners while controlling persistence boundaries."""
    intent = ConsoleDefaultMutationIntent(
        7,
        ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        "openai",
        "model",
        frozenset(),
        {},
        None,
    )
    state = ConsoleDefaultDurabilityState(
        newest_intent_generation=7,
        recovery_intent=intent,
        failure_phase=phase,
    )
    app = SimpleNamespace(console_default_durability_state=state)
    owner = _owner(app, object())
    calls = []
    sync_keys = []
    owner._sync_console_settings_recovery_surfaces = lambda: sync_keys.append(
        set(app.console_default_recovery_inflight)
    )

    def apply(current):
        calls.append("save")
        return ConsoleDefaultMutationOutcome(current.generation, True, True, {}, None)

    def refresh():
        calls.append("refresh")
        return RuntimeConfigPublicationResult(True, {}, None)

    async def publish(current, outcome):
        calls.append("publish")
        return owner._accept_console_default_runtime_publication(
            current.generation,
            current.action,
            outcome.settings_view,
        )

    monkeypatch.setattr(settings_durability, "apply_console_default_intent", apply)
    monkeypatch.setattr(
        settings_durability, "refresh_console_runtime_after_saved_default", refresh
    )
    owner._publish_console_default_outcome_off_event_loop = publish
    action = (
        ConsoleDefaultRecoveryAction.RETRY_SAVE
        if phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
        else ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
    )
    return owner, ConsoleDefaultRecoveryRequest(action, 7), calls, sync_keys


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", tuple(ConsoleDefaultSavePhase))
async def test_default_recovery_excludes_duplicates_through_publication(
    monkeypatch: pytest.MonkeyPatch, phase: ConsoleDefaultSavePhase
) -> None:
    """Keep duplicate recovery out until its exact publication finishes.

    Args:
        monkeypatch: Replace persistence boundaries with controlled outcomes.
        phase: Failed owner whose recovery action is exercised.
    """
    owner, request, calls, sync_keys = _default_recovery_case(monkeypatch, phase)
    entered, release = asyncio.Event(), asyncio.Event()
    publish = owner._publish_console_default_outcome_off_event_loop

    async def held_publish(intent, outcome):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await publish(intent, outcome)

    owner._publish_console_default_outcome_off_event_loop = held_publish
    first = asyncio.create_task(owner._handle_console_default_recovery(request))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = await asyncio.wait_for(
            owner._handle_console_default_recovery(request), 2
        )
        assert duplicate.recovery_intent is not None
        assert calls == [
            "save" if phase is ConsoleDefaultSavePhase.BEFORE_REPLACE else "refresh"
        ]
        assert owner.app_instance.console_default_recovery_inflight == {
            (7, phase.value)
        }
    finally:
        release.set()
        await first
        await owner._console_settings_durability_owner().close_and_drain()
    assert calls[-1] == "publish" and calls.count("publish") == 1
    assert sync_keys == [{(7, phase.value)}]
    assert owner.app_instance.console_default_recovery_inflight == set()
    assert owner._console_default_durability_state().recovery_intent is None
    assert owner.app_instance.console_new_chat_default_generation == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("worker", "publication", "cancel"))
async def test_default_recovery_releases_flight_after_failure_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Every interrupted recovery releases admission for a later valid retry.

    Args:
        monkeypatch: Inject failure at the selected persistence boundary.
        fault: Worker error, publication error, or publication cancellation.
    """
    phase = ConsoleDefaultSavePhase.BEFORE_REPLACE
    owner, request, calls, _ = _default_recovery_case(monkeypatch, phase)
    apply = settings_durability.apply_console_default_intent
    publish = owner._publish_console_default_outcome_off_event_loop

    def failed_apply(intent):
        raise RuntimeError("controlled save failure")

    async def failed_publish(intent, outcome):
        if fault == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("controlled publication failure")

    if fault == "worker":
        monkeypatch.setattr(
            settings_durability, "apply_console_default_intent", failed_apply
        )
    else:
        owner._publish_console_default_outcome_off_event_loop = failed_publish
    if fault == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await owner._handle_console_default_recovery(request)
    else:
        await owner._handle_console_default_recovery(request)
    assert owner.app_instance.console_default_recovery_inflight == set()
    state = owner._console_default_durability_state()
    assert state.recovery_intent is not None
    expected_phase = (
        ConsoleDefaultSavePhase.CACHE_PUBLICATION if fault == "publication" else phase
    )
    assert state.failure_phase is expected_phase
    monkeypatch.setattr(settings_durability, "apply_console_default_intent", apply)
    owner._publish_console_default_outcome_off_event_loop = publish
    action = (
        ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
        if fault == "publication"
        else request.action
    )
    await owner._handle_console_default_recovery(
        ConsoleDefaultRecoveryRequest(action, 7)
    )
    assert owner._console_default_durability_state().recovery_intent is None
    assert owner.app_instance.console_default_recovery_inflight == set()
    assert calls[-1] == "publish"
    await owner._console_settings_durability_owner().close_and_drain()


@pytest.mark.asyncio
async def test_default_recovery_waiter_cancellation_keeps_admitted_flight_until_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown drains accepted publication after its caller stops waiting.

    Args:
        monkeypatch: Hold publication behind a deterministic release event.
    """
    phase = ConsoleDefaultSavePhase.CACHE_PUBLICATION
    owner, request, calls, _ = _default_recovery_case(monkeypatch, phase)
    entered, release = asyncio.Event(), asyncio.Event()
    publish = owner._publish_console_default_outcome_off_event_loop

    async def held_publish(intent, outcome):
        entered.set()
        await release.wait()
        return await publish(intent, outcome)

    owner._publish_console_default_outcome_off_event_loop = held_publish
    waiter = asyncio.create_task(owner._handle_console_default_recovery(request))
    await asyncio.wait_for(entered.wait(), 2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    lifetime = owner._console_settings_durability_owner()
    closing = asyncio.create_task(lifetime.close_and_drain())
    try:
        await asyncio.sleep(0)
        assert not lifetime.accepting and not closing.done()
        assert owner.app_instance.console_default_recovery_inflight == {
            (7, phase.value)
        }
        await owner._handle_console_default_recovery(request)
        assert calls == ["refresh"]
    finally:
        release.set()
        await closing
    assert not lifetime.tasks
    assert owner.app_instance.console_default_recovery_inflight == set()
    assert owner._console_default_durability_state().recovery_intent is None


@pytest.mark.asyncio
async def test_default_recovery_old_publication_failure_preserves_newer_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An obsolete recovery cannot replace a newer default intent's state.

    Args:
        monkeypatch: Advance the app's generation during publication.
    """
    owner, request, _, _ = _default_recovery_case(
        monkeypatch, ConsoleDefaultSavePhase.BEFORE_REPLACE
    )
    newer = ConsoleDefaultDurabilityState(newest_intent_generation=8)

    async def superseded_publish(intent, outcome):
        owner.app_instance.console_default_durability_state = newer
        return False

    owner._publish_console_default_outcome_off_event_loop = superseded_publish
    assert await owner._handle_console_default_recovery(request) is newer
    assert owner.app_instance.console_default_recovery_inflight == set()
    await owner._console_settings_durability_owner().close_and_drain()
