"""Exact ownership and live ports for Console roleplay durability."""

import asyncio
import weakref
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
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
