# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace
import logging
import sys

import pytest
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Input, OptionList, Select

# Exercise the full production app in its supported "optional transcription
# backend absent" configuration. The installed parakeet-mlx wheel aborts the
# interpreter while importing MLX in this test runner, before Textual can mount.
_MISSING_MODULE = object()
_previous_parakeet_mlx = sys.modules.get("parakeet_mlx", _MISSING_MODULE)
sys.modules["parakeet_mlx"] = None

try:
    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import LLMProviderProvider, TldwCli
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleProviderIntent,
        HandoffChannel,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
finally:
    if _previous_parakeet_mlx is _MISSING_MODULE:
        sys.modules.pop("parakeet_mlx", None)
    else:
        sys.modules["parakeet_mlx"] = _previous_parakeet_mlx


PROVIDERS_MODELS = {
    "OpenAI": ["gpt-task-648"],
    "Anthropic": ["claude-task-648"],
}


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _save_initial_provider_config() -> None:
    adapter = SettingsConfigAdapter()
    assert adapter.save_values(
        "chat_defaults",
        {"provider": "OpenAI", "model": "gpt-task-648"},
    )
    assert adapter.save_values(
        "api_settings.openai",
        {"api_key": "TASK_648_TEST_KEY", "model": "gpt-task-648"},
    )
    assert adapter.save_values(
        "api_settings.anthropic",
        {"api_key": "TASK_648_TEST_KEY", "model": "claude-task-648"},
    )


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    _disable_splash(monkeypatch)
    _save_initial_provider_config()
    app = TldwCli()
    app.app_config = load_settings(force_reload=True)
    app.app_config["_first_run"] = False
    app.providers_models = dict(PROVIDERS_MODELS)
    app._initial_tab_value = TAB_CHAT
    return app


async def _wait_for_screen(app: TldwCli, pilot, screen_type):
    for _ in range(300):
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"production TldwCli did not mount {screen_type.__name__}")


async def _wait_for_widget(screen, pilot, selector: str, widget_type):
    for _ in range(300):
        try:
            widget = screen.query_one(selector)
            assert isinstance(widget, widget_type)
            return widget
        except NoMatches:
            pass
        await pilot.pause(0.01)
    raise AssertionError(f"production screen did not mount {selector}")


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_real_console_consumes_typed_provider_intents_and_opens_real_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)
    notifications: list[str] = []

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _wait_for_screen(app, pilot, ChatScreen)
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, *args, **kwargs: notifications.append(str(message)),
            )
            original = screen._ensure_active_console_session_settings()
            store = screen._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            store.replace_session_settings(
                session_id,
                replace(
                    original,
                    provider="anthropic",
                    model="claude-task-648",
                    base_url="https://old-provider.invalid/v1",
                    system_prompt="PRESERVE_TASK_648_SYSTEM_PROMPT",
                    source="user",
                ),
            )

            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROVIDER,
                ConsoleProviderIntent(provider="OpenAI"),
            )
            assert screen.consume_pending_console_provider_intent() is True
            applied = store.session_settings(session_id)
            assert applied is not None
            assert store.active_session_id == session_id
            assert applied.provider == "openai"
            assert applied.model == "gpt-task-648"
            assert applied.base_url is None
            assert applied.system_prompt == "PRESERVE_TASK_648_SYSTEM_PROMPT"
            assert applied.source == "user"

            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROVIDER,
                ConsoleProviderIntent(provider="Unavailable Provider"),
            )
            assert screen.consume_pending_console_provider_intent() is True
            assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROVIDER)
            assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROVIDER) is None
            assert "configured provider" in notifications[-1].lower()
            assert len(notifications[-1]) <= 200

            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROVIDER,
                ConsoleProviderIntent(provider="Anthropic"),
            )
            real_replace = store.replace_session_settings
            replace_attempts = 0

            def fail_once(
                target_session_id: str,
                settings: ConsoleSessionSettings,
            ):
                nonlocal replace_attempts
                replace_attempts += 1
                if replace_attempts == 1:
                    raise RuntimeError("PRIVATE_TRANSIENT_FAILURE")
                return real_replace(target_session_id, settings)

            monkeypatch.setattr(store, "replace_session_settings", fail_once)
            assert screen.consume_pending_console_provider_intent() is False
            assert app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROVIDER)
            assert "PRIVATE_TRANSIENT_FAILURE" not in "\n".join(notifications)
            assert screen.consume_pending_console_provider_intent() is True
            retried = store.session_settings(session_id)
            assert retried is not None
            assert retried.provider == "anthropic"
            assert retried.model == "claude-task-648"

            command_provider = LLMProviderProvider(screen)
            command_provider.handle_llm_command("OpenAI", "switch_OpenAI")
            command_applied = store.session_settings(session_id)
            assert command_applied is not None
            assert command_applied.provider == "openai"
            command_provider.handle_llm_command(None, "show_current")
            assert notifications[-1] == "Current LLM provider: openai"

            await screen.action_open_console_model_popover()
            popover = await _wait_for_screen(app, pilot, ConsoleModelPopover)
            search = await _wait_for_widget(
                popover,
                pilot,
                "#model-search-picker-input",
                Input,
            )
            search.value = "gpt-task"
            await pilot.pause()
            results = popover.query_one(
                "#model-search-picker-results",
                OptionList,
            )
            assert results.display is True
            assert results.option_count == 1
            assert str(results.get_option_at_index(0).prompt) == "gpt-task-648"
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_settings_save_preserves_user_session_then_away_command_hands_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)
    notifications: list[str] = []

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            store = chat._ensure_console_chat_store()
            initial = chat._ensure_active_console_session_settings()
            session_id = store.active_session_id
            assert session_id is not None
            store.replace_session_settings(
                session_id,
                replace(
                    initial,
                    provider="openai",
                    model="gpt-task-648",
                    system_prompt="PRESERVE_ACROSS_SETTINGS",
                    source="user",
                ),
            )

            app.post_message(
                NavigateToScreen(
                    "settings",
                    {"category": SettingsCategoryId.PROVIDERS_MODELS.value},
                )
            )
            settings = await _wait_for_screen(app, pilot, SettingsScreen)
            for _ in range(100):
                if (
                    settings.active_category
                    == SettingsCategoryId.PROVIDERS_MODELS.value
                ):
                    break
                await pilot.pause(0.01)
            assert settings.active_category == SettingsCategoryId.PROVIDERS_MODELS.value

            provider_select = settings.query_one("#settings-provider-value", Select)
            await _wait_for_widget(provider_select, pilot, "#label", Widget)
            provider_select.value = settings._provider_select_value_for_provider(
                "Anthropic"
            )
            await pilot.pause()
            settings.query_one(
                "#settings-model-value",
                Input,
            ).value = "claude-task-648"
            await pilot.pause()
            settings.action_settings_save_category(allow_text_entry_focus=True)
            await pilot.pause()
            assert app.app_config["chat_defaults"]["provider"] == "anthropic"
            assert app.app_config["chat_defaults"]["model"] == "claude-task-648"

            app.post_message(NavigateToScreen("chat"))
            restored_chat = await _wait_for_screen(app, pilot, ChatScreen)
            restored_store = restored_chat._ensure_console_chat_store()
            restored = restored_store.session_settings(session_id)
            assert restored_store.active_session_id == session_id
            assert restored is not None
            assert restored.provider == "openai"
            assert restored.model == "gpt-task-648"
            assert restored.system_prompt == "PRESERVE_ACROSS_SETTINGS"
            assert restored.source == "user"

            app.post_message(NavigateToScreen("settings"))
            settings = await _wait_for_screen(app, pilot, SettingsScreen)
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, *args, **kwargs: notifications.append(str(message)),
            )
            command_provider = LLMProviderProvider(settings)
            command_provider.handle_llm_command(None, "show_current")
            assert notifications[-1] == "Current LLM provider: anthropic"

            command_provider.handle_llm_command("Anthropic", "switch_Anthropic")
            assert app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROVIDER)
            assert "next Console" in notifications[-1]

            app.post_message(NavigateToScreen("chat"))
            handed_off_chat = await _wait_for_screen(app, pilot, ChatScreen)
            for _ in range(100):
                handed_off_store = handed_off_chat._ensure_console_chat_store()
                handed_off = handed_off_store.session_settings(session_id)
                if handed_off is not None and handed_off.provider == "anthropic":
                    break
                await pilot.pause(0.01)
            assert handed_off_store.active_session_id == session_id
            assert handed_off is not None
            assert handed_off.provider == "anthropic"
            assert handed_off.model == "claude-task-648"
            assert handed_off.system_prompt == "PRESERVE_ACROSS_SETTINGS"
            assert handed_off.source == "user"
            assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROVIDER)
    finally:
        await _close_production_app(app)
