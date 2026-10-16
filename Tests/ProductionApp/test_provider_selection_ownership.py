from __future__ import annotations

from dataclasses import replace
import logging

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Input, OptionList

from Tests.app_module_patches import set_app_global
import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import load_settings
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover


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

    set_app_global(monkeypatch, "get_cli_setting", get_cli_setting_without_splash)


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
    _save_initial_provider_config()
    return _production_app_from_saved_config(monkeypatch)


def _production_app_from_saved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> TldwCli:
    """Construct the real app from the sandbox's already-saved config."""

    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config = load_settings(force_reload=True)
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
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
            if widget.region.width > 0 and widget.region.height > 0:
                return widget
        except NoMatches:
            pass
        await pilot.pause(0.01)
    raise AssertionError(f"production screen did not render {selector}")


async def _wait_for_stable_provider_picker(
    settings: SettingsScreen,
    pilot,
) -> OptionList:
    """Return the provider list once its painted Provider control is stable.

    TASK-33007.2, rewritten on purpose: the list is hidden until the user
    types in the one-row Provider control, so the control is what paints.
    """
    for _ in range(300):
        try:
            provider_control = settings.query_one("#settings-provider-search", Input)
            provider_picker = settings.query_one(
                "#settings-provider-picker", OptionList
            )
        except NoMatches:
            await pilot.pause(0.01)
            continue
        if (
            provider_control.is_mounted
            and provider_control.region.width > 0
            and provider_control.region.height > 0
        ):
            await pilot.pause()
            if (
                settings.query_one("#settings-provider-picker", OptionList)
                is provider_picker
            ):
                return provider_picker
        await pilot.pause(0.01)
    raise AssertionError("production Settings provider picker did not stabilize")


async def _wait_until(pilot, predicate, failure: str) -> None:
    for _ in range(300):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(failure)


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
async def test_real_app_restart_routes_saved_global_and_model_profile_to_new_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh TldwCli mount consumes the persisted global/profile defaults."""

    adapter = SettingsConfigAdapter()
    assert adapter.save_sections(
        {
            "chat_defaults": {
                "provider": "Anthropic",
                "model": "claude-task-648",
            },
            "api_settings.anthropic": {
                "api_key": "TASK_648_TEST_KEY",
                "model": "claude-task-648",
                "model_defaults": {
                    "claude-task-648": {
                        "temperature": 0.37,
                        "streaming": False,
                    }
                },
            },
        }
    )
    restarted = _production_app_from_saved_config(monkeypatch)

    try:
        async with restarted.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(restarted, pilot, ChatScreen)
            settings = chat._session._ensure_active_console_session_settings()
            store = chat._ensure_console_chat_store()
            session_id = store.active_session_id

            assert session_id is not None
            assert store.session_settings(session_id) is settings
            assert (
                settings.provider,
                settings.model,
                settings.temperature,
                settings.streaming,
            ) == (
                "anthropic",
                "claude-task-648",
                pytest.approx(0.37),
                False,
            )
    finally:
        await _close_production_app(restarted)


@pytest.mark.asyncio
async def test_real_console_change_model_command_opens_real_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-33001.6: the palette reaches providers only through the model picker."""
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
            original = screen._session._ensure_active_console_session_settings()
            store = screen._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            store.replace_session_settings(
                session_id,
                replace(
                    original,
                    provider="openai",
                    model="gpt-task-648",
                    system_prompt="PRESERVE_TASK_648_SYSTEM_PROMPT",
                    source="user",
                ),
            )

            palette = ConsoleCommandProvider(screen, match_style=None)
            # TASK-33004.7: the entry is named for the surface it opens.
            hits = [hit async for hit in palette.search("switch model")]
            change_model = [
                hit for hit in hits if str(hit.text) == "Console: Switch model…"
            ]
            assert len(change_model) == 1
            # The palette runs a selected command exactly like this.
            app.call_later(change_model[0].command)
            popover = await _wait_for_screen(app, pilot, ConsoleModelPopover)
            # TASK-33004.4: Switch model's Find replaces the popover picker,
            # and a pair row replaces the provider Select.
            search = await _wait_for_widget(
                popover,
                pilot,
                "#console-popover-find",
                Input,
            )
            for _ in range(300):  # readiness resolves in a worker after open
                if popover._first_readiness_done and not popover._readiness_pending:
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError("Switch model readiness did not settle in 3 s")
            search.value = "gpt-task"
            await pilot.pause()
            pairs = [row for row in popover._rows if row.kind == "pair"]
            assert [(row.provider, row.model) for row in pairs] == [
                ("openai", "gpt-task-648")
            ]

            before_apply = screen._session._build_console_turn_execution_context(
                session_id
            )
            search.value = "claude-task"
            await pilot.pause()
            row = popover.highlighted_row()
            assert (row.kind, row.provider, row.model) == (
                "pair",
                "anthropic",
                "claude-task-648",
            )
            apply_button = popover.query_one("#console-popover-apply", Button)
            assert await pilot.click(apply_button) is True
            returned = await _wait_for_screen(app, pilot, ChatScreen)

            applied = store.session_settings(session_id)
            assert returned is screen
            assert store.active_session_id == session_id
            assert applied is not None
            assert (applied.provider, applied.model) == (
                "anthropic",
                "claude-task-648",
            )
            after_apply = screen._session._build_console_turn_execution_context(
                session_id
            )
            assert (
                before_apply.provider_selection.provider,
                before_apply.provider_selection.explicit_model,
            ) == ("openai", "gpt-task-648")
            assert (
                after_apply.provider_selection.provider,
                after_apply.provider_selection.explicit_model,
            ) == ("anthropic", "claude-task-648")
            assert notifications.count("This chat updated") == 1
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_settings_save_preserves_user_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            store = chat._ensure_console_chat_store()
            initial = chat._session._ensure_active_console_session_settings()
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

            provider_picker = await _wait_for_stable_provider_picker(settings, pilot)
            provider_search = await _wait_for_widget(
                settings,
                pilot,
                "#settings-provider-search",
                Input,
            )
            # TASK-33007.2, rewritten on purpose: typed into the focused
            # Provider control; the list never takes focus, Enter chooses.
            provider_search.focus()
            await pilot.press(*"anthropic")
            await _wait_until(
                pilot,
                lambda: (
                    provider_picker.highlighted is not None
                    and getattr(
                        provider_picker.get_option_at_index(
                            provider_picker.highlighted
                        ),
                        "provider_id",
                        None,
                    )
                    == "anthropic"
                ),
                "the provider picker did not highlight Anthropic",
            )
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    settings._provider_setting_values_mapping().get("provider")
                    == "anthropic"
                ),
                "the rendered provider control did not stage Anthropic",
            )
            # TASK-33007.3, rewritten on purpose: the rendered Model control
            # is the Default model picker (#settings-model-value is its hidden
            # adapter); an id no list holds is typed after its Custom ID.
            model_field = await _wait_for_widget(
                settings,
                pilot,
                "#model-search-picker-input",
                Input,
            )
            model_field.focus()
            await pilot.pause()
            settings.query_one("#model-search-picker-custom", Button).press()
            await pilot.pause()
            await pilot.press("home", "shift+end", "backspace", *"claude-task-648")
            await _wait_until(
                pilot,
                lambda: (
                    settings._provider_setting_values_mapping().get("model")
                    == "claude-task-648"
                ),
                "the rendered model control did not stage claude-task-648",
            )
            settings.action_settings_save_category(allow_text_entry_focus=True)
            await _wait_until(
                pilot,
                lambda: (
                    app.app_config["chat_defaults"].get("provider") == "anthropic"
                    and app.app_config["chat_defaults"].get("model")
                    == "claude-task-648"
                ),
                "the production Settings save did not update provider/model defaults",
            )
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
    finally:
        await _close_production_app(app)
