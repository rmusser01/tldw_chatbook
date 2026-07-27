# ruff: noqa: E402

from __future__ import annotations

import logging
import sys

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input

# Exercise the full production app in its supported "optional transcription
# backend absent" configuration. The installed parakeet-mlx wheel aborts the
# interpreter while importing MLX in this test runner, before Textual can mount.
_MISSING_MODULE = object()
_previous_parakeet_mlx = sys.modules.get("parakeet_mlx", _MISSING_MODULE)
sys.modules["parakeet_mlx"] = None

try:
    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )
finally:
    if _previous_parakeet_mlx is _MISSING_MODULE:
        sys.modules.pop("parakeet_mlx", None)
    else:
        sys.modules["parakeet_mlx"] = _previous_parakeet_mlx


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    _disable_splash(monkeypatch)
    adapter = SettingsConfigAdapter()
    assert adapter.save_values(
        "chat_defaults",
        {"provider": "OpenAI", "model": "gpt-task-649"},
    )
    assert adapter.save_values(
        "api_settings.openai",
        {"api_key": "TASK_649_TEST_KEY", "model": "gpt-task-649"},
    )

    app = TldwCli()
    app.app_config = load_settings(force_reload=True)
    app.app_config["_first_run"] = False
    app.providers_models = {"OpenAI": ["gpt-task-649"]}
    app._initial_tab_value = TAB_CHAT
    return app


async def _wait_for_screen(app: TldwCli, pilot, screen_type):
    for _ in range(300):
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"production TldwCli did not mount {screen_type.__name__}")


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
async def test_registered_chat_route_uses_only_native_console_and_restores_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)
    draft = "TASK-649 native Console snapshot"

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            chat.query_one("#console-session-surface", ConsoleSessionSurface)
            composer = chat.query_one("#console-native-composer", ConsoleComposerBar)
            chat.query_one("#console-command-input", Input)
            assert not hasattr(chat, "chat_window")
            with pytest.raises(NoMatches):
                chat.query_one("#chat-window")

            composer.load_draft(draft)
            await pilot.click("#console-composer-collapse")
            assert chat._console_composer_collapsed is True
            await pilot.click("#console-composer-expand")
            assert chat._console_composer_collapsed is False

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            app.post_message(NavigateToScreen("chat"))
            restored_chat = await _wait_for_screen(app, pilot, ChatScreen)

            restored_chat.query_one(
                "#console-session-surface",
                ConsoleSessionSurface,
            )
            restored_composer = restored_chat.query_one(
                "#console-native-composer",
                ConsoleComposerBar,
            )
            assert restored_composer.draft_text() == draft
            assert not hasattr(restored_chat, "chat_window")
            with pytest.raises(NoMatches):
                restored_chat.query_one("#chat-window")
    finally:
        await _close_production_app(app)
