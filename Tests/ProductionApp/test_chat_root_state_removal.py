# ruff: noqa: E402

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pytest
from textual.widgets import Button

# Exercise the full production app in its supported "optional transcription
# backend absent" configuration. The installed parakeet-mlx wheel aborts the
# interpreter while importing MLX in this test runner, before Textual can mount.
_MISSING_MODULE = object()
_previous_parakeet_mlx = sys.modules.get("parakeet_mlx", _MISSING_MODULE)
sys.modules["parakeet_mlx"] = None

try:
    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
    )
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
finally:
    if _previous_parakeet_mlx is _MISSING_MODULE:
        sys.modules.pop("parakeet_mlx", None)
    else:
        sys.modules["parakeet_mlx"] = _previous_parakeet_mlx


REMOVED_CHAT_ROOT_NAMES = (
    "rag_expansion_provider_value",
    "chat_sidebar_collapsed",
    "chat_right_sidebar_collapsed",
    "chat_right_sidebar_width",
    "chat_sidebar_selected_prompt_id",
    "chat_sidebar_selected_prompt_system",
    "chat_sidebar_selected_prompt_user",
    "current_chat_is_ephemeral",
    "current_chat_conversation_id",
    "current_chat_active_character_data",
    "active_chat_tab_id",
    "chat_sessions",
    "chat_sidebar_loaded_prompt_id",
    "chat_sidebar_loaded_prompt_title_text",
    "chat_sidebar_loaded_prompt_system_text",
    "chat_sidebar_loaded_prompt_user_text",
    "chat_sidebar_loaded_prompt_keywords_text",
    "chat_sidebar_prompt_display_visible",
    "chat_settings_mode",
    "chat_settings_search_query",
    "_chat_state_lock",
    "current_ai_message_widget",
    "current_chat_worker",
    "current_chat_is_streaming",
    "current_chat_note_id",
    "current_chat_note_version",
    "_conversation_search_timer",
    "_chat_sidebar_prompt_search_timer",
    "_media_sidebar_search_timer",
    "media_search_current_page",
    "media_search_total_pages",
    "current_sidebar_media_item",
)


class _BlockingProviderGateway:
    """Narrow provider collaborator used by the real production Chat screen."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._block_forever = asyncio.Event()

    async def resolve_for_send(self, selection) -> ConsoleProviderResolution:
        return ConsoleProviderResolution(
            provider=selection.provider,
            base_url="",
            model=(
                selection.explicit_model or selection.configured_model or "gpt-task-650"
            ),
            ready=True,
            execution_key="openai",
        )

    async def stream_chat(self, resolution, messages):
        del resolution, messages
        self.started.set()
        try:
            yield "partial response"
            await self._block_forever.wait()
        finally:
            self.cancelled.set()


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
        {"provider": "OpenAI", "model": "gpt-task-650"},
    )
    assert adapter.save_values(
        "api_settings.openai",
        {"api_key": "TASK_650_TEST_KEY", "model": "gpt-task-650"},
    )

    app = TldwCli()
    app.app_config = load_settings(force_reload=True)
    app.app_config["_first_run"] = False
    app.providers_models = {"OpenAI": ["gpt-task-650"]}
    app._initial_tab_value = TAB_CHAT
    return app


async def _wait_for_screen(app: TldwCli, pilot, screen_type):
    for _ in range(300):
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"production TldwCli did not mount {screen_type.__name__}")


async def _wait_for_session_count(chat: ChatScreen, pilot, expected: int) -> None:
    for _ in range(300):
        if len(chat._ensure_console_chat_store().sessions()) == expected:
            return
        await pilot.pause(0.01)
    raise AssertionError(f"native Console did not reach {expected} sessions")


async def _wait_until(predicate, failure: str) -> None:
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(failure)


def _snapshot_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value}.union(
            *(_snapshot_keys(item) for item in value.values()),
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_snapshot_keys(item) for item in value))
    return set()


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
async def test_native_console_owns_rails_sessions_and_snapshot_without_root_mirrors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(180, 55)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            store = chat._ensure_console_chat_store()
            initial_session_id = store.active_session_id
            assert initial_session_id is not None

            chat.query_one("#console-context-rail-collapse", Button).press()
            chat.query_one("#console-inspector-rail-collapse", Button).press()
            await pilot.pause()
            assert chat._current_console_rail_state().left_open is False
            assert chat._current_console_rail_state().right_open is False
            chat.query_one("#console-context-rail-open", Button).press()
            chat.query_one("#console-inspector-rail-open", Button).press()
            await pilot.pause()
            assert chat._current_console_rail_state().left_open is True
            assert chat._current_console_rail_state().right_open is True

            await pilot.press("ctrl+t")
            await _wait_for_session_count(chat, pilot, 2)
            assert store.active_session_id != initial_session_id
            chat.query_one(f"#console-session-tab-{initial_session_id}", Button).press()
            await pilot.pause()
            assert store.active_session_id == initial_session_id

            snapshot = chat.save_state()
            assert "native_console_state" in snapshot
            assert "chat_state" not in snapshot
            assert snapshot["native_console_state"]["active_session_id"] == (
                initial_session_id
            )
            assert len(snapshot["native_console_state"]["sessions"]) == 2
            assert not set(REMOVED_CHAT_ROOT_NAMES).intersection(
                _snapshot_keys(snapshot)
            )

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            app.post_message(NavigateToScreen("chat"))
            restored_chat = await _wait_for_screen(app, pilot, ChatScreen)
            restored_store = restored_chat._ensure_console_chat_store()
            assert len(restored_store.sessions()) == 2
            assert restored_store.active_session_id == initial_session_id
            assert all(not hasattr(app, name) for name in REMOVED_CHAT_ROOT_NAMES)
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_visible_console_stop_cancels_native_run_without_root_worker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _BlockingProviderGateway()
    app = _production_app(monkeypatch)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    app.console_provider_gateway_factory = lambda: gateway

    try:
        async with app.run_test(size=(180, 55)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            composer = chat.query_one("#console-native-composer", ConsoleComposerBar)
            composer.load_draft("exercise native cancellation")
            chat.query_one("#console-send-message", Button).press()

            await _wait_until(
                gateway.started.is_set,
                "native Console provider stream did not start",
            )
            controller = chat._ensure_console_chat_controller()
            assert controller.run_state.status is ConsoleRunStatus.STREAMING
            assert all(not hasattr(app, name) for name in REMOVED_CHAT_ROOT_NAMES)

            stop_button = chat.query_one("#console-stop-generation", Button)
            await _wait_until(
                lambda: stop_button.display,
                "native Console Stop control did not become visible",
            )
            await pilot.click("#console-stop-generation")
            await _wait_until(
                gateway.cancelled.is_set,
                "visible Console Stop did not cancel the provider stream",
            )
            await _wait_until(
                lambda: controller.run_state.status is ConsoleRunStatus.STOPPED,
                "native Console controller did not enter stopped state",
            )

            store = chat._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            assistant = next(
                message
                for message in reversed(store.messages_for_session(session_id))
                if message.role is ConsoleMessageRole.ASSISTANT
            )
            assert assistant.status == "stopped"
            assert assistant.content == "partial response"
            assert all(not hasattr(app, name) for name in REMOVED_CHAT_ROOT_NAMES)
    finally:
        await _close_production_app(app)
