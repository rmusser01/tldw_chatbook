from __future__ import annotations

import asyncio
import logging

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.config import load_settings
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface


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
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
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


def _chat_handoff(title: str) -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="production-app-test",
        item_type="document",
        title=title,
        body=f"Body for {title}",
        display_summary=f"Summary for {title}",
    )


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


@pytest.mark.asyncio
async def test_native_console_chat_handoff_settles_exact_claim_and_keeps_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)
    first_started = asyncio.Event()
    continue_first = asyncio.Event()

    async def wait_before_native_staging(
        self: ChatScreen,
        payload: ChatHandoffPayload,
    ) -> bool:
        first_started.set()
        await continue_first.wait()
        return False

    monkeypatch.setattr(
        ChatScreen,
        "_start_character_console_session",
        wait_before_native_staging,
    )
    app.pending_handoffs.stage(
        HandoffChannel.CHAT,
        _chat_handoff("first"),
    )

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)
            await first_started.wait()
            app.pending_handoffs.stage(
                HandoffChannel.CHAT,
                _chat_handoff("replacement"),
            )
            continue_first.set()
            for _ in range(300):
                if not chat._handoff_consumption_in_progress:
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError("first Chat handoff did not settle")

            assert app.pending_handoffs.has_pending(HandoffChannel.CHAT)
            assert chat._pending_console_launch_context is not None
            assert chat._pending_console_launch_context.title == "first"

            await chat._consume_pending_chat_handoff()

            assert not app.pending_handoffs.has_pending(HandoffChannel.CHAT)
            assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
            assert chat._pending_console_launch_context is not None
            assert chat._pending_console_launch_context.title == "replacement"

            def fail_native_staging(payload: ChatHandoffPayload) -> None:
                raise RuntimeError("PRIVATE_HANDOFF_FAILURE")

            with monkeypatch.context() as failure_patch:
                failure_patch.setattr(
                    chat,
                    "_stage_handoff_as_console_live_work",
                    fail_native_staging,
                )
                app.pending_handoffs.stage(
                    HandoffChannel.CHAT,
                    _chat_handoff("retry-after-failure"),
                )
                with pytest.raises(RuntimeError, match="PRIVATE_HANDOFF_FAILURE"):
                    await chat._consume_pending_chat_handoff()
            assert app.pending_handoffs.has_pending(HandoffChannel.CHAT)

            await chat._consume_pending_chat_handoff()
            assert not app.pending_handoffs.has_pending(HandoffChannel.CHAT)

            cancellation_started = asyncio.Event()
            hold_cancellation = asyncio.Event()

            async def hold_character_start(payload: ChatHandoffPayload) -> bool:
                cancellation_started.set()
                await hold_cancellation.wait()
                return False

            with monkeypatch.context() as cancellation_patch:
                cancellation_patch.setattr(
                    chat,
                    "_start_character_console_session",
                    hold_character_start,
                )
                app.pending_handoffs.stage(
                    HandoffChannel.CHAT,
                    _chat_handoff("retry-after-cancellation"),
                )
                cancelled_consumer = asyncio.create_task(
                    chat._consume_pending_chat_handoff()
                )
                await cancellation_started.wait()
                cancelled_consumer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled_consumer
            assert app.pending_handoffs.has_pending(HandoffChannel.CHAT)

            await chat._consume_pending_chat_handoff()
            assert not app.pending_handoffs.has_pending(HandoffChannel.CHAT)
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_native_console_prompt_handoff_releases_transient_and_acknowledges_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(app, pilot, ChatScreen)

            monkeypatch.setattr(
                chat, "_console_setup_blocked_reason", lambda: "blocked"
            )
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                "terminal prompt",
            )
            await chat._consume_pending_console_prompt_insert()
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
            assert (
                app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
            )

            monkeypatch.setattr(chat, "_console_setup_blocked_reason", lambda: "")
            monkeypatch.setattr(
                chat,
                "_insert_prompt_text_into_composer",
                lambda text, *, replace: False,
            )
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                "retry prompt",
            )
            await chat._consume_pending_console_prompt_insert()
            assert app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
    finally:
        await _close_production_app(app)
