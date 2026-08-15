from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import time

import pytest
from loguru import logger
from textual.css.query import NoMatches
from textual.widgets import Input

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.config import load_settings
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    fingerprint_system_text,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import VideoStore
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface


@dataclass
class _VideoRetentionConfig:
    retention: str = "session"
    retention_ttl_hours: int = 24
    max_store_mb: int = 2048


@pytest.fixture(autouse=True)
def isolated_video_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    data_root = tmp_path / "user-data"
    config = _VideoRetentionConfig()
    monkeypatch.setattr(
        "tldw_chatbook.Video_Generation.video_store.get_user_data_dir",
        lambda: data_root,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Video_Generation.video_store.get_video_generation_config",
        lambda: config,
    )
    return data_root / "generated_videos", config


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


async def _wait_for_screen(
    app: TldwCli,
    pilot,
    screen_type,
    *,
    selector: str | None = None,
):
    for _ in range(300):
        screen = app.screen
        if isinstance(screen, screen_type) and (
            selector is None or bool(screen.query(selector))
        ):
            await pilot.pause()
            if app.screen is screen and (
                selector is None or bool(screen.query(selector))
            ):
                return screen
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


def test_video_retention_runs_during_app_construction_once(
    monkeypatch: pytest.MonkeyPatch,
    isolated_video_store,
) -> None:
    video_root, _ = isolated_video_store
    prior_store = VideoStore()
    prior_path = prior_store.save(
        "prior-message", "prior-media", b"prior-video", extension="mp4"
    )
    real_enforce_retention = VideoStore.enforce_retention
    retention_calls: list[VideoStore] = []

    def enforce_retention_spy(self: VideoStore):
        retention_calls.append(self)
        return real_enforce_retention(self)

    monkeypatch.setattr(VideoStore, "enforce_retention", enforce_retention_spy)

    app = _production_app(monkeypatch)

    assert retention_calls == [app.generated_video_store]
    assert not prior_path.exists()
    assert app.generated_video_store.root == video_root


def test_next_app_startup_applies_session_retention_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_app = _production_app(monkeypatch)
    metadata = VideoGenerationMetadata(
        name="session-media",
        prompt="session video",
        backend="test-backend",
    )
    message = ConsoleChatMessage(
        id="session-message",
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] session-media",
        video_metadata=metadata,
    )
    current_path = first_app.generated_video_store.save(
        message.id,
        metadata.name,
        b"current-video",
        extension="mp4",
    )

    second_app = _production_app(monkeypatch)

    assert second_app.generated_video_store is not first_app.generated_video_store
    assert not current_path.exists()
    specs = ChatScreen(second_app)._video._build_video_card_specs([message])
    assert specs[message.id].status == "expired"


def test_next_app_startup_keeps_fresh_ttl_video_and_removes_stale(
    monkeypatch: pytest.MonkeyPatch,
    isolated_video_store,
) -> None:
    _, config = isolated_video_store
    config.retention = "ttl"
    config.retention_ttl_hours = 1
    first_app = _production_app(monkeypatch)
    fresh_path = first_app.generated_video_store.save(
        "fresh-message",
        "fresh-media",
        b"fresh-video",
        extension="mp4",
    )
    stale_path = first_app.generated_video_store.save(
        "stale-message",
        "stale-media",
        b"stale-video",
        extension="mp4",
    )
    stale_time = time.time() - 3700
    os.utime(stale_path, (stale_time, stale_time))

    second_app = _production_app(monkeypatch)

    assert fresh_path.exists()
    assert not stale_path.exists()
    assert (
        second_app.generated_video_store.resolve(
            "fresh-message", "fresh-media", extension="mp4"
        )
        == fresh_path
    )


def test_video_retention_startup_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = "/private/generated-videos/customer-42"
    private_message_id = "private-message-id-771"
    private_media_name = "private-media-name-882.mp4"
    retention_calls: list[VideoStore] = []

    def fail_retention(self: VideoStore):
        retention_calls.append(self)
        raise RuntimeError(f"{private_path} {private_message_id} {private_media_name}")

    monkeypatch.setattr(VideoStore, "enforce_retention", fail_retention)
    diagnostics: list[str] = []
    sink_id = logger.add(diagnostics.append, level="WARNING", format="{message}")
    try:
        app = _production_app(monkeypatch)
    finally:
        logger.remove(sink_id)

    assert retention_calls == [app.generated_video_store]
    assert isinstance(app.generated_video_store, VideoStore)
    diagnostic = "".join(diagnostics)
    assert (
        "Generated-video startup retention failed (error_type=RuntimeError)."
        in diagnostic
    )
    assert private_path not in diagnostic
    assert private_message_id not in diagnostic
    assert private_media_name not in diagnostic

    first_screen = ChatScreen(app)
    second_screen = ChatScreen(app)
    assert (
        first_screen._video._ensure_console_video_store() is app.generated_video_store
    )
    assert (
        second_screen._video._ensure_console_video_store() is app.generated_video_store
    )
    assert retention_calls == [app.generated_video_store]


def _prompt_application(
    user_text: str,
    *,
    target_session_id: str = "session-1",
) -> PromptVariableApplication:
    return PromptVariableApplication(
        system_text=None,
        user_text=user_text,
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=target_session_id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )


@pytest.mark.asyncio
async def test_registered_chat_route_uses_only_native_console_and_restores_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_enforce_retention = VideoStore.enforce_retention
    retention_calls: list[VideoStore] = []

    def enforce_retention_spy(self: VideoStore):
        retention_calls.append(self)
        return real_enforce_retention(self)

    monkeypatch.setattr(VideoStore, "enforce_retention", enforce_retention_spy)
    app = _production_app(monkeypatch)
    draft = "TASK-649 native Console snapshot"

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )
            session_id = chat._ensure_console_chat_store().active_session_id
            assert session_id is not None
            chat.query_one("#console-session-surface", ConsoleSessionSurface)
            composer = chat.query_one("#console-native-composer", ConsoleComposerBar)
            chat.query_one("#console-command-input", Input)
            assert not hasattr(chat, "chat_window")
            with pytest.raises(NoMatches):
                chat.query_one("#chat-window")

            message_id = "current-run-video"
            metadata = VideoGenerationMetadata(
                name="current-run-clip",
                prompt="current run",
                backend="comfyui",
                container="webm",
            )
            stored = app.generated_video_store.save(
                message_id,
                metadata.name,
                b"current-run-bytes",
                extension="webm",
            )
            console_store = chat._ensure_console_chat_store()
            assert console_store.active_session_id is not None
            message = console_store.append_video_message(
                console_store.active_session_id,
                video_metadata=metadata,
                message_id=message_id,
            )
            assert (
                chat._video._ensure_console_video_store() is app.generated_video_store
            )
            assert (
                chat._video._build_video_card_specs([message])[message_id].status
                == "ready"
            )

            composer.load_draft(draft)
            store = chat._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            store.set_session_system_prompt(session_id, "projection system")
            assert app.console_prompt_target_projection() is None
            await pilot.click("#console-composer-collapse")
            assert chat._console_composer_collapsed is True
            await pilot.click("#console-composer-expand")
            assert chat._console_composer_collapsed is False

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            projection = app.console_prompt_target_projection()
            assert projection is not None
            assert projection.target_session_id == session_id
            assert projection.system_fingerprint == fingerprint_system_text(
                "projection system"
            )
            app.post_message(NavigateToScreen("chat"))
            restored_chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )

            restored_chat.query_one(
                "#console-session-surface",
                ConsoleSessionSurface,
            )
            restored_composer = restored_chat.query_one(
                "#console-native-composer",
                ConsoleComposerBar,
            )
            assert restored_composer.draft_text() == draft
            assert restored_chat is not chat
            assert (
                restored_chat._video._ensure_console_video_store()
                is app.generated_video_store
            )
            restored_store = restored_chat._ensure_console_chat_store()
            restored_message = restored_store.get_message(message_id)
            assert restored_message.video_metadata == metadata
            spec = restored_chat._video._build_video_card_specs([restored_message])[
                message_id
            ]
            assert spec.status == "ready"
            assert spec.file_path == str(stored)
            assert stored.read_bytes() == b"current-run-bytes"
            assert retention_calls == [app.generated_video_store]
            assert not hasattr(restored_chat, "chat_window")
            with pytest.raises(NoMatches):
                restored_chat.query_one("#chat-window")
    finally:
        await _close_production_app(app)


@pytest.mark.parametrize("save_failure", ["non_callable", "non_mapping", "raises"])
@pytest.mark.asyncio
async def test_failed_console_snapshot_save_discards_stale_projection_only(
    monkeypatch: pytest.MonkeyPatch,
    save_failure: str,
) -> None:
    app = _production_app(monkeypatch)

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )
            store = chat._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            store.set_session_system_prompt(session_id, "old projection system")

            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)
            assert app.console_prompt_target_projection() is not None
            app.post_message(NavigateToScreen("chat"))
            chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )

            runtime_identity = app._current_runtime_identity()
            app.screen_state_store.save(
                "library",
                {"selected": "preserve unrelated route"},
                runtime_identity,
            )

            if save_failure == "non_callable":
                failed_save = None
            elif save_failure == "non_mapping":

                def non_mapping_save():
                    return None

                failed_save = non_mapping_save
            else:

                def raising_save():
                    raise RuntimeError("private snapshot failure")

                failed_save = raising_save
            monkeypatch.setattr(chat, "save_state", failed_save)
            app.post_message(NavigateToScreen("settings"))
            await _wait_for_screen(app, pilot, SettingsScreen)

            assert app.console_prompt_target_projection() is None
            assert app.screen_state_store.restore(TAB_CHAT, runtime_identity) is None
            assert app.screen_state_store.restore("library", runtime_identity) == {
                "selected": "preserve unrelated route"
            }
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
        self: ConsoleSessionController,
        payload: ChatHandoffPayload,
    ) -> bool:
        first_started.set()
        await continue_first.wait()
        return False

    monkeypatch.setattr(
        ConsoleSessionController,
        "_start_character_console_session",
        wait_before_native_staging,
    )

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )
            app.pending_handoffs.stage(
                HandoffChannel.CHAT,
                _chat_handoff("first"),
            )
            first_consumer = asyncio.create_task(chat._consume_pending_chat_handoff())
            await asyncio.wait_for(first_started.wait(), timeout=6.0)
            app.pending_handoffs.stage(
                HandoffChannel.CHAT,
                _chat_handoff("replacement"),
            )
            continue_first.set()
            await asyncio.wait_for(first_consumer, timeout=6.0)

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
                    chat._session,
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
                await asyncio.wait_for(cancellation_started.wait(), timeout=6.0)
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
            chat = await _wait_for_screen(
                app,
                pilot,
                ChatScreen,
                selector="#console-session-surface",
            )
            session_id = chat._ensure_console_chat_store().active_session_id
            assert session_id is not None

            monkeypatch.setattr(
                chat, "_console_setup_blocked_reason", lambda: "blocked"
            )
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                _prompt_application(
                    "terminal prompt",
                    target_session_id=session_id,
                ),
            )
            await chat._consume_pending_console_prompt_insert()
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
            assert (
                app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
            )

            monkeypatch.setattr(chat, "_console_setup_blocked_reason", lambda: "")
            monkeypatch.setattr(chat._prompts, "_composer_accessor", lambda: None)
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                _prompt_application(
                    "retry prompt",
                    target_session_id=session_id,
                ),
            )
            await chat._consume_pending_console_prompt_insert()
            assert app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
    finally:
        await _close_production_app(app)
