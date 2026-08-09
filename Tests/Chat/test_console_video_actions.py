"""Video message actions + command grammar + ephemeral gate (task-3401.5)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_command_grammar import (
    GENERATE_VIDEO_COMMAND_NAME,
    default_console_registry,
)
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import VideoStore


def _video_message():
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] dusk-over-neon-tokyo",
        video_metadata=VideoGenerationMetadata(
            name="dusk-over-neon-tokyo", prompt="p", backend="minimax",
        ),
    )


def _plain_message():
    return ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="hello")


def _video_action_screen(tmp_path):
    native_id = "native-video-message"
    persisted_id = "persisted-video-message"
    store = ConsoleChatStore()
    session = store.create_session(title="Video actions")
    message = store.append_video_message(
        session.id,
        video_metadata=_video_message().video_metadata,
        message_id=native_id,
    )
    message.persisted_message_id = persisted_id

    video_store = VideoStore(root=tmp_path / "generated_videos")
    stored_path = video_store.save(persisted_id, message.video_metadata.name, b"video")
    resolve_calls = []
    real_resolve = video_store.resolve

    def _resolve(message_id, slug, **kwargs):
        resolve_calls.append((message_id, slug))
        return real_resolve(message_id, slug, **kwargs)

    video_store.resolve = _resolve

    screen = ChatScreen.__new__(ChatScreen)
    notifications = []
    pushed = []
    pending_workers = []
    screen.app_instance = SimpleNamespace(
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
        push_screen=pushed.append,
    )
    screen._console_chat_store = store
    screen._console_video_store = video_store
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_video_store = lambda: video_store
    screen._sync_native_console_chat_ui = AsyncMock()
    screen.run_worker = lambda awaitable, **_kwargs: pending_workers.append(awaitable)

    def _unreached(*_args, **_kwargs):
        raise AssertionError("video action test reached an unrelated dependency")

    screen._message = ConsoleMessageController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
        ensure_console_chat_controller=_unreached,
        current_chat_controller_accessor=lambda: None,
        sync_native_console_chat_ui=screen._sync_native_console_chat_ui,
        active_session_is_ephemeral=_unreached,
        active_native_console_session=_unreached,
        current_console_conversation_id=_unreached,
        active_console_provider_model_display=_unreached,
        console_initial_session_title_for_workspace=_unreached,
        console_change_review_run_id=_unreached,
        open_change_review=_unreached,
        start_console_transcript_sync_timer=_unreached,
        clear_native_console_message_selection=_unreached,
        regenerate_console_generation_variant=_unreached,
        select_console_generation_variant=_unreached,
        keep_console_generation_variant=_unreached,
        handle_console_toggle_image_view=_unreached,
        invalidate_console_persisted_rows_cache=_unreached,
        play_console_video=lambda message_id: screen._play_console_video(message_id),
        save_console_video_copy=(
            lambda message_id: screen._save_console_video_copy(message_id)
        ),
    )
    return (
        screen,
        message,
        stored_path,
        resolve_calls,
        pushed,
        pending_workers,
    )


# -- grammar ------------------------------------------------------------------


def test_generate_video_registered_in_default_registry():
    registry = default_console_registry()
    assert GENERATE_VIDEO_COMMAND_NAME in registry.available_names()
    parse = registry.parse("/generate-video :minimax a kite")
    assert parse.name == GENERATE_VIDEO_COMMAND_NAME
    assert parse.args == ":minimax a kite"


def test_unknown_hint_derives_video_name():
    registry = default_console_registry()
    # available_names feeds the unknown-command hint; video must appear.
    assert "generate-video" in registry.available_names()


# -- ephemeral gate -------------------------------------------------------------


def test_generate_video_blocked_in_ephemeral_chat():
    reason = blocked_reason("generate-video", ephemeral=True)
    assert reason is not None and "video" in reason
    assert blocked_reason("generate-video", ephemeral=False) is None


# -- action service ---------------------------------------------------------------


def test_video_actions_offered_on_video_messages():
    service = ConsoleMessageActionService()
    actions = service.available_actions(_video_message(), video_file_available=True)
    ids = [a.action_id for a in actions]
    assert "video-play" in ids and "video-save-copy" in ids


def test_video_actions_absent_on_plain_messages():
    service = ConsoleMessageActionService()
    actions = service.available_actions(_plain_message())
    ids = [a.action_id for a in actions]
    assert "video-play" not in ids and "video-save-copy" not in ids


def test_video_actions_enabled_only_when_file_available():
    service = ConsoleMessageActionService()
    ready = {
        a.action_id: a for a in service.available_actions(
            _video_message(), video_file_available=True
        )
    }
    assert ready["video-play"].enabled
    assert ready["video-save-copy"].enabled

    expired = {
        a.action_id: a for a in service.available_actions(
            _video_message(), video_file_available=False
        )
    }
    assert not expired["video-play"].enabled
    assert "ephemeral video file is gone" in expired["video-play"].disabled_reason


def test_video_action_dispatch_returns_screen_targets():
    service = ConsoleMessageActionService()
    message = _video_message()
    for action_id in ("video-play", "video-save-copy"):
        result = service.dispatch(action_id, message)
        assert result.status == "completed"
        assert result.target_message_id == message.id


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_video_play_with_persisted_storage_id(
    tmp_path, monkeypatch
):
    screen, message, stored_path, resolve_calls, pushed, _pending = (
        _video_action_screen(tmp_path)
    )

    class FakeVideoPlayerScreen:
        def __init__(self, path, *, title):
            self.path = path
            self.title = title

    from tldw_chatbook.Media_Playback import player_pipeline
    from tldw_chatbook.UI.Screens import video_player_screen

    monkeypatch.setattr(
        player_pipeline, "playback_tools_available", lambda: (True, "")
    )
    monkeypatch.setattr(video_player_screen, "VideoPlayerScreen", FakeVideoPlayerScreen)
    button = Button(
        "play", id=f"console-message-action-video-play-{message.id}"
    )

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert message.id == "native-video-message"
    assert resolve_calls == [(message.persisted_message_id, message.video_metadata.name)]
    assert len(pushed) == 1
    assert pushed[0].path == str(stored_path)


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_video_save_with_persisted_storage_id(
    tmp_path, monkeypatch
):
    screen, message, _stored_path, resolve_calls, _pushed, pending_workers = (
        _video_action_screen(tmp_path)
    )
    export_root = tmp_path / "exports"
    monkeypatch.setattr(
        chat_screen_module,
        "get_cli_setting",
        lambda *_args, **_kwargs: str(export_root),
    )
    button = Button(
        "save", id=f"console-message-action-video-save-copy-{message.id}"
    )

    handled = await screen.handle_console_message_action(Button.Pressed(button))
    assert len(pending_workers) == 1
    await pending_workers[0]

    assert handled is True
    assert message.id == "native-video-message"
    assert resolve_calls == [(message.persisted_message_id, message.video_metadata.name)]
    assert (export_root / "dusk-over-neon-tokyo.mp4").read_bytes() == b"video"


def test_guide_segments_name_video_actions():
    from tldw_chatbook.Chat.console_message_actions import action_row_guide

    service = ConsoleMessageActionService()
    actions = service.available_actions(_video_message(), video_file_available=True)
    guide = action_row_guide(actions)
    assert "▶ Play" in guide
    assert "Save copy" in guide
