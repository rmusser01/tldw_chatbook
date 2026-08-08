"""Video message actions + command grammar + ephemeral gate (task-3401.5)."""

from tldw_chatbook.Chat.console_command_grammar import (
    GENERATE_VIDEO_COMMAND_NAME,
    default_console_registry,
)
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


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


def test_guide_segments_name_video_actions():
    from tldw_chatbook.Chat.console_message_actions import action_row_guide

    service = ConsoleMessageActionService()
    actions = service.available_actions(_video_message(), video_file_available=True)
    guide = action_row_guide(actions)
    assert "▶ Play" in guide
    assert "Save copy" in guide
