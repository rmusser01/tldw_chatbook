"""Transcript video-card row wiring (task-3401.5)."""

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_video_card import ConsoleVideoCardSpec


def _video_message(mid="m-vid"):
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] a-red-dragon",
        video_metadata=VideoGenerationMetadata(
            name="a-red-dragon", prompt="a red dragon", backend="minimax",
        ),
        id=mid,
    )


def _spec(message, status="ready"):
    return ConsoleVideoCardSpec(
        message_id=message.id,
        meta=message.video_metadata,
        status=status,
        file_path="/tmp/a-red-dragon.mp4" if status == "ready" else None,
    )


def _rows_for(message, spec):
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.set_video_card_specs({message.id: spec} if spec is not None else {})
    # Media rows are adjuncts nested into the top-level assistant-turn row;
    # exercise the flat row planner that owns the video-card projection.
    return transcript._flat_transcript_rows()


def test_video_message_renders_video_card_row():
    message = _video_message()
    rows = _rows_for(message, _spec(message))
    video_rows = [row for row in rows if row.kind == "video-card"]
    assert len(video_rows) == 1
    row = video_rows[0]
    assert row.key == f"video-card:{message.id}"
    assert row.video_card_spec is not None
    # The signature flips with the status, driving reconcile.
    assert row.signature[0][2] == "ready"


def test_video_message_without_spec_renders_no_media_rows():
    message = _video_message()
    rows = _rows_for(message, None)
    assert not [row for row in rows if row.kind == "video-card"]
    assert not [row for row in rows if row.kind == "generation-card"]
    assert not [row for row in rows if row.kind == "image"]


def test_expired_tombstone_stays_a_row():
    message = _video_message()
    rows = _rows_for(message, _spec(message, status="expired"))
    video_rows = [row for row in rows if row.kind == "video-card"]
    assert len(video_rows) == 1
    assert video_rows[0].signature[0][2] == "expired"


def test_action_kwargs_expose_file_availability():
    message = _video_message()
    transcript = ConsoleTranscript()
    transcript.set_video_card_specs({message.id: _spec(message)})
    kwargs = transcript._generation_action_kwargs(message)
    assert kwargs == {"video_file_available": True}

    transcript.set_video_card_specs({message.id: _spec(message, status="expired")})
    assert transcript._generation_action_kwargs(message) == {"video_file_available": False}


def test_plain_message_action_kwargs_stay_empty():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="hi")
    transcript = ConsoleTranscript()
    assert transcript._generation_action_kwargs(message) == {}
