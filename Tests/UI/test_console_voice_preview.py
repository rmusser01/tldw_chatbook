"""Ephemeral presentation tests for speculative Console voice turns."""

from dataclasses import FrozenInstanceError

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_voice_preview import (
    ConsoleVoicePreview,
    VoicePreviewProjection,
)


class _PreviewHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _projection(
    *,
    epoch: int = 1,
    user_text: str = "rolling user text",
    assistant_text: str = "provisional response",
    status: str = "responding",
) -> VoicePreviewProjection:
    return VoicePreviewProjection(
        turn_id="voice-turn-1",
        attempt_epoch=epoch,
        user_text=user_text,
        assistant_text=assistant_text,
        status=status,
    )


def test_projection_is_frozen_and_repr_hides_provisional_content() -> None:
    projection = _projection()

    with pytest.raises(FrozenInstanceError):
        projection.user_text = "changed"  # type: ignore[misc]

    representation = repr(projection)
    assert "rolling user text" not in representation
    assert "provisional response" not in representation


@pytest.mark.asyncio
async def test_preview_rows_are_separate_from_transcript_messages() -> None:
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        durable = ConsoleChatMessage(
            id="durable-user",
            role=ConsoleMessageRole.USER,
            content="durable text",
        )
        transcript.set_messages([durable], session_id="session-1")
        transcript.set_voice_preview(_projection())
        await pilot.pause()

        preview = transcript.query_one(ConsoleVoicePreview)
        assert preview.display is True
        assert str(
            preview.query_one(".console-voice-preview-user", Static).renderable
        ) == ("You · rolling user text")
        assert (
            str(
                preview.query_one(".console-voice-preview-assistant", Static).renderable
            )
            == "Assistant · provisional response"
        )
        assert [message.id for message in transcript._messages] == ["durable-user"]
        assert all("voice-turn-1" not in key for key in transcript._row_widgets)


@pytest.mark.asyncio
async def test_replacement_clears_only_old_assistant_preview() -> None:
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_voice_preview(_projection())
        await pilot.pause()

        transcript.set_voice_preview(
            _projection(
                epoch=2,
                user_text="rolling user text plus correction",
                assistant_text="",
                status="updating response",
            )
        )
        await pilot.pause()

        preview = transcript.query_one(ConsoleVoicePreview)
        assert str(
            preview.query_one(".console-voice-preview-user", Static).renderable
        ) == ("You · rolling user text plus correction")
        assistant = preview.query_one(".console-voice-preview-assistant", Static)
        assert assistant.display is False
        assert str(
            preview.query_one(".console-voice-preview-status", Static).renderable
        ) == ("Updating response")


@pytest.mark.asyncio
async def test_preview_clears_after_durable_pair_arrives() -> None:
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_voice_preview(_projection(status="speaking"))
        await pilot.pause()

        transcript.set_messages(
            [
                ConsoleChatMessage(
                    id="winning-user",
                    role=ConsoleMessageRole.USER,
                    content="rolling user text",
                ),
                ConsoleChatMessage(
                    id="winning-assistant",
                    role=ConsoleMessageRole.ASSISTANT,
                    content="provisional response",
                ),
            ],
            session_id="session-1",
        )
        transcript.clear_voice_preview()
        await pilot.pause()

        preview = transcript.query_one(ConsoleVoicePreview)
        assert preview.display is False
        assert [message.id for message in transcript._messages] == [
            "winning-user",
            "winning-assistant",
        ]
