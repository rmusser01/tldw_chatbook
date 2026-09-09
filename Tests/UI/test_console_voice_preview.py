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
    CSS = "ConsoleTranscript { height: 10; }"

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
@pytest.mark.parametrize("clear", [False, True])
async def test_first_preview_coalesces_updates_before_mount(clear: bool) -> None:
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        assert not transcript.query(ConsoleVoicePreview)
        transcript.focus()
        focus = app.focused
        transcript.set_voice_preview(_projection())
        transcript.set_voice_preview(_projection(epoch=2, user_text="latest correction"))
        if clear:
            transcript.clear_voice_preview()
        await pilot.pause()

        previews = list(transcript.query(ConsoleVoicePreview))
        assert len(previews) == 1
        preview = previews[0]
        assert preview.display is not clear
        assert str(preview.query_one(".console-voice-preview-user", Static).renderable) == (
            "" if clear else "You · latest correction"
        )
        children = list(transcript.children)
        assert children[children.index(preview) + 1].id == "console-transcript-jump-pill"
        assert app.focused is focus and focus.is_attached


@pytest.mark.asyncio
async def test_active_preview_survives_recompose_and_accepts_the_next_update() -> None:
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_voice_preview(_projection(epoch=2, user_text="current projection"))
        await pilot.pause()
        old_preview = transcript.query_one(ConsoleVoicePreview)
        await transcript.recompose()
        preview = transcript.query_one(ConsoleVoicePreview)
        assert preview is not old_preview
        assert str(preview.query_one(".console-voice-preview-user", Static).renderable) == (
            "You · current projection"
        )
        transcript.set_voice_preview(_projection(epoch=3, user_text="after recompose"))
        await pilot.pause()
        assert len(transcript.query(ConsoleVoicePreview)) == 1
        assert str(preview.query_one(".console-voice-preview-user", Static).renderable) == (
            "You · after recompose"
        )
        transcript.clear_voice_preview()
        await transcript.recompose()
        assert not transcript.query(ConsoleVoicePreview)


@pytest.mark.asyncio
@pytest.mark.parametrize("reading_history", [False, True])
async def test_lazy_preview_preserves_reader_position_and_tail_follow(reading_history):
    app = _PreviewHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([
            ConsoleChatMessage(
                id=f"history-{index}", role=ConsoleMessageRole.ASSISTANT,
                content="First line\nSecond line\nThird line\nFourth line",
            )
            for index in range(12)
        ])
        await transcript.refresh_messages()
        await pilot.pause()
        assert transcript.max_scroll_y > 0
        if reading_history:
            transcript.release_anchor()
            transcript.scroll_to(y=0, animate=False)
            await pilot.pause()
        position = transcript.scroll_y
        transcript.focus()
        focused = app.focused
        transcript.set_voice_preview(_projection())
        await pilot.pause()
        assert transcript.scroll_y == (position if reading_history else transcript.max_scroll_y)
        assert app.focused is focused and focused.is_attached
        transcript.clear_voice_preview()
        await pilot.pause()
        assert transcript.scroll_y == (position if reading_history else transcript.max_scroll_y)


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
