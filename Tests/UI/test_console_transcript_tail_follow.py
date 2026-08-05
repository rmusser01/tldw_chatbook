"""Tests for task-298: transcript tail-follow.

A streamed reply taller than the viewport used to finish below the fold
with no scroll (pre-existing, confirmed by the task-259 base A/B). The
transcript now engages Textual's anchor at mount — pinned to the newest
content while the reader is at the bottom, released when they scroll up,
re-engaged when they return — and a NEW user message (a send) re-anchors
even from a scrolled-up position.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class TailFollowHarness(App):
    """Small viewport so a handful of messages overflow it."""

    CSS = "ConsoleTranscript { height: 10; }"

    def compose(self) -> ComposeResult:
        """Yield the transcript under test.

        Returns:
            The compose result mounting one ConsoleTranscript.
        """
        yield ConsoleTranscript(id="console-native-transcript")


def _msg(
    i: int, role: ConsoleMessageRole = ConsoleMessageRole.ASSISTANT
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=role,
        content="\n".join(f"line {i}.{j}" for j in range(4)),
        id=f"m{i}",
    )


def _messages(n: int) -> list[ConsoleChatMessage]:
    return [
        _msg(i, ConsoleMessageRole.USER if i % 2 == 0 else ConsoleMessageRole.ASSISTANT)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_transcript_anchors_at_mount_and_follows_growth():
    """Content growing past the viewport keeps the view at the bottom."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        assert transcript.is_anchored

        history = _messages(12)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.max_scroll_y > 0, "seed must overflow the viewport"
        assert transcript.scroll_y == transcript.max_scroll_y


@pytest.mark.asyncio
async def test_scrolled_up_reader_is_not_yanked_by_assistant_growth():
    """A reader who scrolled up stays put while the reply keeps streaming."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(12)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()
        reading_position = transcript.scroll_y

        history = history + [_msg(12, ConsoleMessageRole.ASSISTANT)]
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.scroll_y == reading_position, (
            "assistant growth must not yank a scrolled-up reader"
        )


@pytest.mark.asyncio
async def test_new_user_message_reanchors_from_scrolled_up_position():
    """A send jumps to the tail even after the reader scrolled up."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(12)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()

        history = history + [_msg(13, ConsoleMessageRole.USER)]
        # TASK-336: the screen stamps a follow intent at every send/resume/
        # switch site; a user scroll AFTER the intent wins instead of being
        # yanked. Mirror the production choreography here.
        transcript.note_follow_intent()
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.scroll_y == transcript.max_scroll_y, (
            "a new user message (a send) must re-engage tail-follow"
        )


@pytest.mark.asyncio
async def test_send_with_assistant_placeholder_still_reanchors():
    """The REAL send shape: USER + ASSISTANT placeholder appended together.

    The first polled update after a send can already have the assistant
    placeholder at the tail; the user message is one back. Re-anchoring
    must key on newly-seen user ids anywhere, not the tail (PR #697
    review finding).
    """
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(12)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()

        history = history + [
            _msg(13, ConsoleMessageRole.USER),
            _msg(14, ConsoleMessageRole.ASSISTANT),
        ]
        transcript.note_follow_intent()
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.scroll_y == transcript.max_scroll_y, (
            "a send observed with its assistant placeholder must still re-anchor"
        )


@pytest.mark.asyncio
async def test_same_tail_user_message_updates_do_not_reanchor():
    """Ticks that merely UPDATE the tail user message never re-anchor."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(11) + [_msg(11, ConsoleMessageRole.USER)]
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()
        reading_position = transcript.scroll_y

        # Same ids re-set (the streaming tick re-sets the message list).
        transcript.set_messages(list(history))
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.scroll_y == reading_position
