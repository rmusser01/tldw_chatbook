"""TASK-371: a jump-to-latest pill while scrolled up during a streaming reply.

When the reader detaches from the bottom during a run, a docked pill reports the
run state (streaming / stopped / ready) and jumps back to the newest content on
click. It stays hidden while following the tail or when no run is in play.
"""

import pytest
from textual.widgets import Static

from Tests.UI.test_console_transcript_tail_follow import TailFollowHarness, _messages
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


async def _seed_and_detach(pilot, transcript: ConsoleTranscript) -> None:
    """Fill the transcript past the viewport and scroll the reader up."""
    transcript.set_messages(_messages(12))
    await transcript.refresh_messages()
    await pilot.pause()
    transcript.release_anchor()
    transcript.scroll_to(y=0, animate=False)
    await pilot.pause()


def _pill(app) -> Static:
    return app.query_one("#console-transcript-jump-pill", Static)


@pytest.mark.asyncio
async def test_jump_pill_shows_run_state_while_scrolled_up():
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _seed_and_detach(pilot, transcript)

        transcript.sync_jump_indicator("streaming")
        pill = _pill(app)
        assert pill.display is True
        assert "streaming below" in str(pill.renderable)
        assert "jump to latest" in str(pill.renderable)

        transcript.sync_jump_indicator("stopped")
        assert _pill(app).display is True
        assert "stopped" in str(_pill(app).renderable)

        transcript.sync_jump_indicator("completed")
        assert "reply ready" in str(_pill(app).renderable)


@pytest.mark.asyncio
async def test_jump_pill_hidden_while_following_tail():
    """At the bottom (following) the pill stays hidden even during streaming."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(12))
        await transcript.refresh_messages()
        await pilot.pause()
        assert transcript.is_anchored  # following the tail

        transcript.sync_jump_indicator("streaming")
        assert _pill(app).display is False


@pytest.mark.asyncio
async def test_jump_pill_hidden_when_idle_even_if_scrolled_up():
    """No run in play -> no pill, even when detached."""
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _seed_and_detach(pilot, transcript)

        transcript.sync_jump_indicator("idle")
        assert _pill(app).display is False


@pytest.mark.asyncio
async def test_jump_to_latest_reattaches_and_hides_pill():
    app = TailFollowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _seed_and_detach(pilot, transcript)
        transcript.sync_jump_indicator("streaming")
        assert _pill(app).display is True

        transcript.jump_to_latest()
        await pilot.pause()

        assert _pill(app).display is False
        assert transcript.scroll_y == transcript.max_scroll_y
