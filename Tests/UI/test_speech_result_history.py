"""Generated takes, kept so options can actually be compared."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Speech.speech_result_history import (
    SpeechResultHistory,
    SpeechTake,
)


class _Harness(App[None]):
    def __init__(self, takes=()):
        super().__init__()
        self._takes = takes

    def compose(self) -> ComposeResult:
        yield SpeechResultHistory(takes=self._takes)


@pytest.mark.asyncio
async def test_an_empty_history_says_why_it_is_empty():
    """An unexplained empty region reads as broken.

    "No takes yet" plus what to do is the difference between an empty state
    and a screen that looks like it failed to load.
    """
    app = _Harness()
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        empty = app.query_one("#speech-history-empty", Static)
        assert "Generate" in str(empty.renderable)


@pytest.mark.asyncio
async def test_newest_take_is_first():
    """Comparison works backwards from the most recent attempt."""
    takes = (
        SpeechTake("t1", "Server default", "mp3", 12.0, "14:00"),
        SpeechTake("t2", "Nova", "wav", 4.0, "14:02"),
    )
    app = _Harness(takes)
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        rows = list(app.query(".speech-take-row"))
        assert len(rows) == 2
        # `render_line` on a Horizontal renders the container's own
        # background, not its children -- read the summary widget itself.
        summaries = [
            str(w.renderable) for w in app.query(".speech-take-summary").results(Static)
        ]
        assert "Nova" in summaries[0], f"newest take not first: {summaries}"
        assert "Server default" in summaries[1]


@pytest.mark.asyncio
async def test_every_take_keeps_its_own_play_and_export():
    """The point of a history is re-hearing an earlier option.

    Per-row controls are what make that possible; a single player bound to
    the latest take would force a re-generate to compare.
    """
    takes = (
        SpeechTake("t1", "Server default", "mp3", 12.0, "14:00"),
        SpeechTake("t2", "Nova", "wav", 4.0, "14:02"),
    )
    app = _Harness(takes)
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        for take_id in ("t1", "t2"):
            assert app.query(f"#speech-take-play-{take_id}")
            assert app.query(f"#speech-take-export-{take_id}")


@pytest.mark.asyncio
async def test_a_takes_summary_states_the_variables_being_compared():
    """Voice and format are what the user is choosing between, so a row that
    omits them cannot be compared against its neighbour."""
    take = SpeechTake("t1", "Nova", "wav", 91.0, "14:02")
    assert "Nova" in take.summary
    assert "wav" in take.summary
    assert "1:31" in take.summary, "duration not rendered as m:ss"
    assert "14:02" in take.summary


@pytest.mark.asyncio
async def test_adding_a_take_shows_it_without_losing_the_previous_ones():
    app = _Harness((SpeechTake("t1", "Server default", "mp3", 12.0, "14:00"),))
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        history = app.query_one(SpeechResultHistory)
        history.add_take(SpeechTake("t2", "Nova", "wav", 4.0, "14:02"))
        await pilot.pause()
        await pilot.pause()
        assert len(list(app.query(".speech-take-row"))) == 2
        assert app.query("#speech-take-play-t1")
        assert not app.query("#speech-history-empty")
