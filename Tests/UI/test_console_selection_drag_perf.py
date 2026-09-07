"""Drag-selection per-MouseMove cost bounds (TASK-21114).

A drag delivers MouseMove at 50-100 Hz. The per-event work must be O(changed
state), not O(message body): the wrap table for the origin row's body is
memoized per (text, width), an unchanged offset re-renders nothing, and the
stale-highlight sweep over every mounted row runs once per drag (at arm time),
not per move. Each test drives the REAL transcript handlers with synthetic
mouse events over a ~20 KB plain row -- the evidence-doc scenario
(Docs/Design/2026-08-22-holistic-perf-review.md, finding 21114).
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
import tldw_chatbook.Widgets.Console.console_transcript as console_transcript_module
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)

#: ~20 KB multi-line body (28 paragraphs that each wrap over several lines).
_BIG_BODY = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit " * 13 + "\n"
) * 28
assert 18_000 < len(_BIG_BODY) < 24_000


class _DragPerfApp(App[None]):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content=_BIG_BODY, id="big"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="short other row", id="other"
                ),
            ]
        )
        yield transcript


def _mouse_event(event_cls, widget, *, screen_x: int, screen_y: int, button: int = 1):
    """Synthetic mouse event addressed by absolute screen cell (the harness
    convention shared with test_console_selection_transcript.py)."""
    return event_cls(
        widget=widget,
        x=screen_x - widget.region.x,
        y=screen_y - widget.region.y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


async def _armed_drag(pilot):
    """Mount the big row, arm a drag on it, and return (transcript, row, body)."""
    app = pilot.app
    transcript = app.query_one(ConsoleTranscript)
    await transcript.refresh_messages()
    await pilot.pause()
    row = app.query_one("#console-message-big", ConsoleTranscriptMessage)
    body = row.query_one(".console-transcript-message-body", Static)
    transcript.on_mouse_down(
        _mouse_event(
            MouseDown, row, screen_x=body.region.x + 2, screen_y=body.region.y + 1
        )
    )
    assert transcript.selection_manager.state.active is True
    return transcript, row, body


def _sweeping_moves(transcript, body, count: int):
    """``count`` MouseMoves across distinct cells of the wrapped body."""
    region = body.region
    return [
        _mouse_event(
            MouseMove,
            transcript,
            screen_x=region.x + 2 + (i % 50),
            screen_y=region.y + 1 + (i // 10) % min(region.height, 20),
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_drag_wraps_the_body_at_most_once_per_text_and_width():
    """30 sweeping moves over a 20 KB row must not re-wrap the body per event.

    The wrap table is memoized per (text, width): one ``Content.wrap`` for the
    whole drag (the arm-time mapping), regardless of how many moves follow.
    """
    wrap_calls = 0
    real_content = console_transcript_module.Content

    class _CountingContent(real_content):
        def wrap(self, *args, **kwargs):
            nonlocal wrap_calls
            wrap_calls += 1
            return super().wrap(*args, **kwargs)

    app = _DragPerfApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, row, body = await _armed_drag(pilot)
        wrap_table = getattr(console_transcript_module, "_body_wrap_table", None)
        if wrap_table is not None and hasattr(wrap_table, "cache_clear"):
            wrap_table.cache_clear()
        console_transcript_module.Content = _CountingContent
        try:
            for event in _sweeping_moves(transcript, body, 30):
                transcript.on_mouse_move(event)
        finally:
            console_transcript_module.Content = real_content
        # One wrap rebuilds the memoized table after the cache_clear; the
        # other 29 moves must reuse it. (Pre-fix: one wrap per move = 30.)
        assert wrap_calls <= 2, f"expected memoized wrap table, got {wrap_calls} wraps"
        # The drag still works: the selection followed the last move.
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert selection.end > selection.start


@pytest.mark.asyncio
async def test_moves_at_an_unchanged_offset_do_not_rerender_the_body():
    """Same-cell moves (an idle pointer between cells) must not re-render.

    One offset change = one body update; ten further moves mapping to the
    same offset add zero. (Pre-fix: every move re-rendered the full body.)
    """
    app = _DragPerfApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, row, body = await _armed_drag(pilot)
        update_calls = 0
        real_update = body.update

        def counting_update(*args, **kwargs):
            nonlocal update_calls
            update_calls += 1
            return real_update(*args, **kwargs)

        body.update = counting_update
        target = _mouse_event(
            MouseMove,
            transcript,
            screen_x=body.region.x + 11,
            screen_y=body.region.y + 1,
        )
        try:
            transcript.on_mouse_move(target)
            assert update_calls == 1, "the first offset change must render"
            for _ in range(10):
                transcript.on_mouse_move(target)
        finally:
            body.update = real_update
        assert update_calls == 1, (
            f"unchanged-offset moves re-rendered the body {update_calls - 1} times"
        )
        # The selection the moves produced is intact and quotable.
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert row.get_selection_text() != ""


@pytest.mark.asyncio
async def test_moves_do_not_sweep_other_rows_per_event():
    """The stale-highlight sweep runs at drag arm, never per MouseMove."""
    app = _DragPerfApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript, row, body = await _armed_drag(pilot)
        other = app.query_one("#console-message-other", ConsoleTranscriptMessage)
        clear_calls = 0
        real_clear = other.clear_selection

        def counting_clear(*args, **kwargs):
            nonlocal clear_calls
            clear_calls += 1
            return real_clear(*args, **kwargs)

        other.clear_selection = counting_clear
        try:
            for event in _sweeping_moves(transcript, body, 20):
                transcript.on_mouse_move(event)
        finally:
            other.clear_selection = real_clear
        assert clear_calls == 0, (
            f"MouseMove swept other rows {clear_calls} times (sweep belongs at arm time)"
        )


@pytest.mark.asyncio
async def test_arm_time_sweep_still_clears_a_stale_highlight():
    """Moving the sweep to arm time must not orphan a prior row's highlight.

    A finished drag's highlight on row A must disappear when a new drag arms
    on row B -- the guarantee the per-move sweep used to (redundantly)
    re-assert on every event.
    """
    app = _DragPerfApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await transcript.refresh_messages()
        await pilot.pause()
        row_a = app.query_one("#console-message-other", ConsoleTranscriptMessage)
        row_b = app.query_one("#console-message-big", ConsoleTranscriptMessage)
        body_a = row_a.query_one(".console-transcript-message-body", Static)
        body_b = row_b.query_one(".console-transcript-message-body", Static)
        # Drag on row A and finish it: A holds a highlight.
        transcript.on_mouse_down(
            _mouse_event(
                MouseDown, row_a, screen_x=body_a.region.x + 1, screen_y=body_a.region.y
            )
        )
        transcript.on_mouse_move(
            _mouse_event(
                MouseMove,
                transcript,
                screen_x=body_a.region.x + 6,
                screen_y=body_a.region.y,
            )
        )
        transcript.on_mouse_up(
            _mouse_event(
                MouseUp, transcript, screen_x=body_a.region.x + 6, screen_y=body_a.region.y
            )
        )
        await pilot.pause()
        assert row_a.get_selection_text() != ""
        # Arm a new drag on row B: A's stale highlight must clear.
        transcript.on_mouse_down(
            _mouse_event(
                MouseDown, row_b, screen_x=body_b.region.x + 2, screen_y=body_b.region.y
            )
        )
        assert row_a.get_selection_text() == ""
        transcript.on_mouse_up(
            _mouse_event(
                MouseUp, transcript, screen_x=body_b.region.x + 2, screen_y=body_b.region.y
            )
        )
        await pilot.pause()
