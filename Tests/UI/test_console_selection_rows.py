"""Selection protocol on plain Console transcript rows (console selection phase 1).

``ConsoleTranscriptMessage`` gains a four-method selection protocol over its
BODY text domain (the speaker header is a separate child widget and excluded):
``get_display_text`` / ``get_selection_text`` / ``set_selection_range`` /
``clear_selection``, plus clamp-on-sync so streaming updates never leave a
stored range pointing past the new text.
"""

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsolePresentationContext,
    resolve_console_message_presentation,
)
from tldw_chatbook.Widgets.Console.console_selection import SELECTION_QUOTE_CAP
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscriptMessage,
    _message_body_render_text,
)


def _make_message(body: str, **overrides) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content=body, id="m1", **overrides
    )


def _presentation_for(message: ConsoleChatMessage):
    # The transcript always syncs with a freshly resolved presentation
    # (``ConsoleTranscript._update_row_widget``); mirror that here so a new
    # message body actually reaches the row.
    return resolve_console_message_presentation(message, ConsolePresentationContext())


class _RowApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscriptMessage(_make_message("hello selection world"))


def _body_static(row: ConsoleTranscriptMessage) -> Static:
    return row.query_one(".console-transcript-message-body", Static)


@pytest.mark.asyncio
async def test_display_text_is_plain_body():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        assert row.get_display_text() == "hello selection world"


@pytest.mark.asyncio
async def test_selection_range_highlights_and_quotes():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        assert row.get_selection_text() == "selection"
        row.clear_selection()
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_selection_range_accepts_reversed_order():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(15, 6)
        assert row.get_selection_text() == "selection"


@pytest.mark.asyncio
async def test_selection_range_clamps_to_text_length():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(0, 999)
        assert row.get_selection_text() == "hello selection world"


@pytest.mark.asyncio
async def test_selection_text_caps_long_bodies():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.sync_message(
            long_message := _make_message("x" * 5000),
            _presentation_for(long_message),
        )
        row.set_selection_range(0, 5000)
        quoted = row.get_selection_text()
        assert len(quoted) == SELECTION_QUOTE_CAP
        assert quoted.endswith("[truncated]")


@pytest.mark.asyncio
async def test_set_selection_renders_reverse_span_on_body_static():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        renderable = _body_static(row).renderable
        assert isinstance(renderable, Text)
        assert str(renderable) == "hello selection world"
        reverse_spans = [
            span
            for span in renderable.spans
            if span.start == 6 and span.end == 15 and "reverse" in str(span.style)
        ]
        assert reverse_spans, f"no reverse span over [6, 15): {renderable.spans}"


@pytest.mark.asyncio
async def test_clear_selection_restores_plain_body_renderable():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        row.clear_selection()
        renderable = _body_static(row).renderable
        expected = _message_body_render_text(row._message, row._presentation)
        assert str(renderable) == str(expected)
        assert not isinstance(renderable, Text) or not renderable.spans


@pytest.mark.asyncio
async def test_sync_message_clamps_selection_end_when_text_shrinks():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(0, 15)
        shorter = _make_message("hi there")
        row.sync_message(shorter, _presentation_for(shorter))
        assert row.get_selection_text() == "hi there"
        # The clamped range stays highlighted (rendered), not just stored.
        renderable = _body_static(row).renderable
        assert isinstance(renderable, Text)
        assert any(
            "reverse" in str(span.style) for span in renderable.spans
        ), f"expected a reverse span after clamp-on-sync: {renderable.spans}"


@pytest.mark.asyncio
async def test_sync_message_clears_selection_when_text_shrinks_past_start():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        tiny = _make_message("hi")
        row.sync_message(tiny, _presentation_for(tiny))
        assert row.get_selection_text() == ""
        # Body returns to the un-highlighted plain renderable.
        renderable = _body_static(row).renderable
        assert str(renderable) == "hi"


@pytest.mark.asyncio
async def test_sync_message_keeps_selection_when_text_grows():
    app = _RowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        longer = _make_message("hello selection world and more")
        row.sync_message(longer, _presentation_for(longer))
        assert row.get_selection_text() == "selection"


def test_selection_protocol_is_safe_before_mount():
    row = ConsoleTranscriptMessage(_make_message("unmounted"))
    row.set_selection_range(0, 4)  # must not raise (body Static not composed yet)
    assert row.get_selection_text() == "unmo"
    row.clear_selection()
    assert row.get_selection_text() == ""
