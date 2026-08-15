"""Selection protocol on Console transcript rows (console selection phase 1).

``ConsoleTranscriptMessage`` gains a four-method selection protocol over its
BODY text domain (the speaker header is a separate child widget and excluded):
``get_display_text`` / ``get_selection_text`` / ``set_selection_range`` /
``clear_selection``, plus clamp-on-sync so streaming updates never leave a
stored range pointing past the new text.

``ConsoleMarkdownMessage`` implements the same protocol at LINE granularity
(task G): the selection domain is the markdown SOURCE (``_body_text``),
offsets snap outward to whole source lines, and the highlight renders as a
reverse-video Static strip below the Markdown widget rather than restyling
the Markdown renderer's internals.
"""

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

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
    ConsoleMarkdownMessage,
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
    async with app.run_test():
        row = app.query_one(ConsoleTranscriptMessage)
        assert row.get_display_text() == "hello selection world"


@pytest.mark.asyncio
async def test_selection_range_highlights_and_quotes():
    app = _RowApp()
    async with app.run_test():
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(6, 15)
        assert row.get_selection_text() == "selection"
        row.clear_selection()
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_selection_range_accepts_reversed_order():
    app = _RowApp()
    async with app.run_test():
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(15, 6)
        assert row.get_selection_text() == "selection"


@pytest.mark.asyncio
async def test_selection_range_clamps_to_text_length():
    app = _RowApp()
    async with app.run_test():
        row = app.query_one(ConsoleTranscriptMessage)
        row.set_selection_range(0, 999)
        assert row.get_selection_text() == "hello selection world"


@pytest.mark.asyncio
async def test_selection_text_caps_long_bodies():
    app = _RowApp()
    async with app.run_test():
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
    async with app.run_test():
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
    async with app.run_test():
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
    async with app.run_test():
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
    async with app.run_test():
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
    async with app.run_test():
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


# -- Markdown rows: LINE-granularity selection protocol (task G) --------------

_MARKDOWN_SOURCE = "# Title\n\nbody line\nmore text"


def _make_assistant_message(body: str, **overrides) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content=body, id="a1", **overrides
    )


class _MarkdownRowApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleMarkdownMessage(_make_assistant_message(_MARKDOWN_SOURCE))


def _selection_strip(row: ConsoleMarkdownMessage) -> Static:
    return row.query_one(".console-markdown-selection-strip", Static)


@pytest.mark.asyncio
async def test_markdown_display_text_is_markdown_source():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        assert row.get_display_text() == _MARKDOWN_SOURCE


@pytest.mark.asyncio
async def test_markdown_row_selects_at_line_granularity():
    """Offsets INSIDE a line snap outward to the whole source line."""
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 2, body_start + 5)
        assert row.get_selection_text() == "body line"
        row.clear_selection()
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_markdown_row_selection_spans_whole_lines():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        more_start = _MARKDOWN_SOURCE.index("more")
        row.set_selection_range(body_start + 3, more_start + 2)
        assert row.get_selection_text() == "body line\nmore text"


@pytest.mark.asyncio
async def test_markdown_row_selection_accepts_reversed_order():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 5, body_start + 2)
        assert row.get_selection_text() == "body line"


@pytest.mark.asyncio
async def test_markdown_row_selection_clamps_to_source_length():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        row.set_selection_range(0, 999)
        assert row.get_selection_text() == _MARKDOWN_SOURCE


@pytest.mark.asyncio
async def test_markdown_row_selection_text_caps_long_bodies():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        long_message = _make_assistant_message("x" * 5000)
        row.sync_message(
            long_message,
            resolve_console_message_presentation(
                long_message, ConsolePresentationContext()
            ),
        )
        row.set_selection_range(0, 5000)
        quoted = row.get_selection_text()
        assert len(quoted) == SELECTION_QUOTE_CAP
        assert quoted.endswith("[truncated]")


@pytest.mark.asyncio
async def test_markdown_row_selection_renders_reverse_strip():
    """The highlight is a reverse-video Static strip below the Markdown child."""
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        strip = _selection_strip(row)
        assert strip.display is False  # hidden until a selection exists

        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 2, body_start + 5)
        assert strip.display is True
        renderable = strip.renderable
        assert isinstance(renderable, Text)
        assert str(renderable) == "body line"
        assert "reverse" in str(renderable.style)

        # The strip sits below the Markdown widget, not inside it.
        markdown = row.query_one(Markdown)
        children = list(row.children)
        assert children.index(markdown) < children.index(strip)

        row.clear_selection()
        assert strip.display is False


@pytest.mark.asyncio
async def test_markdown_row_sync_clamps_selection_when_body_shrinks():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 2, body_start + 5)
        shorter = _make_assistant_message("# Title\n\nbody")
        row.sync_message(
            shorter,
            resolve_console_message_presentation(shorter, ConsolePresentationContext()),
        )
        assert row.get_selection_text() == "body"  # clamped to the new end


@pytest.mark.asyncio
async def test_markdown_row_sync_clears_selection_when_body_shrinks_past_start():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        more_start = _MARKDOWN_SOURCE.index("more")
        row.set_selection_range(more_start + 1, more_start + 3)
        tiny = _make_assistant_message("hi")
        row.sync_message(
            tiny,
            resolve_console_message_presentation(tiny, ConsolePresentationContext()),
        )
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_markdown_row_sync_keeps_selection_when_body_grows():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        seed = _make_assistant_message("# Title\n\nbody line")
        row.sync_message(
            seed,
            resolve_console_message_presentation(seed, ConsolePresentationContext()),
        )
        body_start = seed.content.index("body")
        row.set_selection_range(body_start + 2, body_start + 5)
        grown = _make_assistant_message("# Title\n\nbody line\nmore text")
        row.sync_message(
            grown,
            resolve_console_message_presentation(grown, ConsolePresentationContext()),
        )
        assert row.get_selection_text() == "body line"


def test_markdown_selection_protocol_is_safe_before_mount():
    row = ConsoleMarkdownMessage(_make_assistant_message("unmounted"))
    row.set_selection_range(2, 4)  # must not raise (row not composed yet)
    assert row.get_selection_text() == "unmounted"
    row.clear_selection()
    assert row.get_selection_text() == ""
