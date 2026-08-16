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

``ConsoleToolDiffRow`` implements the same protocol at LINE granularity over
a deterministic unified-diff projection (phase 3, task 1): the domain is
``difflib.unified_diff`` of the row's ``tool_diff`` contents, offsets snap
outward to whole diff lines, and the highlight renders as a reverse-video
Static strip below the DiffView rather than restyling its internals. Diff
content is immutable, so there is no streaming clamp; row removal rides the
existing reconciliation guard.
"""

import asyncio
import difflib
import time

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Markdown, Static
from textual_diff_view import DiffView

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
    ConsoleToolDiffRow,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
    _diff_cell_to_offset,
    _message_body_render_text,
    _tool_diff_display_text,
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
async def test_markdown_row_selects_at_char_granularity():
    """Character ranges map verbatim (live-spike change: whole-line snapping
    made any partial drag on a one-line reply select the entire message)."""
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 2, body_start + 5)
        assert row.get_selection_text() == "dy "
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
        assert row.get_selection_text() == "y line\nmo"


@pytest.mark.asyncio
async def test_markdown_row_selection_accepts_reversed_order():
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        body_start = _MARKDOWN_SOURCE.index("body")
        row.set_selection_range(body_start + 5, body_start + 2)
        assert row.get_selection_text() == "dy "


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
        assert str(renderable) == "dy "
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
        assert row.get_selection_text() == "dy"  # clamped to the new end


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
        assert row.get_selection_text() == "dy "


@pytest.mark.asyncio
async def test_markdown_row_sync_non_prefix_shrink_clamps_cleanly():
    """A non-prefix body replace under a stored char range must clamp to
    the new length without misalignment (the old whole-line re-snap existed
    to keep line-aligned quotes; char ranges clamp directly)."""
    app = _MarkdownRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleMarkdownMessage)
        source = "line one\nline two\nline three"
        seeded = _make_assistant_message(source)
        row.sync_message(
            seeded,
            resolve_console_message_presentation(seeded, ConsolePresentationContext()),
        )
        two = source.index("line two")
        three = source.index("line three")
        row.set_selection_range(two + 2, three + 2)
        assert row.get_selection_text() == "ne two\nli"

        # Non-prefix replace: offsets clamp to the new body length; the
        # slice stays aligned to the stored characters (no stray newline).
        replaced = _make_assistant_message("abcdefghi\nXY")
        row.sync_message(
            replaced,
            resolve_console_message_presentation(replaced, ConsolePresentationContext()),
        )
        assert row.get_selection_text() == "Y"


@pytest.mark.asyncio
async def test_markdown_selection_strip_caps_huge_selections():
    """The strip caps its display like the quote (select-all must not
    duplicate a huge body below itself)."""
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
        strip = _selection_strip(row)
        renderable = strip.renderable
        assert isinstance(renderable, Text)
        assert len(str(renderable)) == SELECTION_QUOTE_CAP
        assert str(renderable).endswith("[truncated]")


def test_markdown_selection_protocol_is_safe_before_mount():
    row = ConsoleMarkdownMessage(_make_assistant_message("unmounted"))
    row.set_selection_range(2, 4)  # must not raise (row not composed yet)
    assert row.get_selection_text() == "mo"
    row.clear_selection()
    assert row.get_selection_text() == ""


# -- Tool diff rows: LINE-granularity selection protocol (phase 3, task 1) ----

_DIFF_PATH = "/tmp/a.py"
_DIFF_OLD = "alpha\nbeta\ngamma\n"
_DIFF_NEW = "alpha\nBETA\ngamma\ndelta\n"


def _expected_unified_diff(path: str, old: str, new: str) -> str:
    """The plan-specified deterministic projection (keepends, fromfile=tofile)."""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


async def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until true or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


class _ToolDiffRowApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleToolDiffRow("m1", (_DIFF_PATH, _DIFF_OLD, _DIFF_NEW))


class _ToolDiffTranscriptApp(App[None]):
    """Transcript harness with an expanded file-write TOOL marker."""

    def __init__(self) -> None:
        super().__init__()
        self.selected_events: list[ConsoleTranscript.TranscriptTextSelected] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL,
                    content="write_file → /tmp/a.py",
                    id="m1",
                    tool_diff=(_DIFF_PATH, _DIFF_OLD, _DIFF_NEW),
                )
            ]
        )
        yield transcript

    def on_console_transcript_transcript_text_selected(
        self, event: ConsoleTranscript.TranscriptTextSelected
    ) -> None:
        self.selected_events.append(event)


def _diff_strip(row: ConsoleToolDiffRow) -> Static:
    return row.query_one(".console-tool-diff-selection-strip", Static)


def _mouse_event(event_cls, widget, *, screen_x: int, screen_y: int, button: int = 1):
    """Build a mouse event addressed to ``widget`` with absolute coordinates."""
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


def test_tool_diff_display_text_projection_is_deterministic():
    text = _tool_diff_display_text((_DIFF_PATH, _DIFF_OLD, _DIFF_NEW))
    assert text == _expected_unified_diff(_DIFF_PATH, _DIFF_OLD, _DIFF_NEW)
    assert text.startswith(f"--- {_DIFF_PATH}\n+++ {_DIFF_PATH}\n@@")
    assert "-beta\n" in text
    assert "+BETA\n" in text


@pytest.mark.asyncio
async def test_tool_diff_display_text_is_unified_diff_projection():
    app = _ToolDiffRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleToolDiffRow)
        assert row.get_display_text() == _expected_unified_diff(
            _DIFF_PATH, _DIFF_OLD, _DIFF_NEW
        )


@pytest.mark.asyncio
async def test_tool_diff_selection_snaps_to_whole_diff_lines():
    app = _ToolDiffRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleToolDiffRow)
        text = row.get_display_text()
        start = text.index("-beta") + 2  # mid-line offsets...
        end = text.index("+BETA") + 2
        row.set_selection_range(start, end)
        # ...snap outward to whole diff lines (newlines stay inside the quote).
        assert row.get_selection_text() == "-beta\n+BETA"
        row.clear_selection()
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_tool_diff_selection_accepts_reversed_order():
    app = _ToolDiffRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleToolDiffRow)
        text = row.get_display_text()
        row.set_selection_range(text.index("+BETA") + 2, text.index("-beta") + 2)
        assert row.get_selection_text() == "-beta\n+BETA"


@pytest.mark.asyncio
async def test_tool_diff_selection_clamps_to_projection_length():
    app = _ToolDiffRowApp()
    async with app.run_test():
        row = app.query_one(ConsoleToolDiffRow)
        row.set_selection_range(0, 9999)
        assert row.get_selection_text() == row.get_display_text()


@pytest.mark.asyncio
async def test_tool_diff_selection_renders_reverse_strip_below_diff_view():
    app = _ToolDiffRowApp()
    async with app.run_test() as pilot:
        row = app.query_one(ConsoleToolDiffRow)
        assert await _wait_for(lambda: bool(row.query(DiffView)))
        await pilot.pause()
        strip = _diff_strip(row)
        assert strip.display is False  # hidden until a selection exists

        text = row.get_display_text()
        row.set_selection_range(text.index("-beta") + 1, text.index("+BETA") + 1)
        assert strip.display is True
        renderable = strip.renderable
        assert isinstance(renderable, Text)
        assert str(renderable) == "-beta\n+BETA"
        assert "reverse" in str(renderable.style)

        # The strip sits below the DiffView, not above it (and never inside it).
        diff_view = row.query_one(DiffView)
        children = list(row.children)
        assert children.index(diff_view) < children.index(strip)

        row.clear_selection()
        assert strip.display is False


@pytest.mark.asyncio
async def test_tool_diff_selection_text_caps_long_diffs():
    huge_old = "x" * 5000 + "\n"
    huge_new = "y" * 5000 + "\n"

    class _HugeDiffApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ConsoleToolDiffRow("m2", (_DIFF_PATH, huge_old, huge_new))

    app = _HugeDiffApp()
    async with app.run_test():
        row = app.query_one(ConsoleToolDiffRow)
        row.set_selection_range(0, len(row.get_display_text()))
        quoted = row.get_selection_text()
        assert len(quoted) == SELECTION_QUOTE_CAP
        assert quoted.endswith("[truncated]")


def test_tool_diff_selection_protocol_is_safe_before_mount():
    row = ConsoleToolDiffRow("m1", (_DIFF_PATH, _DIFF_OLD, _DIFF_NEW))
    text = row.get_display_text()
    row.set_selection_range(text.index("-beta") + 1, text.index("-beta") + 3)
    assert row.get_selection_text() == "-beta"
    row.clear_selection()
    assert row.get_selection_text() == ""


def test_diff_cell_to_offset_distributes_lines_and_clamps():
    text = _expected_unified_diff(_DIFF_PATH, _DIFF_OLD, _DIFF_NEW)
    lines = text.split("\n")
    height = len(lines)
    # Above the diff body clamps to the projection start.
    assert _diff_cell_to_offset(text, height, 0, -1) == 0
    # Below it clamps to the end.
    assert _diff_cell_to_offset(text, height, 0, height) == len(text)
    # The last rendered row maps to the last line (nearest clamp).
    assert _diff_cell_to_offset(text, height, 0, height - 1) == len(text) - len(
        lines[-1]
    )
    # Interior rows map within their distributed line (monotone in cell_x).
    first_line = lines[0]
    assert _diff_cell_to_offset(text, height, 3, 0) == min(3, len(first_line))
    assert _diff_cell_to_offset(text, height, 99, 0) == len(first_line)


def test_diff_cell_to_offset_single_line_degrades_to_monotone():
    assert _diff_cell_to_offset("abc", 1, 99, 0) == 3  # single line
    assert _diff_cell_to_offset("abc", 0, 2, 5) == 2  # not laid out


@pytest.mark.asyncio
async def test_transcript_resolves_diff_view_drag_to_line_selection():
    """The transcript resolves a DiffView press to its diff row (widened row
    union) and a drag arms/extends at whole-diff-line granularity."""
    app = _ToolDiffTranscriptApp()
    async with app.run_test(size=(80, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await transcript.refresh_messages()
        await pilot.pause()
        transcript.toggle_tool_output("m1")  # expand: mounts the diff row
        assert await _wait_for(lambda: bool(transcript.query(ConsoleToolDiffRow)))
        assert await _wait_for(lambda: bool(transcript.query(DiffView)))
        await pilot.pause()

        diff_row = transcript.query_one(ConsoleToolDiffRow)
        diff_view = transcript.query_one(DiffView)
        region = diff_view.region
        assert region.height > 1

        # A press on the DiffView resolves to the diff row via the widened
        # ``_selection_row_for`` union.
        assert transcript._selection_row_for(diff_view) is diff_row

        text = diff_row.get_display_text()
        diff_row.post_message(
            _mouse_event(
                MouseDown, diff_row, screen_x=region.x, screen_y=region.y
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is True
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert selection.row_key == diff_row.id

        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=region.x + 3, screen_y=region.y + 2
            )
        )
        await pilot.pause()
        quoted = diff_row.get_selection_text()
        assert quoted, "drag over the diff produced no line selection"
        # Line granularity: the quote is whole projection lines (starts on a
        # line boundary, ends on one), and dragging from the top row selects
        # from the projection start.
        assert quoted.startswith(f"--- {_DIFF_PATH}")
        start = text.find(quoted)
        assert start == 0 or text[start - 1] == "\n"
        after = text[start + len(quoted) :]
        assert after == "" or after.startswith("\n")
        # The strip highlight carries the same whole-line quote.
        strip = _diff_strip(diff_row)
        assert strip.display is True

        transcript.post_message(
            _mouse_event(
                MouseUp, transcript, screen_x=region.x + 3, screen_y=region.y + 2
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is False
        assert len(app.selected_events) == 1
        assert app.selected_events[0].selection.row_key == diff_row.id
