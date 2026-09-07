"""TASK-1994: j/k/space/b scroll keys on read-only markdown panes."""

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Widgets.reader_scroll import ReaderVerticalScroll


class ScrollHarness(ConsolidatedCSSApp):
    CSS = "ReaderVerticalScroll { height: 10; }"

    def compose(self) -> ComposeResult:
        with ReaderVerticalScroll(id="reader"):
            yield Static("line\n" * 60, id="tall")


@pytest.mark.asyncio
async def test_reader_keys_scroll_line_and_page():
    app = ScrollHarness()
    async with app.run_test(size=(60, 14)) as pilot:
        reader = app.query_one(ReaderVerticalScroll)
        reader.focus()
        await pilot.pause()
        assert reader.scroll_y == 0

        await pilot.press("j")
        await pilot.pause()
        assert reader.scroll_y == 1
        await pilot.press("k")
        await pilot.pause()
        assert reader.scroll_y == 0

        await pilot.press("space")
        for _ in range(10):
            await pilot.pause()
            if reader.scroll_y > 1:
                break
        page_pos = reader.scroll_y
        assert page_pos > 1
        await pilot.press("b")
        for _ in range(10):
            await pilot.pause()
            if reader.scroll_y < page_pos:
                break
        assert reader.scroll_y < page_pos


def test_console_transcript_keeps_selection_bindings():
    """j/k on the transcript remain SELECTION keys, not scroll keys."""
    rendered = [
        tuple(b) if isinstance(b, tuple) else (b.key, b.action)
        for b in ConsoleTranscript.BINDINGS
    ]
    keys = {entry[0]: entry[1] for entry in rendered}
    assert keys.get("down,j") == "select_next"
    assert keys.get("up,k") == "select_previous"
    assert not issubclass(ConsoleTranscript, ReaderVerticalScroll)
