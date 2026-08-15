"""End-to-end console selection: quote routing into the composer (task F).

Closes the loop task E's menu opened: ``ConsoleSelectionQuoteRequested``
bubbles from the transcript to ``ChatScreen``, which inserts the selection
as a block quote into the native composer at the caret. Also covers the
screen-level click-outside dismissal of a mounted selection menu (a click
on any non-transcript widget, e.g. the composer, folds the menu).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class _ComposerApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(id="console-native-composer")


@pytest.mark.asyncio
async def test_insert_quote_prepends_quote_markers():
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("line one\nline two")
        assert "> line one\n> line two" in composer.draft_text()


@pytest.mark.asyncio
async def test_insert_quote_blank_lines_get_bare_marker():
    """An empty selection line quotes as a bare ``>`` (a real block quote)."""
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("first\n\nlast")
        assert composer.draft_text().endswith("> first\n>\n> last")


@pytest.mark.asyncio
async def test_insert_quote_empty_selection_is_noop():
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_text("existing")
        composer.insert_quote("")
        composer.insert_quote("\n")
        assert composer.draft_text() == "existing"


@pytest.mark.asyncio
async def test_insert_quote_lands_at_caret_not_end():
    """The quote splices at the caret, wherever it sits in the draft."""
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_text("hello world")
        for _ in range(6):
            composer.move_cursor_left()  # caret between "hello" and " world"
        composer.insert_quote("X")
        assert "hello> X world" in composer.draft_text()


@pytest.mark.asyncio
async def test_insert_quote_unfocused_composer_lands_at_end():
    """An unfocused composer still inserts: the caret is not focus-bound.

    Spec fallback: the caret always exists in the segment model and sits at
    the end of a freshly initialised (never focused) draft.
    """
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("tail insert")
        assert composer.draft_text().endswith("> tail insert")


@pytest.mark.asyncio
async def test_screen_routes_quote_request_into_composer():
    """ChatScreen consumes ConsoleSelectionQuoteRequested into the draft."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        screen.post_message(ConsoleSelectionQuoteRequested(quote="hello world"))
        await pilot.pause()
        assert "> hello world" in composer.draft_text()


@pytest.mark.asyncio
async def test_click_outside_transcript_dismisses_selection_menu():
    """A click on a non-transcript widget (the composer) folds the menu."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
        # Mount the menu exactly as a real selection release would (onto the
        # transcript); only the click-outside seam is under test here.
        await transcript.mount(ConsoleSelectionMenu(local_x=2, local_y=2))
        await pilot.pause()
        assert screen.query_one(ConsoleSelectionMenu)

        await pilot.click("#console-native-composer")
        await pilot.pause()
        assert not screen.query(ConsoleSelectionMenu)
