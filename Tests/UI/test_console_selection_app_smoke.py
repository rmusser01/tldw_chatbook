"""Console selection phase-1 smoke: the full loop on the REAL ChatScreen.

Drives the production Console harness (``make_console_pilot``, the same one
the rail/shell tests use): a plain-row drag selects text and mounts the
floating menu, ``Add to chat`` lands a ``> ``-quoted block at the composer's
caret, a markdown-row drag quotes whole source lines, and a genuine plain
click still toggles message selection without opening the menu.
"""

import pytest
from textual.widgets import Markdown, Static

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)

_PLAIN_BODY = "hello smoke world"
_MARKDOWN_SOURCE = "line one\nline two\nline three"


async def _seed_rows(pilot, messages: list[ConsoleChatMessage]) -> ConsoleTranscript:
    """Mount ``messages`` into the real screen's transcript.

    The screen's 0.2s transcript-sync poll reconciles the widget against the
    (empty) store, so seeding waits for the poll to self-stop first -- the
    same settle the fleet-wake freshness test relies on -- or the rows would
    be wiped by the next tick.
    """
    screen = pilot.app.screen
    for _ in range(50):
        if screen._console_transcript_sync_timer is None:
            break
        await pilot.pause(0.1)
    assert screen._console_transcript_sync_timer is None
    transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.set_messages(messages)
    await transcript.refresh_messages()
    await pilot.pause()
    return transcript


async def _drag(pilot, widget, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Real pilot drag: press, move, release (release synthesizes the Click)."""
    await pilot.mouse_down(widget, offset=start)
    await pilot.hover(widget, offset=end)
    await pilot.mouse_up(widget, offset=end)
    await pilot.pause()


@pytest.mark.asyncio
async def test_plain_drag_menu_add_to_chat_quotes_at_composer_caret():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        transcript = await _seed_rows(
            pilot,
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content=_PLAIN_BODY, id="smoke-plain"
                )
            ],
        )
        row = screen.query_one("#console-message-smoke-plain", ConsoleTranscriptMessage)
        body = row.query_one(".console-transcript-message-body", Static)
        composer.insert_text("note: ")
        await pilot.pause()

        await _drag(pilot, body, (3, 0), (11, 0))

        # The drag release toggled no message selection (a real terminal
        # suppresses the click after a drag, so the suppression flag stays
        # set until the next press -- exactly the guarded outcome).
        assert transcript.selected_message_id is None
        assert row.get_selection_text() == "lo smoke"
        screen.query_one(ConsoleSelectionMenu)  # menu mounted at the release cell

        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()

        # The quote spliced in at the caret (end of the seeded "note: " run).
        assert composer.draft_text() == "note: > lo smoke"
        assert row.get_selection_text() == ""  # selection cleaned up
        assert not screen.query(ConsoleSelectionMenu)


@pytest.mark.asyncio
async def test_markdown_drag_add_to_chat_quotes_whole_lines():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        await _seed_rows(
            pilot,
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content=_MARKDOWN_SOURCE,
                    id="smoke-md",
                )
            ],
        )
        row = screen.query_one("#console-message-smoke-md", ConsoleMarkdownMessage)
        body = row.query_one(Markdown)
        bottom = max(0, body.region.height - 1)
        # Far-right visible cell of the body row; the offset map clamps
        # beyond the line length to the line end.
        bottom_x = max(0, body.region.width - 1)

        # Press at the body origin, release at the last line's far edge
        # (clamped by the offset map to the line end): the char-level
        # range covers the entire source.
        await _drag(pilot, body, (0, 0), (bottom_x, bottom))

        assert row.get_selection_text() == _MARKDOWN_SOURCE
        screen.query_one(ConsoleSelectionMenu)

        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()

        assert composer.draft_text() == "> line one\n> line two\n> line three"


@pytest.mark.asyncio
async def test_plain_click_toggles_selection_without_menu():
    """Control: a genuine click is not a drag -- no menu, toggle still works."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = await _seed_rows(
            pilot,
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content=_PLAIN_BODY, id="smoke-plain"
                )
            ],
        )
        row = screen.query_one("#console-message-smoke-plain", ConsoleTranscriptMessage)
        body = row.query_one(".console-transcript-message-body", Static)

        await pilot.click(body, offset=(4, 0))
        await pilot.pause()

        assert transcript.selected_message_id == "smoke-plain"
        assert transcript.selection_manager.state.active is False
        assert row.get_selection_text() == ""
        assert not screen.query(ConsoleSelectionMenu)
