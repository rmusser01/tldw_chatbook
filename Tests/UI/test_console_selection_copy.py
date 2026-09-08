"""Copy a highlighted Console span without changing drafts or truncating text."""

import pytest
from textual.app import ComposeResult
from textual.widgets import Markdown, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_left_rail import make_console_pilot
from Tests.UI.test_console_selection_app_smoke import _drag, _seed_rows
from Tests.UI.test_console_selection_menu import (
    _finish_drag_selection,
    _TranscriptMenuApp,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleToolDiffRow,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", [ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT]
)
async def test_mouse_copy_uses_only_the_highlight_and_preserves_the_draft(role):
    async with make_console_pilot(production_styles=True) as pilot:
        screen = pilot.app.screen
        composer = screen.query_one(ConsoleComposerBar)
        transcript = await _seed_rows(
            pilot,
            [ConsoleChatMessage(role=role, content="hello smoke world", id="copy-me")],
        )
        row = screen.query_one("#console-message-copy-me")
        body = (
            row.query_one(Markdown)
            if isinstance(row, ConsoleMarkdownMessage)
            else row.query_one(".console-transcript-message-body", Static)
        )
        composer.insert_text("keep this draft")
        transcript.focus()
        await pilot.pause()

        await _drag(pilot, body, (3, 0), (11, 0))
        assert row.get_selection_text() == "lo smoke"
        await pilot.click("#console-selection-copy")
        await pilot.pause()

        assert pilot.app.clipboard == "lo smoke"
        assert composer.draft_text() == "keep this draft"
        assert row.get_selection_text() == ""
        assert transcript.selection_manager.state.selection is None
        assert transcript.selected_message_id is None
        assert not screen.query(ConsoleSelectionMenu)
        assert screen.focused is transcript


@pytest.mark.asyncio
async def test_keyboard_selection_enter_copies_and_keeps_the_selected_message():
    async with make_console_pilot(production_styles=True) as pilot:
        screen = pilot.app.screen
        transcript = await _seed_rows(
            pilot,
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="keyboard journey target",
                    id="copy-keyboard",
                )
            ],
        )
        transcript.focus()
        await pilot.press("j", "s", "l", "l", "enter")
        await pilot.pause()
        screen.query_one(ConsoleSelectionMenu)
        await pilot.press("enter")
        await pilot.pause()

        assert pilot.app.clipboard == "key"
        assert transcript.selected_message_id == "copy-keyboard"
        assert not screen.query(ConsoleSelectionMenu)
        assert transcript.selection_manager.state.selection is None
        assert screen.focused is transcript


class LongSelectionApp(ConsolidatedCSSApp):
    def __init__(self, message: ConsoleChatMessage) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages([self.message])
        yield transcript


@pytest.mark.asyncio
@pytest.mark.parametrize("row_kind", ["plain", "markdown", "diff"])
async def test_copy_preserves_long_selection_while_quotes_remain_capped(row_kind):
    selected = "  café λ " * 600 + "\nlast line  "
    message = ConsoleChatMessage(
        role=(
            ConsoleMessageRole.USER
            if row_kind == "plain"
            else ConsoleMessageRole.ASSISTANT
        ),
        content=f"before:{selected}:after",
        id="copy-long",
    )
    if row_kind == "diff":
        message = ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="write_file → /tmp/a.py",
            id="copy-long",
            tool_diff=("/tmp/a.py", "x" * 5000 + "\n", "y\n"),
        )
        selected = "-" + "x" * 5000

    app = LongSelectionApp(message)
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await transcript.refresh_messages()
        if row_kind == "diff":
            transcript.toggle_tool_output(message.id)
            await transcript.refresh_messages()
        await pilot.pause()
        row = (
            transcript.query_one(ConsoleToolDiffRow)
            if row_kind == "diff"
            else transcript.query_one(f"#console-message-{message.id}")
        )
        start = row.get_display_text().index(selected)
        end = start + len(selected)
        transcript.selection_manager.begin_drag(row.id, start)
        transcript.selection_manager.extend_drag(row.id, end)
        row.set_selection_range(start, end)
        selection = transcript.selection_manager.finish_drag()
        assert selection is not None
        assert len(row.get_selection_text()) == 4000
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(selection, 2, 2)
        )
        await pilot.pause()
        await pilot.click("#console-selection-copy")
        await pilot.pause()

        assert app.clipboard == selected
        assert row.get_selection_text() == ""
        assert not app.query(ConsoleSelectionMenu)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection_change", ["empty", "removed"])
async def test_stale_selection_does_not_replace_the_clipboard(selection_change):
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        app.copy_to_clipboard("previous clipboard")
        await _finish_drag_selection(pilot)
        row = app.query_one(ConsoleTranscriptMessage)
        if selection_change == "removed":
            await row.remove()
        else:
            row.clear_selection()
        await pilot.click("#console-selection-copy")
        await pilot.pause()

        assert app.clipboard == "previous clipboard"
        assert not app.query(ConsoleSelectionMenu)
        assert (
            app.query_one(ConsoleTranscript).selection_manager.state.selection is None
        )
