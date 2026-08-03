"""ConsoleCommandPopup widget behavior; ChatScreen integration (Tasks 3-4)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_command_suggestions import CommandSuggestion
from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup

SUGGESTIONS = [
    CommandSuggestion(insert_text="/a ", label="/a", description="first"),
    CommandSuggestion(insert_text="/b ", label="/b", description="second"),
]


class _PopupApp(App):
    def compose(self) -> ComposeResult:
        # The popup repositions against whatever carries this id; a Static
        # suffices for widget-level tests.
        yield Static("anchor", id="console-native-composer")
        yield ConsoleCommandPopup()


@pytest.mark.asyncio
async def test_popup_show_highlight_accept_hide():
    app = _PopupApp()
    async with app.run_test(size=(80, 24)) as pilot:
        popup = app.screen.query_one(ConsoleCommandPopup)
        assert not popup.is_open

        popup.show_suggestions(SUGGESTIONS)
        await pilot.pause()
        assert popup.is_open
        assert popup.accept_selected().label == "/a"

        popup.move_highlight(1)
        assert popup.accept_selected().label == "/b"

        popup.move_highlight(1)  # wraps
        assert popup.accept_selected().label == "/a"

        popup.hide()
        await pilot.pause()
        assert not popup.is_open
        assert popup.accept_selected() is None
