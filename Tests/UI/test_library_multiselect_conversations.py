from types import SimpleNamespace
import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen, LIBRARY_ROW_BROWSE_CONVERSATIONS
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationsCanvasState,
    LibraryConversationRow,
)
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)


def _fake(select_mode):
    return SimpleNamespace(
        _library_conversations_select_mode=select_mode,
        _library_conversations_row_selection=RowSelection("conversations"),
        _selected_conversation_id="", _library_selected_row_id="", _refreshed=0, _opened=[],
    )


def test_convo_row_select_mode_toggles():
    fake = _fake(True); fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    ev = SimpleNamespace(button=SimpleNamespace(conversation_id="c5"), stop=lambda: None)
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._library_conversations_row_selection.is_selected("c5")
    assert fake._selected_conversation_id == ""     # did NOT open/select the detail
    assert fake._refreshed == 1


def test_convo_row_normal_mode_selects():
    fake = _fake(False); fake.refresh = lambda **k: None
    ev = SimpleNamespace(button=SimpleNamespace(conversation_id="c5"), stop=lambda: None)
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._selected_conversation_id == "c5"
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS


@pytest.mark.asyncio
async def test_convo_export_selected_scope():
    fake = _fake(True); fake._library_conversations_row_selection.select_all(["c2", "c1"])
    async def _open(s): fake._opened.append(s)
    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_conversations_export_selected(fake, SimpleNamespace(stop=lambda: None))
    assert fake._opened == [ExportScope(kind="conversations", ids=("c1", "c2"))]


def _select_mode_canvas_state() -> LibraryConversationsCanvasState:
    rows = (
        LibraryConversationRow(
            conversation_id="c1",
            title="First conversation",
            secondary="today",
            checked=False,
        ),
        LibraryConversationRow(
            conversation_id="c2",
            title="Second conversation",
            secondary="today",
            checked=False,
        ),
    )
    return LibraryConversationsCanvasState(
        rows=rows,
        query="",
        status_copy="",
        empty_copy="No conversations in your Library yet.",
        selected_id="",
        preview_lines=(),
        select_mode=True,
        selected_count=0,
    )


class _ConversationsCanvasApp(App):
    def compose(self):
        yield LibraryConversationsCanvas(
            canvas=_select_mode_canvas_state(), id="library-conversations-canvas"
        )


@pytest.mark.asyncio
async def test_canvas_select_mode_renders_action_row_and_disables_export():
    app = _ConversationsCanvasApp()
    async with app.run_test() as pilot:
        select_all_btn = pilot.app.query_one("#library-conversations-select-all", Button)
        assert select_all_btn is not None
        assert "2 shown" in str(select_all_btn.label)
        export_selected_btn = pilot.app.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_selected_btn.disabled is True
