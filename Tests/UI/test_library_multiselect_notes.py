from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_notes_state import (
    LibraryNotesListRow,
    LibraryNotesListState,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.Library.library_notes_session import (
    NoteFlushOutcome,
    NoteFlushOutcomeKind,
)


def _fake(select_mode):
    return SimpleNamespace(
        _library_notes_select_mode=select_mode,
        _library_notes_row_selection=RowSelection("notes"),
        _selected_note_id="",
        _library_note_dirty=False,
        _refreshed=0,
        _opened=[],
        _flushed=0,
        _library_notes_view="list",
    )


@pytest.mark.asyncio
async def test_notes_row_select_mode_toggles_and_does_not_open_editor():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)

    async def _flush():
        fake._flushed += 1
        return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(button=SimpleNamespace(note_id="n9"), stop=lambda: None)
    await LibraryScreen.handle_library_notes_row(fake, ev)
    assert fake._library_notes_row_selection.is_selected("n9")
    assert fake._library_notes_view == "list"  # editor NOT opened
    assert fake._refreshed == 1


@pytest.mark.asyncio
async def test_notes_export_selected_scope():
    fake = _fake(True)
    fake._library_notes_row_selection.select_all(["n2", "n1"])

    async def _open(s):
        fake._opened.append(s)

    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_notes_export_selected(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert fake._opened == [ExportScope(kind="notes", ids=("n1", "n2"))]


# -- F-018: "Export selected" explains its disabled state -----------------


def _select_mode_notes_state(selected_count: int = 0) -> LibraryNotesListState:
    return LibraryNotesListState(
        rows=(
            LibraryNotesListRow(
                note_id="n1",
                title="First note",
                age_label="today",
                checked=False,
            ),
        ),
        header_copy="Notes (1)",
        status_copy="",
        empty_copy="",
        select_mode=True,
        selected_count=selected_count,
    )


class _NotesCanvasApp(App):
    def __init__(self, selected_count: int = 0):
        super().__init__()
        self._selected_count = selected_count

    def compose(self):
        yield LibraryNotesCanvas(
            list_state=_select_mode_notes_state(self._selected_count),
            id="library-notes-canvas",
        )


@pytest.mark.asyncio
async def test_export_selected_tooltip_follows_its_disabled_state():
    """F-018: "Export selected" disabled with zero selection says WHY;
    with a selection the tooltip describes the action."""
    async with _NotesCanvasApp(selected_count=0).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-notes-export-selected", Button)
        assert export_btn.disabled is True
        assert "select" in str(export_btn.tooltip).lower()

    async with _NotesCanvasApp(selected_count=1).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-notes-export-selected", Button)
        assert export_btn.disabled is False
        assert "export" in str(export_btn.tooltip).lower()
