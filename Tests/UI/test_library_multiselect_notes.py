from types import SimpleNamespace
import pytest
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def _fake(select_mode):
    return SimpleNamespace(
        _library_notes_select_mode=select_mode,
        _library_notes_row_selection=RowSelection("notes"),
        _selected_note_id="", _library_note_dirty=False, _refreshed=0, _opened=[], _flushed=0,
        _library_notes_view="list",
    )


@pytest.mark.asyncio
async def test_notes_row_select_mode_toggles_and_does_not_open_editor():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    async def _flush(): fake._flushed += 1
    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(button=SimpleNamespace(note_id="n9"), stop=lambda: None)
    await LibraryScreen.handle_library_notes_row(fake, ev)
    assert fake._library_notes_row_selection.is_selected("n9")
    assert fake._library_notes_view == "list"      # editor NOT opened
    assert fake._refreshed == 1


@pytest.mark.asyncio
async def test_notes_export_selected_scope():
    fake = _fake(True); fake._library_notes_row_selection.select_all(["n2", "n1"])
    async def _open(s): fake._opened.append(s)
    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_notes_export_selected(fake, SimpleNamespace(stop=lambda: None))
    assert fake._opened == [ExportScope(kind="notes", ids=("n1", "n2"))]
