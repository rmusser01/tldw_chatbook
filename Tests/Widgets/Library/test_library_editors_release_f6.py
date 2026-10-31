"""The Library's note-body editors leave F6 to the app (TASK-34000.28 N-32, fix round 1 I-1).

Textual 8.2.8's ``TextArea`` binds ``f6`` to ``select_line`` (non-priority),
and the app's own F6 -- "next pane" -- is non-priority by ADR-031, so with a
note body focused F6 selected the current line and focus stayed put.
LibraryScreen keeps F6 app-owned (no screen binding); each Library body
editor releases its own ``f6`` instead (``release_pane_key``).

A bare Textual ``App`` with the app's shape of binding (non-priority ``f6``)
and a real keypress: the key must reach the app's action, and the body must
keep its text with nothing selected. A stock ``TextArea`` is the negative
control -- it still claims the key -- so the fixture is sensitive to exactly
the binding the helper removes.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import TextArea

from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FileNotesEditorTextArea,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import NoteEditorTextArea

pytestmark = pytest.mark.asyncio

_TEXT = "first line\nsecond line"


class _PaneKeyApp(App[None]):
    """An app whose F6 is a non-priority binding, as ``app.py``'s is."""

    BINDINGS = [Binding("f6", "next_pane", "Next pane")]

    def __init__(self, editor: TextArea) -> None:
        super().__init__()
        self.editor = editor
        self.pane_moves = 0

    def compose(self) -> ComposeResult:
        yield self.editor

    def action_next_pane(self) -> None:
        self.pane_moves += 1


async def _press_f6_in(editor: TextArea) -> _PaneKeyApp:
    app = _PaneKeyApp(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        assert app.focused is editor
        editor.move_cursor((1, 3))
        await pilot.press("f6")
        await pilot.pause()
    return app


@pytest.mark.parametrize(
    "editor_type", [NoteEditorTextArea, FileNotesEditorTextArea], ids=lambda t: t.__name__
)
async def test_f6_in_a_library_body_editor_reaches_the_app_binding(editor_type) -> None:
    """F6 runs the app's pane action; the body keeps its text, nothing selected."""
    editor = editor_type(_TEXT)
    app = await _press_f6_in(editor)
    assert app.pane_moves == 1
    assert editor.selection.start == editor.selection.end == (1, 3), editor.selection
    assert editor.text == _TEXT
    # Only F6 was released: the editor's own keys are all still bound.
    assert "f6" not in editor._bindings.key_to_bindings
    assert "ctrl+end" in editor._bindings.key_to_bindings
    assert "shift+home" in editor._bindings.key_to_bindings


async def test_a_stock_text_area_still_claims_f6() -> None:
    """Negative control: Textual's own TextArea selects the line on F6.

    Also pins that the release is per instance -- a stock TextArea built
    after the Library editors above still has its ``f6`` binding.
    """
    editor = TextArea(_TEXT)
    app = await _press_f6_in(editor)
    assert app.pane_moves == 0
    assert editor.selection.start == (1, 0)
    assert editor.selection.end == (1, len("second line"))
