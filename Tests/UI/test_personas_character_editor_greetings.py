"""Roleplay P3b Task 3: alternate-greetings list editor (character editor).

Replaces the single newline-joined alt-greetings ``TextArea`` with a real
list editor (``DataTable`` + edit area + Add/Update/Delete/Move buttons),
mirroring ``personas_lore_detail.py``'s entry editor. Each greeting is a
discrete ``str`` mutated in place - never blob-joined/split - so a greeting
containing embedded newlines round-trips byte-identical. This is the
anchor guarantee the old TextArea-splitting approach could not make.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, TextArea

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


async def test_greetings_load_add_delete_reorder_and_multiline_fidelity():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        multi = "Hello there.\n\nA second paragraph."  # a greeting WITH newlines
        ed.load_character({"name": "A", "alternate_greetings": [multi, "Hi!"]})
        await pilot.pause()
        # unedited round-trip is byte-identical (the fidelity guarantee)
        assert ed.get_character_data()["alternate_greetings"] == [multi, "Hi!"]
        # add
        ed._greetings_add("Third")
        assert ed.get_character_data()["alternate_greetings"] == [multi, "Hi!", "Third"]
        # update index 1
        ed._greetings_update(1, "Hi again!")
        assert ed.get_character_data()["alternate_greetings"][1] == "Hi again!"
        # move index 2 up
        ed._greetings_move(2, -1)
        assert ed.get_character_data()["alternate_greetings"] == [
            multi,
            "Third",
            "Hi again!",
        ]
        # delete index 0 (the multi-line one)
        ed._greetings_delete(0)
        assert ed.get_character_data()["alternate_greetings"] == ["Third", "Hi again!"]


async def test_greetings_table_renders_a_row_per_greeting():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "alternate_greetings": ["One", "Two"]})
        await pilot.pause()
        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        assert table.row_count == 2


async def test_greetings_edit_textarea_does_not_mark_dirty_on_selection():
    """Selecting a row loads it into the scratch edit TextArea - that alone
    must not flip the editor dirty (it isn't a persisted field by itself)."""
    dirty_events = []

    class CaptureApp(App):
        def compose(self) -> ComposeResult:
            yield PersonasCharacterEditorWidget()

        def on_editor_content_changed(self, message) -> None:
            dirty_events.append(message)

    app = CaptureApp()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "alternate_greetings": ["One", "Two"]})
        await pilot.pause()
        edit_area = ed.query_one("#personas-char-editor-greeting-edit", TextArea)
        edit_area.text = "Programmatically loaded"
        await pilot.pause()
        assert dirty_events == []


async def test_greetings_buttons_and_row_highlighted_selection_via_pilot_click():
    app = _Host()
    async with app.run_test(size=(120, 60)) as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "alternate_greetings": ["First", "Second"]})
        await pilot.pause()
        # The greetings editor lives in the Advanced section, which loads
        # collapsed (display: none) - open it so pilot.click can hit it.
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()

        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        # Arrow-key navigation only fires RowHighlighted (not RowSelected) -
        # move the cursor to row 1 ("Second") and confirm selection tracking.
        table.focus()
        await pilot.pause()
        table.move_cursor(row=1)
        await pilot.pause()
        assert ed._selected_greeting_index == 1
        edit_area = ed.query_one("#personas-char-editor-greeting-edit", TextArea)
        assert edit_area.text == "Second"

        # Update the selected greeting via the Update button.
        edit_area.text = "Second, edited"
        await pilot.click("#personas-char-editor-greeting-update")
        await pilot.pause()
        assert ed.get_character_data()["alternate_greetings"][1] == "Second, edited"

        # Add a new greeting via the Add button (uses the edit TextArea text).
        edit_area.text = "Third"
        await pilot.click("#personas-char-editor-greeting-add")
        await pilot.pause()
        assert ed.get_character_data()["alternate_greetings"] == [
            "First",
            "Second, edited",
            "Third",
        ]

        # Move the newly-selected ("Third", index 2) row up.
        assert ed._selected_greeting_index == 2
        await pilot.click("#personas-char-editor-greeting-move-up")
        await pilot.pause()
        assert ed.get_character_data()["alternate_greetings"] == [
            "First",
            "Third",
            "Second, edited",
        ]

        # Delete the currently-selected ("Third", now index 1) row.
        assert ed._selected_greeting_index == 1
        await pilot.click("#personas-char-editor-greeting-delete")
        await pilot.pause()
        assert ed.get_character_data()["alternate_greetings"] == [
            "First",
            "Second, edited",
        ]


async def test_new_character_clears_greetings():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "alternate_greetings": ["One"]})
        await pilot.pause()
        ed.new_character()
        await pilot.pause()
        assert ed.get_character_data()["alternate_greetings"] == []
        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        assert table.row_count == 0
