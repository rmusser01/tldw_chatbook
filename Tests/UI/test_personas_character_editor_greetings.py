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


async def test_greetings_update_keeps_selection_and_edit_box_in_sync():
    """Regression: Task 3's cursor-race fix (``_select_greeting_row`` after a
    mutation's re-render) was applied to Add/Move but NOT to Update. Without
    it, the async ``RowHighlighted(row=0)`` queued by ``_render_greetings_table``
    (via ``clear()``/``add_row()``) lands one tick later and silently reverts
    both ``_selected_greeting_index`` and the scratch edit box to row 0 - so a
    follow-up Update would commit to the wrong greeting."""
    app = _Host()
    async with app.run_test(size=(120, 60)) as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character(
            {"name": "A", "alternate_greetings": ["First", "Second", "Third"]}
        )
        await pilot.pause()
        # The greetings editor lives in the Advanced section, which loads
        # collapsed (display: none) - open it so pilot.click can hit it.
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()

        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        table.focus()
        await pilot.pause()
        table.move_cursor(row=1)
        await pilot.pause()
        assert ed._selected_greeting_index == 1

        edit_area = ed.query_one("#personas-char-editor-greeting-edit", TextArea)
        edit_area.text = "Second, edited"
        await pilot.click("#personas-char-editor-greeting-update")
        # Let the async RowHighlighted(row=0) queued by the re-render drain -
        # this is exactly the tick where the pre-fix code reverted to row 0.
        await pilot.pause()

        assert ed._selected_greeting_index == 1
        assert edit_area.text == "Second, edited"
        assert ed.get_character_data()["alternate_greetings"] == [
            "First",
            "Second, edited",
            "Third",
        ]


async def test_greetings_delete_selects_surviving_neighbor():
    """Regression: same cursor-race gap as Update, but for Delete. Deleting
    the selected row must leave the surviving neighbor at that position
    selected (and loaded into the edit box), not silently revert to row 0
    once the async RowHighlighted(row=0) from the re-render drains."""
    app = _Host()
    async with app.run_test(size=(120, 60)) as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character(
            {"name": "A", "alternate_greetings": ["First", "Second", "Third"]}
        )
        await pilot.pause()
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()

        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        table.focus()
        await pilot.pause()
        table.move_cursor(row=1)
        await pilot.pause()
        assert ed._selected_greeting_index == 1

        await pilot.click("#personas-char-editor-greeting-delete")
        await pilot.pause()  # let the async RowHighlighted(row=0) drain

        assert ed.get_character_data()["alternate_greetings"] == ["First", "Third"]
        # min(1, len-1) == 1: "Third" is now the surviving neighbor at index 1.
        assert ed._selected_greeting_index == 1
        edit_area = ed.query_one("#personas-char-editor-greeting-edit", TextArea)
        assert edit_area.text == "Third"


async def test_greetings_delete_last_remaining_clears_selection_and_edit_box():
    """Deleting the only remaining greeting must land on the documented empty
    state (no selection, empty edit box) - there is no row left for an async
    RowHighlighted to fire against, so this must be set directly."""
    app = _Host()
    async with app.run_test(size=(120, 60)) as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "alternate_greetings": ["Only"]})
        await pilot.pause()
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()

        table = ed.query_one("#personas-char-editor-greetings-table", DataTable)
        table.focus()
        await pilot.pause()
        table.move_cursor(row=0)
        await pilot.pause()
        assert ed._selected_greeting_index == 0

        await pilot.click("#personas-char-editor-greeting-delete")
        await pilot.pause()

        assert ed.get_character_data()["alternate_greetings"] == []
        assert ed._selected_greeting_index is None
        edit_area = ed.query_one("#personas-char-editor-greeting-edit", TextArea)
        assert edit_area.text == ""


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
