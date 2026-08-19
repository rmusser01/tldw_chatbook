"""Console review notes modal (task-18515 review-note management, task 2).

Covers: rows rendering comment + read-only quote/date preview, the
edit round-trip (Save calls ``on_edit`` with the annotation id and new
text, then re-renders the row; Cancel restores the original text
untouched), the delete round-trip (Delete pushes ``ConfirmationDialog``;
Cancel skips ``on_delete`` and keeps the row; Confirm calls ``on_delete``
and removes the row; deleting the last note dismisses ``True``
immediately), Escape layering (a mid-edit Escape first closes the open
editor, a second Escape dismisses the modal), and that the quote preview
is never made editable.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Static, TextArea

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_review_notes_modal import (
    ConsoleReviewNotesModal,
)

NOTE_1: dict[str, Any] = {
    "annotation_id": "ann-1",
    "conversation_id": "conv-1",
    "row_key": "row-1",
    "message_id": "msg-1",
    "quote_text": "the first quoted excerpt",
    "comment": "original comment one",
    "created_at": "2026-08-01T00:00:00",
    "updated_at": "2026-08-01T00:00:00",
}
NOTE_2: dict[str, Any] = {
    "annotation_id": "ann-2",
    "conversation_id": "conv-1",
    "row_key": "row-2",
    "message_id": "msg-1",
    "quote_text": "the second quoted excerpt",
    "comment": "original comment two",
    "created_at": "2026-08-02T00:00:00",
    "updated_at": "2026-08-02T00:00:00",
}


class _RecordingEdit:
    def __init__(self, *, result: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self._result = result

    async def __call__(self, annotation_id: str, new_comment: str) -> bool:
        # Async because the real callables run their SQLite write off the UI
        # event loop (the modal awaits them).
        self.calls.append((annotation_id, new_comment))
        return self._result


class _RecordingDelete:
    def __init__(self, *, result: bool = True) -> None:
        self.calls: list[str] = []
        self._result = result

    async def __call__(self, annotation_id: str) -> bool:
        self.calls.append(annotation_id)
        return self._result


class _ReviewNotesModalApp(App[None]):
    CSS = """
    Screen { align: center middle; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


def _modal(
    notes: list[dict[str, Any]],
    *,
    on_edit: _RecordingEdit | None = None,
    on_delete: _RecordingDelete | None = None,
) -> ConsoleReviewNotesModal:
    return ConsoleReviewNotesModal(
        notes,
        on_edit or _RecordingEdit(),
        on_delete or _RecordingDelete(),
    )


def _static_text(modal: ConsoleReviewNotesModal, selector: str) -> str:
    return str(modal.query_one(selector, Static).render())


# ---------------------------------------------------------------------------
# Rows render comment + read-only quote/date preview
# ---------------------------------------------------------------------------


async def test_rows_render_comment_quote_and_date() -> None:
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1, NOTE_2])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert (
            _static_text(modal, "#console-review-notes-comment-ann-1")
            == NOTE_1["comment"]
        )
        meta_1 = _static_text(modal, "#console-review-notes-meta-ann-1")
        assert NOTE_1["quote_text"] in meta_1
        assert NOTE_1["created_at"] in meta_1

        assert (
            _static_text(modal, "#console-review-notes-comment-ann-2")
            == NOTE_2["comment"]
        )
        meta_2 = _static_text(modal, "#console-review-notes-meta-ann-2")
        assert NOTE_2["quote_text"] in meta_2
        assert NOTE_2["created_at"] in meta_2

        # The quote preview is a Static, never an editable widget.
        quote_widget = modal.query_one("#console-review-notes-meta-ann-1")
        assert isinstance(quote_widget, Static)


# ---------------------------------------------------------------------------
# Edit round-trip
# ---------------------------------------------------------------------------


async def test_edit_round_trip_calls_on_edit_and_rerenders() -> None:
    on_edit = _RecordingEdit(result=True)
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_edit=on_edit)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-edit-button-ann-1")
        await pilot.pause()

        edit_area = modal.query_one("#console-review-notes-edit-ann-1", TextArea)
        assert edit_area.display is True
        comment_static = modal.query_one(
            "#console-review-notes-comment-ann-1", Static
        )
        assert comment_static.display is False
        edit_area.text = "updated comment text"

        await pilot.click("#console-review-notes-save-button-ann-1")
        await pilot.pause()

        assert on_edit.calls == [("ann-1", "updated comment text")]
        assert (
            _static_text(modal, "#console-review-notes-comment-ann-1")
            == "updated comment text"
        )
        assert comment_static.display is True
        assert edit_area.display is False
        # The quote preview is untouched by an edit.
        assert NOTE_1["quote_text"] in _static_text(
            modal, "#console-review-notes-meta-ann-1"
        )

    assert app.results == []


async def test_edit_cancel_restores_original_text() -> None:
    on_edit = _RecordingEdit()
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_edit=on_edit)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-edit-button-ann-1")
        await pilot.pause()

        edit_area = modal.query_one("#console-review-notes-edit-ann-1", TextArea)
        edit_area.text = "throwaway edit that must not stick"

        await pilot.click("#console-review-notes-cancel-button-ann-1")
        await pilot.pause()

        assert edit_area.text == NOTE_1["comment"]
        assert edit_area.display is False
        assert (
            modal.query_one("#console-review-notes-comment-ann-1", Static).display
            is True
        )
        assert on_edit.calls == []
        assert (
            _static_text(modal, "#console-review-notes-comment-ann-1")
            == NOTE_1["comment"]
        )

    assert app.results == []


async def test_on_edit_returning_false_keeps_editor_open_and_skips_rerender() -> None:
    on_edit = _RecordingEdit(result=False)
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_edit=on_edit)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-edit-button-ann-1")
        await pilot.pause()

        edit_area = modal.query_one("#console-review-notes-edit-ann-1", TextArea)
        edit_area.text = "rejected edit"

        await pilot.click("#console-review-notes-save-button-ann-1")
        await pilot.pause()

        assert on_edit.calls == [("ann-1", "rejected edit")]
        # A rejected save leaves the editor open -- nothing to silently lose.
        assert edit_area.display is True
        assert (
            _static_text(modal, "#console-review-notes-comment-ann-1")
            == NOTE_1["comment"]
        )


# ---------------------------------------------------------------------------
# Delete round-trip
# ---------------------------------------------------------------------------


async def test_delete_shows_confirmation_dialog() -> None:
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1, NOTE_2])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-delete-button-ann-1")
        await pilot.pause()

        assert isinstance(app.screen_stack[-1], ConfirmationDialog)


async def test_delete_cancel_keeps_row_and_skips_on_delete() -> None:
    on_delete = _RecordingDelete()
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1, NOTE_2], on_delete=on_delete)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-delete-button-ann-1")
        await pilot.pause()
        assert isinstance(app.screen_stack[-1], ConfirmationDialog)

        await pilot.click("#cancel-button")
        await pilot.pause()

        assert on_delete.calls == []
        # Both rows survive a cancelled delete.
        assert modal.query_one("#console-review-notes-row-ann-1", Vertical)
        assert modal.query_one("#console-review-notes-row-ann-2", Vertical)

    assert app.results == []


async def test_delete_confirm_calls_on_delete_and_removes_row() -> None:
    on_delete = _RecordingDelete()
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1, NOTE_2], on_delete=on_delete)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-delete-button-ann-1")
        await pilot.pause()
        assert isinstance(app.screen_stack[-1], ConfirmationDialog)

        await pilot.click("#confirm-button")
        await pilot.pause()

        assert on_delete.calls == ["ann-1"]
        with pytest.raises(NoMatches):
            modal.query_one("#console-review-notes-row-ann-1", Vertical)
        assert modal.query_one("#console-review-notes-row-ann-2", Vertical)

    # The modal itself is still open -- one note remains.
    assert app.results == []


async def test_on_delete_returning_false_keeps_row() -> None:
    on_delete = _RecordingDelete(result=False)
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_delete=on_delete)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-delete-button-ann-1")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert on_delete.calls == ["ann-1"]
        assert modal.query_one("#console-review-notes-row-ann-1", Vertical)

    assert app.results == []


async def test_deleting_last_note_dismisses_true() -> None:
    on_delete = _RecordingDelete()
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_delete=on_delete)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-delete-button-ann-1")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

    assert on_delete.calls == ["ann-1"]
    assert app.results == [True]


# ---------------------------------------------------------------------------
# Escape layering: mid-edit Escape closes the editor first, then dismisses
# ---------------------------------------------------------------------------


async def test_escape_first_cancels_open_editor_second_dismisses() -> None:
    on_edit = _RecordingEdit()
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_edit=on_edit)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-edit-button-ann-1")
        await pilot.pause()

        edit_area = modal.query_one("#console-review-notes-edit-ann-1", TextArea)
        edit_area.text = "will be discarded on escape"

        await pilot.press("escape")
        await pilot.pause()

        # First Escape: editor closes, modal stays open, nothing saved.
        assert app.results == []
        assert edit_area.display is False
        assert (
            modal.query_one("#console-review-notes-comment-ann-1", Static).display
            is True
        )
        assert edit_area.text == NOTE_1["comment"]
        assert on_edit.calls == []

        await pilot.press("escape")
        await pilot.pause()

    # Second Escape: modal dismisses -- no edits committed, so False.
    assert app.results == [False]


async def test_escape_with_no_open_editor_dismisses_immediately() -> None:
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [False]


async def test_escape_after_committed_edit_dismisses_true() -> None:
    on_edit = _RecordingEdit(result=True)
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1], on_edit=on_edit)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-edit-button-ann-1")
        await pilot.pause()
        edit_area = modal.query_one("#console-review-notes-edit-ann-1", TextArea)
        edit_area.text = "a committed edit"
        await pilot.click("#console-review-notes-save-button-ann-1")
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [True]


# ---------------------------------------------------------------------------
# Close button routes through the same safe-dismiss path as Escape
# ---------------------------------------------------------------------------


async def test_close_button_dismisses_with_no_changes() -> None:
    app = _ReviewNotesModalApp()
    modal = _modal([NOTE_1])

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#console-review-notes-close")
        await pilot.pause()

    assert app.results == [False]
