"""Console review notes modal.

task-18515 review-note management, task 2. Pushed by the owning screen
(task 3) in response to ``ConsoleReviewNotesRequested`` -- this module
never imports DB code: ``on_edit``/``on_delete`` are sync callables the
screen wires to its own off-thread execution against
``CharactersRAGDB.get_transcript_annotations`` and friends.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

_EDIT_BUTTON_PREFIX = "console-review-notes-edit-button-"
_DELETE_BUTTON_PREFIX = "console-review-notes-delete-button-"
_SAVE_BUTTON_PREFIX = "console-review-notes-save-button-"
_CANCEL_BUTTON_PREFIX = "console-review-notes-cancel-button-"
_CLOSE_BUTTON_ID = "console-review-notes-close"


def _row_id(prefix: str, annotation_id: str) -> str:
    return f"console-review-notes-{prefix}-{annotation_id}"


def _row_selector(prefix: str, annotation_id: str) -> str:
    return f"#{_row_id(prefix, annotation_id)}"


def _format_meta(note: dict[str, Any]) -> str:
    """Render the dim, read-only quote + created-date preview line."""
    quote = (note.get("quote_text") or "").strip()
    created_at = note.get("created_at") or ""
    if quote and created_at:
        return f'"{quote}" — {created_at}'
    if quote:
        return f'"{quote}"'
    return created_at


class ConsoleReviewNotesModal(SafeModalDismissMixin, ModalScreen[bool]):
    """Browse, edit, and delete the review notes anchored to one message.

    One row per note: a multi-line comment, a dim read-only quote+date
    preview, and Edit/Delete buttons. Edit swaps the comment ``Static``
    for a prefilled ``TextArea`` with Save/Cancel; Save calls ``on_edit``
    and re-renders the row in place, Cancel restores the original text.
    Delete pushes ``ConfirmationDialog``; a confirmed delete calls
    ``on_delete`` and removes the row -- when the last row goes, the
    modal dismisses ``True`` immediately.

    Escape/backdrop/Close all funnel through the safe-cancel path: a
    mid-edit request first cancels the open editor (transient surface
    first), a second request dismisses the modal with whatever changed
    so far. The quote preview is always read-only.

    Dismiss result is ``True`` if any edit/delete actually committed (the
    owning screen reloads its previews from that), ``False``/``None``
    otherwise.
    """

    BUNDLED_CSS = """
    ConsoleReviewNotesModal {
        align: center middle;
    }

    #console-review-notes-modal {
        width: 76;
        max-height: 90%;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-review-notes-header {
        text-style: bold;
        margin-bottom: 1;
    }

    #console-review-notes-list {
        height: auto;
        max-height: 26;
    }

    .console-review-notes-row {
        height: auto;
        border: round gray;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .console-review-notes-comment {
        height: auto;
        margin-bottom: 1;
    }

    .console-review-notes-edit {
        height: 5;
        margin-bottom: 1;
    }

    .console-review-notes-meta {
        color: $text-muted;
        height: auto;
        margin-bottom: 1;
    }

    .console-review-notes-actions {
        height: 3;
        min-height: 3;
        align-horizontal: right;
    }

    .console-review-notes-actions Button {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
        margin-left: 1;
    }

    #console-review-notes-close-bar {
        height: 3;
        min-height: 3;
        margin-top: 1;
        align-horizontal: right;
    }

    #console-review-notes-close {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    SAFE_MODAL_CONTENT = "#console-review-notes-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        notes: list[dict[str, Any]],
        on_edit: Callable[[str, str], Awaitable[bool]],
        on_delete: Callable[[str], Awaitable[bool]],
    ) -> None:
        super().__init__()
        self._order: list[str] = [str(note["annotation_id"]) for note in notes]
        self._notes: dict[str, dict[str, Any]] = {
            str(note["annotation_id"]): dict(note) for note in notes
        }
        self._on_edit = on_edit
        self._on_delete = on_delete
        self._changed = False
        self._editing_id: str | None = None

    def compose(self) -> ComposeResult:
        """Build the header, the scrolling note list, and the close bar.

        Returns:
            The modal's widgets: one row per note (comment, hidden editor,
            metadata, actions) inside a scroll container, so a heavily
            annotated message cannot outgrow the modal.
        """
        with Vertical(id="console-review-notes-modal"):
            yield Static("Review notes", id="console-review-notes-header")
            with VerticalScroll(id="console-review-notes-list"):
                for annotation_id in self._order:
                    yield from self._compose_row(self._notes[annotation_id])
            with Horizontal(id="console-review-notes-close-bar"):
                yield Button("Close", id=_CLOSE_BUTTON_ID)

    def _compose_row(self, note: dict[str, Any]) -> ComposeResult:
        annotation_id = str(note["annotation_id"])
        with Vertical(
            id=_row_id("row", annotation_id),
            classes="console-review-notes-row",
        ):
            yield Static(
                note.get("comment") or "",
                id=_row_id("comment", annotation_id),
                classes="console-review-notes-comment",
                markup=False,
            )
            edit_area = TextArea(
                note.get("comment") or "",
                id=_row_id("edit", annotation_id),
                classes="console-review-notes-edit",
            )
            edit_area.display = False
            yield edit_area
            yield Static(
                _format_meta(note),
                id=_row_id("meta", annotation_id),
                classes="console-review-notes-meta",
                markup=False,
            )
            with Horizontal(
                id=_row_id("actions", annotation_id),
                classes="console-review-notes-actions",
            ):
                yield Button(
                    "Edit",
                    id=_row_id("edit-button", annotation_id),
                )
                yield Button(
                    "Delete",
                    id=_row_id("delete-button", annotation_id),
                    variant="error",
                )
                save_button = Button(
                    "Save",
                    id=_row_id("save-button", annotation_id),
                    variant="primary",
                )
                save_button.display = False
                yield save_button
                cancel_button = Button(
                    "Cancel",
                    id=_row_id("cancel-button", annotation_id),
                )
                cancel_button.display = False
                yield cancel_button

    # ------------------------------------------------------------------
    # Button routing -- row ids are per-note, so one handler dispatches
    # by id prefix rather than declaring a `@on` per dynamic id.
    # ------------------------------------------------------------------

    @on(Button.Pressed)
    async def _on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        if button_id == _CLOSE_BUTTON_ID:
            await self.request_safe_cancel(source="button")
            return
        if button_id.startswith(_EDIT_BUTTON_PREFIX):
            self._start_edit(button_id[len(_EDIT_BUTTON_PREFIX) :])
            return
        if button_id.startswith(_DELETE_BUTTON_PREFIX):
            self._request_delete(button_id[len(_DELETE_BUTTON_PREFIX) :])
            return
        if button_id.startswith(_SAVE_BUTTON_PREFIX):
            await self._save_edit(button_id[len(_SAVE_BUTTON_PREFIX) :])
            return
        if button_id.startswith(_CANCEL_BUTTON_PREFIX):
            self._cancel_edit(button_id[len(_CANCEL_BUTTON_PREFIX) :])
            return

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def _start_edit(self, annotation_id: str) -> None:
        if annotation_id not in self._notes:
            return
        if self._editing_id is not None and self._editing_id != annotation_id:
            self._cancel_edit(self._editing_id)
        edit_area = self.query_one(_row_selector("edit", annotation_id), TextArea)
        edit_area.text = self._notes[annotation_id].get("comment") or ""
        self._set_row_editing(annotation_id, editing=True)
        self._editing_id = annotation_id
        edit_area.focus()

    async def _save_edit(self, annotation_id: str) -> None:
        note = self._notes.get(annotation_id)
        if note is None:
            return
        edit_area = self.query_one(_row_selector("edit", annotation_id), TextArea)
        new_text = edit_area.text
        # Awaited, not called: the injected callable performs a SQLite write,
        # and the screen runs it off-thread. A synchronous call here would put
        # that write on the UI event loop, where a contended writer waits out
        # the connection's 15s busy timeout with the interface frozen.
        if not await self._on_edit(annotation_id, new_text):
            return
        note["comment"] = new_text
        self.query_one(_row_selector("comment", annotation_id), Static).update(
            new_text
        )
        self._changed = True
        self._set_row_editing(annotation_id, editing=False)
        if self._editing_id == annotation_id:
            self._editing_id = None

    def _cancel_edit(self, annotation_id: str) -> None:
        note = self._notes.get(annotation_id)
        if note is not None:
            try:
                edit_area = self.query_one(
                    _row_selector("edit", annotation_id), TextArea
                )
            except Exception:  # noqa: BLE001 - defensive, row may be gone
                edit_area = None
            if edit_area is not None:
                edit_area.text = note.get("comment") or ""
        self._set_row_editing(annotation_id, editing=False)
        if self._editing_id == annotation_id:
            self._editing_id = None

    def _set_row_editing(self, annotation_id: str, *, editing: bool) -> None:
        try:
            self.query_one(_row_selector("comment", annotation_id), Static).display = (
                not editing
            )
            self.query_one(_row_selector("edit", annotation_id), TextArea).display = (
                editing
            )
            self.query_one(
                _row_selector("edit-button", annotation_id), Button
            ).display = not editing
            self.query_one(
                _row_selector("delete-button", annotation_id), Button
            ).display = not editing
            self.query_one(
                _row_selector("save-button", annotation_id), Button
            ).display = editing
            self.query_one(
                _row_selector("cancel-button", annotation_id), Button
            ).display = editing
        except Exception:  # noqa: BLE001 - defensive, row may already be gone
            pass

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _request_delete(self, annotation_id: str) -> None:
        if annotation_id not in self._notes:
            return
        dialog = ConfirmationDialog(
            title="Delete review note?",
            message="This review note will be removed from the transcript.",
            confirm_label="Delete",
            cancel_label="Keep",
        )
        self.app.push_screen(
            dialog,
            callback=partial(self._apply_delete_confirmation, annotation_id),
        )

    async def _apply_delete_confirmation(
        self, annotation_id: str, confirmed: bool | None
    ) -> None:
        if confirmed is not True:
            return
        if annotation_id not in self._notes:
            return
        # Awaited for the same reason as the edit write above.
        if not await self._on_delete(annotation_id):
            return
        self._changed = True
        del self._notes[annotation_id]
        self._order.remove(annotation_id)
        if self._editing_id == annotation_id:
            self._editing_id = None
        try:
            self.query_one(_row_selector("row", annotation_id), Vertical).remove()
        except Exception:  # noqa: BLE001 - defensive, row may already be gone
            pass
        if not self._notes and self.is_mounted and self.app.screen is self:
            self.dismiss(True)

    # ------------------------------------------------------------------
    # Safe cancel: an open editor is a transient surface -- close it
    # first, only dismiss the modal on the next request.
    # ------------------------------------------------------------------

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._editing_id is not None:
            self._cancel_edit(self._editing_id)
            return
        self.dismiss_safe_once(self._changed)
