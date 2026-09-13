"""Permanent Database Notes work pane for Edit, Preview, Info, and tasks."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Static

from .library_notes_canvas import LibraryNotesCanvas

#: Modes whose child canvas paints its own header. The work pane adds no
#: authority line of its own for these -- stacking one produced four headers
#: over the same sentence on Add from files (task-32063).
_CHILD_OWNED_HEADER_MODES = frozenset({"import", "lasting_add", "lasting_roots"})


class LibraryNoteWorkPane(LibraryNotesCanvas):
    """Render non-list Notes content while the concrete list stays mounted."""

    class EditorReady(Message):
        """Notify the screen that mounted editor children are safe to arm."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["authority_id"] = "library-note-work-authority"
        super().__init__(**kwargs)
        self.remove_class("library-adaptive-reader-items")

    def _authority_prefix(self) -> str:
        """Name storage when compact Work hides the list's authority.

        task-32063: both panes painted "Library notes · Library database …",
        so a wide Notes session showed the same authority sentence twice.
        """
        return super()._authority_prefix() if self.compact else ""

    def compose(self) -> ComposeResult:
        """Compose the active task or a stable no-selection work surface."""
        if self.mode == "list":
            yield Static(
                "Select a note to edit it here.",
                id="library-note-work-empty",
                classes="destination-purpose",
                markup=False,
            )
            return
        if self.mode in _CHILD_OWNED_HEADER_MODES:
            composed = super().compose()
            next(composed, None)  # drop this pane's authority line
            yield from composed
            return
        yield from super().compose()

    def _after_recompose(self) -> None:
        """Apply presentation state, then announce a mounted editor subtree."""
        super()._after_recompose()
        if (
            self.mode == "editor"
            and self.query("#library-note-title")
            and self.query("#library-note-body")
        ):
            self.post_message(self.EditorReady())
