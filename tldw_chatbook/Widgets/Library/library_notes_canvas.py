"""Library notes canvas: list mode (rows + filter + sort) and editor mode."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_notes_state import (
    LibraryNoteEditorState,
    LibraryNotesListState,
)

_SORT_LABELS = {"newest": "Newest", "oldest": "Oldest", "title": "Title"}


class LibraryNotesCanvas(Vertical):
    """Render the Library notes canvas: the list view, or the note editor.

    Attributes:
        list_state: List-view display state (header, filter, sort, rows).
            Only used when ``mode == "list"``.
        sort_mode: Current notes sort mode key (``"newest"``/``"oldest"``/
            ``"title"``), used to label the sort control.
        filter_value: Current notes filter text, prefilled into the filter
            ``Input``.
        mode: ``"list"`` renders the notes list; ``"editor"`` renders the
            in-canvas note editor for ``editor_state``.
        editor_state: The note to render in editor mode. Required when
            ``mode == "editor"``.
        preview: Reserved for the Markdown preview toggle (a later task);
            accepted now so callers can start passing it, but it has no
            effect on rendering yet.
    """

    def __init__(
        self,
        list_state: LibraryNotesListState | None = None,
        *,
        sort_mode: str = "newest",
        filter_value: str = "",
        mode: str = "list",
        editor_state: LibraryNoteEditorState | None = None,
        preview: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.editor_state = editor_state
        self.preview = preview
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        yield from self._compose_list()

    def _compose_list(self) -> ComposeResult:
        list_state = self.list_state
        if list_state is None:
            return
        yield Static(list_state.header_copy, id="library-notes-header",
                     classes="destination-section", markup=False)
        yield Input(
            placeholder="Filter notes… (Enter)",
            id="library-notes-filter",
            value=self.filter_value,
        )
        yield Button(
            f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
            id="library-notes-sort", classes="library-canvas-action", compact=True,
        )
        if list_state.status_copy:
            yield Static(list_state.status_copy, id="library-notes-status", markup=False)
        if not list_state.rows:
            yield Static(list_state.empty_copy, id="library-notes-empty", markup=False)
            return
        with Vertical(id="library-notes-list"):
            for index, row in enumerate(list_state.rows):
                button = Button(
                    f"{row.title}\n{row.age_label}" if row.age_label else row.title,
                    id=f"library-notes-row-{index}",
                    classes="library-notes-row", compact=True,
                )
                button.note_id = row.note_id
                yield button

    def _compose_editor(self) -> ComposeResult:
        """Render the note editor: Back, title, body, keywords, meta, actions.

        Stacked full-width widgets (mirroring ``LibraryMediaViewer.compose``)
        plus a single plain ``ds-toolbar`` action row -- the render-safe
        shape already proven by the media viewer canvas. All fields render
        with their current values; the action buttons are wired with ids
        only here (Save/Preview/Use in Console/Export/Copy/Delete stay
        inert until later tasks add their handlers).
        """
        editor_state = self.editor_state
        if editor_state is None:
            return
        yield Button(
            "‹ Back to list",
            id="library-note-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Input(
            value=editor_state.title,
            id="library-note-title",
        )
        yield TextArea(
            editor_state.content,
            id="library-note-body",
        )
        yield Input(
            value=editor_state.keywords_text,
            placeholder="Keywords (comma-separated)",
            id="library-note-keywords",
        )
        yield Static(
            editor_state.meta_line,
            id="library-note-meta",
            markup=False,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "Save",
                id="library-note-save",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Preview",
                id="library-note-preview",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Use in Console",
                id="library-note-use-in-console",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Export .md",
                id="library-note-export-md",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Export .txt",
                id="library-note-export-txt",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Copy",
                id="library-note-copy",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Delete",
                id="library-note-delete",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
