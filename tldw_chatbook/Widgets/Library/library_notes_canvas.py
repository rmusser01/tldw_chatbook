"""Library notes canvas: list mode (rows + filter + sort), editor mode, and
create mode (Blank note + template rows)."""

from __future__ import annotations

from collections.abc import Mapping
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
            in-canvas note editor for ``editor_state``; ``"create"`` renders
            the Blank note / template picker reached from the rail's
            Create > New note row.
        editor_state: The note to render in editor mode. Required when
            ``mode == "editor"``.
        preview: Reserved for the Markdown preview toggle (a later task);
            accepted now so callers can start passing it, but it has no
            effect on rendering yet.
        conflict: When ``True`` (editor mode only), renders the save
            conflict banner -- a quiet explanatory line plus Overwrite/
            Reload actions -- in addition to the normal editor fields.
            ``editor_state`` must already reflect the user's kept text
            (never the server's stale detail) when this is set.
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
        conflict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.editor_state = editor_state
        self.preview = preview
        self.conflict = conflict
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        if self.mode == "create":
            yield from self._compose_create()
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
        if self.conflict:
            yield Static(
                "This note changed elsewhere — Overwrite saves your text; "
                "Reload discards it.",
                id="library-note-conflict-copy",
                classes="destination-purpose",
                markup=False,
            )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            if self.conflict:
                yield Button(
                    "Overwrite",
                    id="library-note-conflict-overwrite",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Reload",
                    id="library-note-conflict-reload",
                    classes="library-canvas-action",
                    compact=True,
                )
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

    def _compose_create(self) -> ComposeResult:
        """Render the notes canvas in create mode: Blank note + template rows.

        Reached via the rail's Create > New note row (canvas kind
        ``"notes-create"``). The Blank note action and every template row
        are stacked, full-width, compact buttons styled like the list
        view's note rows (``library-notes-create-row`` copies the
        ``library-notes-row`` look) so the create view reads as more note
        rows rather than a distinct toolbar -- a *different* class on
        purpose: reusing ``library-notes-row`` itself would also match the
        list view's ``.library-notes-row`` press handler (selecting a note
        row and opening the editor for it), double-dispatching alongside
        this view's own create handlers on every press. Templates come
        from ``NOTE_TEMPLATES`` (imported locally to match the existing
        deferred-import convention used elsewhere for this module-level
        dict), sorted by key for a stable order; each row's
        ``template_key`` attribute (mirroring ``note_id`` on list rows) is
        read by the screen's press handler to resolve the template's
        fields via ``_library_note_template_fields`` -- this widget only
        needs the key and a human label, never the raw title/content.
        """
        yield Static(
            "New note",
            id="library-notes-create-header",
            classes="destination-section",
            markup=False,
        )
        yield Button(
            "Blank note",
            id="library-notes-create-blank",
            classes="library-notes-create-row",
            compact=True,
        )
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES
        from tldw_chatbook.Widgets.Note_Widgets.notes_workbench_panes import (
            template_display_label,
        )

        for index, (key, template) in enumerate(sorted(NOTE_TEMPLATES.items())):
            label = (
                template_display_label(key, template)
                if isinstance(template, Mapping)
                else str(key).replace("_", " ")
            )
            button = Button(
                label,
                id=f"library-notes-template-{index}",
                classes="library-notes-create-row library-notes-template-row",
                compact=True,
            )
            button.template_key = key
            yield button
