"""Library notes canvas: list mode (rows + filter + sort), editor mode, and
create mode (Blank note + template rows)."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Markdown, Static, TextArea

from tldw_chatbook.Library.library_notes_state import (
    LibraryNoteEditorState,
    LibraryNotesListState,
    build_library_note_template_rows,
)
from tldw_chatbook.Library.library_notes_sync_state import (
    LibraryNotesSyncState,
    auto_sync_label,
    sync_conflict_label,
    sync_direction_label,
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
            Create > New note row; ``"sync"`` renders the in-canvas notes
            sync panel for ``sync_state``.
        editor_state: The note to render in editor mode. Required when
            ``mode == "editor"``.
        sync_state: The sync panel's display state. Required when
            ``mode == "sync"``.
        preview: When ``True`` (editor mode only), renders ``editor_state``'s
            content as read-only ``Markdown`` in place of the editable
            ``TextArea`` -- the screen is responsible for threading the
            live (possibly unsaved) body text into ``editor_state.content``
            before toggling this on, so switching to preview never drops
            in-progress edits. The Preview/Edit action button's label
            reflects this flag.
        conflict: When ``True`` (editor mode only), renders the save
            conflict banner -- a quiet explanatory line plus Overwrite/
            Reload actions -- in addition to the normal editor fields.
            ``editor_state`` must already reflect the user's kept text
            (never the server's stale detail) when this is set.
        confirming_delete: When ``True`` (editor mode only, and only when
            ``conflict`` is not also set), renders the inline delete
            confirmation affordance -- a quiet explanatory line plus
            Delete/Cancel actions -- in place of the normal action row.
            Mirrors ``LibraryMediaViewer.confirming_delete``.
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
        confirming_delete: bool = False,
        sync_state: LibraryNotesSyncState | None = None,
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
        self.confirming_delete = confirming_delete
        self.sync_state = sync_state
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        if self.mode == "create":
            yield from self._compose_create()
            return
        if self.mode == "sync":
            yield from self._compose_sync()
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
        select_mode = list_state.select_mode
        # Gate/label off the RENDERED rows, not any total-count field -- only
        # rendered rows are selectable, matching the media/conversations
        # canvases' ``len(rows)`` convention.
        rendered_count = len(list_state.rows)
        # One horizontal ds-toolbar row for sort/Sync/Import note/Export…/
        # Select (2026-07 UAT: the previous bare stacked Buttons rendered as
        # an overlapped vertical pile eating into the first list row). Safe
        # here because every child is a fixed-width compact Button -- the
        # known non-rendering failure mode for this canvas family is only
        # a Horizontal mixing a 1fr sibling with fixed-width children,
        # exactly the ds-toolbar shape `_compose_editor` already proves out.
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
                id="library-notes-sort", classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Sync", id="library-notes-sync-open",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Import note", id="library-notes-import",
                classes="library-canvas-action", compact=True,
            )
            export_btn = Button(
                "Export…", id="library-notes-export",
                classes="library-canvas-action", compact=True,
            )
            export_btn.display = not select_mode
            yield export_btn
            select_btn = Button(
                "Done" if select_mode else "Select",
                id="library-notes-select-toggle",
                classes="library-canvas-action", compact=True,
            )
            # Disable only when nothing to select AND not already in select mode
            # -- in select mode "Done" must stay pressable so the user can exit
            # even if the rows dropped to zero (e.g. a background refresh).
            select_btn.disabled = rendered_count == 0 and not select_mode
            yield select_btn
        if select_mode:
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                yield Static(f"{list_state.selected_count} selected",
                             id="library-notes-selected-count", markup=False)
                yield Button(f"Select all {rendered_count} shown",
                             id="library-notes-select-all",
                             classes="library-canvas-action", compact=True)
                yield Button("Clear", id="library-notes-select-clear",
                             classes="library-canvas-action", compact=True)
                export_selected = Button("Export selected",
                                         id="library-notes-export-selected",
                                         classes="library-canvas-action", compact=True)
                export_selected.disabled = list_state.selected_count == 0
                yield export_selected
        if list_state.status_copy:
            yield Static(list_state.status_copy, id="library-notes-status", markup=False)
        if not list_state.rows:
            yield Static(list_state.empty_copy, id="library-notes-empty", markup=False)
            return
        with Vertical(id="library-notes-list"):
            for index, row in enumerate(list_state.rows):
                # Button labels are parsed as Rich markup: escape the
                # user-supplied title so "[draft] Q3 plan [wip]" renders
                # verbatim instead of eating bracketed segments as tags
                # (or crashing on an unmatched closing tag) -- the same
                # fix class as the escaped search-history Button labels.
                title = escape_markup(row.title)
                if select_mode:
                    # Notes rows had no marker at all before select mode
                    # existed -- normal mode keeps that markerless label
                    # (no ``▸``, unlike the media/conversations rows). The 2-col
                    # glyph shifts line 1, so indent the age line by 2 to keep it
                    # aligned under the title rather than under the checkbox.
                    glyph = "☑ " if row.checked else "☐ "
                    label = f"{glyph}{title}\n  {row.age_label}" if row.age_label else f"{glyph}{title}"
                else:
                    label = f"{title}\n{row.age_label}" if row.age_label else title
                button = Button(
                    label,
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
        if self.preview:
            yield Markdown(
                editor_state.content,
                id="library-note-preview-body",
            )
        else:
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
        confirming_delete = self.confirming_delete and not self.conflict
        if confirming_delete:
            # A single full-width Static above the toolbar, not inside it --
            # mixing a Static with the toolbar's Buttons is the known
            # non-rendering failure mode called out on the media viewer's
            # ``compose`` (the pattern this mirrors).
            yield Static(
                "Delete this note? This cannot be undone from Library.",
                id="library-note-delete-confirm-copy",
                markup=False,
            )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            if confirming_delete:
                yield Button(
                    "Delete",
                    id="library-note-delete-confirm",
                    classes="library-canvas-action library-media-action-danger",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-note-delete-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
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
                    "Edit" if self.preview else "Preview",
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

        # The pure builder excludes the "blank" template (it duplicates the
        # Blank note action above) and pre-resolves each template's title so
        # the row's muted secondary line shows the exact title the created
        # note will get (date placeholders already substituted).
        rows = build_library_note_template_rows(NOTE_TEMPLATES)
        yield Static(
            "From a template",
            id="library-notes-template-section",
            classes="destination-section",
            markup=False,
        )
        for index, row in enumerate(rows):
            label = (
                f"{row.label}\n{row.resolved_title}"
                if row.resolved_title
                else row.label
            )
            button = Button(
                label,
                id=f"library-notes-template-{index}",
                classes="library-notes-create-row library-notes-template-row",
                compact=True,
            )
            button.template_key = row.template_key
            yield button

    def _compose_sync(self) -> ComposeResult:
        """Render the notes sync panel: folder, direction, conflicts, activity.

        Every control here is a plain, stacked, full-width Button/Input/
        Static -- the render-safe shape already proven by list/editor/create
        mode in this canvas. Notably absent: ``Select`` (the retired
        standalone Notes screen's Direction/Conflict dropdowns) and ``Switch``
        (its auto-sync toggle) -- neither renders reliably in this canvas,
        so both become cycling/toggle Buttons instead, matching the
        pattern the media type filter and notes sort control already use.
        """
        sync_state = self.sync_state
        if sync_state is None:
            return
        yield Button(
            "‹ Back to notes",
            id="library-notes-sync-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static(
            "Notes sync",
            id="library-notes-sync-header",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            "Mirror notes between a folder on disk and the Library.",
            id="library-notes-sync-purpose",
            markup=False,
        )
        yield Static("folder", id="library-notes-sync-folder-label", markup=False)
        yield Input(
            value=sync_state.folder,
            placeholder="Folder to sync…",
            id="library-notes-sync-folder",
        )
        yield Button(
            "Browse…",
            id="library-notes-sync-browse",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            f"direction: {sync_direction_label(sync_state.direction)} ▸",
            id="library-notes-sync-direction",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            f"conflicts: {sync_conflict_label(sync_state.conflict)} ▸",
            id="library-notes-sync-conflict",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            auto_sync_label(sync_state.auto_sync),
            id="library-notes-sync-auto",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            "Syncing…" if sync_state.running else "Sync now",
            id="library-notes-sync-run",
            classes="library-canvas-action",
            compact=True,
            disabled=sync_state.running,
        )
        # ``sync_status_line``'s own tested contract is that a failed status
        # always starts with the literal prefix "failed" -- safe to key the
        # error styling off that prefix here.
        yield Static(
            sync_state.status_line,
            id="library-notes-sync-status",
            classes=(
                "library-notes-sync-status-failed"
                if sync_state.status_line.startswith("failed")
                else ""
            ),
            markup=False,
        )
        yield Static(
            "\n".join(sync_state.activity_lines),
            id="library-notes-sync-activity",
            markup=False,
        )
