"""Library notes canvas: list mode (rows + filter + sort), editor mode, and
create mode (Blank note + template rows)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static, TextArea

from tldw_chatbook.Library.library_notes_state import (
    LibraryNoteSessionSnapshot,
    LibraryNotesListState,
    build_library_note_template_rows,
    ellipsize_note_title_cells,
)
from tldw_chatbook.Library.library_note_import_state import LibraryNoteImportSnapshot
from tldw_chatbook.Library.library_notes_sync_state import (
    LibraryNotesSyncState,
    auto_sync_label,
    sync_conflict_label,
    sync_direction_label,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_disabled_action_label,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

_SORT_LABELS = {"newest": "Newest", "oldest": "Oldest", "title": "Title"}
_COMPACT_SYNC_DIRECTION_LABELS = {
    "bidirectional": "Both",
    "disk_to_db": "Disk → Lib",
    "db_to_disk": "Lib → Disk",
}
_COMPACT_SYNC_CONFLICT_LABELS = {
    "newer_wins": "Newest",
    "disk_wins": "Disk",
    "db_wins": "Library",
}


@dataclass(frozen=True)
class LibraryNotePresentationState:
    """Immutable presentation input for one mounted Database Note canvas.

    The coordinator snapshot is the only source of draft text. Everything
    else describes how that draft is presented; applying this state must not
    perform persistence, navigation, or draft mutation.
    """

    snapshot: LibraryNoteSessionSnapshot
    metadata_line: str
    status_line: str
    region: Literal["editor", "context"] = "editor"
    presentation: Literal["edit", "preview"] = "edit"
    compact: bool = False
    validation: bool = False
    conflict: bool = False
    conflict_running: bool = False
    confirming_delete: bool = False
    destructive_running: bool = False
    discard_new_note: bool = False
    transfer_status: str = ""
    transfer_running: bool = False


class LibraryNotesCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library notes canvas: the list view, or the note editor.

    Attributes:
        list_state: List-view display state (header, filter, sort, rows).
            Only used when ``mode == "list"``.
        sort_mode: Current notes sort mode key (``"newest"``/``"oldest"``/
            ``"title"``), used to label the sort control.
        filter_value: Current notes filter text, prefilled into the filter
            ``Input``.
        mode: ``"list"`` renders the notes list; ``"loading"`` renders the
            editor loading/retry receipt; ``"editor"`` renders the in-canvas
            note editor for ``presentation_state``; ``"create"`` renders the
            Blank note / template picker reached from the rail's Create > New
            note row; ``"sync"`` renders the in-canvas notes sync panel for
            ``sync_panel_state``.
        presentation_state: Canonical snapshot plus presentation-only state.
            Required when ``mode == "editor"``.
        sync_panel_state: The sync panel's display state. Required when
            ``mode == "sync"``. This deliberately does not use the name
            ``sync_state`` because that name belongs to the mounted canvas's
            targeted update hook.
        import_snapshot: Reviewed one-time import presentation state. Required
            when ``mode == "import"``.
        import_receipt_available: Whether the latest same-session receipt can
            reopen from list mode.
        title_placeholder_only: When ``True`` (editor mode only), the title
            ``Input`` renders empty with an "Untitled" placeholder instead
            of a literal editable "Untitled" value -- LIB-14's fix for a
            just-created, never-touched "Blank note": with a literal
            ``value="Untitled"``, typing right after opening the note
            landed after the existing text instead of replacing it (e.g.
            "UntitledAtlas follow-ups"). An empty value with a placeholder
            sidesteps the ambiguity entirely -- there is no text to land
            after. The screen sets this only while the open note is still
            its own pristine "Blank note" (the same condition that also
            arms it for GC-on-exit -- see
            ``_library_note_pending_blank_gc_id``); it never applies to a
            note whose title happens to equal the word "Untitled" by the
            user's own choice.
        compact: Whether compact, 60-column-safe action labels are active.
        create_running: Whether a Create request is currently in flight.
        create_status: Visible Create completion or recovery status.
    """

    def __init__(
        self,
        list_state: LibraryNotesListState | None = None,
        *,
        sort_mode: str = "newest",
        filter_value: str = "",
        mode: str = "list",
        presentation_state: LibraryNotePresentationState | None = None,
        sync_panel_state: LibraryNotesSyncState | None = None,
        import_snapshot: LibraryNoteImportSnapshot | None = None,
        import_receipt_available: bool = False,
        tree_projection: LibraryNotesTreeProjection | None = None,
        tree_selected_placement_id: str = "",
        tree_deleted_folder_available: bool = False,
        title_placeholder_only: bool = False,
        compact: bool = False,
        create_running: bool = False,
        create_status: str = "",
        load_state: str = "loading",
        load_message: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize one list, editor, create, sync, or import canvas.

        Args:
            list_state: List-mode rows, counts, selection, and empty-state copy.
            sort_mode: Active list sort key.
            filter_value: Text prefilled into the list filter.
            mode: Canvas surface to compose: list, loading, editor, create,
                or sync.
            presentation_state: Canonical editor snapshot and UI-only flags.
            sync_panel_state: Display state for the sync surface.
            import_snapshot: Display state for the one-time import surface.
            tree_projection: Placement-aware folder rows for list mode.
            tree_selected_placement_id: Context row for folder actions.
            tree_deleted_folder_available: Whether Undo folder removal is available.
            title_placeholder_only: Render an empty title with an Untitled
                placeholder for a pristine newly-created note.
            compact: Whether 60-column-safe controls and labels are active.
            create_running: Whether note creation is in progress.
            create_status: Visible creation completion or recovery status.
            load_state: Editor-load state (``"loading"`` or ``"failed"``).
            load_message: Recovery copy shown after an editor-load failure.
            **kwargs: Additional keyword arguments forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.presentation_state = presentation_state
        self.sync_panel_state = sync_panel_state
        self.import_snapshot = import_snapshot
        self.import_receipt_available = import_receipt_available
        self.tree_projection = tree_projection
        self.tree_selected_placement_id = tree_selected_placement_id
        self.tree_deleted_folder_available = tree_deleted_folder_available
        self.title_placeholder_only = title_placeholder_only
        self.compact = compact
        self.create_running = create_running
        self.create_status = create_status
        self.load_state = load_state
        self.load_message = load_message
        self.styles.width = "1fr"
        self.styles.min_width = 40
        self.add_class(f"library-notes-mode-{mode}")

    def _after_recompose(self) -> None:
        """Re-run the post-compose wiring ``on_mount`` does.

        ``on_mount`` fires once, when the canvas itself mounts -- a
        ``refresh(recompose=True)`` remounts this widget's CHILDREN without
        re-firing it, so ``sync_state``'s recompose would otherwise leave the
        editor's stable subtree (populated by ``apply_session_state``) and the
        compact label rewrites showing compose-time defaults.

        Implemented as ``PostRecomposeCallback``'s hook rather than a
        ``recompose()`` override so it runs BEFORE any queued follow-up
        (task-15457 review round 1, minor 5): with the override form, a
        ``then=`` that focused a control saw its pre-compact label.
        """
        self._apply_post_compose_state()

    def compose(self) -> ComposeResult:
        yield Static(
            self._authority_copy(),
            id="library-notes-authority",
            markup=False,
        )
        if self.mode == "loading":
            yield from self._compose_loading()
            return
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        if self.mode == "create":
            yield from self._compose_create()
            return
        if self.mode == "sync":
            yield from self._compose_sync()
            return
        if self.mode == "import":
            if self.import_snapshot is not None:
                yield LibraryNoteImportCanvas(
                    self.import_snapshot,
                    id="library-note-import-canvas",
                )
            yield Button(
                "Back to Notes",
                id="library-notes-import-back",
                classes="library-canvas-action",
                compact=True,
            )
            return
        yield from self._compose_list()

    def _authority_copy(self) -> str:
        """Describe Library storage, current status, and the next action."""
        prefix = "Library notes · Library database"
        if self.mode == "loading":
            if self.load_state == "failed":
                status = self.load_message or "Could not load note."
                return f"{prefix} · {status} · Next: Retry loading."
            return f"{prefix} · Loading note… · Next: Wait for loading to finish."
        if self.mode == "editor":
            state = self.presentation_state
            if state is None:
                return f"{prefix} · Editor unavailable · Next: Back to notes."
            status = state.status_line or "Ready"
            transfer = (
                f" · {state.transfer_status}"
                if state.transfer_status and state.transfer_status != status
                else ""
            )
            if state.conflict:
                next_action = "Resolve the conflict or reload the note."
            elif state.snapshot.saving:
                next_action = "Wait for saving to finish."
            elif state.transfer_running:
                next_action = "Wait for export to finish."
            elif "failed" in f"{status} {state.transfer_status}".lower():
                next_action = "Review the error, then keep editing."
            else:
                next_action = "Keep editing; changes save automatically."
            return f"{prefix} · {status}{transfer} · Next: {next_action}"
        if self.mode == "create":
            status = self.create_status or (
                "Creating note…" if self.create_running else "Ready"
            )
            next_action = (
                "Wait for creation to finish."
                if self.create_running
                else "Choose Blank note or a template."
            )
            return f"{prefix} · {status} · Next: {next_action}"
        if self.mode == "sync":
            state = self.sync_panel_state
            status = "Sync unavailable" if state is None else f"Sync {state.status_line}"
            if state is None or state.status_line.startswith("failed"):
                next_action = "Review the error, then Sync now."
            elif state.running:
                next_action = "Wait for sync to finish."
            else:
                next_action = "Choose a folder, then Sync now."
            return f"{prefix} · {status} · Next: {next_action}"
        if self.mode == "import":
            state = self.import_snapshot
            status = "Import unavailable" if state is None else state.status_line
            return (
                f"{prefix} · Import once · {status} · "
                "Next: Review the import workflow."
            )
        state = self.list_state
        status = state.operation_status if state is not None else ""
        running = state is not None and state.operation_running
        status = status or ("Updating notes…" if running else "Ready")
        next_action = (
            "Wait for the running notes operation to finish."
            if running
            else "Create, Sync, or Import."
        )
        return f"{prefix} · {status} · Next: {next_action}"

    def sync_state(
        self,
        *,
        list_state: LibraryNotesListState | None,
        sort_mode: str,
        filter_value: str,
        mode: str,
        presentation_state: LibraryNotePresentationState | None,
        sync_panel_state: LibraryNotesSyncState | None,
        import_snapshot: LibraryNoteImportSnapshot | None = None,
        import_receipt_available: bool = False,
        tree_projection: LibraryNotesTreeProjection | None,
        tree_selected_placement_id: str,
        tree_deleted_folder_available: bool,
        title_placeholder_only: bool,
        compact: bool,
        create_running: bool,
        create_status: str,
        load_state: str,
        load_message: str,
    ) -> None:
        """Apply a complete screen-owned snapshot within this canvas only.

        The method intentionally replaces every compose input before asking
        Textual to rebuild this widget's children. Keeping the update complete
        prevents list/editor/sync conditionals from retaining values from the
        previous surface while the Library shell, rail, and footer retain
        identity.

        Args:
            list_state: Notes list snapshot, or ``None`` outside list mode.
            sort_mode: Active Notes sort identifier.
            filter_value: Current Notes filter text.
            mode: Canvas surface to render.
            presentation_state: Note editor/create presentation snapshot.
            sync_panel_state: Notes folder-sync panel snapshot.
            import_snapshot: Reviewed one-time import presentation snapshot.
            import_receipt_available: Whether the latest same-session receipt can reopen.
            tree_projection: Placement-aware folder rows for list mode.
            tree_selected_placement_id: Context row for folder actions.
            tree_deleted_folder_available: Whether Undo folder removal is available.
            title_placeholder_only: Whether the title is placeholder-only.
            compact: Whether compact editor controls are enabled.
            create_running: Whether note creation is in progress.
            create_status: Current note-creation status copy.
            load_state: Current note-loading state identifier.
            load_message: Current note-loading status or error copy.
        """
        previous_mode = self.mode
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.presentation_state = presentation_state
        self.sync_panel_state = sync_panel_state
        self.import_snapshot = import_snapshot
        self.import_receipt_available = import_receipt_available
        self.tree_projection = tree_projection
        self.tree_selected_placement_id = tree_selected_placement_id
        self.tree_deleted_folder_available = tree_deleted_folder_available
        self.title_placeholder_only = title_placeholder_only
        self.compact = compact
        self.create_running = create_running
        self.create_status = create_status
        self.load_state = load_state
        self.load_message = load_message
        if previous_mode != mode:
            self.remove_class(f"library-notes-mode-{previous_mode}")
            self.add_class(f"library-notes-mode-{mode}")
        self.refresh(recompose=True)

    def _compose_loading(self) -> ComposeResult:
        """Render the existing note-loading/retry surface inside the canvas."""
        with Vertical(id="library-note-load-state"):
            with Horizontal(id="library-note-load-heading"):
                yield Button(
                    "‹ Notes",
                    id="library-note-back",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Static(
                    "Edit note",
                    id="library-note-loading-title",
                    markup=False,
                )
            load_copy = (
                self.load_message if self.load_state == "failed" else "Loading note…"
            )
            yield Static(
                load_copy,
                id="library-note-loading",
                classes="destination-purpose",
                markup=False,
            )
            with Vertical(id="library-note-loading-viewport"):
                if self.load_state == "failed":
                    yield Button(
                        "Retry",
                        id="library-note-load-retry",
                        classes="library-canvas-action",
                        compact=True,
                    )

    def _compose_list(self) -> ComposeResult:
        list_state = self.list_state
        if list_state is None:
            return
        yield Static(
            list_state.header_copy,
            id="library-notes-header",
            classes="destination-section",
            markup=False,
        )
        # Database mode persists notes in the Library; Files edits a folder
        # directly, while Sync mirrors a folder into this database.
        database_purpose = Static(
            "These notes live in the Library's own database — for notes "
            "that live in a folder on disk, switch to Files, or use Sync "
            "to mirror one in.",
            id="library-notes-database-purpose",
            markup=False,
        )
        database_purpose.display = not self.compact
        yield database_purpose
        with Horizontal(id="library-notes-filter-row"):
            yield Static("Filter", id="library-notes-filter-label", markup=False)
            yield Input(
                placeholder="Filter notes… (Enter)",
                id="library-notes-filter",
                value=self.filter_value,
            )
        select_mode = list_state.select_mode
        # Gate/label off the RENDERED rows, not any total-count field -- only
        # rendered rows are selectable, matching the media/conversations
        # canvases' ``len(rows)`` convention.
        rendered_note_ids = (
            {
                row.note_id
                for row in self.tree_projection.rows
                if row.kind == "note" and row.note_id
            }
            if self.tree_projection is not None
            else {row.note_id for row in list_state.rows}
        )
        rendered_count = len(rendered_note_ids)
        if select_mode:
            action_row = Horizontal(
                id="library-notes-selection-actions", classes="ds-toolbar"
            )
            action_row.styles.height = "auto"
            with action_row:
                # task-2853 review round 2: the SAME unbounded-width defect
                # proved live in the Media canvas's identical counter (see
                # library_media_canvas.py's compose()) also affects this
                # canvas's counter -- fixed generally via the shared
                # ``library-toolbar-count`` class (css/components/
                # _agentic_terminal.tcss's ``width: auto``) rather than a
                # per-canvas one-off.
                yield Static(
                    f"{list_state.selected_count} selected",
                    id="library-notes-selected-count",
                    classes="library-toolbar-count",
                    markup=False,
                )
                yield Button(
                    "Done",
                    id="library-notes-select-toggle",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    (
                        f"All {rendered_count}"
                        if self.compact
                        else f"Select all {rendered_count} shown"
                    ),
                    id="library-notes-select-all",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Clear",
                    id="library-notes-select-clear",
                    classes="library-canvas-action",
                    compact=True,
                )
                export_base = "Export" if self.compact else "Export selected"
                export_disabled = list_state.selected_count == 0
                export_selected = Button(
                    # task-4023 AC#1 (RC-07): "○" disabled marker; base
                    # label stashed for `_apply_library_row_toggle`'s
                    # in-place patch (compact and full spellings differ,
                    # so the patcher must not hard-code either).
                    library_disabled_action_label(export_base, export_disabled),
                    id="library-notes-export-selected",
                    classes="library-canvas-action",
                    compact=True,
                )
                export_selected._library_disabled_marker_base = export_base
                export_selected.disabled = export_disabled
                # F-018: a disabled action says why.
                export_selected.tooltip = (
                    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
                    if export_selected.disabled
                    else LIBRARY_EXPORT_SELECTED_TOOLTIP
                )
                yield export_selected
            yield Static(
                f"{list_state.selected_count} selected",
                id="library-notes-selection-status",
                markup=False,
            )
        else:
            browse_actions = Horizontal(
                id="library-notes-browse-actions", classes="ds-toolbar"
            )
            browse_actions.styles.height = "auto"
            browse_actions.display = not list_state.sort_choices_visible
            with browse_actions:
                # task-4023 AC#1 (RC-07): every disabled toolbar action
                # carries the non-colour "○" marker plus an F-018 reason.
                running = list_state.operation_running
                running_tooltip = "Wait for the running notes operation to finish."
                yield Button(
                    library_disabled_action_label("New", running),
                    id="library-notes-new",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=running,
                    tooltip=running_tooltip if running else None,
                )
                sort_base = f"Sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')}"
                yield Button(
                    library_disabled_action_label(sort_base, running),
                    id="library-notes-sort",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=running,
                    tooltip=running_tooltip if running else None,
                )
                select_disabled = rendered_count == 0 or running
                yield Button(
                    library_disabled_action_label("Select", select_disabled),
                    id="library-notes-select-toggle",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=select_disabled,
                    tooltip=(
                        (
                            running_tooltip
                            if running
                            else LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP
                        )
                        if select_disabled
                        else None
                    ),
                )
            if list_state.sort_choices_visible:
                # task-14902: composed through the ONE shared strip builder
                # (this control is the pattern's precedent; the media type /
                # prompts sort / skills sort / export quality strips share
                # the same mechanism).
                yield from compose_library_choice_strip(
                    strip_id="library-notes-sort-choices",
                    choice_class="library-notes-sort-choice",
                    options=tuple(
                        (f"library-notes-sort-{mode}", mode, label)
                        for mode, label in _SORT_LABELS.items()
                    ),
                    active_value=self.sort_mode,
                )
            import_phase = (
                self.import_snapshot.phase if self.import_snapshot is not None else ""
            )
            import_label = (
                "View import"
                if import_phase == "importing"
                else "Continue import"
                if import_phase in {"destination", "checking", "review"}
                or (
                    import_phase == "select"
                    and bool(self.import_snapshot and self.import_snapshot.selected_names)
                )
                else "Import"
            )
            transfer_actions = Horizontal(
                id="library-notes-transfer-actions", classes="ds-toolbar"
            )
            transfer_actions.styles.height = "auto"
            with transfer_actions:
                for label, button_id in (
                    ("Sync", "library-notes-sync-open"),
                    (import_label, "library-notes-import"),
                    ("Export", "library-notes-export"),
                ):
                    view_import = (
                        button_id == "library-notes-import"
                        and import_phase == "importing"
                    )
                    disabled = list_state.operation_running and not view_import
                    yield Button(
                        library_disabled_action_label(label, disabled),
                        id=button_id,
                        classes="library-canvas-action",
                        compact=True,
                        disabled=disabled,
                        tooltip=running_tooltip if disabled else None,
                    )
                if self.import_receipt_available:
                    yield Button(
                        "Last import",
                        id="library-notes-import-receipt",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=list_state.operation_running,
                    )
            if self.tree_projection is not None:
                yield from self._compose_tree_actions(
                    operation_running=list_state.operation_running
                )
        status_row = Horizontal(id="library-notes-status-row")
        status_row.styles.height = "auto"
        status_row.display = not select_mode
        with status_row:
            status = Static(
                list_state.status_copy,
                id="library-notes-status",
                markup=False,
            )
            status.display = not select_mode
            yield status
            if self.filter_value:
                yield Button(
                    "Clear filter",
                    id="library-notes-filter-clear",
                    classes="library-canvas-action",
                    compact=True,
                )
        receipt = list_state.delete_receipt
        if receipt is not None:
            title = ellipsize_note_title_cells(
                receipt.title or "Untitled", 18 if self.compact else 42
            )
            receipt_row = Horizontal(
                id="library-notes-delete-receipt", classes="ds-toolbar"
            )
            receipt_row.styles.height = "auto"
            with receipt_row:
                yield Static(
                    f"✓ deleted · {title}",
                    id="library-notes-delete-receipt-copy",
                    classes="library-toolbar-count",
                    markup=False,
                )
                yield Button(
                    "Undo",
                    id="library-notes-delete-undo",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=list_state.operation_running,
                )
                yield Button(
                    "Dismiss",
                    id="library-notes-delete-receipt-dismiss",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=list_state.operation_running,
                )
        if self.tree_projection is not None:
            yield from self._compose_tree_rows(list_state)
            return
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
                    label_rest = (
                        f"{title}\n  {row.age_label}" if row.age_label else title
                    )
                    label = f"{glyph}{label_rest}"
                else:
                    label_rest = f"{title}\n{row.age_label}" if row.age_label else title
                    label = label_rest
                button = Button(
                    label,
                    id=f"library-notes-row-{index}",
                    classes="library-notes-row",
                    compact=True,
                    disabled=list_state.operation_running,
                )
                button.note_id = row.note_id
                # task-281 (PR #665 review): raw marker-less label for the
                # in-place toggle (reading it back off the Button un-escapes
                # user titles).
                button._library_row_label_rest = label_rest
                yield button

    def _compose_tree_rows(self, list_state: LibraryNotesListState) -> ComposeResult:
        """Render placement-aware rows while retaining legacy note handlers."""
        projection = self.tree_projection
        if projection is None:
            return
        if not projection.rows:
            yield Static(
                list_state.empty_copy,
                id="library-notes-empty",
                markup=False,
            )
            return
        checked_ids = {row.note_id for row in list_state.rows if row.checked}
        with Vertical(id="library-notes-list", classes="library-notes-tree"):
            for index, row in enumerate(projection.rows):
                indent = "  " * row.depth
                if row.kind in {"folder", "unfiled"}:
                    glyph = "▾" if row.expanded else "▸"
                    label = f"{indent}{glyph} {escape_markup(row.label)}"
                    if row.status_text:
                        label = f"{label}  {row.status_text}"
                    classes = "library-notes-folder-row"
                    if row.semantic_status == "connected":
                        classes += " library-notes-tree-connected"
                    elif row.semantic_status == "needs_attention":
                        classes += " library-notes-tree-needs-attention"
                    button = Button(
                        label,
                        id=f"library-notes-tree-folder-{index}",
                        classes=classes,
                        compact=True,
                        tooltip=row.breadcrumb,
                    )
                    if row.placement_id == self.tree_selected_placement_id:
                        button.add_class("is-selected")
                    self._set_tree_row_metadata(button, row)
                    yield button
                    continue

                title = f"{indent}{escape_markup(row.label)}"
                if self.filter_value and row.breadcrumb:
                    parent_breadcrumb = row.breadcrumb.rsplit(" / ", 1)[0]
                    title = f"{title}  — {escape_markup(parent_breadcrumb)}"
                if row.status_text:
                    title = f"{title}  {row.status_text}"
                if list_state.select_mode:
                    marker = "☑ " if row.note_id in checked_ids else "☐ "
                    label_rest = title
                    label = f"{marker}{label_rest}"
                else:
                    label_rest = title
                    label = label_rest
                classes = "library-notes-row library-notes-tree-note-row"
                if row.semantic_status == "connected":
                    classes += " library-notes-tree-connected"
                elif row.semantic_status == "needs_attention":
                    classes += " library-notes-tree-needs-attention"
                button = Button(
                    label,
                    id=f"library-notes-tree-note-{index}",
                    classes=classes,
                    compact=True,
                    tooltip=row.breadcrumb,
                    disabled=list_state.operation_running,
                )
                if row.placement_id == self.tree_selected_placement_id:
                    button.add_class("is-selected")
                self._set_tree_row_metadata(button, row)
                button._library_row_label_rest = label_rest
                yield button
        if projection.has_more:
            yield Button(
                "Load more folder contents",
                id="library-notes-tree-more",
                classes="library-canvas-action",
                compact=True,
            )

    def _compose_tree_actions(self, *, operation_running: bool) -> ComposeResult:
        """Render actions appropriate to the selected folder-tree placement."""
        projection = self.tree_projection
        selected = (
            projection.row(self.tree_selected_placement_id)
            if projection is not None and self.tree_selected_placement_id
            else None
        )
        selected_folder_protected = bool(
            selected is not None and selected.kind == "folder" and selected.protected
        )
        protected_reason = (
            "This folder is managed by sync; change its sync root instead."
        )
        with Horizontal(id="library-notes-tree-actions", classes="ds-toolbar"):
            yield Button(
                "New folder",
                id="library-notes-folder-new",
                classes="library-canvas-action",
                compact=True,
                disabled=operation_running or selected_folder_protected,
                tooltip=(protected_reason if selected_folder_protected else None),
            )
            if selected is not None and selected.kind == "folder":
                for label, button_id in (
                    ("Rename", "library-notes-folder-rename"),
                    ("Move", "library-notes-folder-move"),
                    ("Remove", "library-notes-folder-remove"),
                ):
                    yield Button(
                        label,
                        id=button_id,
                        classes="library-canvas-action",
                        compact=True,
                        disabled=operation_running or selected.protected,
                        tooltip=protected_reason if selected.protected else None,
                    )
            elif selected is not None and selected.kind == "note":
                protected = selected.protected
                protected_placement_reason = (
                    "This placement is managed by sync; change its sync root instead."
                )
                yield Button(
                    "Add to folder",
                    id="library-notes-placement-add",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=operation_running,
                )
                yield Button(
                    "Move note",
                    id="library-notes-placement-move",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=operation_running or protected,
                    tooltip=protected_placement_reason if protected else None,
                )
                yield Button(
                    "Remove placement",
                    id="library-notes-placement-remove",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=(
                        operation_running or protected or not selected.membership_id
                    ),
                    tooltip=(
                        protected_placement_reason
                        if protected
                        else (
                            "Unfiled is shown automatically; move the note into a folder."
                            if not selected.membership_id
                            else None
                        )
                    ),
                )
            if self.tree_deleted_folder_available:
                yield Button(
                    "Restore folder",
                    id="library-notes-folder-restore",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=operation_running,
                )

    @staticmethod
    def _set_tree_row_metadata(button: Button, row: LibraryNotesTreeRow) -> None:
        """Attach stable domain identities without encoding them in DOM ids."""
        button.tree_kind = row.kind
        button.placement_id = row.placement_id
        button.note_id = row.note_id or ""
        button.folder_id = row.folder_id or ""
        button.membership_id = row.membership_id or ""
        button.breadcrumb = row.breadcrumb
        button.ownership = row.ownership or ""
        button.owner_active = row.owner_active
        button.protected_placement = row.protected
        button.folder_version = row.version

    def _compose_editor(self) -> ComposeResult:
        """Mount every editor-session presentation surface exactly once."""
        presentation_state = self.presentation_state
        if presentation_state is None:
            return
        snapshot = presentation_state.snapshot
        title = snapshot.title
        content = snapshot.body
        keywords_text = snapshot.keywords_text
        metadata_line = presentation_state.metadata_line
        status_line = presentation_state.status_line

        # File-synced notes may carry YAML front matter; consume it instead
        # of rendering the delimiter block as note content.
        from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory

        with Horizontal(id="library-note-heading"):
            yield Button(
                "‹ Notes",
                id="library-note-back",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "‹ Note",
                id="library-note-context-back",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static(
                "Edit note",
                id="library-note-editor-title",
                markup=False,
            )
            yield Static(
                ellipsize_note_title_cells(title, 72),
                id="library-note-preview-title",
                markup=False,
            )
            yield Static(
                ellipsize_note_title_cells(title, 72),
                id="library-note-context-title",
                markup=False,
            )
        with Vertical(id="library-note-editor-region"):
            with Horizontal(id="library-note-title-row"):
                yield Static("Title", id="library-note-title-label", markup=False)
                yield Input(
                    value="" if self.title_placeholder_only else title,
                    placeholder="Untitled" if self.title_placeholder_only else "",
                    id="library-note-title",
                )
            yield Static("Body", id="library-note-body-label", markup=False)
            yield TextArea(content, id="library-note-body")

        with VerticalScroll(id="library-note-preview-region", can_focus=True):
            yield Markdown(
                content,
                id="library-note-preview-body",
                parser_factory=front_matter_parser_factory(),
            )
        yield Static(
            status_line,
            id="library-note-context-status",
            markup=False,
        )
        with VerticalScroll(id="library-note-context-region", can_focus=True):
            yield Static("Properties", classes="destination-section", markup=False)
            with Horizontal(id="library-note-context-keywords-row"):
                yield Static(
                    "Keywords", id="library-note-context-keywords-label", markup=False
                )
                yield Input(
                    value=keywords_text,
                    placeholder="Comma-separated keywords",
                    id="library-note-context-keywords",
                )
            yield Static("Metadata", classes="destination-section", markup=False)
            yield Static(metadata_line, id="library-note-context-meta", markup=False)
            yield Static("Chatbook", classes="destination-section", markup=False)
            yield Button(
                "Use in Console",
                id="library-note-context-use-in-console",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static("Utilities", classes="destination-section", markup=False)
            yield Button(
                "Copy",
                id="library-note-context-copy",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Export Markdown",
                id="library-note-context-export-md",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Export text",
                id="library-note-context-export-txt",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static(
                presentation_state.transfer_status
                if presentation_state is not None
                else "",
                id="library-note-context-transfer-status",
                markup=False,
            )
            yield Static("Danger zone", classes="destination-section", markup=False)
            yield Button(
                "Delete",
                id="library-note-context-delete",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
        yield Static(status_line, id="library-note-status", markup=False)

        primary_actions = Horizontal(
            id="library-note-primary-actions", classes="ds-toolbar"
        )
        primary_actions.styles.height = "auto"
        with primary_actions:
            yield Button(
                "Save",
                id="library-note-save",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Edit" if presentation_state.presentation == "preview" else "Preview",
                id="library-note-preview",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Context",
                id="library-note-context",
                classes="library-canvas-action",
                compact=True,
            )
            discard_new = Button(
                "Discard new note",
                id="library-note-discard-new",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
            discard_new.display = presentation_state.discard_new_note
            discard_new.disabled = presentation_state.destructive_running
            yield discard_new
        yield Static(
            presentation_state.transfer_status,
            id="library-note-transfer-status",
            markup=False,
        )

        with Vertical(id="library-note-wide-utilities"):
            yield Static("Keywords", id="library-note-keywords-label", markup=False)
            yield Input(
                value=keywords_text,
                placeholder="Comma-separated keywords",
                id="library-note-keywords",
            )
            yield Static(metadata_line, id="library-note-meta", markup=False)
            wide_actions = Horizontal(classes="ds-toolbar")
            wide_actions.styles.height = "auto"
            with wide_actions:
                yield Button(
                    "Use in Console",
                    id="library-note-use-in-console",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Export Markdown",
                    id="library-note-export-md",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Export text",
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

        with Vertical(id="library-note-conflict-region"):
            yield Static(
                "This note changed elsewhere — Overwrite saves your text; "
                "Reload discards it.",
                id="library-note-conflict-copy",
                classes="destination-purpose",
                markup=False,
            )
            conflict_actions = Horizontal(
                id="library-note-conflict-actions", classes="ds-toolbar"
            )
            conflict_actions.styles.height = "auto"
            with conflict_actions:
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

        with Vertical(id="library-note-delete-confirmation"):
            yield Static(
                "Delete this note? Undo will be available in the Notes list.",
                id="library-note-delete-confirm-copy",
                markup=False,
            )
            delete_actions = Horizontal(
                id="library-note-delete-actions", classes="ds-toolbar"
            )
            delete_actions.styles.height = "auto"
            with delete_actions:
                yield Button(
                    "Cancel",
                    id="library-note-delete-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Delete",
                    id="library-note-delete-confirm",
                    classes="library-canvas-action library-media-action-danger",
                    compact=True,
                )

    def on_mount(self) -> None:
        """Apply initial visibility after the stable editor subtree mounts."""
        self._apply_post_compose_state()

    def _apply_post_compose_state(self) -> None:
        """Post-compose wiring shared by ``on_mount`` and ``_after_recompose``.

        Gated on the MOUNTED CHILDREN, not on ``self.mode``. ``on_mount``
        could trust the mode because it fires once, immediately after its own
        compose. ``_after_recompose`` cannot: ``sync_state`` mutates the
        fields and only SCHEDULES the rebuild, so a second ``sync_state``
        landing while the first recompose is still awaiting ``mount_all``
        leaves this hook reading the newer state against the older children.
        Observed exactly that on the list -> loading -> editor row-press
        sequence: ``mode`` was already "editor" with a presentation state set
        while the mounted child was still ``#library-note-load-state``, and
        ``apply_session_state``'s ``query_one("#library-note-title")`` raised
        into the sync's whole-screen fallback. The newer state's own
        recompose is already queued and applies it a moment later.
        """
        self.apply_compact_presentation(self.compact)
        if self.mode != "editor" or self.presentation_state is None:
            return
        if not self.query("#library-note-title"):
            return
        self.apply_session_state(self.presentation_state)

    def apply_compact_presentation(self, compact: bool) -> None:
        """Update responsive copy without remounting the canvas."""
        self.compact = compact
        if not self.is_mounted:
            return
        if self.mode == "list" and self.list_state is not None:
            database_purpose = self.query("#library-notes-database-purpose")
            if database_purpose:
                database_purpose.first(Static).display = not compact
            rendered_count = len(self.list_state.rows)
            select_all = self.query("#library-notes-select-all")
            if select_all:
                select_all.first(Button).label = (
                    f"All {rendered_count}"
                    if compact
                    else f"Select all {rendered_count} shown"
                )
            export_selected = self.query("#library-notes-export-selected")
            if export_selected:
                # Whole-branch review IMPORTANT-1: this in-place rewrite must
                # compose through the same marker helper as compose() and the
                # screen's `_patch_library_disabled_marker_label`, and re-tier
                # the stashed base -- a plain rewrite stripped the AC#1 "○"
                # marker on every compact-boundary crossing while disabled,
                # and left the stash at the wrong-tier spelling for the next
                # in-place patch.
                button = export_selected.first(Button)
                export_base = "Export" if compact else "Export selected"
                button._library_disabled_marker_base = export_base
                button.label = library_disabled_action_label(
                    export_base, button.disabled
                )
            return
        if self.mode != "sync" or self.sync_panel_state is None:
            return
        for value in ("bidirectional", "disk_to_db", "db_to_disk"):
            label = (
                _COMPACT_SYNC_DIRECTION_LABELS[value]
                if compact
                else sync_direction_label(value)
            )
            prefix = "✓ " if value == self.sync_panel_state.direction else ""
            choices = self.query(f"#library-notes-sync-direction-{value}")
            if choices:
                choices.first(Button).label = f"{prefix}{label}"
        for value in ("newer_wins", "disk_wins", "db_wins"):
            label = (
                _COMPACT_SYNC_CONFLICT_LABELS[value]
                if compact
                else sync_conflict_label(value)
            )
            prefix = "✓ " if value == self.sync_panel_state.conflict else ""
            choices = self.query(f"#library-notes-sync-conflict-{value}")
            if choices:
                choices.first(Button).label = f"{prefix}{label}"
        auto_label = auto_sync_label(self.sync_panel_state.auto_sync)
        if compact:
            auto_label = auto_label.replace("auto-sync: every ", "Auto ", 1)
        auto = self.query("#library-notes-sync-auto")
        if auto:
            auto.first(Button).label = auto_label

    @staticmethod
    def _static_text(widget: Static) -> str:
        renderable = widget.renderable
        return getattr(renderable, "plain", str(renderable))

    def apply_session_state(self, state: LibraryNotePresentationState) -> None:
        """Synchronize stable editor surfaces from one immutable snapshot.

        Value assignments are difference-checked so repeated application is
        idempotent. The screen owns the presentation-sync guard around calls
        that may assign ``Input`` or ``TextArea`` values.
        """
        if self.mode != "editor" or not self.is_mounted:
            self.presentation_state = state
            self.compact = state.compact
            return
        self.presentation_state = state
        self.compact = state.compact
        authority = self.query_one("#library-notes-authority", Static)
        authority_copy = self._authority_copy()
        if self._static_text(authority) != authority_copy:
            authority.update(authority_copy)
        snapshot = state.snapshot
        conflict = state.conflict
        confirming_delete = state.confirming_delete and not conflict
        show_context = (
            state.region == "context" and not conflict and not confirming_delete
        )
        show_preview = (
            not show_context
            and not conflict
            and not confirming_delete
            and state.presentation == "preview"
        )
        show_editor = not show_context and not show_preview

        title_input = self.query_one("#library-note-title", Input)
        body_input = self.query_one("#library-note-body", TextArea)
        wide_keywords = self.query_one("#library-note-keywords", Input)
        context_keywords = self.query_one("#library-note-context-keywords", Input)
        presented_title = "" if self.title_placeholder_only else snapshot.title
        if title_input.value != presented_title:
            with title_input.prevent(Input.Changed):
                title_input.value = presented_title
        title_input.placeholder = "Untitled" if self.title_placeholder_only else ""
        if body_input.text != snapshot.body:
            with body_input.prevent(TextArea.Changed):
                body_input.text = snapshot.body
        if wide_keywords.value != snapshot.keywords_text:
            with wide_keywords.prevent(Input.Changed):
                wide_keywords.value = snapshot.keywords_text
        if context_keywords.value != snapshot.keywords_text:
            with context_keywords.prevent(Input.Changed):
                context_keywords.value = snapshot.keywords_text

        title_width = 52 if state.compact else 72
        title = ellipsize_note_title_cells(snapshot.title, title_width)
        for selector in ("#library-note-preview-title", "#library-note-context-title"):
            widget = self.query_one(selector, Static)
            if self._static_text(widget) != title:
                widget.update(title)

        preview_body = self.query_one("#library-note-preview-body", Markdown)
        # Markdown.update() parses and remounts asynchronously. Keep the
        # hidden Preview stale while typing, then perform one canonical update
        # when Preview becomes the active surface so edits cannot queue an
        # unbounded hidden-render backlog.
        if show_preview and preview_body.source != snapshot.body:
            preview_body.update(snapshot.body)
        compact_status = (
            state.transfer_status
            if state.compact and state.transfer_status
            else state.status_line
        )
        for selector in ("#library-note-status", "#library-note-context-status"):
            widget = self.query_one(selector, Static)
            if self._static_text(widget) != compact_status:
                widget.update(compact_status)
        for selector in ("#library-note-meta", "#library-note-context-meta"):
            widget = self.query_one(selector, Static)
            if self._static_text(widget) != state.metadata_line:
                widget.update(state.metadata_line)
        for selector in (
            "#library-note-transfer-status",
            "#library-note-context-transfer-status",
        ):
            transfer = self.query_one(selector, Static)
            if self._static_text(transfer) != state.transfer_status:
                transfer.update(state.transfer_status)
            transfer.display = bool(state.transfer_status) and not state.compact

        self.apply_compact_presentation(state.compact)
        self.set_class(state.validation, "library-note-validation")
        self.query_one("#library-note-back").display = not show_context
        self.query_one("#library-note-context-back").display = show_context
        self.query_one("#library-note-editor-title").display = (
            show_editor and state.compact
        )
        self.query_one("#library-note-preview-title").display = show_preview
        self.query_one("#library-note-context-title").display = show_context
        self.query_one("#library-note-editor-region").display = show_editor
        self.query_one("#library-note-preview-region").display = show_preview
        self.query_one("#library-note-context-status").display = show_context
        self.query_one("#library-note-context-region").display = show_context
        self.query_one("#library-note-status").display = not show_context
        self.query_one("#library-note-primary-actions").display = (
            not show_context and not conflict and not confirming_delete
        )
        self.query_one("#library-note-wide-utilities").display = (
            not state.compact
            and not show_context
            and not conflict
            and not confirming_delete
        )
        self.query_one("#library-note-conflict-region").display = conflict
        self.query_one("#library-note-delete-confirmation").display = confirming_delete

        locked = confirming_delete or state.destructive_running
        title_input.disabled = not show_editor or locked
        body_input.disabled = not show_editor or locked
        wide_keywords.disabled = state.compact or show_context or locked
        context_keywords.disabled = not show_context or locked
        preview_body.can_focus = False
        self.query_one("#library-note-preview-region").can_focus = show_preview
        self.query_one("#library-note-context-region").can_focus = show_context

        preview_button = self.query_one("#library-note-preview", Button)
        preview_label = "Edit" if state.presentation == "preview" else "Preview"
        if str(preview_button.label) != preview_label:
            preview_button.label = preview_label

        for selector in (
            "#library-note-save",
            "#library-note-preview",
            "#library-note-context",
            "#library-note-use-in-console",
            "#library-note-export-md",
            "#library-note-export-txt",
            "#library-note-copy",
            "#library-note-delete",
            "#library-note-context-use-in-console",
            "#library-note-context-export-md",
            "#library-note-context-export-txt",
            "#library-note-context-copy",
            "#library-note-context-delete",
        ):
            self.query_one(selector, Button).disabled = state.destructive_running
        for selector in (
            "#library-note-use-in-console",
            "#library-note-export-md",
            "#library-note-export-txt",
            "#library-note-copy",
            "#library-note-context-use-in-console",
            "#library-note-context-export-md",
            "#library-note-context-export-txt",
            "#library-note-context-copy",
        ):
            self.query_one(selector, Button).disabled = (
                state.destructive_running or state.transfer_running
            )
        discard_new = self.query_one("#library-note-discard-new", Button)
        discard_new.display = state.discard_new_note
        discard_new.disabled = state.destructive_running
        for selector in (
            "#library-note-conflict-overwrite",
            "#library-note-conflict-reload",
        ):
            self.query_one(selector, Button).disabled = (
                state.destructive_running or state.conflict_running
            )
        for selector in (
            "#library-note-delete-confirm",
            "#library-note-delete-cancel",
        ):
            self.query_one(selector, Button).disabled = state.destructive_running

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
        with Horizontal(id="library-notes-create-heading"):
            yield Button(
                "‹ Notes",
                id="library-notes-create-back",
                classes="library-canvas-action",
                compact=True,
                disabled=self.create_running,
            )
            yield Static(
                "New note",
                id="library-notes-create-header",
                classes="destination-section",
                markup=False,
            )
        with VerticalScroll(id="library-notes-create-viewport"):
            yield Button(
                "Blank note",
                id="library-notes-create-blank",
                classes="library-notes-create-row",
                compact=True,
                disabled=self.create_running,
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
                button.disabled = self.create_running
                yield button
            yield Static(
                self.create_status,
                id="library-notes-create-status",
                markup=False,
            )

    def _compose_sync(self) -> ComposeResult:
        """Render the notes sync panel: folder, direction, conflicts, activity.

        Direction and conflict policy use explicit compact choice groups so
        every available value and the current selection remain visible.
        Auto-sync stays a direct toggle button. All mutable controls are
        disabled while a sync run is active.
        """
        sync_state = self.sync_panel_state
        if sync_state is None:
            return
        with Horizontal(id="library-notes-sync-heading"):
            yield Button(
                "‹ Notes",
                id="library-notes-sync-back",
                classes="library-canvas-action",
                compact=True,
                disabled=sync_state.running,
            )
            yield Static(
                "Notes sync",
                id="library-notes-sync-header",
                classes="destination-section",
                markup=False,
            )
        with VerticalScroll(id="library-notes-sync-viewport"):
            yield Static(
                "Mirror notes between a folder on disk and the Library — "
                "unlike Files mode, which edits that folder directly without "
                "mirroring it in.",
                id="library-notes-sync-purpose",
                markup=False,
            )
            with Horizontal(id="library-notes-sync-folder-row"):
                yield Static(
                    "Folder", id="library-notes-sync-folder-label", markup=False
                )
                yield Input(
                    value=sync_state.folder,
                    placeholder="Folder to sync…",
                    id="library-notes-sync-folder",
                    disabled=sync_state.running,
                )
                yield Button(
                    "Browse…",
                    id="library-notes-sync-browse",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=sync_state.running,
                )
            with Horizontal(id="library-notes-sync-direction-row"):
                yield Static(
                    "Direction", id="library-notes-sync-direction-label", markup=False
                )
                direction_choices = Horizontal(
                    id="library-notes-sync-direction-choices", classes="ds-toolbar"
                )
                direction_choices.styles.height = "auto"
                with direction_choices:
                    for value in ("bidirectional", "disk_to_db", "db_to_disk"):
                        label = (
                            _COMPACT_SYNC_DIRECTION_LABELS[value]
                            if self.compact
                            else sync_direction_label(value)
                        )
                        button = Button(
                            f"{'✓ ' if value == sync_state.direction else ''}{label}",
                            id=f"library-notes-sync-direction-{value}",
                            classes=(
                                "library-canvas-action "
                                "library-notes-sync-direction-choice"
                            ),
                            compact=True,
                            disabled=sync_state.running,
                        )
                        button.choice_value = value
                        yield button
            with Horizontal(id="library-notes-sync-conflict-row"):
                yield Static(
                    "Conflicts", id="library-notes-sync-conflict-label", markup=False
                )
                conflict_choices = Horizontal(
                    id="library-notes-sync-conflict-choices", classes="ds-toolbar"
                )
                conflict_choices.styles.height = "auto"
                with conflict_choices:
                    for value in ("newer_wins", "disk_wins", "db_wins"):
                        label = (
                            _COMPACT_SYNC_CONFLICT_LABELS[value]
                            if self.compact
                            else sync_conflict_label(value)
                        )
                        button = Button(
                            f"{'✓ ' if value == sync_state.conflict else ''}{label}",
                            id=f"library-notes-sync-conflict-{value}",
                            classes=(
                                "library-canvas-action "
                                "library-notes-sync-conflict-choice"
                            ),
                            compact=True,
                            disabled=sync_state.running,
                        )
                        button.choice_value = value
                        yield button
            with Horizontal(id="library-notes-sync-actions"):
                auto_label = auto_sync_label(sync_state.auto_sync)
                if self.compact:
                    auto_label = auto_label.replace("auto-sync: every ", "Auto ", 1)
                yield Button(
                    auto_label,
                    id="library-notes-sync-auto",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=sync_state.running,
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
