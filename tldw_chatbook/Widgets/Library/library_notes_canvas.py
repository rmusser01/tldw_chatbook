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
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_disabled_action_label,
)
from tldw_chatbook.Library.library_notes_sync_state import (
    LibraryNotesSyncState,
    auto_sync_label,
    sync_conflict_label,
    sync_direction_label,
)
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)

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
            in-canvas note editor for ``presentation_state``; ``"create"`` renders
            the Blank note / template picker reached from the rail's
            Create > New note row; ``"sync"`` renders the in-canvas notes
            sync panel for ``sync_state``.
        presentation_state: Canonical snapshot plus presentation-only state.
            Required when ``mode == "editor"``.
        sync_state: The sync panel's display state. Required when
            ``mode == "sync"``.
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
        sync_state: LibraryNotesSyncState | None = None,
        title_placeholder_only: bool = False,
        compact: bool = False,
        create_running: bool = False,
        create_status: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize one list, editor, create, or sync canvas.

        Args:
            list_state: List-mode rows, counts, selection, and empty-state copy.
            sort_mode: Active list sort key.
            filter_value: Text prefilled into the list filter.
            mode: Canvas surface to compose: list, editor, create, or sync.
            presentation_state: Canonical editor snapshot and UI-only flags.
            sync_state: Display state for the sync surface.
            title_placeholder_only: Render an empty title with an Untitled
                placeholder for a pristine newly-created note.
            compact: Whether 60-column-safe controls and labels are active.
            create_running: Whether note creation is in progress.
            create_status: Visible creation completion or recovery status.
            **kwargs: Additional keyword arguments forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.presentation_state = presentation_state
        self.sync_state = sync_state
        self.title_placeholder_only = title_placeholder_only
        self.compact = compact
        self.create_running = create_running
        self.create_status = create_status
        self.styles.width = "1fr"
        self.styles.min_width = 40
        self.add_class(f"library-notes-mode-{mode}")

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
        rendered_count = len(list_state.rows)
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
            transfer_actions = Horizontal(
                id="library-notes-transfer-actions", classes="ds-toolbar"
            )
            transfer_actions.styles.height = "auto"
            with transfer_actions:
                for label, button_id in (
                    ("Sync", "library-notes-sync-open"),
                    ("Import", "library-notes-import"),
                    ("Export", "library-notes-export"),
                ):
                    yield Button(
                        label,
                        id=button_id,
                        classes="library-canvas-action",
                        compact=True,
                        disabled=list_state.operation_running,
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
                )
                button.note_id = row.note_id
                # task-281 (PR #665 review): raw marker-less label for the
                # in-place toggle (reading it back off the Button un-escapes
                # user titles).
                button._library_row_label_rest = label_rest
                yield button

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
                "Delete this note? This cannot be undone from Library.",
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
        self.apply_compact_presentation(self.compact)
        if self.mode == "editor" and self.presentation_state is not None:
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
        if self.mode != "sync" or self.sync_state is None:
            return
        for value in ("bidirectional", "disk_to_db", "db_to_disk"):
            label = (
                _COMPACT_SYNC_DIRECTION_LABELS[value]
                if compact
                else sync_direction_label(value)
            )
            prefix = "✓ " if value == self.sync_state.direction else ""
            choices = self.query(f"#library-notes-sync-direction-{value}")
            if choices:
                choices.first(Button).label = f"{prefix}{label}"
        for value in ("newer_wins", "disk_wins", "db_wins"):
            label = (
                _COMPACT_SYNC_CONFLICT_LABELS[value]
                if compact
                else sync_conflict_label(value)
            )
            prefix = "✓ " if value == self.sync_state.conflict else ""
            choices = self.query(f"#library-notes-sync-conflict-{value}")
            if choices:
                choices.first(Button).label = f"{prefix}{label}"
        auto_label = auto_sync_label(self.sync_state.auto_sync)
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
        sync_state = self.sync_state
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
