"""Library notes canvas: list mode (rows + filter + sort), editor mode, and
create mode (Blank note + template rows)."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static, TextArea

from tldw_chatbook.Library.library_notes_state import (
    LibraryNoteSessionSnapshot,
    LibraryNotesListState,
    LibraryNotesTrashState,
    build_library_note_template_rows,
    ellipsize_note_title_cells,
)
from tldw_chatbook.Library.library_note_import_state import LibraryNoteImportSnapshot
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LibraryNotesLastingSyncSnapshot,
)
from tldw_chatbook.Notes.agent_lessons import (
    AGENT_LESSONS_FOLDER,
    AGENT_LESSONS_FOLDER_GLOSS,
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
from tldw_chatbook.Widgets.Library.library_rail import LibraryRailSearchInput
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
    library_row_button,
)
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

_SORT_LABELS = {"newest": "Newest", "oldest": "Oldest", "title": "Title"}

#: Columns the list pane needs before the browse and transfer toolbars share
#: one row. Their widest composition -- New, Select, Add from files…, Export,
#: Manage sync folders, Last import, plus both toolbars' own padding -- is 97
#: cells, so under this width the merged row clips its last action off the
#: pane, which is worse than the third row it saves (task-32127, review 1).
_TOOLBAR_MERGE_MIN_WIDTH = 100

#: Columns a single action group needs to stay on one row. The transfer group
#: (Add from files…, Export, Last import) is 47 cells and the folder actions
#: (New folder, Rename, Move, Remove) are 46, so at the 44-column pane a
#: 130-column terminal gives the list beside an open note, "Last import" and
#: "Remove" were painted past the pane's right edge and could not be pressed
#: (task-32127, final review). Below this width each group stacks, the way the
#: delete receipt already stacks its recovery actions (task-32123) -- Textual
#: toolbars do not wrap. NOT in a compact shell: there the split screen sheet
#: pins these rows to `height: 1; overflow-x: hidden`, so a stacked column
#: would be clipped to its first button instead of merely running off-pane.
_TOOLBAR_STACK_MIN_WIDTH = 48


def _toolbar_shape(pane_width: int, compact: bool) -> tuple[bool, bool]:
    """Return the two toolbar decisions one pane width drives.

    A width of 0 means "not measured yet": the canvas is composed before the
    reader shell it lives in exists, so the first frame of a Notes visit has
    no resolved Items width. That frame takes the conservative shape -- one
    row per group, neither merged nor stacked -- which is what the list
    rendered before either threshold existed; the first state sync then
    composes at the real width (task-32127, review round 2).

    Args:
        pane_width: Columns the list pane has, or 0 when unmeasured.
        compact: Whether the compact shell pins the toolbars to one row.

    Returns:
        ``(merged, stacked)`` -- whether the browse and transfer groups
        share a row, and whether a group stacks its actions vertically.
    """
    return (
        pane_width >= _TOOLBAR_MERGE_MIN_WIDTH,
        0 < pane_width < _TOOLBAR_STACK_MIN_WIDTH and not compact,
    )


def compose_note_row_label(
    title: str, *, folder_label: str = "", age_label: str = ""
) -> str:
    """Render one Notes list row label: title, folder, then age.

    The single renderer for BOTH list paths (task-32137). The flat list used
    to put the age on a second line of its own -- a branch nothing reached
    once a folder tree existed -- while the tree rows carried no age at all,
    so two notes titled "Reading list" rendered as identical rows.

    Args:
        title: The note title, already markup-escaped.
        folder_label: The row's parent folder ("Unfiled", "Work / Q3"),
            included only when the row needs telling apart from a sibling
            with the same title, or when a filter has scattered the rows.
        age_label: Relative age of the note ("3m", "1d"), if known.

    Returns:
        The row label, its present parts joined with " · ".
    """
    return " · ".join(part for part in (title, folder_label, age_label) if part)


#: Backlink rows Info renders at most (task-32145). The loader asks for one
#: more than this so an over-cap result can say "50+" rather than claim an
#: exact 50 that is not the real number.
LIBRARY_NOTE_BACKLINK_DISPLAY_CAP = 50


def library_note_backlink_header(
    backlinks: tuple[tuple[str, str], ...],
    status: str = "ready",
) -> str:
    """The "Linked from" heading for one note's inbound links.

    Args:
        backlinks: ``(note_id, title)`` rows, possibly one over the display
            cap (see ``LIBRARY_NOTE_BACKLINK_DISPLAY_CAP``).
        status: ``"loading"``, ``"ready"`` or ``"failed"`` -- the count is
            only claimed once the query has actually answered, so a pending
            or failed lookup never reads as a verified zero.

    Returns:
        The heading, which names the count and, when there are none, says
        so in words rather than leaving a bare ``(0)`` to be read as a
        failed load.
    """
    if status == "loading":
        return "Linked from — checking…"
    if status == "failed":
        return "Linked from — couldn't check"
    if not backlinks:
        return "Linked from (0) — no notes link here yet"
    if len(backlinks) > LIBRARY_NOTE_BACKLINK_DISPLAY_CAP:
        return f"Linked from ({LIBRARY_NOTE_BACKLINK_DISPLAY_CAP}+)"
    return f"Linked from ({len(backlinks)})"


def _library_note_back_label(compact: bool) -> str:
    """The single Back wording (task-32139), sized by ``compact``.

    PR #2547 review (Qodo finding 1): compose time and state-apply time
    each inlined this same ternary; a wording change could update one
    rendering path and leave the other stale. One function, both callers.
    """
    return "‹ Back to list" if compact else "‹ Notes"


#: The storage authority every Database Notes surface answers to. Painted once
#: per screen: the mounted list pane owns it, and a work pane beside it drops
#: it rather than repeating the same sentence (task-32063).
NOTES_AUTHORITY_PREFIX = "Library notes · Library database"

#: Every control a reader types a note into. A refresh that would recompose
#: this canvas while one of them has focus is deferred instead (task-32062).
#: The keyword boxes belong here for the same reason the title does -- review
#: of PR #2531: both are live ``Input``s, and only one of the two is mounted
#: visible at a time (wide editor vs. context region).
_NOTE_EDITOR_INPUT_IDS = frozenset(
    {
        "library-note-title",
        "library-note-body",
        "library-note-keywords",
        "library-note-context-keywords",
    }
)


#: task-32106 AC#1: shared by every field of the note editor -- see
#: ``NoteEditorInput`` below for the mechanism and the measurement.
_NOTE_FIELD_TAB_BINDINGS = [
    Binding("tab", "screen.focus_next", show=False, priority=True),
    Binding("shift+tab", "screen.focus_previous", show=False, priority=True),
]


class NoteEditorInput(Input):
    """A note field whose Tab moves focus BEFORE the next key is forwarded.

    task-32106 AC#1: ``Screen.BINDINGS``' ``Binding("tab", "app.focus_next")``
    is not ``priority=True``, so ``Key(tab)`` is posted to the focused
    ``Input`` and has to bubble one message-queue hop per ancestor up to the
    Screen -- while the App keeps dequeuing the following keys and forwarding
    each to ``self.focused``, which is still this field. Typed fast enough
    (one terminal read, or a message queue backed up behind a busy Library
    screen) the body lands in the title. Reproduced in stock Textual 8 with
    nothing from this repo in it:

        pilot  elapsed=1268.9ms  title='My first note'      body='hello'
        burst  elapsed=   0.2ms  title='My first notehello' body=''

    A priority binding makes the App resolve the focus move before it
    forwards the next key. The action is namespaced to the SCREEN so it
    resolves to ``LibraryScreen.action_focus_next``, which cycles inside
    ``#screen-content`` (task-32052 AC#3) -- ``app.focus_next`` would walk
    the nav bar instead, and a bare ``focus_next`` resolves against this
    ``Input``, which has no such action, so the binding never fires.
    """

    BINDINGS = _NOTE_FIELD_TAB_BINDINGS


class NoteEditorTextArea(TextArea):
    """The note body, with the same synchronous Tab as the fields around it.

    The body has the identical defect one widget over (coordinator addendum
    from a peer session): bursting ``hello`` + Tab + ``world`` into it left
    BOTH words in the body -- measured here as
    ``'helloworldalpha budget line'`` -- because Tab's focus move landed
    after the burst.

    Safe only while ``tab_behavior`` is ``"focus"`` -- Textual's default,
    and what this editor wants: the body is prose, not code, and Tab is how
    a reader leaves it. Under ``"indent"`` this binding would steal the key
    the ``TextArea`` needs, so the pin asserts that behaviour rather than
    trusting the default.
    """

    BINDINGS = _NOTE_FIELD_TAB_BINDINGS


@dataclass(frozen=True)
class NotesStatusChannels:
    """Independent Notes header channels with one optional recovery action."""

    content_recovery: str
    authority_git: str
    safe_next_action: str | None = None


def resolve_database_note_status_channels(
    *,
    conflict: bool = False,
    unavailable: bool = False,
    read_only: bool = False,
    save_failed: bool = False,
    saving: bool = False,
    dirty: bool = False,
) -> NotesStatusChannels:
    """Resolve Database Notes status in the approved deterministic order.

    Args:
        conflict: Whether the database note and editor draft conflict.
        unavailable: Whether the Library database cannot currently be reached.
        read_only: Whether the active note cannot be edited.
        save_failed: Whether the latest save attempt failed.
        saving: Whether a save is currently running.
        dirty: Whether the editor contains unsaved changes.

    Returns:
        The independent content, authority, and safe-action status channels.
    """
    if conflict:
        content = "Conflict — the note changed elsewhere; your draft is preserved."
        safe = "Review recovery"
    elif unavailable:
        content = (
            "Unavailable — the database cannot be reached; your draft is preserved."
        )
        safe = "Retry when storage is available"
    elif read_only:
        content = "Read-only — this note cannot be changed; your draft is preserved."
        safe = "Keep the draft"
    elif save_failed:
        content = "Save failed — your draft remains in the editor."
        safe = "Retry Save"
    elif saving:
        content, safe = "Saving…", None
    elif dirty:
        content, safe = "Unsaved changes", None
    else:
        content, safe = "Saved", None
    return NotesStatusChannels(content, "Database Notes · Library database", safe)


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
    bulk_read_only: bool = False
    bulk_included: bool = False
    status_channels: NotesStatusChannels | None = None
    #: ``(note_id, title)`` for each note linking to this one (task-32145).
    backlinks: tuple[tuple[str, str], ...] = ()
    #: Whether the backlink query has answered yet -- see
    #: ``library_note_backlink_header``.
    backlinks_status: str = "loading"


class _LibraryNotesTreePagerButton(Button):
    """Keep semantic focus during an inert loading-state replacement."""

    _retain_disabled_focus = False

    @property
    def focusable(self) -> bool:
        """Allow only the recompose restorer to retain disabled focus."""
        return self._retain_disabled_focus or super().focusable

    def retain_semantic_focus(self) -> None:
        """Restore focus without making a disabled pager activatable."""
        self._retain_disabled_focus = True
        self.focus()
        self.app.call_later(self._clear_disabled_focus_override)

    def _clear_disabled_focus_override(self) -> None:
        self._retain_disabled_focus = False


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
        presentation_state: Canonical snapshot plus presentation-only state.
            Required when ``mode == "editor"``.
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
            ``_notes_state.pending_blank_gc_id``); it never applies to a
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
        import_snapshot: LibraryNoteImportSnapshot | None = None,
        import_receipt_available: bool = False,
        lasting_sync_snapshot: LibraryNotesLastingSyncSnapshot | None = None,
        tree_projection: LibraryNotesTreeProjection | None = None,
        tree_selected_placement_id: str = "",
        tree_deleted_folder_available: bool = False,
        trash: LibraryNotesTrashState | None = None,
        title_placeholder_only: bool = False,
        compact: bool = False,
        pane_width: int = 0,
        create_running: bool = False,
        create_status: str = "",
        load_state: str = "loading",
        load_message: str = "",
        authority_id: str = "library-notes-authority",
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
            import_snapshot: Display state for the one-time import surface.
            tree_projection: Placement-aware folder rows for list mode.
            tree_selected_placement_id: Context row for folder actions.
            tree_deleted_folder_available: Whether Undo folder removal is available.
            title_placeholder_only: Render an empty title with an Untitled
                placeholder for a pristine newly-created note.
            compact: Whether 60-column-safe controls and labels are active.
            pane_width: Columns the mounted list pane has, from the resolved
                reader layout. Only the toolbar reads it, to decide whether
                one row can hold both action groups; ``0`` (unknown) keeps
                them on their own rows.
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
        self.import_snapshot = import_snapshot
        self.import_receipt_available = import_receipt_available
        self.lasting_sync_snapshot = lasting_sync_snapshot
        self.tree_projection = tree_projection
        self.tree_selected_placement_id = tree_selected_placement_id
        self.tree_deleted_folder_available = tree_deleted_folder_available
        self.trash = trash
        self.title_placeholder_only = title_placeholder_only
        self.compact = compact
        self.pane_width = pane_width
        self.create_running = create_running
        self.create_status = create_status
        self.load_state = load_state
        self.load_message = load_message
        self.authority_id = authority_id
        self._tree_pager_focus_id: str | None = None
        self._tree_pager_focus_guard: Callable[[], bool] | None = None
        self._tree_pager_focus_generation = 0
        #: Which backlink rows the mounted Info panel currently holds, so a
        #: sync only remounts them when the set actually changed (task-32145).
        self._rendered_backlinks: tuple[tuple[str, str], ...] = ()
        self._tree_focus_intent_generation: Callable[[], int] | None = None
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
        focus_id = self._tree_pager_focus_id
        guard = self._tree_pager_focus_guard
        generation = self._tree_pager_focus_generation
        self._tree_pager_focus_id = None
        self._tree_pager_focus_guard = None
        if not focus_id or not self._tree_pager_authority_is_current(guard, generation):
            return
        matches = self.query(f"#{focus_id}")
        if not matches:
            return
        pager = matches.first(_LibraryNotesTreePagerButton)
        if pager.disabled:
            self.call_after_refresh(
                self._retain_tree_pager_focus,
                pager,
                guard,
                generation,
            )
        elif self._tree_pager_authority_is_current(guard, generation):
            pager.focus()

    def _tree_pager_authority_is_current(
        self,
        guard: Callable[[], bool] | None,
        generation: int,
    ) -> bool:
        """Return whether one sync still owns semantic pager focus."""
        return generation == self._tree_pager_focus_generation and (
            guard is None or guard()
        )

    def _retain_tree_pager_focus(
        self,
        pager: _LibraryNotesTreePagerButton,
        guard: Callable[[], bool] | None,
        generation: int,
    ) -> None:
        """Retain disabled pager focus only for the originating sync."""
        if not self._tree_pager_authority_is_current(guard, generation):
            return
        if not pager.is_attached:
            return
        pager.retain_semantic_focus()

    async def recompose(self) -> None:
        """Preserve a newer in-canvas focus when pager authority expires."""
        newest_focus_id: str | None = None
        focus_generation = self._tree_pager_focus_generation
        focus_intent_generation: int | None = None
        focus_intent_generation_getter = self._tree_focus_intent_generation
        pager_focus_id = self._tree_pager_focus_id
        if pager_focus_id and not self._tree_pager_authority_is_current(
            self._tree_pager_focus_guard,
            self._tree_pager_focus_generation,
        ):
            focused = self.app.focused
            if (
                focused is not None
                and focused.id
                and focused.id != pager_focus_id
                and self in focused.ancestors_with_self
            ):
                newest_focus_id = focused.id
                if focus_intent_generation_getter is not None:
                    focus_intent_generation = focus_intent_generation_getter()
        await super().recompose()
        if (
            not newest_focus_id
            or not self.is_attached
            or focus_generation != self._tree_pager_focus_generation
            or (
                focus_intent_generation is not None
                and focus_intent_generation_getter is not None
                and focus_intent_generation_getter() != focus_intent_generation
            )
        ):
            return
        matches = self.query(f"#{newest_focus_id}")
        if matches:
            matches.first().screen.set_focus(matches.first())

    def compose(self) -> ComposeResult:
        yield Static(
            self._authority_copy(),
            id=self.authority_id,
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
        if self.mode == "trash":
            yield from self._compose_trash()
            return
        if self.mode == "import":
            if self.import_snapshot is not None:
                yield LibraryNoteImportCanvas(
                    self.import_snapshot,
                    compact=self.compact,
                    id="library-note-import-canvas",
                )
            yield Button(
                "Back to Notes",
                id="library-notes-import-back",
                classes="library-canvas-action",
                compact=True,
            )
            return
        if self.mode == "lasting_add":
            if self.lasting_sync_snapshot is not None:
                yield LibraryNotesAddFromFilesCanvas(
                    self.lasting_sync_snapshot,
                    id="library-notes-lasting-add-canvas",
                )
            return
        if self.mode == "lasting_roots":
            if self.lasting_sync_snapshot is not None:
                yield LibraryNotesSyncRootsCanvas(
                    self.lasting_sync_snapshot,
                    id="library-notes-lasting-roots-canvas",
                )
            return
        yield from self._compose_list()

    def _authority_prefix(self) -> str:
        """Return the authority this canvas names before its own status.

        Subclasses that render beside a pane already naming the authority
        return ``""`` -- see ``LibraryNoteWorkPane`` (task-32063).
        """
        return NOTES_AUTHORITY_PREFIX

    def _authority_copy(self) -> str:
        """Describe Library storage, current status, and the next action."""
        prefix = self._authority_prefix()
        def line(*parts: str) -> str:
            """Join the non-empty clauses this state actually has."""
            return " · ".join(part for part in (prefix, *parts) if part)

        if self.mode == "loading":
            if self.load_state == "failed":
                status = self.load_message or "Could not load note."
                return line(status, "Next: Retry loading.")
            # task-32063: a "Next:" clause names a control the reader can
            # press. "Wait for loading to finish" names none, so this state
            # ends at its status.
            return line("Loading note…")
        if self.mode == "editor":
            state = self.presentation_state
            if state is None:
                return line("Editor unavailable", "Next: Back to notes.")
            status = state.status_line or "Ready"
            transfer = (
                f" · {state.transfer_status}"
                if state.transfer_status and state.transfer_status != status
                else ""
            )
            if state.conflict:
                next_action = "Resolve the conflict or reload the note."
            elif state.snapshot.saving or state.transfer_running:
                next_action = ""
            elif "failed" in f"{status} {state.transfer_status}".lower():
                next_action = "Review the error, then keep editing."
            else:
                next_action = "Keep editing; changes save automatically."
            return line(
                f"{status}{transfer}",
                f"Next: {next_action}" if next_action else "",
            )
        if self.mode == "create":
            status = self.create_status or (
                "Creating note…" if self.create_running else "Ready"
            )
            next_action = (
                "" if self.create_running else "Choose Blank note or a template."
            )
            return line(status, f"Next: {next_action}" if next_action else "")
        if self.mode == "import":
            state = self.import_snapshot
            status = "Import unavailable" if state is None else state.status_line
            return line("Import once", status, "Next: Review the import workflow.")
        if self.mode in {"lasting_add", "lasting_roots"}:
            state = self.lasting_sync_snapshot
            status = "Unavailable" if state is None else state.status_line
            if state is not None and state.phase == "choose":
                # task-32125: naming one of the two relationships before the
                # reader has picked either presumed the answer.
                return line(
                    "Add from files",
                    status,
                    "Next: Choose Import once or Keep a folder synced.",
                )
            next_action = (
                "Use Import once."
                if state is None or not state.lasting_available
                else "Review the current lasting-sync step."
            )
            return line("Lasting sync", status, f"Next: {next_action}")
        state = self.list_state
        status = state.operation_status if state is not None else ""
        running = state is not None and state.operation_running
        status = status or ("Updating notes…" if running else "Ready")
        next_action = "" if running else "Create a note or add from files."
        return line(status, f"Next: {next_action}" if next_action else "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Refuse row/action presses while this canvas is resident but hidden.

        Phase C keeps both browse canvases mounted and toggles ``display``
        (``UI/Library_Modules/library_browse_route_swap.py``). Textual 8.2.8's
        ``Button.press()`` consults the BUTTON's own ``disabled``/``display``
        and nothing above it, so a press aimed at a hidden canvas's row still
        bubbles to the screen and would act on a route the user has left --
        the design record's verified finding #3. Stop it here, at the canvas
        that owns the residency state, rather than in each of the screen's
        row handlers.

        **Scope, stated because it is narrower than it looks.** This gates
        ``Button.Pressed`` and nothing else. Still ungated, deliberately:
        ``Input.Changed`` / ``Input.Submitted`` (a hidden widget is not in the
        focus chain, so a user cannot type into one, and no code drives these
        programmatically off-route). Unlike ``LibraryMediaCanvas``, this
        canvas has no row-geometry message of its own -- there is no Notes
        analogue of ``LibraryMediaRowGeometryChanged`` to gate or fence
        (verified: that message is defined and posted only in
        ``library_media_canvas.py``). If a future change makes a hidden
        canvas focusable, drives its Inputs from code, or adds a Notes
        geometry message, this gate does NOT cover it.

        Args:
            event: The bubbling press.

        Returns:
            None.
        """
        if not self.display:
            event.stop()
            event.prevent_default()

    def sync_state(
        self,
        *,
        list_state: LibraryNotesListState | None,
        sort_mode: str,
        filter_value: str,
        mode: str,
        presentation_state: LibraryNotePresentationState | None,
        import_snapshot: LibraryNoteImportSnapshot | None = None,
        import_receipt_available: bool = False,
        lasting_sync_snapshot: LibraryNotesLastingSyncSnapshot | None = None,
        tree_projection: LibraryNotesTreeProjection | None,
        tree_selected_placement_id: str,
        tree_deleted_folder_available: bool,
        trash: LibraryNotesTrashState | None = None,
        title_placeholder_only: bool,
        compact: bool,
        pane_width: int = 0,
        create_running: bool,
        create_status: str,
        load_state: str,
        load_message: str,
        deferred_guard: Callable[[], bool] | None = None,
        focus_intent_generation: Callable[[], int] | None = None,
    ) -> None:
        """Apply a complete screen-owned snapshot within this canvas only.

        The method replaces every compose input before updating the visible
        surface. A retained import child accepts same-route snapshots directly
        so active text input keeps identity; other routes rebuild this widget's
        children.

        Args:
            list_state: Notes list snapshot, or ``None`` outside list mode.
            sort_mode: Active Notes sort identifier.
            filter_value: Current Notes filter text.
            mode: Canvas surface to render.
            presentation_state: Note editor/create presentation snapshot.
            import_snapshot: Reviewed one-time import presentation snapshot.
            import_receipt_available: Whether the latest same-session receipt can reopen.
            tree_projection: Placement-aware folder rows for list mode.
            tree_selected_placement_id: Context row for folder actions.
            tree_deleted_folder_available: Whether Undo folder removal is available.
            title_placeholder_only: Whether the title is placeholder-only.
            compact: Whether compact editor controls are enabled.
            pane_width: Columns the mounted list pane has (see ``__init__``).
                Defaults to the same ``0`` the constructor uses for "not
                measured yet", so a caller outside the screen's reader
                layout gets the conservative toolbar shape rather than a
                ``TypeError`` (PR #2549 review, finding 6).
            create_running: Whether note creation is in progress.
            create_status: Current note-creation status copy.
            load_state: Current note-loading state identifier.
            load_message: Current note-loading status or error copy.
            deferred_guard: Authority predicate for pager focus restoration
                scheduled by this exact sync.
            focus_intent_generation: Current screen focus-intent generation,
                read before and after an awaited recompose.
        """
        previous_mode = self.mode
        focused = self.app.focused
        self._tree_pager_focus_generation += 1
        self._tree_focus_intent_generation = focus_intent_generation
        if (
            focused is not None
            and focused.id
            and focused.has_class("library-notes-tree-pager")
            and self in focused.ancestors_with_self
        ):
            self._tree_pager_focus_id = focused.id
            self._tree_pager_focus_guard = deferred_guard
        else:
            self._tree_pager_focus_id = None
            self._tree_pager_focus_guard = None
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.presentation_state = presentation_state
        self.import_snapshot = import_snapshot
        self.import_receipt_available = import_receipt_available
        self.lasting_sync_snapshot = lasting_sync_snapshot
        self.tree_projection = tree_projection
        self.tree_selected_placement_id = tree_selected_placement_id
        self.tree_deleted_folder_available = tree_deleted_folder_available
        self.trash = trash
        self.title_placeholder_only = title_placeholder_only
        self.compact = compact
        self.pane_width = pane_width
        self.create_running = create_running
        self.create_status = create_status
        self.load_state = load_state
        self.load_message = load_message
        if previous_mode != mode:
            self.remove_class(f"library-notes-mode-{previous_mode}")
            self.add_class(f"library-notes-mode-{mode}")
        import_canvases = self.query("#library-note-import-canvas")
        if (
            previous_mode == mode == "import"
            and import_snapshot is not None
            and import_canvases
        ):
            authority = self.query(f"#{self.authority_id}")
            if authority:
                authority.first(Static).update(self._authority_copy())
            child = import_canvases.first(LibraryNoteImportCanvas)
            child.compact = compact
            callback = self._post_recompose_callback
            self._post_recompose_callback = None
            child.queue_after_recompose(callback)
            child.sync_state(import_snapshot)
            if not getattr(child, "_recompose_required", False):
                child.queue_after_recompose(None)
                if callback is not None:
                    self.call_after_refresh(callback)
            return
        lasting_canvases = self.query("#library-notes-lasting-add-canvas")
        if (
            previous_mode == mode == "lasting_add"
            and lasting_sync_snapshot is not None
            and lasting_canvases
        ):
            authority = self.query(f"#{self.authority_id}")
            if authority:
                authority.first(Static).update(self._authority_copy())
            lasting_canvases.first(LibraryNotesAddFromFilesCanvas).sync_state(
                lasting_sync_snapshot
            )
            return
        if previous_mode == mode and self.editor_has_focus():
            # task-32062: a Notes refresh (a save landing, the first note
            # reaching the list, an evidence-driven reload) recomposed the
            # surface the reader was typing into: the title Input and body
            # TextArea were rebuilt and focus fell onto the list grip, so the
            # next keystrokes went somewhere else. Reported as a title that
            # swallowed the body. Every compose input above is already stored,
            # so the next refresh from outside the field paints this state --
            # only the rebuild is skipped, and only while the surface is
            # STAYING put: a mode change is a navigation the reader asked for
            # and must paint immediately.
            #
            # Known ceiling: a banner this recompose would have painted (the
            # "changed elsewhere" conflict, say) waits for the next refresh
            # that arrives with the reader's hands off the field. Interrupting
            # a sentence to show it costs them their keystrokes, which is the
            # worse half of the trade.
            #
            # The caller must not queue a post-recompose follow-up for a
            # rebuild that is not happening -- see the `notes_editor_owned`
            # guard in ``canvas_sync._sync_library_canvas``, without which the
            # Notes focus restore sat here and fired at the NEXT recompose,
            # dragging focus back out of whatever field the reader moved to.
            return
        self.refresh(recompose=True)

    def editor_has_focus(self) -> bool:
        """Whether a field of THIS canvas's note editor currently has focus.

        Returns:
            ``True`` while the focused widget is one of this canvas's own
            editable note fields (title, body, or either keyword box).
        """
        try:
            focused = self.app.focused
        except Exception:
            return False
        if focused is None or focused.id not in _NOTE_EDITOR_INPUT_IDS:
            return False
        try:
            return self in focused.ancestors_with_self
        except Exception:
            return False

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
        # Database mode persists notes in the Library; Folder files edits a
        # folder directly, while Add from files owns import/sync setup.
        database_purpose = Static(
            "These notes live in the Library's own database — for notes "
            "that live in a folder on disk, switch to Folder files. To copy "
            "or keep a folder synced, choose Add from files.",
            id="library-notes-database-purpose",
            markup=False,
        )
        database_purpose.display = not self.compact
        yield database_purpose
        with Horizontal(id="library-notes-filter-row"):
            yield Static("Filter", id="library-notes-filter-label", markup=False)
            # task-32131: a plain ``Input`` here let a SECOND "/" -- pressed
            # while the filter already had focus -- insert a literal slash
            # (Screen.on_key's "/" handling bails as soon as an Input owns
            # focus, so it never gets a chance to redirect). Reuse the rail
            # search box's widget instead of re-solving it -- but with
            # ``swallow_slash_on_focus=False`` (fix round 1 Important 4,
            # controller ruling): notes filter content can legitimately
            # contain "/" (folder-style filters like "Work/Q3"), so once
            # this box has focus "/" must be a plain typeable character,
            # not an accelerator that swallows it. The screen-level "/"
            # handler already only fires while this box is NOT focused.
            yield LibraryRailSearchInput(
                placeholder="Filter notes… (Enter)",
                id="library-notes-filter",
                value=self.filter_value,
                swallow_slash_on_focus=False,
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
                    # task-31959: the enabled spelling reserves the
                    # marker's own width, so the word holds its column
                    # when the first selection enables this in place.
                    library_disabled_action_label(
                        export_base, export_disabled, align=True
                    ),
                    id="library-notes-export-selected",
                    classes="library-canvas-action",
                    compact=True,
                )
                export_selected._library_disabled_marker_base = export_base
                export_selected._library_disabled_marker_align = True
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
            # task-4023 AC#1 (RC-07): every disabled toolbar action carries
            # the non-colour "○" marker plus an F-018 reason.
            running = list_state.operation_running
            running_tooltip = "Wait for the running notes operation to finish."
            # task-32128: the folder tree's row order is a repository
            # contract -- `page_note_placements` is ORDER BY title COLLATE
            # NOCASE and every page offset (including the deep-link
            # locator's) is computed against it -- so a Sort control there
            # could only reorder the loaded window and lie about the rest.
            # It stays on the flat list, which sorts its own records.
            sort_available = self.tree_projection is None
            sort_choices_visible = sort_available and list_state.sort_choices_visible
            # task-32127: the browse and transfer actions share ONE row, so
            # the toolbar is two rows rather than three -- but only where the
            # pane can hold both groups. Below the threshold the merged row
            # clipped its last action off the pane, which is worse than the
            # third row it saves (review round 1).
            merged, stacked = _toolbar_shape(self.pane_width, self.compact)
            action_rows: Horizontal | None = None
            if merged:
                action_rows = Horizontal(id="library-notes-action-rows")
                action_rows.styles.height = "auto"
            with action_rows or nullcontext():
                browse_actions = Horizontal(
                    id="library-notes-browse-actions", classes="ds-toolbar"
                )
                browse_actions.styles.height = "auto"
                if action_rows is not None:
                    browse_actions.styles.width = "auto"
                browse_actions.display = not sort_choices_visible
                with browse_actions:
                    yield Button(
                        library_disabled_action_label("New", running),
                        id="library-notes-new",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=running,
                        tooltip=running_tooltip if running else None,
                    )
                    if sort_available:
                        sort_base = (
                            f"Sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')}"
                        )
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
                if sort_choices_visible:
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
                    self.import_snapshot.phase
                    if self.import_snapshot is not None
                    else ""
                )
                transfer_actions = (Vertical if stacked else Horizontal)(
                    id="library-notes-transfer-actions", classes="ds-toolbar"
                )
                transfer_actions.styles.height = "auto"
                with transfer_actions:
                    for label, button_id in (
                        ("Add from files…", "library-notes-add-from-files"),
                        ("Export", "library-notes-export"),
                    ):
                        view_import = (
                            button_id == "library-notes-add-from-files"
                            and import_phase == "importing"
                        )
                        disabled = list_state.operation_running and not view_import
                        yield Button(
                            library_disabled_action_label(
                                "View import" if view_import else label, disabled
                            ),
                            id=button_id,
                            classes="library-canvas-action",
                            compact=True,
                            disabled=disabled,
                            tooltip=running_tooltip if disabled else None,
                        )
                    if self.lasting_sync_snapshot is not None and (
                        self.lasting_sync_snapshot.roots
                        or self.lasting_sync_snapshot.root_page_count > 1
                    ):
                        yield Button(
                            "Manage sync folders",
                            id="library-notes-manage-sync-folders",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=list_state.operation_running,
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
            # task-32123: the copy and the two recovery actions are stacked,
            # not laid side by side. A one-row receipt needed the title's 42
            # cells PLUS both buttons, so in a narrow list pane Undo -- the
            # only recovery path there is, with no Trash browser -- was
            # painted past the pane edge and could not be pressed.
            receipt_row = Vertical(
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
                receipt_actions = Horizontal(
                    id="library-notes-delete-receipt-actions"
                )
                receipt_actions.styles.height = "auto"
                with receipt_actions:
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
            # The opener is the row list's last row -- see
            # ``_compose_trash_opener``.
            yield from self._compose_tree_rows(list_state)
            return
        if not list_state.rows:
            yield Static(list_state.empty_copy, id="library-notes-empty", markup=False)
            yield from self._compose_trash_opener()
            return
        with Vertical(id="library-notes-list"):
            for index, row in enumerate(list_state.rows):
                # Button labels are parsed as Rich markup: escape the
                # user-supplied title so "[draft] Q3 plan [wip]" renders
                # verbatim instead of eating bracketed segments as tags
                # (or crashing on an unmatched closing tag) -- the same
                # fix class as the escaped search-history Button labels.
                title = escape_markup(row.title)
                label_rest = compose_note_row_label(title, age_label=row.age_label)
                if select_mode:
                    # Notes rows had no marker at all before select mode
                    # existed -- normal mode keeps that markerless label
                    # (no ``▸``, unlike the media/conversations rows).
                    glyph = "☑ " if row.checked else "☐ "
                    label = f"{glyph}{label_rest}"
                else:
                    label = label_rest
                # task-31945: shared row press behaviour (no 0.2s flash
                # swallowing the next click on the same row).
                button = library_row_button(
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
            yield from self._compose_trash_opener()

    def _compose_trash_opener(self) -> ComposeResult:
        """Name the Trash under the tree, and only while it holds something.

        task-32144: the delete receipt was the ONLY way back from a delete,
        so dismissing it stranded the note. This row is the standing second
        net; at zero it is absent rather than disabled -- an empty Trash has
        nothing to say and a dead affordance is worse than none.

        Composed as the row list's LAST ROW, not as a sibling after it: the
        list is ``height: 1fr``, so a sibling docked to the foot of the pane
        with a dozen blank rows between it and the tree (measured live at
        235x52) -- the same detached-affordance shape task-28015 fixed in the
        Media Trash. Every path that has no row list yields it directly.
        """
        trash = self.trash
        if trash is None or trash.total <= 0:
            return
        yield Button(
            f"Recently deleted ({trash.total})",
            id="library-notes-trash-open",
            classes="library-canvas-action",
            compact=True,
        )

    def _compose_trash(self) -> ComposeResult:
        """Render the soft-deleted notes with one Restore each, and no more.

        Restore is the only mutation this view offers: it commits through
        the same ``_undo_library_note_delete`` seam the receipt's Undo uses,
        so the row returns to its folder (or Unfiled) and the rail count
        moves exactly as an Undo would. There is deliberately no permanent
        delete here -- ADR-055 keeps destruction behind its own receipt, and
        this surface exists to recover.
        """
        trash = self.trash or LibraryNotesTrashState()
        yield Static(
            "Recently deleted",
            id="library-notes-trash-header",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            "Deleted notes stay here until you restore them. Restore puts a "
            "note back where it was; nothing is removed for good from here.",
            id="library-notes-trash-purpose",
            markup=False,
        )
        yield Button(
            _library_note_back_label(self.compact),
            id="library-notes-trash-back",
            classes="library-canvas-action",
            compact=True,
        )
        if not trash.rows:
            yield Static(
                "Nothing deleted recently. Deleted notes appear here — press "
                "Escape to go back to the list.",
                id="library-notes-trash-empty",
                markup=False,
            )
            return
        with Vertical(id="library-notes-trash-list"):
            for index, row in enumerate(trash.rows):
                trash_row = Horizontal(classes="library-notes-trash-row")
                trash_row.styles.height = "auto"
                with trash_row:
                    yield Static(
                        compose_note_row_label(row.title, age_label=row.age_label),
                        classes="library-notes-trash-row-copy",
                        markup=False,
                    )
                    button = Button(
                        "Restore",
                        id=f"library-notes-trash-restore-{index}",
                        classes=(
                            "library-canvas-action library-notes-trash-restore"
                        ),
                        compact=True,
                    )
                    button.note_id = row.note_id
                    button.note_title = row.title
                    button.note_version = row.version
                    yield button
        if trash.total > len(trash.rows):
            yield Static(
                f"Showing the {len(trash.rows)} most recently deleted of "
                f"{trash.total}. Restore one to see the rest.",
                id="library-notes-trash-more",
                markup=False,
            )

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
            yield from self._compose_trash_opener()
            return
        # task-32126: a seeded folder (Agent_Lessons) gives the tree
        # projection rows even when the library holds zero notes, so the
        # "no rows" check above never fires and the empty state never
        # renders. Render it above the tree whenever the library itself is
        # empty, independent of whether the projection has folder rows.
        # PR #2538 review (Qodo finding 5): a fresh visit renders with
        # empty_kind == "source-empty" before the bulk source-count lookup
        # resolves, while the tree's own root "folders"/"placements" slices
        # are still loading (a loading pager row, not an empty projection).
        # Skip the banner while any row is still in flight so a user who
        # does have notes never sees "No notes yet" flash ahead of the
        # load result.
        if list_state.empty_kind == "source-empty" and not any(
            row.loading for row in projection.rows
        ):
            yield Static(
                list_state.empty_copy,
                id="library-notes-empty",
                markup=False,
            )
        checked_ids = {row.note_id for row in list_state.rows if row.checked}
        # task-32137: only a title that repeats under the SAME parent earns
        # a folder suffix -- two rows with one title in two folders already
        # sit under their own folder rows, and spending the width there
        # ellipsized the semantic sync status instead.
        sibling_counts = Counter(
            (row.folder_id or "", row.label)
            for row in projection.rows
            if row.kind == "note"
        )
        duplicate_siblings = {
            sibling for sibling, count in sibling_counts.items() if count > 1
        }
        with Vertical(id="library-notes-list", classes="library-notes-tree"):
            for index, row in enumerate(projection.rows):
                indent = "  " * row.depth
                if row.kind == "pager":
                    button = _LibraryNotesTreePagerButton(
                        Text(f"{indent}{row.label}"),
                        id=row.focus_id,
                        classes="library-notes-tree-pager library-canvas-action",
                        compact=True,
                        disabled=row.disabled,
                    )
                    button.placement_id = row.placement_id
                    button.parent_folder_id = row.parent_folder_id
                    button.content_kind = row.content_kind
                    button.paging_action = row.paging_action
                    button.retry_direction = row.retry_direction
                    button.range_copy = row.range_copy
                    button.pager_status = row.status_text
                    button.action_copy = row.action_copy
                    button.paging_loading = row.loading
                    yield button
                    continue
                if row.kind in {"folder", "unfiled"}:
                    glyph = "▾" if row.expanded else "▸"
                    label = f"{indent}{glyph} {escape_markup(row.label)}"
                    # task-32126: gloss the seeded Agent_Lessons folder while
                    # the library holds zero notes -- a first-time user's
                    # first question is what this folder is and whether they
                    # made it themselves.
                    # ponytail: keyed off "the whole library is empty"
                    # rather than "this folder has no children" (no per-
                    # folder note count is loaded for a collapsed row) --
                    # exactly the scoped scenario this task covers; widen to
                    # a real per-folder empty check if Agent_Lessons ever
                    # needs the gloss while sibling notes exist elsewhere.
                    if (
                        row.label == AGENT_LESSONS_FOLDER
                        and list_state.empty_kind == "source-empty"
                    ):
                        label = f"{label} — {AGENT_LESSONS_FOLDER_GLOSS} (empty)"
                    if row.status_text:
                        label = f"{label}  {row.status_text}"
                    classes = "library-notes-folder-row"
                    if row.semantic_status == "connected":
                        classes += " library-notes-tree-connected"
                    elif row.semantic_status == "needs_attention":
                        classes += " library-notes-tree-needs-attention"
                    button = library_row_button(
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

                # A filter scatters rows out of their folders, and a
                # repeated title is otherwise an identical row: both name
                # their folder (task-32137).
                folder_label = (
                    row.breadcrumb.rsplit(" / ", 1)[0]
                    if row.breadcrumb
                    and (
                        self.filter_value
                        or (row.folder_id or "", row.label) in duplicate_siblings
                    )
                    else ""
                )
                title = indent + compose_note_row_label(
                    escape_markup(row.label),
                    folder_label=escape_markup(folder_label),
                    age_label=row.age_label,
                )
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
                button = library_row_button(
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
            yield from self._compose_trash_opener()

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
        selected_branch_stale = bool(
            selected is not None and selected.unsafe_mutation_disabled
        )
        protected_reason = (
            "This folder is managed by sync; change its sync root instead."
        )
        stale_reason = "This branch may be out of date; retry it before changing it."
        _, stacked = _toolbar_shape(self.pane_width, self.compact)
        with (Vertical if stacked else Horizontal)(
            id="library-notes-tree-actions", classes="ds-toolbar"
        ):
            yield Button(
                "New folder",
                id="library-notes-folder-new",
                classes="library-canvas-action",
                compact=True,
                disabled=(
                    operation_running
                    or selected_folder_protected
                    or selected_branch_stale
                ),
                tooltip=(
                    stale_reason
                    if selected_branch_stale
                    else protected_reason
                    if selected_folder_protected
                    else None
                ),
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
                        disabled=(
                            operation_running
                            or selected.protected
                            or selected_branch_stale
                        ),
                        tooltip=(
                            stale_reason
                            if selected_branch_stale
                            else protected_reason
                            if selected.protected
                            else None
                        ),
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
                    disabled=operation_running or selected_branch_stale,
                    tooltip=stale_reason if selected_branch_stale else None,
                )
                yield Button(
                    "Move note",
                    id="library-notes-placement-move",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=operation_running or protected or selected_branch_stale,
                    tooltip=(
                        stale_reason
                        if selected_branch_stale
                        else protected_placement_reason
                        if protected
                        else None
                    ),
                )
                yield Button(
                    "Remove placement",
                    id="library-notes-placement-remove",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=(
                        operation_running
                        or protected
                        or selected_branch_stale
                        or not selected.membership_id
                    ),
                    tooltip=(
                        stale_reason
                        if selected_branch_stale
                        else (
                            protected_placement_reason
                            if protected
                            else (
                                "Unfiled is shown automatically; move the note into a folder."
                                if not selected.membership_id
                                else None
                            )
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
    def _backlink_buttons(
        backlinks: tuple[tuple[str, str], ...],
    ) -> list[Button]:
        """Build one activatable row per inbound link (task-32145).

        The single renderer for both paths -- compose, and the remount in
        ``apply_session_state`` for backlinks that arrive after the editor
        is already on screen (their loader runs after the note opens, and a
        recompose is deferred while the reader is typing).

        Args:
            backlinks: ``(note_id, title)`` rows, possibly one over the cap.

        Returns:
            Row buttons carrying their own ``note_id``, capped for display.
        """
        buttons: list[Button] = []
        for note_id, title in backlinks[:LIBRARY_NOTE_BACKLINK_DISPLAY_CAP]:
            button = library_row_button(
                escape_markup(ellipsize_note_title_cells(title, 60) or "Untitled"),
                classes="library-canvas-action library-note-backlink",
                compact=True,
            )
            button.note_id = note_id
            buttons.append(button)
        return buttons

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
        channels = presentation_state.status_channels or NotesStatusChannels(
            status_line or "Saved",
            "Database Notes · Library database",
        )

        # File-synced notes may carry YAML front matter; consume it instead
        # of rendering the delimiter block as note content.
        from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory

        # task-32139: Edit/Preview said "‹ Notes", Info said "‹ Note" (two
        # wordings for the identical Back action, live-caught at 235x52),
        # and BOTH said "‹ Notes" on a compact terminal where the guide
        # documents "‹ Back to list" (60x24). One label now, sized by
        # ``self.compact`` like the guide's own compact-vs-wide split.
        back_label = _library_note_back_label(self.compact)
        with Horizontal(id="library-note-heading"):
            yield Button(
                back_label,
                id="library-note-back",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                back_label,
                id="library-note-context-back",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static(
                ellipsize_note_title_cells(title, 72),
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
            yield Static(
                channels.authority_git,
                id="library-note-authority-git-status",
                markup=False,
            )
        yield Static(
            "Included in bulk selection"
            if presentation_state.bulk_included
            else "Not included in bulk selection",
            id="library-note-bulk-status",
            classes="destination-purpose",
            markup=False,
        )
        with Horizontal(id="library-note-header-second-row"):
            yield Static(
                channels.content_recovery,
                id="library-note-status",
                markup=False,
            )
            primary_actions = Horizontal(
                id="library-note-primary-actions", classes="ds-toolbar"
            )
            primary_actions.styles.height = "auto"
            with primary_actions:
                with Horizontal(id="library-note-mode-controls", classes="ds-toolbar"):
                    yield Button(
                        "Edit",
                        id="library-note-edit",
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
                        "Info",
                        id="library-note-context",
                        classes="library-canvas-action",
                        compact=True,
                    )
                with Horizontal(id="library-note-task-actions", classes="ds-toolbar"):
                    yield Button(
                        "Save",
                        id="library-note-save",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield Button(
                        "Use in Console",
                        id="library-note-use-in-console",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    discard_new = Button(
                        "Discard" if self.compact else "Discard new note",
                        id="library-note-discard-new",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    discard_new.display = presentation_state.discard_new_note
                    discard_new.disabled = presentation_state.destructive_running
                    yield discard_new
        with Vertical(id="library-note-editor-region"):
            with Horizontal(id="library-note-title-row"):
                yield Static("Title", id="library-note-title-label", markup=False)
                yield NoteEditorInput(
                    value="" if self.title_placeholder_only else title,
                    placeholder="Untitled" if self.title_placeholder_only else "",
                    id="library-note-title",
                )
            yield Static("Body", id="library-note-body-label", markup=False)
            yield NoteEditorTextArea(content, id="library-note-body")

        with VerticalScroll(id="library-note-preview-region", can_focus=True):
            # task-32142 AC#1: the shared heading row's title Static (above)
            # sits in a crowded strip with the mode buttons and Back --
            # easy to miss, and NOT part of the scrolling content, so it
            # never reads as the document's own title the way Edit's Title
            # field does. This one renders INSIDE the preview, immediately
            # above the rendered body, like a document heading.
            yield Static(
                ellipsize_note_title_cells(title, 72),
                id="library-note-preview-body-title",
                classes="destination-section",
                markup=False,
            )
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
                yield NoteEditorInput(
                    value=keywords_text,
                    placeholder="Comma-separated keywords",
                    id="library-note-context-keywords",
                )
            yield Static(metadata_line, id="library-note-context-meta", markup=False)
            backlinks = (
                presentation_state.backlinks if presentation_state is not None else ()
            )
            self._rendered_backlinks = tuple(backlinks)
            yield Static(
                library_note_backlink_header(
                    self._rendered_backlinks,
                    presentation_state.backlinks_status
                    if presentation_state is not None
                    else "loading",
                ),
                id="library-note-context-backlinks-title",
                markup=False,
            )
            backlink_rows = Vertical(id="library-note-context-backlinks")
            # Auto height or the empty container claims the whole Info
            # scroll region and pushes Reuse & Export off the pane.
            backlink_rows.styles.height = "auto"
            with backlink_rows:
                yield from self._backlink_buttons(self._rendered_backlinks)
            yield Static("Reuse & Export", classes="destination-section", markup=False)
            legacy_use_in_console = Button(
                "Use in Console",
                id="library-note-context-use-in-console",
                classes="library-canvas-action",
                compact=True,
            )
            # Retain the incumbent selector/handler for compatibility while the
            # single visible affordance lives in the primary task group.
            legacy_use_in_console.display = False
            yield legacy_use_in_console
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
            yield Static("Danger", classes="destination-section", markup=False)
            yield Button(
                "Delete",
                id="library-note-context-delete",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
        yield Static(
            presentation_state.transfer_status,
            id="library-note-transfer-status",
            markup=False,
        )

        with Vertical(id="library-note-wide-utilities"):
            yield Static("Keywords", id="library-note-keywords-label", markup=False)
            yield NoteEditorInput(
                value=keywords_text,
                placeholder="Comma-separated keywords",
                id="library-note-keywords",
            )
            yield Static(metadata_line, id="library-note-meta", markup=False)
            wide_actions = Horizontal(classes="ds-toolbar")
            wide_actions.styles.height = "auto"
            with wide_actions:
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
        if self.mode == "lasting_add":
            self.call_after_refresh(self._focus_initial_lasting_sync_control)

    def _focus_initial_lasting_sync_control(self) -> None:
        """Focus the safe first control once, after the wrapper initially mounts."""

        canvases = self.query("#library-notes-lasting-add-canvas")
        if canvases:
            canvases.first(LibraryNotesAddFromFilesCanvas).focus_first_safe_control()

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
        self.styles.min_width = 0 if compact else 40
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
                    export_base, button.disabled, align=True
                )
            return
        header_rows = self.query("#library-note-header-second-row")
        if header_rows:
            heading = self.query_one("#library-note-heading")
            second_row = header_rows.first(Horizontal)
            status = self.query_one("#library-note-status", Static)
            authority = self.query_one("#library-note-authority-git-status", Static)
            primary = self.query_one("#library-note-primary-actions", Horizontal)
            mode_controls = self.query_one("#library-note-mode-controls", Horizontal)
            task_actions = self.query_one("#library-note-task-actions", Horizontal)
            heading.styles.layout = "horizontal"
            heading.styles.height = 1 if compact else 3
            heading.styles.min_height = 1 if compact else 3
            heading.styles.max_height = 1 if compact else 3
            second_row.styles.layout = "vertical" if compact else "horizontal"
            second_row.styles.height = "auto" if compact else 3
            second_row.styles.min_height = 3
            second_row.styles.max_height = 5 if compact else 3
            status.styles.width = "1fr"
            status.styles.height = "auto" if compact else 3
            status.styles.min_height = 1 if compact else 3
            status.styles.max_height = 3
            status.styles.text_wrap = "wrap"
            status.styles.text_overflow = "clip"
            primary.styles.layout = "vertical" if compact else "horizontal"
            primary.styles.width = "100%" if compact else "auto"
            primary.styles.height = 2 if compact else 3
            primary.styles.min_height = 2 if compact else 3
            primary.styles.max_height = 2 if compact else 3
            for actions in (mode_controls, task_actions):
                actions.styles.width = "100%" if compact else "auto"
                actions.styles.height = 1 if compact else 3
                actions.styles.min_height = 1 if compact else 3
                actions.styles.max_height = 1 if compact else 3
            authority.styles.width = 18 if compact else "auto"
            authority.styles.min_width = 12 if compact else 0
            authority.styles.max_width = 18 if compact else None
            authority.styles.height = 1 if compact else 3
            authority.styles.text_wrap = "nowrap" if compact else "wrap"
            authority.styles.text_overflow = "ellipsis" if compact else "clip"
            for button in primary.query(Button):
                button.styles.width = "auto"
                button.styles.height = 1 if compact else 3
                button.styles.min_height = 1 if compact else 3
                button.styles.max_height = 1 if compact else 3
        discard_new = self.query("#library-note-discard-new")
        if discard_new:
            discard_new.first(Button).label = (
                "Discard" if compact else "Discard new note"
            )

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
        authority = self.query_one(f"#{self.authority_id}", Static)
        authority_copy = self._authority_copy()
        if self._static_text(authority) != authority_copy:
            authority.update(authority_copy)
        snapshot = state.snapshot
        conflict = state.conflict
        confirming_delete = state.confirming_delete and not conflict
        bulk_read_only = state.bulk_read_only
        # task-32132: Delete is only reachable from the Info Danger section
        # (the Edit pane's own Delete lives in ``library-note-wide-
        # utilities``, permanently hidden below). Confirming used to force
        # ``show_context`` off unconditionally, snapping the pane to Edit --
        # "delete this note?" painted 14 rows away, under a body editor the
        # user never opened. Info stays put while confirming; only Preview
        # (which never hosts a Delete button) still yields to Edit.
        show_context = (
            state.region == "context"
            and not conflict
            and not bulk_read_only
        )
        show_preview = bulk_read_only or (
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
        # task-32062: the field the reader is typing in is its OWN authority.
        # This patch used to overwrite it from a snapshot that could be one
        # keystroke behind -- and assigning `Input.value` clamps the cursor to
        # the shorter text, so the rest of the sentence was then inserted at
        # that stale position. Live: "My first note", Tab, a body typed within
        # ~0.4 s stored the title "Mhello from jordan, testing the libraryy
        # first note" with an empty body. A focused field is skipped; its own
        # Changed events are what the snapshot is built from anyway.
        if title_input.value != presented_title and not title_input.has_focus:
            with title_input.prevent(Input.Changed):
                title_input.value = presented_title
        title_input.placeholder = "Untitled" if self.title_placeholder_only else ""
        if body_input.text != snapshot.body and not body_input.has_focus:
            with body_input.prevent(TextArea.Changed):
                body_input.text = snapshot.body
        # Same rule as the title above: a keyword box the reader is typing in
        # is its own authority, and the snapshot is built from its own Changed
        # events anyway (review of PR #2531).
        if (
            wide_keywords.value != snapshot.keywords_text
            and not wide_keywords.has_focus
        ):
            with wide_keywords.prevent(Input.Changed):
                wide_keywords.value = snapshot.keywords_text
        if (
            context_keywords.value != snapshot.keywords_text
            and not context_keywords.has_focus
        ):
            with context_keywords.prevent(Input.Changed):
                context_keywords.value = snapshot.keywords_text

        title_width = 52 if state.compact else 72
        title = ellipsize_note_title_cells(snapshot.title, title_width)
        for selector in (
            "#library-note-editor-title",
            "#library-note-preview-title",
            "#library-note-context-title",
            "#library-note-preview-body-title",
        ):
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
        channels = state.status_channels or NotesStatusChannels(
            state.status_line or "Saved",
            "Database Notes · Library database",
        )
        content_copy = channels.content_recovery
        if channels.safe_next_action:
            content_copy = f"{content_copy} Next: {channels.safe_next_action}."
        for selector in ("#library-note-status", "#library-note-context-status"):
            widget = self.query_one(selector, Static)
            if self._static_text(widget) != content_copy:
                widget.update(content_copy)
        authority_status = self.query_one("#library-note-authority-git-status", Static)
        if self._static_text(authority_status) != channels.authority_git:
            authority_status.update(channels.authority_git)
        for selector in ("#library-note-meta", "#library-note-context-meta"):
            widget = self.query_one(selector, Static)
            if self._static_text(widget) != state.metadata_line:
                widget.update(state.metadata_line)
        # task-32145: backlinks are loaded by their own worker AFTER the note
        # opens, so they land on an editor that is already composed -- and a
        # recompose is deferred for as long as the reader owns a field
        # (task-32062). Reconciling them here is what makes them appear at
        # all: switching Edit -> Info is a `display` flip on this same
        # composition, not a rebuild.
        backlinks = tuple(state.backlinks)
        if backlinks != self._rendered_backlinks:
            self._rendered_backlinks = backlinks
            container = self.query_one("#library-note-context-backlinks", Vertical)
            container.remove_children()
            rows = self._backlink_buttons(backlinks)
            if rows:
                container.mount_all(rows)
        backlink_title = self.query_one(
            "#library-note-context-backlinks-title", Static
        )
        backlink_copy = library_note_backlink_header(
            backlinks, state.backlinks_status
        )
        if self._static_text(backlink_title) != backlink_copy:
            backlink_title.update(backlink_copy)
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
        # task-32139: one Back label, sized by compact -- see the matching
        # compose-time comment above.
        back_label = _library_note_back_label(state.compact)
        back_button = self.query_one("#library-note-back", Button)
        if str(back_button.label) != back_label:
            back_button.label = back_label
        back_button.display = not show_context and not bulk_read_only
        back_button.disabled = confirming_delete
        context_back_button = self.query_one("#library-note-context-back", Button)
        if str(context_back_button.label) != back_label:
            context_back_button.label = back_label
        context_back_button.display = show_context
        # PR #2547 review (Qodo finding 4): Back was left out of the
        # disabled-selector loops below, so it stayed live behind the
        # confirmation prompt. A press ran the Back handler, which clears
        # ``_library_note_context`` without cancelling the pending
        # admission -- displacing the prompt instead of leaving Info in
        # place like every other Danger/Reuse & Export action.
        context_back_button.disabled = confirming_delete
        self.query_one("#library-note-editor-title").display = show_editor
        self.query_one("#library-note-preview-title").display = show_preview
        self.query_one("#library-note-context-title").display = show_context
        bulk_status = self.query_one("#library-note-bulk-status", Static)
        bulk_status.display = bulk_read_only
        bulk_copy = (
            "Read-only preview · Included in bulk selection"
            if state.bulk_included
            else "Read-only preview · Not included in bulk selection"
        )
        if self._static_text(bulk_status) != bulk_copy:
            bulk_status.update(bulk_copy)
        self.query_one("#library-note-editor-region").display = show_editor
        self.query_one("#library-note-preview-region").display = show_preview
        # task-32142 AC#2: this Static repeats the identical text
        # ``#library-note-status`` (the header-second-row status, visible
        # in every mode) already shows -- Info printed "Saved" twice, once
        # in the header and once again immediately above the panel.
        self.query_one("#library-note-context-status").display = False
        self.query_one("#library-note-context-region").display = show_context
        self.query_one("#library-note-edit", Button).set_class(show_editor, "is-active")
        self.query_one("#library-note-preview", Button).set_class(
            show_preview and not bulk_read_only, "is-active"
        )
        self.query_one("#library-note-context", Button).set_class(
            show_context, "is-active"
        )
        self.query_one("#library-note-primary-actions").display = (
            not conflict and not confirming_delete
        )
        self.query_one("#library-note-wide-utilities").display = False
        self.query_one("#library-note-conflict-region").display = conflict
        self.query_one("#library-note-delete-confirmation").display = confirming_delete

        locked = confirming_delete or state.destructive_running or bulk_read_only
        title_input.disabled = not show_editor or locked
        body_input.disabled = not show_editor or locked
        wide_keywords.disabled = state.compact or show_context or locked
        context_keywords.disabled = not show_context or locked
        preview_body.can_focus = False
        self.query_one("#library-note-preview-region").can_focus = show_preview
        self.query_one("#library-note-context-region").can_focus = show_context

        for selector in (
            "#library-note-edit",
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
            self.query_one(selector, Button).disabled = (
                state.destructive_running or bulk_read_only or confirming_delete
            )
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
                state.destructive_running
                or state.transfer_running
                or bulk_read_only
                # task-32132 fix round 1 Important 5: Info stays visible
                # while confirming (this fix's own AC#1), so its Danger/
                # Reuse & Export buttons -- Delete, Copy, Export, Use in
                # Console -- were still live behind the confirmation
                # prompt; a press could navigate away (Use in Console) or
                # mutate (Copy/Export) with the delete admission still
                # pending.
                or confirming_delete
            )
        discard_new = self.query_one("#library-note-discard-new", Button)
        discard_new.display = state.discard_new_note
        discard_new.disabled = state.destructive_running or bulk_read_only
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
