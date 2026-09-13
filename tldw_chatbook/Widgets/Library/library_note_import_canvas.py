"""Render-only canvas for the reviewed one-time Notes import workflow."""

from __future__ import annotations

from itertools import groupby
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Button, Collapsible, Input, Static

from tldw_chatbook.Library.library_note_import_state import (
    UNIFORM_RUN_MIN,
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_GLYPH_SELECTED,
    LIBRARY_GLYPH_UNSELECTED,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    NON_IMPORTABLE_CLASSIFICATIONS,
    REVIEW_CLASSIFICATION_ORDER,
)
from tldw_chatbook.Utils.Utils import elide_path_middle
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)


_CLASSIFICATION_LABELS = {
    "new": "New",
    "unchanged_repeat": "Unchanged repeat",
    "changed_repeat": "Changed repeat",
    "uncertain_match": "Uncertain match",
    "unsupported": "Unsupported",
    "skipped": "Skipped",
    "empty": "Empty",
    "failed": "Failed",
}

# Classifications that carry no payload: their row states the reason instead
# of an effect, and their group offers no Create all (task-32130/32135).
# Derived from the planner's own set (task-32176): the same four names were
# spelled out by hand here, in the plan model and in the parser's issue
# contract, which is three places to forget when a classification is added.
_NON_IMPORTABLE = frozenset(
    classification.value for classification in NON_IMPORTABLE_CLASSIFICATIONS
)

_REVIEW_ORDER = tuple(
    classification.value for classification in REVIEW_CLASSIFICATION_ORDER
)
"""The pager's group order, which this canvas renders in (task-32250)."""


def _choice_label(*, selected: bool, text: str) -> str:
    """Return a monochrome-readable selected/unselected action label.

    task-32235 AC#2: these are checkboxes, so they take the legend's
    checkbox pair. They used to render "✓"/"○", which collided with the
    settled-outcome "✓" one row above them in the same receipt and with the
    blocked-action "○" on the buttons beside them.
    """
    glyph = LIBRARY_GLYPH_SELECTED if selected else LIBRARY_GLYPH_UNSELECTED
    return f"{glyph} {text}"


def _disabled_action_label(text: str, *, disabled: bool, reason: str = "") -> str:
    """Keep an unavailable action's reason readable at the control itself.

    task-32257: the reason was on the tooltip only, so the control stated
    that it was unavailable and nothing else. The grammar is the one this
    screen already uses four panels away ("Unavailable — server sync-folder
    capability not installed"): the blocker is the label's own text.
    """
    if not disabled:
        return text
    if not reason:
        return f"{text} unavailable"
    return f"{text} unavailable — {reason.rstrip('.')}"


_ROW_NAME_BUDGET = 56
"""Display-width budget for the path at the head of one review row.

task-32250: the row's job is to state the resulting title, its keywords and
its link count. Those came last and were the first thing the row lost.
"""

# The threshold is the pager's (task-32250): it budgets a page by rendered
# rows, and a collapsed run renders as one, so a second threshold here would
# make every page count wrong.
_UNIFORM_RUN_MIN = UNIFORM_RUN_MIN


def _bounded_row_name(name: str) -> str:
    """Keep a review row's path recognizable without spending the whole row."""
    return elide_path_middle(name, budget=_ROW_NAME_BUDGET)


def _group_heading(label: str, *, rendered: int, total: int) -> str:
    """Name what a group heading's count means on this page.

    task-32250: "New (23)" on page 1 and "New (22)" on page 2 were the same
    words for different numbers, with the group's real size nowhere on screen.
    """
    if rendered >= total:
        return f"{label} ({total})"
    return f"{label} ({rendered} of {total} on this page)"


def _run_key(item: LibraryNoteImportItemSnapshot) -> tuple[str, ...]:
    """Return what makes two review rows interchangeable at a glance."""
    folder, _, _ = item.name.rpartition("/")
    return (
        folder,
        item.classification,
        item.action,
        item.reason if item.classification in _NON_IMPORTABLE else "",
        item.membership_summary,
    )


def _uniform_runs(
    items: tuple[LibraryNoteImportItemSnapshot, ...],
) -> tuple[tuple[LibraryNoteImportItemSnapshot, ...], ...]:
    """Split one rendered group into consecutive interchangeable runs."""
    return tuple(tuple(run) for _, run in groupby(items, key=_run_key))


def _run_disclosure(title: str, *, dom_token: str) -> Collapsible:
    """Return a one-line disclosure for a collapsed run of identical rows.

    The app-wide ``Collapsible`` rule (a round border, a 3-row title, a 3-row
    floor and a bottom margin) is app-tier and beats this widget's own
    BUNDLED_CSS whatever its specificity -- five lines of chrome for a summary
    that has to cost the page one, because the pager budgets by rendered rows.
    Inline styles are the one tier above it (task-32250).
    """
    disclosure = Collapsible(
        # A vault folder name is untrusted text: `CollapsibleTitle` runs it
        # through `Content.from_text`, whose markup parameter defaults to ON,
        # so a folder called "[@click=app.quit]" turned the whole summary into
        # an action link and dropped its own name from the row. A `Content`
        # instance comes back from `from_text` unmodified -- this is the
        # markup=False every other Static on this canvas already sets.
        title=Content(title),
        id=f"note-import-run-{dom_token}",
        classes="note-import-run",
        collapsed=True,
    )
    disclosure.styles.min_height = 1
    disclosure.styles.margin = 0
    disclosure.styles.padding = 0
    disclosure.styles.border = ("none", "transparent")
    return disclosure


def _run_summary(
    run: tuple[LibraryNoteImportItemSnapshot, ...],
    total: int | None = None,
) -> str:
    """Return the one row that stands for a collapsed run of identical rows.

    Args:
        run: The rows of this run that are on the rendered page.
        total: How many the whole run holds, when the page only shows part of
            it. A run bigger than the mount ceiling is the one case a page
            break falls inside one, and its summary then has to say which of
            the two numbers it means -- "200 files" on one page and "50 files"
            on the next is the shape task-32250 was filed about.

    Returns:
        The summary line for the collapsed disclosure's title.
    """
    first = run[0]
    folder, _, _ = first.name.rpartition("/")
    where = _bounded_row_name(folder) if folder else "the selection"
    count = (
        f"{len(run)} files"
        if total is None or total <= len(run)
        else f"{len(run)} of {total} files"
    )
    if first.classification in _NON_IMPORTABLE:
        return f"{where} · {count} · {first.reason.rstrip(' .')}"
    verb = "Skip" if first.action == "skip" else "Create"
    destination = first.membership_summary.rstrip(" .")
    return f"{where} · {count} · {verb} all · {destination}"


_SOURCE_NAME_BUDGET = 48
"""Shared display-width budget for a selected source name (review round 4:

was repeated as the literal 48 twice plus a hand-derived 47 in the
head-truncate branch below -- one named constant keeps the folder-path and
bare-filename branches from drifting to different limits independently).
"""


def _bounded_source_name(name: str) -> str:
    """Keep one selected source name useful without dominating compact layouts.

    A folder's absolute path (contains "/" or "\\\\" -- a folder selection
    is projected as a native ``str(Path)``, which is backslash-separated on
    Windows; review round 4) middle-elides (task-32122 Step 3) so it keeps
    its basename -- the name a user actually picked -- intact instead of
    showing an unrecognizable path prefix. A bare filename (the files list;
    no separator) has no basename/prefix split to preserve, so
    `elide_path_middle` would fall through to keeping its *tail* instead --
    a silent behavior change for the files list (review round 2 escalated
    minor). Head-truncate those as before.
    """
    if "/" in name or "\\" in name:
        return elide_path_middle(name, budget=_SOURCE_NAME_BUDGET)
    if len(name) <= _SOURCE_NAME_BUDGET:
        return name
    return f"{name[: _SOURCE_NAME_BUDGET - 1]}…"


class _ImportBody(VerticalScroll):
    """Keyboard-focusable scroll owner for the changing import detail."""

    can_focus = True


class LibraryNoteImportCanvas(PostRecomposeCallback, Vertical):
    """Render one immutable import snapshot and post typed physical intents."""

    BUNDLED_CSS = """
    $ds-status-error-readable: $text-error;

    LibraryNoteImportCanvas {
        width: 1fr;
        min-width: 40;
        height: 1fr;
        overflow: hidden;
    }

    LibraryNoteImportCanvas #note-import-heading,
    LibraryNoteImportCanvas #note-import-status,
    LibraryNoteImportCanvas #note-import-overflow-hint,
    LibraryNoteImportCanvas .note-import-primary {
        height: auto;
    }

    LibraryNoteImportCanvas #note-import-body {
        height: 1fr;
        overflow-y: auto;
    }

    LibraryNoteImportCanvas Static,
    LibraryNoteImportCanvas Input,
    LibraryNoteImportCanvas Button {
        width: 1fr;
    }

    LibraryNoteImportCanvas .note-import-group-heading {
        text-style: bold;
        width: 1fr;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    LibraryNoteImportCanvas .note-import-item-name {
        text-style: bold;
    }

    LibraryNoteImportCanvas .note-import-quiet {
        color: $text-muted;
    }

    LibraryNoteImportCanvas .note-import-error {
        color: $ds-status-error-readable;
        text-style: bold;
    }

    /* task-32135: one review row is one line -- path, effect and destination
       beside the controls that change them. */
    /* An empty container defaults to 1fr and would push every group to the
       bottom of the body when no vault is detected and the toggle is absent. */
    LibraryNoteImportCanvas #notes-import-review-options {
        height: auto;
    }

    LibraryNoteImportCanvas .note-import-row {
        height: 1;
        width: 1fr;
    }

    LibraryNoteImportCanvas .note-import-group-row {
        height: 1;
        width: 1fr;
        margin-top: 1;
    }

    LibraryNoteImportCanvas .note-import-row-text {
        width: 1fr;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    LibraryNoteImportCanvas .note-import-row-destination {
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    LibraryNoteImportCanvas .note-import-row-action {
        width: auto;
        min-width: 6;
        margin-left: 1;
    }

    LibraryNoteImportCanvas .note-import-skipped-row {
        height: auto;
        color: $text-muted;
    }
    """

    class AddSourceRequested(Message):
        """Request one more physical file selection."""

    class ChangeSourceRequested(Message):
        """Request replacing the current selection with a new picker result."""

    class ClearSourceRequested(Message):
        """Request dropping the current selection without leaving the flow."""

    class GroupActionRequested(Message):
        """Report one Skip all / Create all over a rendered review group."""

        def __init__(self, classification: str, action: str) -> None:
            """Carry one group bulk action to the owning controller.

            Args:
                classification: The ``ImportClassification`` value naming the
                    pressed group's heading.
                action: The ``ImportAction`` value to apply to that group.
                    The controller validates both and refuses an unknown one.
            """
            super().__init__()
            self.classification = classification
            self.action = action

    class DestinationChanged(Message):
        """Report the proposed, not-yet-created Notes destination."""

        def __init__(self, destination: str) -> None:
            super().__init__()
            self.destination = destination

    class CheckRequested(Message):
        """Request read-only discovery and planning."""

    class ObsidianModeToggled(Message):
        """Report the requested Obsidian-vault reading mode."""

        def __init__(self, enabled: bool) -> None:
            """Carry one toggle press.

            Args:
                enabled: True when the review should read the source as a vault.
            """
            super().__init__()
            self.enabled = enabled

    class CollisionChoiceRequested(Message):
        """Report one explicit imported-root collision choice."""

        def __init__(self, choice: str) -> None:
            super().__init__()
            self.choice = choice

    class CollisionNameChanged(Message):
        """Report the proposed replacement root label."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    class ItemActionRequested(Message):
        """Report one item-scoped Skip/Create/Update selection."""

        def __init__(self, item_id: str, action: str) -> None:
            super().__init__()
            self.item_id = item_id
            self.action = action

    class ItemChoiceRequested(Message):
        """Report an independent content or membership decision."""

        def __init__(self, item_id: str, choice: str, enabled: bool) -> None:
            super().__init__()
            self.item_id = item_id
            self.choice = choice
            self.enabled = enabled

    class UncertainMatchConfirmed(Message):
        """Request confirmation of one uncertain existing-note match."""

        def __init__(self, item_id: str) -> None:
            super().__init__()
            self.item_id = item_id

    class ImportRequested(Message):
        """Request approval and execution of the exact reviewed plan."""

    class CancelRequested(Message):
        """Request cooperative cancellation of checking or execution."""

    class RetryRequested(Message):
        """Request retry of only receipt-reported retryable failures."""

    class PageRequested(Message):
        """Request a bounded preview-page change."""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    def __init__(
        self,
        snapshot: LibraryNoteImportSnapshot,
        *,
        compact: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.snapshot = snapshot
        # A compact shell clips the one-line row, so it repeats the
        # destination underneath (review of task-32135).
        # ponytail: read at compose time only. A resize across the compact
        # threshold mid-review leaves the second line stale until the next
        # recompose -- recomposing on the flag instead unmounted the
        # destination and collision inputs while the user was typing in them.
        self.compact = compact
        self._destination_value = snapshot.destination
        self._collision_name = (
            snapshot.collision_rename_input or snapshot.collision_name
        )

    def sync_state(self, snapshot: LibraryNoteImportSnapshot) -> None:
        """Apply one snapshot without replacing an actively edited input."""
        previous = self.snapshot
        self.snapshot = snapshot
        self._destination_value = snapshot.destination
        self._collision_name = (
            snapshot.collision_rename_input or snapshot.collision_name
        )
        if not self.is_attached or previous.phase != snapshot.phase:
            self.refresh(recompose=True)
            return
        if snapshot.phase in {"select", "destination"}:
            self._sync_destination_controls(snapshot)
            return
        collision_only = (
            snapshot.phase == "review"
            and previous.preview_items == snapshot.preview_items
            and previous.page == snapshot.page
        )
        if collision_only:
            self._sync_collision_controls(snapshot)
            return
        self.refresh(recompose=True)

    def _sync_destination_controls(self, snapshot: LibraryNoteImportSnapshot) -> None:
        try:
            destination = self.query_one("#note-import-destination", Input)
            if destination.value != snapshot.destination:
                destination.value = snapshot.destination
            self.query_one("#note-import-destination-error", Static).update(
                snapshot.destination_error
            )
            check = self.query_one("#note-import-check", Button)
            check.disabled = not snapshot.can_check
            check.label = _disabled_action_label(
                "Check selection",
                disabled=not snapshot.can_check,
                reason=snapshot.check_disabled_reason,
            )
            check.tooltip = snapshot.check_disabled_reason or (
                "Check the selected sources without changing Notes."
            )
        except Exception:
            self.refresh(recompose=True)

    def _sync_collision_controls(self, snapshot: LibraryNoteImportSnapshot) -> None:
        try:
            self.query_one("#note-import-collision-heading", Static).update(
                f"Folder collision: {snapshot.collision_name}"
            )
            self.query_one("#note-import-collision-reason", Static).update(
                snapshot.collision_reason
            )
            rename_input = self.query_one("#note-import-collision-name", Input)
            if rename_input.value != snapshot.collision_rename_input:
                rename_input.value = snapshot.collision_rename_input
            self.query_one("#note-import-collision-rename-error", Static).update(
                snapshot.collision_rename_error
            )
            rename = self.query_one("#note-import-collision-rename", Button)
            rename.disabled = not snapshot.collision_rename_available
            for choice, button_id, label in (
                ("use_existing", "use-existing", "Use existing folder"),
                ("unique_sibling", "unique", "Create a unique sibling"),
                ("renamed_root", "rename", "Use another name"),
            ):
                button = self.query_one(f"#note-import-collision-{button_id}", Button)
                button.label = _choice_label(
                    selected=snapshot.collision_choice == choice,
                    text=label,
                )
            submit = self.query_one("#note-import-import", Button)
            submit.disabled = not snapshot.can_import
            submit.label = _disabled_action_label(
                "Import selected items",
                disabled=not snapshot.can_import,
                reason=snapshot.import_disabled_reason,
            )
            submit.tooltip = snapshot.import_disabled_reason or (
                "Import the exact choices shown in this review."
            )
        except Exception:
            self.refresh(recompose=True)

    def on_mount(self) -> None:
        self._tighten_run_disclosures()
        self.call_after_refresh(self._update_overflow_hint)

    def _after_recompose(self) -> None:
        self._tighten_run_disclosures()
        self.call_after_refresh(self._update_overflow_hint)

    def _tighten_run_disclosures(self) -> None:
        """Keep a collapsed run's title one line, as the pager budgeted for."""
        for title in self.query(".note-import-run > CollapsibleTitle"):
            title.styles.height = 1
            title.styles.padding = 0

    def _update_overflow_hint(self) -> None:
        try:
            body = self.query_one("#note-import-body", VerticalScroll)
            hint = self.query_one("#note-import-overflow-hint", Static)
            hint.display = body.virtual_size.height > body.container_size.height
        except Exception:
            return

    def compose(self) -> ComposeResult:
        state = self.snapshot
        yield Static(
            "Import once",
            id="note-import-heading",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            state.status_line,
            id="note-import-status",
            markup=False,
        )
        with _ImportBody(id="note-import-body"):
            if state.phase in {"select", "destination"}:
                yield from self._compose_selection(state)
                # task-32259: "Check selection" acts on the summary three
                # rows up, and floating it under a `1fr` body put it at
                # screen row 49 of 52 with a selection summary at rows
                # 7-11. This phase's content is bounded (a summary line,
                # up to three source buttons, one destination field), so
                # the action stands with it. The scrolling phases below
                # keep the pinned floor -- their lists are unbounded and
                # an action that scrolls away is worse than a far one.
                yield from self._compose_primary_action(state)
            elif state.phase == "review":
                yield from self._compose_review(state)
            elif state.phase == "importing":
                yield from self._compose_importing(state)
            elif state.phase == "receipt":
                yield from self._compose_receipt(state)
        hint = Static(
            "More below — focus this panel and use Up/Down to scroll.",
            id="note-import-overflow-hint",
            classes="note-import-quiet",
            markup=False,
        )
        hint.display = False
        yield hint
        if state.phase not in {"select", "destination"}:
            yield from self._compose_primary_action(state)

    def _compose_primary_action(
        self, state: LibraryNoteImportSnapshot
    ) -> ComposeResult:
        if state.phase in {"select", "destination"}:
            check = Button(
                _disabled_action_label(
                    "Check selection",
                    disabled=not state.can_check,
                    reason=state.check_disabled_reason,
                ),
                id="note-import-check",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_check,
            )
            check.tooltip = state.check_disabled_reason or (
                "Check the selected sources without changing Notes."
            )
            yield check
        elif state.phase in {"checking", "importing"}:
            yield Button(
                (
                    "Cancel check"
                    if state.phase == "checking" and state.can_cancel
                    else "Cancel import"
                    if state.can_cancel
                    else "Stopping…"
                ),
                id="note-import-cancel",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_cancel,
            )
        elif state.phase == "review":
            submit = Button(
                _disabled_action_label(
                    "Import selected items",
                    disabled=not state.can_import,
                    reason=state.import_disabled_reason,
                ),
                id="note-import-import",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_import,
            )
            submit.tooltip = state.import_disabled_reason or (
                "Import the exact choices shown in this review."
            )
            yield submit
        elif state.phase == "receipt" and (
            state.retry_available or state.retryable_failures
        ):
            noun = "failure" if state.retryable_failures == 1 else "failures"
            yield Button(
                state.retry_label or f"Retry {state.retryable_failures} {noun}",
                id="note-import-retry",
                classes="library-canvas-action note-import-primary",
                compact=True,
            )

    def _compose_selection(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        count = len(state.selected_names)
        if not count:
            source_copy = "No source selected."
        elif state.selection_kind == "folder":
            source_copy = (
                f"1 folder selected: {_bounded_source_name(state.selected_names[0])}"
            )
        else:
            noun = "file" if count == 1 else "files"
            visible_names = tuple(
                _bounded_source_name(name) for name in state.selected_names[:3]
            )
            remainder = count - len(visible_names)
            more = f"; and {remainder} more" if remainder else ""
            source_copy = f"{count} {noun} selected: {', '.join(visible_names)}{more}"
        yield Static(
            source_copy,
            id="note-import-source-summary",
            markup=False,
        )

        if state.selection_kind != "folder":
            yield Button(
                "Add another file" if count else "Choose a file or folder",
                id="note-import-add-source",
                classes="library-canvas-action",
                compact=True,
            )
        if count:
            # task-32134: a wrong source used to be unreachable -- the phase
            # offered only Check selection and Back to Notes.
            yield Button(
                "Change selection",
                id="note-import-change-source",
                classes="library-canvas-action",
                compact=True,
                tooltip="Choose a different file or folder and replace this selection.",
            )
            yield Button(
                "Clear",
                id="note-import-clear-source",
                classes="library-canvas-action",
                compact=True,
                tooltip="Drop this selection and start choosing again.",
            )
        if state.selection_kind == "files":
            yield Static(
                "Notes destination",
                id="note-import-destination-label",
                markup=False,
            )
            yield Input(
                value=state.destination,
                placeholder="Existing or new folder path",
                id="note-import-destination",
            )
            yield Static(
                state.destination_error,
                id="note-import-destination-error",
                classes="note-import-error",
                markup=False,
            )

    def _compose_review(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        if state.collision_kind:
            yield Static(
                f"Folder collision: {state.collision_name}",
                id="note-import-collision-heading",
                classes="note-import-item-name",
                markup=False,
            )
            yield Static(
                state.collision_reason,
                id="note-import-collision-reason",
                markup=False,
            )
            for choice, button_id, label in (
                ("use_existing", "use-existing", "Use existing folder"),
                ("unique_sibling", "unique", "Create a unique sibling"),
                ("renamed_root", "rename", "Use another name"),
            ):
                yield Button(
                    _choice_label(
                        selected=state.collision_choice == choice, text=label
                    ),
                    id=f"note-import-collision-{button_id}",
                    name=choice,
                    classes="library-canvas-action note-import-collision-choice",
                    compact=True,
                    disabled=(
                        choice == "renamed_root"
                        and not state.collision_rename_available
                    ),
                )
            yield Input(
                # task-32262: an untouched field carries its placeholder, not
                # the colliding name with an error already painted under it.
                value=state.collision_rename_input,
                placeholder="New top-level folder name",
                id="note-import-collision-name",
            )
            yield Static(
                state.collision_rename_error,
                id="note-import-collision-rename-error",
                classes="note-import-error",
                markup=False,
            )

        # One options slot above the groups (task-32135). It stays an empty,
        # zero-height Vertical unless a vault was detected, in which case the
        # Obsidian toggle and its reason line live here (task-32129).
        with Vertical(id="notes-import-review-options"):
            if state.obsidian_available:
                yield Button(
                    _choice_label(selected=state.obsidian_mode, text="Obsidian vault"),
                    id="note-import-obsidian-mode",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Static(
                    state.obsidian_reason,
                    id="note-import-obsidian-reason",
                    classes="note-import-quiet",
                    markup=False,
                )

        # The pager fills a page assuming THIS order (task-32250); a second
        # hand-kept order here would put groups on a page in an order it did
        # not plan for. One sequence, both readers.
        order = _REVIEW_ORDER
        sorted_items = sorted(
            state.preview_items,
            key=lambda item: order.index(item.classification),
        )
        dom_tokens = {
            item.item_id: f"item-{index}"
            for index, item in enumerate(sorted_items, start=1)
        }
        group_totals = dict(state.group_totals)
        run_totals = dict(state.run_totals)
        for classification, grouped in groupby(
            sorted_items,
            key=lambda item: item.classification,
        ):
            items = tuple(grouped)
            with Horizontal(classes="note-import-group-row ds-toolbar"):
                yield Static(
                    _group_heading(
                        _CLASSIFICATION_LABELS[classification],
                        rendered=len(items),
                        total=group_totals.get(classification, len(items)),
                    ),
                    classes="note-import-group-heading",
                    markup=False,
                )
                # task-32135: settling 71 rows one at a time is not review.
                # task-32176: the action settles the rendered page, and the
                # group heading counts that page, so the label says so.
                yield Button(
                    "Skip all on this page",
                    id=f"note-import-group-{classification}-skip",
                    name=f"{classification}:skip",
                    classes=(
                        "library-canvas-action note-import-row-action "
                        "note-import-group-action"
                    ),
                    compact=True,
                )
                if classification not in _NON_IMPORTABLE:
                    yield Button(
                        "Create all on this page",
                        id=f"note-import-group-{classification}-create",
                        name=f"{classification}:create_new",
                        classes=(
                            "library-canvas-action note-import-row-action "
                            "note-import-group-action"
                        ),
                        compact=True,
                    )
            for run in _uniform_runs(items):
                if len(run) < _UNIFORM_RUN_MIN:
                    for item in run:
                        yield from self._compose_review_item(
                            item, dom_tokens[item.item_id]
                        )
                    continue
                # task-32250: 23 near-identical rows are not a review. One
                # summary row states the shared outcome; the disclosure keeps
                # every individual decision one press away.
                with _run_disclosure(
                    _run_summary(run, run_totals.get(_run_key(run[0]))),
                    dom_token=dom_tokens[run[0].item_id],
                ):
                    for item in run:
                        yield from self._compose_review_item(
                            item, dom_tokens[item.item_id]
                        )

        if state.page_count > 1:
            # task-32250: a disabled pager looked like an active one in
            # monochrome, so it says which end of the review it is at.
            at_start = state.page <= 1
            at_end = state.page >= state.page_count
            previous = Button(
                _disabled_action_label(
                    "Previous page", disabled=at_start, reason="this is the first page"
                ),
                id="note-import-page-previous",
                classes="library-canvas-action",
                compact=True,
                disabled=at_start,
            )
            yield previous
            yield Static(
                f"Page {state.page} of {state.page_count}",
                id="note-import-page",
                markup=False,
            )
            next_button = Button(
                _disabled_action_label(
                    "Next page", disabled=at_end, reason="this is the last page"
                ),
                id="note-import-page-next",
                classes="library-canvas-action",
                compact=True,
                disabled=at_end,
            )
            yield next_button

    @staticmethod
    def _review_row_summary(item: LibraryNoteImportItemSnapshot) -> str:
        """Return one line: path, what happens, and where it lands.

        task-32250: the path was spent first and in full, so a 120-character
        filename pushed the whole outcome clause off the row and every other
        row ended in "· ke…" -- hiding the keywords and link count the row
        exists to state. The path middle-elides to a budget instead; the
        decision-bearing half is what has to survive.
        """
        parts = (
            (_bounded_row_name(item.name), item.reason)
            if item.classification in _NON_IMPORTABLE
            else (
                _bounded_row_name(item.name),
                item.effect_summary,
                item.membership_summary,
            )
        )
        return " · ".join(part.rstrip(" .") for part in parts if part)

    def _compose_review_item(
        self,
        item: LibraryNoteImportItemSnapshot,
        dom_token: str,
    ) -> ComposeResult:
        # task-32135: this was five stacked lines per item, with the controls
        # right-aligned about 130 columns from the path they governed.
        with Horizontal(classes="note-import-row"):
            summary = Static(
                self._review_row_summary(item),
                classes="note-import-row-text",
                markup=False,
            )
            # A narrow terminal clips the line to the path, so the whole
            # sentence stays reachable on hover and, in the compact layout,
            # on a second line below (review of task-32135).
            summary.tooltip = self._review_row_summary(item)
            yield summary
            yield Button(
                _choice_label(selected=item.action == "skip", text="Skip"),
                id=f"note-import-action-{dom_token}-skip",
                name=f"{item.item_id}:skip",
                classes=(
                    "library-canvas-action note-import-row-action "
                    "note-import-item-action"
                ),
                compact=True,
            )
            if item.classification not in _NON_IMPORTABLE:
                yield Button(
                    _choice_label(
                        selected=item.action == "create_new", text="Create new"
                    ),
                    id=f"note-import-action-{dom_token}-create",
                    name=f"{item.item_id}:create_new",
                    classes=(
                        "library-canvas-action note-import-row-action "
                        "note-import-item-action"
                    ),
                    compact=True,
                )
            if item.can_update or item.uncertain:
                update = Button(
                    _choice_label(
                        selected=item.action == "update_existing",
                        text="Update existing",
                    ),
                    id=f"note-import-action-{dom_token}-update",
                    name=f"{item.item_id}:update_existing",
                    classes=(
                        "library-canvas-action note-import-row-action "
                        "note-import-item-action"
                    ),
                    compact=True,
                    disabled=not item.can_update,
                )
                update.tooltip = (
                    "Update the confirmed existing note."
                    if item.can_update
                    else "Confirm the match before updating."
                )
                yield update
        # The follow-on choices go on their own line. Five controls plus a path
        # need about 120 columns; below that the trailing ones used to render
        # entirely outside the body and could not be clicked (review of 32135).
        if (item.uncertain and not item.confirmed) or item.action == "update_existing":
            with Horizontal(classes="note-import-row"):
                if item.uncertain and not item.confirmed:
                    yield Button(
                        "Confirm this match",
                        id=f"note-import-confirm-{dom_token}",
                        name=item.item_id,
                        classes=(
                            "library-canvas-action note-import-row-action "
                            "note-import-confirm-match"
                        ),
                        compact=True,
                    )
                if item.action == "update_existing":
                    yield Button(
                        _choice_label(
                            selected=item.replace_content,
                            text="Replace note content",
                        ),
                        id=f"note-import-replace-{dom_token}",
                        name=f"{item.item_id}:replace_content",
                        classes=(
                            "library-canvas-action note-import-row-action "
                            "note-import-item-choice"
                        ),
                        compact=True,
                    )
                    yield Button(
                        _choice_label(
                            selected=item.add_membership,
                            text="Add folder placement",
                        ),
                        id=f"note-import-membership-{dom_token}",
                        name=f"{item.item_id}:add_membership",
                        classes=(
                            "library-canvas-action note-import-row-action "
                            "note-import-item-choice"
                        ),
                        compact=True,
                    )
        if (
            self.compact
            and item.classification not in _NON_IMPORTABLE
            and item.membership_summary
        ):
            # Only the compact shell needs it: at full width the row already
            # ends in the destination.
            yield Static(
                item.membership_summary,
                classes="note-import-row-destination note-import-quiet",
                markup=False,
            )
        # Only a matched item carries these, so the bulk of a review stays
        # exactly one line per source.
        for detail in (item.target_label, item.content_diff):
            if detail:
                yield Static(
                    detail,
                    classes="note-import-quiet",
                    markup=False,
                )

    def _compose_importing(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        detail = f" · {state.progress_detail}" if state.progress_detail else ""
        # task-32258: "67 of 67 complete" above a review that counted 66 read
        # as a contradiction. The unit is named, because it is not the same
        # unit: one planned change per note a source creates, one per skip.
        noun = "planned change" if state.progress_total == 1 else "planned changes"
        yield Static(
            f"{state.progress_completed} of {state.progress_total} {noun} complete"
            f"{detail}",
            id="note-import-progress",
            markup=False,
        )

    def _compose_receipt(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        yield Static(
            state.receipt_line,
            id="note-import-receipt",
            markup=False,
        )
        yield Static(
            state.receipt_detail,
            id="note-import-receipt-detail",
            classes="note-import-quiet",
            markup=False,
        )
        if not state.skipped_count:
            return
        # task-32130: the counts named no file, so nothing explained which
        # sources were left behind or why.
        with Collapsible(
            title=f"Skipped ({state.skipped_count})",
            id="note-import-skipped",
            collapsed=True,
        ):
            for index, (path, reason) in enumerate(state.skipped_items):
                yield Static(
                    f"{path} · {reason}",
                    id=f"note-import-skipped-{index}",
                    classes="note-import-skipped-row",
                    markup=False,
                )
            listed = len(state.skipped_items)
            if listed < state.skipped_count:
                yield Static(
                    f"Showing the first {listed} of {state.skipped_count}.",
                    classes="note-import-skipped-row",
                    markup=False,
                )

    @on(Input.Changed, "#note-import-destination")
    def _destination_changed(self, event: Input.Changed) -> None:
        if event.value == self._destination_value:
            return
        self._destination_value = event.value
        self.post_message(self.DestinationChanged(event.value))

    @on(Input.Changed, "#note-import-collision-name")
    def _collision_name_changed(self, event: Input.Changed) -> None:
        if event.value == self._collision_name:
            return
        self._collision_name = event.value
        self.post_message(self.CollisionNameChanged(event.value))

    @on(Button.Pressed, "#note-import-add-source")
    def _add_source(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.AddSourceRequested())

    @on(Button.Pressed, "#note-import-change-source")
    def _change_source(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ChangeSourceRequested())

    @on(Button.Pressed, "#note-import-clear-source")
    def _clear_source(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ClearSourceRequested())

    @on(Button.Pressed, ".note-import-group-action")
    def _choose_group_action(self, event: Button.Pressed) -> None:
        event.stop()
        classification, separator, action = (event.button.name or "").rpartition(":")
        if separator and classification and action:
            self.post_message(self.GroupActionRequested(classification, action))

    @on(Button.Pressed, "#note-import-check")
    def _check(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CheckRequested())

    @on(Button.Pressed, "#note-import-obsidian-mode")
    def _toggle_obsidian_mode(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ObsidianModeToggled(not self.snapshot.obsidian_mode))

    @on(Button.Pressed, ".note-import-collision-choice")
    def _choose_collision(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.name:
            self.post_message(self.CollisionChoiceRequested(event.button.name))

    @on(Button.Pressed, ".note-import-item-action")
    def _choose_item_action(self, event: Button.Pressed) -> None:
        event.stop()
        item_id, separator, action = (event.button.name or "").rpartition(":")
        if separator and item_id and action:
            self.post_message(self.ItemActionRequested(item_id, action))

    @on(Button.Pressed, ".note-import-item-choice")
    def _choose_item_effect(self, event: Button.Pressed) -> None:
        event.stop()
        item_id, separator, choice = (event.button.name or "").rpartition(":")
        if not separator or not item_id or not choice:
            return
        current = next(
            (item for item in self.snapshot.preview_items if item.item_id == item_id),
            None,
        )
        if current is None:
            return
        enabled = (
            not current.replace_content
            if choice == "replace_content"
            else not current.add_membership
        )
        self.post_message(self.ItemChoiceRequested(item_id, choice, enabled))

    @on(Button.Pressed, ".note-import-confirm-match")
    def _confirm_match(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.name:
            self.post_message(self.UncertainMatchConfirmed(event.button.name))

    @on(Button.Pressed, "#note-import-import")
    def _import(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ImportRequested())

    @on(Button.Pressed, "#note-import-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CancelRequested())

    @on(Button.Pressed, "#note-import-retry")
    def _retry(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RetryRequested())

    @on(Button.Pressed, "#note-import-page-previous")
    def _previous_page(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.PageRequested(-1))

    @on(Button.Pressed, "#note-import-page-next")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.PageRequested(1))
