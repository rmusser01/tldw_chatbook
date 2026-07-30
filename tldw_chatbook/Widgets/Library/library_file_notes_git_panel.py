"""Presentation-only Session Git controls for the File Notes navigator."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from rich.cells import cell_len, split_graphemes
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, ListItem, ListView, Static, TextArea

from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitIncludedNote,
    CommitOutcome,
    CommitRecoveryProjection,
    CommitReviewChangeType,
    CommitReviewProjection,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    HeadIdentity,
    SessionGitRow,
    SessionGitStatus,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

_CHANGE_VERBS = {
    "created": "CREATED",
    "modified": "EDITED",
    "moved": "MOVED",
    "deleted": "DELETED",
    "restored": "RESTORED",
}

CommitPanelPhase = Literal[
    "list",
    "form",
    "checking",
    "review",
    "confirming",
    "executing",
    "result",
]


@dataclass(frozen=True, slots=True)
class CommitDraftProjection:
    """Binding-scoped, presentation-only commit draft."""

    binding_key: object
    branch: str
    staged_note_count: int
    subject: str = ""
    body: str = ""
    form_error: str | None = None
    subject_error: str | None = None
    body_error: str | None = None

    def __post_init__(self) -> None:
        if self.staged_note_count < 0:
            raise ValueError("staged_note_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CommitReviewNoteProjection:
    """One sanitized included-note row for literal rendering."""

    note: CommitIncludedNote

    @property
    def group_id(self) -> int:
        """Return the exact reviewed session-group identity."""
        return self.note.group_id

    @property
    def change_type(self) -> CommitReviewChangeType:
        """Return the immutable Git-semantic review label."""
        return self.note.change_type

    @property
    def display_path(self) -> str:
        """Return the exact service-sanitized display text."""
        return self.note.display_text


@dataclass(frozen=True, slots=True)
class CommitPanelReviewProjection:
    """Sanitized review facts paired with explicit note change types."""

    review: CommitReviewProjection
    included_notes: tuple[CommitReviewNoteProjection, ...]

    def __post_init__(self) -> None:
        included_notes = tuple(self.included_notes)
        if (
            tuple(item.note for item in included_notes)
            != self.review.included_notes
        ):
            raise ValueError(
                "included note projections must exactly match the review"
            )
        object.__setattr__(self, "included_notes", included_notes)

    @property
    def change_counts(self) -> tuple[tuple[str, int], ...]:
        """Return non-zero change counts in the stable review order."""
        counts = {
            change_type: sum(
                note.change_type == change_type
                for note in self.included_notes
            )
            for change_type in ("New", "Modified", "Deleted", "Moved")
        }
        return tuple((label, count) for label, count in counts.items() if count)


@dataclass(frozen=True, slots=True)
class CommitExecutionProjection:
    """Sanitized progress facts for one started commit."""

    staged_note_count: int

    def __post_init__(self) -> None:
        if self.staged_note_count < 0:
            raise ValueError("staged_note_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CommitResultProjection:
    """Sanitized terminal or recoverable commit presentation."""

    outcome: CommitOutcome
    recovery: CommitRecoveryProjection | None = None


def _repository_path_for_display(path: str, *, markup: bool = False) -> str:
    """Make path controls visible and optionally escape Rich markup."""
    parts: list[str] = []
    replacements = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
    for character in path:
        if character in replacements:
            parts.append(replacements[character])
            continue
        codepoint = ord(character)
        if 0xDC80 <= codepoint <= 0xDCFF:
            parts.append(f"\\x{codepoint - 0xDC00:02x}")
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            parts.append(
                f"\\x{codepoint:02x}"
                if codepoint <= 0xFF
                else (
                    f"\\u{codepoint:04x}"
                    if codepoint <= 0xFFFF
                    else f"\\U{codepoint:08x}"
                )
            )
        else:
            parts.append(character)
    display = "".join(parts)
    return escape_markup(display) if markup else display


def _grapheme_spans(text: str) -> list[tuple[int, int, int]]:
    """Return Rich spans with regional-indicator pairs kept together."""
    spans, _ = split_graphemes(text)
    grouped: list[tuple[int, int, int]] = []
    index = 0
    while index < len(spans):
        start, end, width = spans[index]
        if index + 1 < len(spans):
            next_start, next_end, next_width = spans[index + 1]
            current_is_indicator = (
                end == start + 1
                and 0x1F1E6 <= ord(text[start]) <= 0x1F1FF
            )
            next_is_indicator = (
                next_end == next_start + 1
                and 0x1F1E6 <= ord(text[next_start]) <= 0x1F1FF
            )
            if current_is_indicator and next_is_indicator:
                grouped.append((start, next_end, width + next_width))
                index += 2
                continue
        grouped.append((start, end, width))
        index += 1
    return grouped


def _middle_elide_cells(text: str, width: int) -> str:
    """Middle-elide on grapheme boundaries without exceeding cell width."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width <= 3:
        return "." * width

    graphemes = _grapheme_spans(text)
    remaining = width - 3
    left_budget = (remaining + 1) // 2
    right_budget = remaining - left_budget
    first_width = graphemes[0][2]
    last_width = graphemes[-1][2]
    if first_width > left_budget and first_width + last_width <= remaining:
        left_budget = first_width
        right_budget = remaining - left_budget
    if last_width > right_budget and first_width + last_width <= remaining:
        right_budget = last_width
        left_budget = remaining - right_budget

    left_end = 0
    used = 0
    for _start, end, grapheme_width in graphemes:
        if used + grapheme_width > left_budget:
            break
        left_end = end
        used += grapheme_width

    right_start = len(text)
    used = 0
    for start, _end, grapheme_width in reversed(graphemes):
        if used + grapheme_width > right_budget:
            break
        right_start = start
        used += grapheme_width

    return f"{text[:left_end]}...{text[right_start:]}"


def _fit_two_line_copy(text: str, width: int) -> str:
    """Fit copy to explicit lines so word wrapping cannot create a third."""
    text = (
        text.replace("\r\n", r"\n")
        .replace("\r", r"\r")
        .replace("\n", r"\n")
    )
    if width <= 0 or cell_len(text) <= width:
        return text

    fitted = _middle_elide_cells(text, max(width, width * 2 - 3))
    if cell_len(fitted) <= width:
        return fitted

    graphemes = _grapheme_spans(fitted)
    total_width = sum(grapheme_width for _, _, grapheme_width in graphemes)
    first_width = 0
    split_end = 0
    split_key = (2, total_width)
    ellipsis_start = fitted.find("...")
    for _start, end, grapheme_width in graphemes:
        first_width += grapheme_width
        if first_width > width:
            break
        if total_width - first_width <= width:
            if ellipsis_start >= 0 and ellipsis_start < end < ellipsis_start + 3:
                continue
            key = (
                0 if fitted[end - 1].isspace() else 1,
                abs(total_width - first_width * 2),
            )
            if key < split_key:
                split_key = key
                split_end = end
    if split_end:
        return f"{fitted[:split_end]}\n{fitted[split_end:]}"
    return _middle_elide_cells(fitted, width)


def _group_path_for_display(row: SessionGitRow) -> str:
    """Project one immutable group to display-only note path copy."""
    source = _repository_path_for_display(row.group.source_path)
    destination = row.group.destination_path
    if destination is None:
        return source
    return f"{source} -> {_repository_path_for_display(destination)}"


def _row_primary_copy(row: SessionGitRow) -> str:
    """Lead one row with the note intent represented by its latest change."""
    verb = _CHANGE_VERBS[row.group.latest_action]
    return f"{verb:<9} {_group_path_for_display(row)}"


def _row_secondary_parts(
    row: SessionGitRow,
) -> tuple[str, str, str, str]:
    """Return semantic prefix, detail, recovery, and narrow recovery."""
    state = row.state
    if state == "unstaged":
        return "READY TO STAGE · Git: unstaged", "", "", ""
    if state == "owned":
        return "STAGED · by Chatbook", "", "", ""
    if state == "owned_newer_edits":
        return (
            "UPDATE AVAILABLE · newer note edits are not staged",
            "",
            "",
            "",
        )
    if state == "owned_topology_changed":
        return (
            "UPDATE REQUIRED · stage the moved note before unstaging",
            "",
            "",
            "",
        )
    if state in {"external_staged", "external_partial"}:
        return (
            "BLOCKED · ",
            "already staged outside Chatbook",
            "; manage this path in Git, then Refresh",
            "; use Git, then Refresh",
        )
    if state == "clean":
        return "NO ACTION · matches HEAD", "", "", ""
    if state == "ignored":
        return (
            "BLOCKED · ",
            "ignored by Git",
            ("; change the ignore rule or stage outside Chatbook, then Refresh"),
            "; fix ignore, then Refresh",
        )

    reason = _repository_path_for_display(
        row.disabled_reason
        or {
            "conflict": "Git conflict",
            "unsupported": "unsupported Git index state",
            "nested_repository": "nested repository",
            "unsafe_closure": "unsafe session path closure",
            "ambiguous_lineage": "ambiguous session path lineage",
            "unavailable": "Git unavailable",
            "error": "Git status failed",
        }[state]
    )
    if state == "conflict":
        return (
            "BLOCKED · ",
            reason,
            "; resolve the Git conflict outside Chatbook, then Refresh",
            "Conflict: use Git; Refresh",
        )
    if state in {
        "unsupported",
        "nested_repository",
        "unsafe_closure",
        "ambiguous_lineage",
    }:
        return (
            "BLOCKED · ",
            reason,
            "; resolve it outside Chatbook, then Refresh",
            "; use Git, then Refresh",
        )
    if state == "unavailable":
        return (
            "BLOCKED · ",
            reason,
            "; restore Git, then Refresh",
            "Restore Git first; Refresh",
        )
    return (
        "FAILED · ",
        reason,
        "; retry Git status, then Refresh",
        "; retry, Refresh",
    )


def _row_secondary_copy(row: SessionGitRow, width: int | None = None) -> str:
    """Project Git state while reserving semantic and recovery cells."""
    prefix, detail, recovery, narrow_recovery = _row_secondary_parts(row)
    full = f"{prefix}{detail}{recovery}"
    if width is None or cell_len(full) <= width:
        return full
    if not detail:
        return _middle_elide_cells(full, width)

    suffix = recovery
    if narrow_recovery and cell_len(prefix) + cell_len(suffix) > width:
        suffix = narrow_recovery
    detail_width = max(
        0,
        width - cell_len(prefix) - cell_len(suffix),
    )
    fitted_detail = _middle_elide_cells(detail, detail_width)
    fitted = f"{prefix}{fitted_detail}{suffix}"
    return fitted if cell_len(fitted) <= width else _middle_elide_cells(fitted, width)


class _SessionGitListItem(ListItem):
    """One selectable row whose action policy remains in ``SessionGitRow``."""

    def __init__(self, row: SessionGitRow) -> None:
        super().__init__(classes="file-notes-git-row")
        self.row = row

    def compose(self) -> ComposeResult:
        yield Static(
            _row_primary_copy(self.row),
            classes="file-notes-git-row-primary",
            markup=False,
        )
        yield Static(
            _row_secondary_copy(self.row),
            classes="file-notes-git-row-secondary",
            markup=False,
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self._fit_copy)

    def on_resize(self, _event: Resize) -> None:
        """Reproject display-only labels against the mounted row width."""
        self.call_after_refresh(self._fit_copy)

    def _fit_copy(self) -> None:
        if not self.is_mounted or not self.children:
            return
        primary = self.query_one(".file-notes-git-row-primary", Static)
        secondary = self.query_one(".file-notes-git-row-secondary", Static)
        width = primary.content_region.width
        if width <= 0:
            return
        primary.update(_middle_elide_cells(_row_primary_copy(self.row), width))
        secondary.update(_row_secondary_copy(self.row, width))


class _CommitIncludedNoteListItem(ListItem):
    """One review row with a stable change label and fitted path."""

    def __init__(self, note: CommitReviewNoteProjection) -> None:
        super().__init__(classes="file-notes-git-commit-included-row")
        self.note = note

    def compose(self) -> ComposeResult:
        yield Static(
            self._copy(),
            classes="file-notes-git-commit-included-copy",
            markup=False,
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self._fit_copy)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self._fit_copy)

    def _copy(self, width: int | None = None) -> str:
        prefix = f"{self.note.change_type}: "
        path = _repository_path_for_display(self.note.display_path)
        if width is None:
            return f"{prefix}{path}"
        return f"{prefix}{_middle_elide_cells(path, max(0, width - cell_len(prefix)))}"

    def _fit_copy(self) -> None:
        if not self.is_mounted or not self.children:
            return
        copy = self.query_one(".file-notes-git-commit-included-copy", Static)
        width = copy.content_region.width
        if width > 0:
            copy.update(self._copy(width))


class _CommitWorkflowScroll(VerticalScroll):
    """Keyboard-scrollable focus target for constrained workflow states."""

    can_focus = True


class LibraryFileNotesGitPanel(Vertical):
    """Render immutable Session Git policy and emit typed user intent."""

    BINDINGS = [
        Binding("escape", "back_to_files", "Back to Files", show=False),
    ]

    DEFAULT_CSS = """
    $ds-focus-bg: #51677e;
    $ds-focus-fg: $text;

    LibraryFileNotesGitPanel {
        display: none;
        height: 1fr;
        min-height: 6;
        min-width: 0;
    }

    #file-notes-git-header,
    #file-notes-git-selected-actions,
    #file-notes-git-bulk-actions,
    #file-notes-git-commit-actions {
        height: auto;
        min-height: 1;
    }

    #file-notes-git-repository,
    #file-notes-git-title,
    #file-notes-git-scope,
    #file-notes-git-guide,
    #file-notes-git-empty,
    #file-notes-git-status,
    #file-notes-git-selected-note,
    #file-notes-git-action-status {
        height: auto;
        min-height: 1;
    }

    #file-notes-git-repository,
    #file-notes-git-status,
    #file-notes-git-selected-note,
    #file-notes-git-action-status {
        max-height: 2;
        overflow: hidden hidden;
        text-wrap: nowrap;
    }

    #file-notes-git-repository {
        text-style: bold;
    }

    #file-notes-git-title {
        text-style: bold;
    }

    #file-notes-git-scope,
    #file-notes-git-guide,
    #file-notes-git-empty,
    #file-notes-git-status,
    #file-notes-git-selected-note,
    #file-notes-git-action-status {
        color: $text-muted;
    }

    #file-notes-git-action-status,
    #file-notes-git-selected-note {
        display: none;
    }

    #file-notes-git-empty {
        display: none;
    }

    #file-notes-git-rows {
        height: 1fr;
        min-height: 1;
    }

    .file-notes-git-row {
        height: 2;
        min-height: 2;
        padding: 0 1;
    }

    .file-notes-git-row:focus,
    .file-notes-git-row.-highlight {
        background: $ds-focus-bg;
        color: $ds-focus-fg;
        text-style: bold underline;
        outline: none;
    }

    .file-notes-git-row-primary,
    .file-notes-git-row-secondary {
        height: 1;
        min-height: 1;
        overflow: hidden hidden;
    }

    LibraryFileNotesGitPanel Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
    }

    LibraryFileNotesGitPanel Button.-style-default:focus {
        background: $ds-focus-bg;
        color: $ds-focus-fg;
        text-style: bold underline;
        outline: none;
    }

    LibraryFileNotesGitPanel.-stack-actions #file-notes-git-header,
    LibraryFileNotesGitPanel.-stack-actions #file-notes-git-selected-actions,
    LibraryFileNotesGitPanel.-stack-actions #file-notes-git-bulk-actions,
    LibraryFileNotesGitPanel.-stack-actions #file-notes-git-commit-actions {
        layout: vertical;
    }

    LibraryFileNotesGitPanel.-stack-actions Button {
        width: 1fr;
    }

    #file-notes-git-list-surface,
    #file-notes-git-commit-workflow {
        height: 1fr;
        min-height: 1;
        min-width: 0;
    }

    #file-notes-git-commit-workflow {
        display: none;
    }

    #file-notes-git-commit-zero {
        display: none;
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #file-notes-git-commit-staged {
        display: none;
    }

    #file-notes-git-commit-body {
        height: 1fr;
        min-height: 1;
        min-width: 0;
        scrollbar-size: 1 1;
    }

    #file-notes-git-commit-body:focus {
        outline: heavy $ds-focus-bg;
    }

    .file-notes-git-commit-phase {
        display: none;
        height: auto;
        min-height: 1;
        min-width: 0;
    }

    .file-notes-git-commit-copy,
    .file-notes-git-commit-label,
    .file-notes-git-commit-error {
        height: auto;
        min-height: 1;
        text-wrap: wrap;
    }

    .file-notes-git-commit-label {
        text-style: bold;
    }

    .file-notes-git-commit-error {
        display: none;
        color: $error;
    }

    #file-notes-git-commit-subject {
        height: 3;
        min-height: 3;
        border: round $surface-lighten-1;
        background: $surface-darken-1;
    }

    #file-notes-git-commit-body-input {
        height: 5;
        min-height: 3;
        border: round $surface-lighten-1;
        background: $surface-darken-1;
    }

    #file-notes-git-commit-subject.-invalid,
    #file-notes-git-commit-body-input.-invalid {
        border: round $error;
        background: $error 10%;
    }

    #file-notes-git-commit-review-message {
        padding: 0 1;
        background: $surface-darken-1;
    }

    #file-notes-git-commit-review-promise,
    #file-notes-git-commit-result-state {
        text-style: bold;
    }

    #file-notes-git-commit-check-again-reason {
        display: none;
        color: $text-muted;
    }

    #file-notes-git-commit-included-toggle {
        display: none;
    }

    #file-notes-git-commit-included-notes {
        display: none;
        height: auto;
        min-height: 1;
        max-height: 8;
    }

    .file-notes-git-commit-included-row {
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    .file-notes-git-commit-included-row:focus,
    .file-notes-git-commit-included-row.-highlight {
        background: $ds-focus-bg;
        color: $ds-focus-fg;
        text-style: bold underline;
        outline: none;
    }

    .file-notes-git-commit-included-copy {
        height: 1;
        min-height: 1;
        overflow: hidden hidden;
    }

    #file-notes-git-commit-included-selected {
        display: none;
    }

    #file-notes-git-commit-footer {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 1;
        height: 1;
        min-height: 1;
        min-width: 0;
    }

    #file-notes-git-commit-footer Button {
        display: none;
        width: 1fr;
    }

    LibraryFileNotesGitPanel.-commit-footer-narrow
    #file-notes-git-commit-footer {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1 1;
        height: 2;
        min-height: 2;
    }

    LibraryFileNotesGitPanel.-commit-footer-narrow
    #file-notes-git-commit-confirm {
        column-span: 2;
    }
    """

    class BackRequested(Message):
        """Request return to the retained Files/search navigator."""

    class RefreshRequested(Message):
        """Request a visible Session Git status refresh."""

    class TrustRequested(Message):
        """Request process-only trust confirmation."""

    class StageRequested(Message):
        """Request Stage for exact stable session group identities."""

        def __init__(
            self,
            group_ids: tuple[int, ...],
            *,
            bulk: bool = False,
        ) -> None:
            self.group_ids = group_ids
            self.bulk = bulk
            super().__init__()

    class UnstageRequested(Message):
        """Request Unstage for exact stable session group identities."""

        def __init__(
            self,
            group_ids: tuple[int, ...],
            *,
            bulk: bool = False,
        ) -> None:
            self.group_ids = group_ids
            self.bulk = bulk
            super().__init__()

    class CommitStagedRequested(Message):
        """Request entry into the guarded commit flow for one binding."""

        def __init__(self, binding_key: object) -> None:
            self.binding_key = binding_key
            super().__init__()

    class CommitDraftChanged(Message):
        """Publish literal draft edits back to their binding-scoped owner."""

        def __init__(
            self,
            binding_key: object,
            subject: str,
            body: str,
        ) -> None:
            self.binding_key = binding_key
            self.subject = subject
            self.body = body
            super().__init__()

    class ReviewCommitRequested(Message):
        """Request guarded preflight for the current literal draft."""

        def __init__(
            self,
            binding_key: object,
            subject: str,
            body: str,
        ) -> None:
            self.binding_key = binding_key
            self.subject = subject
            self.body = body
            super().__init__()

    class EditCommitMessageRequested(Message):
        """Request return from review or recovery to the draft form."""

    class CancelCommitRequested(Message):
        """Request cancellation before branch mutation starts."""

        def __init__(self, from_phase: CommitPanelPhase) -> None:
            self.from_phase = from_phase
            super().__init__()

    class ConfirmCommitRequested(Message):
        """Request confirmation of the workspace-owned review capability."""

    class CheckCommitAgainRequested(Message):
        """Request safe inspection of one retained uncertain attempt."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("id", "file-notes-git-panel")
        super().__init__(**kwargs)
        self._rows: tuple[SessionGitRow, ...] = ()
        self._selected_group_id: int | None = None
        self._trusted = False
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
        self._row_render_generation = 0
        self._replacing_rows = False
        self._repository_text = "Repository: not checked"
        self._current_status_text = "Status: NOT CHECKED"
        self._last_action_text = ""
        self._selected_note_text = ""
        self._commit_phase: CommitPanelPhase = "list"
        self._commit_availability: CommitDraftProjection | None = None
        self._active_commit_draft: CommitDraftProjection | None = None
        self._commit_review: CommitPanelReviewProjection | None = None
        self._commit_result: CommitResultProjection | None = None
        self._commit_notes: tuple[CommitReviewNoteProjection, ...] = ()
        self._commit_included_expanded = False
        self._commit_note_render_generation = 0
        self._commit_list_focus_pending = False
        self._commit_list_preferred_group_id: int | None = None
        self._commit_list_focus_selector: str | None = None
        self._commit_entry_focus: tuple[object, str] | None = None

    @property
    def selected_group_id(self) -> int | None:
        """Return the stable identity of the currently selected row."""
        return self._selected_group_id

    @property
    def rows(self) -> tuple[SessionGitRow, ...]:
        """Return the immutable row snapshot currently presented to the user."""
        return self._rows

    @property
    def commit_phase(self) -> CommitPanelPhase:
        """Return the currently presented commit workflow phase."""
        return self._commit_phase

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="file-notes-git-list-surface"):
            with Horizontal(id="file-notes-git-header"):
                yield Button(
                    "Back to navigator",
                    id="file-notes-git-back",
                    compact=True,
                )
                yield Button(
                    "Refresh",
                    id="file-notes-git-refresh",
                    compact=True,
                )
                yield Button(
                    "Trust and check status",
                    id="file-notes-git-trust",
                    compact=True,
                )
            yield Static(
                "Prepare session for commit",
                id="file-notes-git-title",
                markup=False,
            )
            yield Static(
                "Repository: not checked",
                id="file-notes-git-repository",
                markup=False,
            )
            yield Static(
                "Session paths only · stages complete file state",
                id="file-notes-git-scope",
                markup=False,
            )
            yield Static(
                "Up/Down Select | Tab Actions | Enter Run | Esc Back",
                id="file-notes-git-guide",
                markup=False,
            )
            yield ListView(id="file-notes-git-rows")
            yield Static(
                "No current-session Git changes.",
                id="file-notes-git-empty",
                markup=False,
            )
            yield Static(
                "Status: NOT CHECKED",
                id="file-notes-git-status",
                markup=False,
            )
            yield Static(
                "",
                id="file-notes-git-action-status",
                markup=False,
            )
            yield Static(
                "",
                id="file-notes-git-selected-note",
                markup=False,
            )
            with Horizontal(id="file-notes-git-selected-actions"):
                yield Button(
                    "Stage",
                    id="file-notes-git-stage-selected",
                    compact=True,
                )
                yield Button(
                    "Unstage",
                    id="file-notes-git-unstage-selected",
                    compact=True,
                )
            with Horizontal(id="file-notes-git-bulk-actions"):
                yield Button(
                    "Stage all (0)",
                    id="file-notes-git-stage-all",
                    compact=True,
                )
                yield Button(
                    "Unstage all (0)",
                    id="file-notes-git-unstage-all",
                    compact=True,
                )
            with Horizontal(id="file-notes-git-commit-actions"):
                yield Button(
                    "Commit staged (0)",
                    id="file-notes-git-commit-staged",
                    compact=True,
                    disabled=True,
                )
            yield Static(
                "Stage at least one session note to commit",
                id="file-notes-git-commit-zero",
                markup=False,
            )

        with Vertical(id="file-notes-git-commit-workflow"):
            with _CommitWorkflowScroll(id="file-notes-git-commit-body"):
                with Vertical(
                    id="file-notes-git-commit-form",
                    classes="file-notes-git-commit-phase",
                ):
                    yield Static(
                        "",
                        id="file-notes-git-commit-form-meta",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-form-error",
                        classes="file-notes-git-commit-error",
                        markup=False,
                    )
                    yield Static(
                        "Subject",
                        classes="file-notes-git-commit-label",
                        markup=False,
                    )
                    yield Input(
                        id="file-notes-git-commit-subject",
                        placeholder="Required commit subject",
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-subject-error",
                        classes="file-notes-git-commit-error",
                        markup=False,
                    )
                    yield Static(
                        "Body (optional)",
                        classes="file-notes-git-commit-label",
                        markup=False,
                    )
                    yield TextArea(
                        "",
                        id="file-notes-git-commit-body-input",
                        soft_wrap=True,
                        tab_behavior="focus",
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-body-error",
                        classes="file-notes-git-commit-error",
                        markup=False,
                    )

                with Vertical(
                    id="file-notes-git-commit-checking",
                    classes="file-notes-git-commit-phase",
                ):
                    yield Static(
                        "Checking commit...",
                        id="file-notes-git-commit-checking-copy",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )

                with Vertical(
                    id="file-notes-git-commit-review-surface",
                    classes="file-notes-git-commit-phase",
                ):
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-branch",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "Exact commit message",
                        classes="file-notes-git-commit-label",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-message",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-identity-primary",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-identity-secondary",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-promise",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-scope",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-change-counts",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Button(
                        "Show included notes (0)",
                        id="file-notes-git-commit-included-toggle",
                        compact=True,
                    )
                    yield ListView(
                        id="file-notes-git-commit-included-notes",
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-included-selected",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-policy",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-review-complete-state",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )

                with Vertical(
                    id="file-notes-git-commit-execution",
                    classes="file-notes-git-commit-phase",
                ):
                    yield Static(
                        "",
                        id="file-notes-git-commit-execution-title",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-execution-detail",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )

                with Vertical(
                    id="file-notes-git-commit-result",
                    classes="file-notes-git-commit-phase",
                ):
                    yield Static(
                        "",
                        id="file-notes-git-commit-result-state",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-result-message",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-result-qualification",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )
                    yield Static(
                        "",
                        id="file-notes-git-commit-check-again-reason",
                        classes="file-notes-git-commit-copy",
                        markup=False,
                    )

            with Grid(id="file-notes-git-commit-footer"):
                yield Button(
                    "Edit message",
                    id="file-notes-git-commit-edit",
                    compact=True,
                )
                yield Button(
                    "Cancel commit",
                    id="file-notes-git-commit-cancel",
                    compact=True,
                )
                yield Button(
                    "Review commit",
                    id="file-notes-git-commit-review",
                    compact=True,
                )
                yield Button(
                    "Confirm commit",
                    id="file-notes-git-commit-confirm",
                    compact=True,
                )
                yield Button(
                    "Check again",
                    id="file-notes-git-commit-check-again",
                    compact=True,
                )

    def on_mount(self) -> None:
        self._sync_action_layout(self.size.width)
        self._sync_commit_footer_layout(self.size.width)
        self._show_commit_phase(self._commit_phase)
        self.call_after_refresh(self._sync_commit_availability)
        self._update_actions()
        self.call_after_refresh(self._fit_fixed_regions)

    def on_resize(self, event: Resize) -> None:
        """Recompute mounted copy and actions from real available geometry."""
        self._sync_action_layout(event.size.width)
        self._sync_commit_footer_layout(event.size.width)
        self.call_after_refresh(self._fit_fixed_regions)

    def _sync_action_layout(self, width: int) -> None:
        """Stack only when a visible action row's real labels do not fit."""
        if not self.is_mounted:
            return
        needs_stack = any(
            self._visible_action_cells(selector)
            > self._action_row_width(selector, width)
            for selector in (
                "#file-notes-git-header",
                "#file-notes-git-selected-actions",
                "#file-notes-git-bulk-actions",
                "#file-notes-git-commit-actions",
            )
        )
        self.set_class(needs_stack, "-stack-actions")

    def _sync_commit_footer_layout(self, width: int) -> None:
        """Stack the guarded review footer when all three labels cannot fit."""
        required = sum(
            cell_len(label) + 2
            for label in ("Edit message", "Cancel commit", "Confirm commit")
        )
        self.set_class(width < required, "-commit-footer-narrow")

    def _visible_action_cells(self, selector: str) -> int:
        row = self.query_one(selector)
        return sum(
            cell_len(str(button.label)) + button.styles.padding.width
            for button in row.query(Button)
            if button.display
        )

    def _action_row_width(self, selector: str, fallback: int) -> int:
        row = self.query_one(selector)
        return row.content_region.width or fallback

    def _fit_fixed_regions(self) -> None:
        for selector, text in (
            ("#file-notes-git-repository", self._repository_text),
            ("#file-notes-git-status", self._current_status_text),
            ("#file-notes-git-action-status", self._last_action_text),
            ("#file-notes-git-selected-note", self._selected_note_text),
        ):
            widget = self.query_one(selector, Static)
            width = widget.content_region.width or self.content_region.width
            fitted = text if width <= 0 else _fit_two_line_copy(text, width)
            widget.update(fitted)

    def _set_repository_text(self, text: str) -> None:
        self._repository_text = text
        self._fit_fixed_regions()

    def render_commit_availability(
        self,
        projection: CommitDraftProjection,
    ) -> None:
        """Project list availability without replacing an active form draft."""
        self._commit_availability = projection
        self._sync_commit_availability()

    def clear_commit_availability(
        self,
        *,
        discard_draft: bool = False,
    ) -> None:
        """Remove a staged-count projection whose authority was invalidated."""
        self._commit_availability = None
        if discard_draft:
            self._active_commit_draft = None
        self._sync_commit_availability()

    def _sync_commit_availability(self) -> None:
        if not self.is_mounted:
            return
        available = self._commit_availability is not None
        count = (
            self._commit_availability.staged_note_count
            if available
            else 0
        )
        button = self.query_one("#file-notes-git-commit-staged", Button)
        zero = self.query_one("#file-notes-git-commit-zero", Static)
        button.label = f"Commit staged ({count})"
        button.disabled = count == 0 or self._mutating
        button.display = available
        zero.display = available and count == 0
        self._sync_action_layout(self.content_region.width)

    def render_commit_form(self, projection: CommitDraftProjection) -> None:
        """Render the literal binding-scoped draft and its inline errors."""
        self._active_commit_draft = projection
        branch = _repository_path_for_display(projection.branch)
        self.query_one(
            "#file-notes-git-commit-form-meta",
            Static,
        ).update(
            f"Branch: {branch} · "
            f"{projection.staged_note_count} session notes staged"
        )
        subject = self.query_one("#file-notes-git-commit-subject", Input)
        body = self.query_one("#file-notes-git-commit-body-input", TextArea)
        subject.value = projection.subject
        body.load_text(projection.body)
        form_error = self.query_one(
            "#file-notes-git-commit-form-error",
            Static,
        )
        form_error.update(projection.form_error or "")
        form_error.display = projection.form_error is not None
        self._render_commit_form_error(
            "#file-notes-git-commit-subject-error",
            subject,
            projection.subject_error,
        )
        self._render_commit_form_error(
            "#file-notes-git-commit-body-error",
            body,
            projection.body_error,
        )
        self._show_commit_phase("form")
        if projection.body_error is not None:
            target_selector = "#file-notes-git-commit-body-input"
        elif projection.subject_error is not None:
            target_selector = "#file-notes-git-commit-subject"
        elif projection.form_error is not None:
            target_selector = "#file-notes-git-commit-review"
        else:
            target_selector = "#file-notes-git-commit-subject"
        self._focus_commit_control(target_selector)

    def _render_commit_form_error(
        self,
        selector: str,
        field: Widget,
        detail: str | None,
    ) -> None:
        error = self.query_one(selector, Static)
        error.update(detail or "")
        error.display = detail is not None
        field.set_class(detail is not None, "-invalid")

    def render_commit_checking(self) -> None:
        """Render cancelable read-only preflight progress."""
        self.query_one(
            "#file-notes-git-commit-checking-copy",
            Static,
        ).update("Checking commit...")
        self._show_commit_phase("checking")
        self._focus_commit_control("#file-notes-git-commit-cancel")

    def render_commit_review(
        self,
        projection: CommitPanelReviewProjection,
    ) -> None:
        """Render one immutable, literal review without owning its authority."""
        self._commit_review = projection
        review = projection.review
        branch = _repository_path_for_display(review.branch)
        self.query_one(
            "#file-notes-git-commit-review-branch",
            Static,
        ).update(f"Branch: {branch} · Parent: {review.old_commit[:12]}")
        self.query_one(
            "#file-notes-git-commit-review-message",
            Static,
        ).update(review.message)

        identities = review.identity_display
        primary_label, primary_value = identities[0]
        primary = self.query_one(
            "#file-notes-git-commit-review-identity-primary",
            Static,
        )
        primary.update(f"{primary_label}: {primary_value}")
        secondary = self.query_one(
            "#file-notes-git-commit-review-identity-secondary",
            Static,
        )
        if len(identities) > 1:
            secondary_label, secondary_value = identities[1]
            secondary.update(f"{secondary_label}: {secondary_value}")
            secondary.display = True
        else:
            secondary.update("")
            secondary.display = False

        count = review.included_note_count
        self.query_one(
            "#file-notes-git-commit-review-promise",
            Static,
        ).update(
            f"{count} session notes will be committed; "
            "unrelated changes untouched"
        )
        self.query_one(
            "#file-notes-git-commit-review-scope",
            Static,
        ).update(
            "No unrelated staged content will be committed; "
            "Chatbook will select no unrelated worktree paths"
        )
        counts = " · ".join(
            f"{label} {amount}"
            for label, amount in projection.change_counts
        )
        self.query_one(
            "#file-notes-git-commit-review-change-counts",
            Static,
        ).update(f"Changes: {counts}")
        hook_copy = (
            "Git hooks will not run"
            if review.hooks_bypassed
            else "Git hook behavior is not confirmed"
        )
        signing_copy = (
            "Commit will be unsigned"
            if review.unsigned
            else "Commit signing is not confirmed"
        )
        self.query_one(
            "#file-notes-git-commit-review-policy",
            Static,
        ).update(f"Commit policy: {hook_copy} · {signing_copy}")
        self.query_one(
            "#file-notes-git-commit-review-complete-state",
            Static,
        ).update(
            "Included notes use their complete staged file state, "
            "not only edits made in Chatbook"
        )

        self._commit_notes = projection.included_notes
        self._commit_included_expanded = False
        disclosure = self.query_one(
            "#file-notes-git-commit-included-toggle",
            Button,
        )
        disclosure.label = f"Show included notes ({count})"
        self._replace_commit_review_notes()
        self._show_commit_phase("review")
        self._focus_commit_control("#file-notes-git-commit-edit")

    def render_commit_confirming(self) -> None:
        """Render cancelable final checking before the commit child starts."""
        self.query_one(
            "#file-notes-git-commit-checking-copy",
            Static,
        ).update("Checking commit...")
        self._show_commit_phase("confirming")
        self._focus_commit_control("#file-notes-git-commit-cancel")

    def render_commit_executing(
        self,
        projection: CommitExecutionProjection,
    ) -> None:
        """Render non-cancelable branch mutation progress."""
        self.query_one(
            "#file-notes-git-commit-execution-title",
            Static,
        ).update(
            f"Committing {projection.staged_note_count} session notes..."
        )
        self.query_one(
            "#file-notes-git-commit-execution-detail",
            Static,
        ).update(
            "Git is updating the branch; cancellation is unavailable."
        )
        self._show_commit_phase("executing")

    def render_commit_recovery_checking(self) -> None:
        """Render non-cancelable proof-only recovery without implying a commit."""
        self.query_one(
            "#file-notes-git-commit-execution-title",
            Static,
        ).update("Checking the retained commit attempt...")
        self.query_one(
            "#file-notes-git-commit-execution-detail",
            Static,
        ).update("No new commit will be started.")
        self._show_commit_phase("executing")

    def render_commit_result(
        self,
        projection: CommitResultProjection,
    ) -> None:
        """Render a complete typed outcome without fixed-region elision."""
        self._commit_result = projection
        outcome = projection.outcome
        state_labels = {
            "cancelled": "Cancelled",
            "blocked": "Blocked",
            "succeeded": "Succeeded",
            "failed_unchanged": "Failed unchanged",
            "uncertain": "Uncertain",
        }
        self.query_one(
            "#file-notes-git-commit-result-state",
            Static,
        ).update(state_labels[outcome.state])
        self.query_one(
            "#file-notes-git-commit-result-message",
            Static,
        ).update(outcome.message)
        qualification = self.query_one(
            "#file-notes-git-commit-result-qualification",
            Static,
        )
        qualification.update(outcome.qualification or "")
        qualification.display = outcome.qualification is not None
        recovery_unavailable = (
            outcome.state == "uncertain"
            and projection.recovery is not None
            and not projection.recovery.can_check_again
        )
        recovery_reason = self.query_one(
            "#file-notes-git-commit-check-again-reason",
            Static,
        )
        recovery_reason.update(
            (
                "Check again performs a proof-only recheck and never starts "
                "a new commit. If the exact Git child is still settling or "
                "Git has a relevant lock or operation, the result remains "
                "uncertain."
            )
            if recovery_unavailable
            else ""
        )
        recovery_reason.display = recovery_unavailable
        self._show_commit_phase("result")
        if outcome.state in {"blocked", "failed_unchanged"}:
            self._focus_commit_control("#file-notes-git-commit-edit")
        elif (
            outcome.state == "uncertain"
            and projection.recovery is not None
        ):
            self._focus_commit_control(
                "#file-notes-git-commit-check-again"
            )

    def return_to_commit_list(
        self,
        *,
        preferred_group_id: int | None = None,
        restore_entry_focus: bool = False,
    ) -> None:
        """Leave a retained result/workflow and focus a stable list target."""
        self._commit_review = None
        self._commit_result = None
        self._commit_notes = ()
        self._commit_included_expanded = False
        self._commit_list_focus_pending = True
        self._commit_list_preferred_group_id = preferred_group_id
        self._commit_list_focus_selector = None
        if restore_entry_focus:
            active = self._active_commit_draft
            entry_focus = self._commit_entry_focus
            if (
                active is not None
                and entry_focus is not None
                and entry_focus[0] == active.binding_key
            ):
                self._commit_list_focus_selector = entry_focus[1]
            self._commit_entry_focus = None
        self._show_commit_phase("list")
        if not self._replacing_rows:
            self._settle_commit_list_focus()

    def invalidate_commit_binding(self) -> None:
        """Discard both binding-scoped projections and leave the workflow."""
        self._commit_availability = None
        self._active_commit_draft = None
        self._commit_entry_focus = None
        self._sync_commit_availability()
        self.return_to_commit_list()

    def _show_commit_phase(self, phase: CommitPanelPhase) -> None:
        """Switch between retained list and workflow surfaces without remount."""
        self._commit_phase = phase
        list_surface = self.query_one("#file-notes-git-list-surface")
        workflow = self.query_one("#file-notes-git-commit-workflow")
        list_surface.display = phase == "list"
        workflow.display = phase != "list"

        phase_selectors = {
            "form": "#file-notes-git-commit-form",
            "checking": "#file-notes-git-commit-checking",
            "confirming": "#file-notes-git-commit-checking",
            "review": "#file-notes-git-commit-review-surface",
            "executing": "#file-notes-git-commit-execution",
            "result": "#file-notes-git-commit-result",
        }
        visible_selector = phase_selectors.get(phase)
        for widget in self.query(".file-notes-git-commit-phase"):
            widget.display = (
                visible_selector is not None
                and f"#{widget.id}" == visible_selector
            )

        disclosure = self.query_one(
            "#file-notes-git-commit-included-toggle",
            Button,
        )
        included = self.query_one(
            "#file-notes-git-commit-included-notes",
            ListView,
        )
        selected = self.query_one(
            "#file-notes-git-commit-included-selected",
            Static,
        )
        disclosure.display = phase == "review"
        included.display = phase == "review" and self._commit_included_expanded
        selected.display = included.display and bool(self._commit_notes)

        visible_footer: tuple[str, ...]
        if phase == "form":
            visible_footer = (
                "#file-notes-git-commit-cancel",
                "#file-notes-git-commit-review",
            )
        elif phase in {"checking", "confirming"}:
            visible_footer = ("#file-notes-git-commit-cancel",)
        elif phase == "review":
            visible_footer = (
                "#file-notes-git-commit-edit",
                "#file-notes-git-commit-cancel",
                "#file-notes-git-commit-confirm",
            )
        elif (
            phase == "result"
            and self._commit_result is not None
            and self._commit_result.outcome.state
            in {"blocked", "failed_unchanged"}
        ):
            visible_footer = (
                "#file-notes-git-commit-edit",
                "#file-notes-git-commit-cancel",
            )
        elif (
            phase == "result"
            and self._commit_result is not None
            and self._commit_result.outcome.state == "uncertain"
        ):
            visible_footer = (
                "#file-notes-git-commit-check-again",
            )
        else:
            visible_footer = ()
        self._set_commit_footer_buttons(visible_footer)
        self._sync_commit_footer_layout(
            self.content_region.width or self.size.width
        )

    def _set_commit_footer_buttons(
        self,
        visible_selectors: tuple[str, ...],
    ) -> None:
        for button in self.query_one(
            "#file-notes-git-commit-footer"
        ).query(Button):
            selector = f"#{button.id}"
            button.display = selector in visible_selectors
            button.disabled = False
        if (
            "#file-notes-git-commit-check-again" in visible_selectors
            and self._commit_result is not None
        ):
            recovery = self._commit_result.recovery
            self.query_one(
                "#file-notes-git-commit-check-again",
                Button,
            ).disabled = recovery is None

    def _focus_commit_control(self, selector: str) -> None:
        self.call_after_refresh(
            partial(
                self._focus_commit_control_if_current,
                self._commit_phase,
                selector,
            )
        )

    def _focus_commit_control_if_current(
        self,
        phase: CommitPanelPhase,
        selector: str,
    ) -> None:
        """Ignore deferred focus after its workflow phase became stale."""
        if self._commit_phase != phase:
            return
        control = self.query_one(selector, Widget)
        if any(
            isinstance(node, Widget) and not node.display
            for node in control.ancestors_with_self
        ):
            return
        control.focus()

    def _replace_commit_review_notes(self) -> None:
        self._commit_note_render_generation += 1
        generation = self._commit_note_render_generation
        notes = self._commit_notes
        self.run_worker(
            partial(
                self._render_commit_review_notes,
                generation,
                notes,
            ),
            name="file-notes-git-render-commit-notes",
            group="file-notes-git-render-commit-notes",
            exclusive=True,
        )

    async def _render_commit_review_notes(
        self,
        generation: int,
        notes: tuple[CommitReviewNoteProjection, ...],
    ) -> None:
        list_view = self.query_one(
            "#file-notes-git-commit-included-notes",
            ListView,
        )
        await list_view.clear()
        if generation != self._commit_note_render_generation:
            return
        await list_view.extend(
            _CommitIncludedNoteListItem(note)
            for note in notes
        )
        if generation != self._commit_note_render_generation:
            return
        list_view.index = 0 if notes else None
        self._update_commit_included_selection()

    def _update_commit_included_selection(self) -> None:
        selected = self.query_one(
            "#file-notes-git-commit-included-selected",
            Static,
        )
        list_view = self.query_one(
            "#file-notes-git-commit-included-notes",
            ListView,
        )
        index = list_view.index
        if index is None or not 0 <= index < len(self._commit_notes):
            selected.update("")
            selected.display = False
            return
        note = self._commit_notes[index]
        path = _repository_path_for_display(note.display_path)
        selected.update(f"{note.change_type}: {path}")
        selected.display = (
            self._commit_phase == "review"
            and self._commit_included_expanded
        )

    def render_status(
        self,
        status: SessionGitStatus,
        *,
        retain_rows: bool = False,
    ) -> None:
        """Render one immutable owner-published status result.

        Args:
            status: Owner-published status to project.
            retain_rows: Whether the caller has proved that stale or failed
                status still belongs to the currently trusted authority.
        """
        prior_group_id = self._selected_group_id
        authority_available = status.repository is not None
        self._trusted = authority_available
        self._trust_available = False
        self._status_ready = status.state == "ready" and authority_available
        self._mutating = False
        if not authority_available:
            repository_text = "Repository: unavailable"
        else:
            assert status.repository is not None
            repository_text = (
                "Repository: "
                f"{_repository_path_for_display(status.repository.worktree_root)}"
                f" · {self._head_label(status.head)}"
            )
        self._set_repository_text(repository_text)

        if self._status_ready:
            self._rows = status.rows
            self._replace_rows(prior_group_id)
            stage_count = sum(row.stage_eligible for row in self._rows)
            unstage_count = sum(row.unstage_eligible for row in self._rows)
            self.set_current_status(
                "Status: CURRENT · READY — "
                f"{stage_count} can be staged · "
                f"{unstage_count} can be unstaged."
            )
        elif authority_available and status.state in {"stale", "error"} and retain_rows:
            if status.rows:
                self._rows = status.rows
            self._replace_rows(prior_group_id)
            token = "STALE" if status.state == "stale" else "STALE · ERROR"
            detail = status.message or (
                "Session notes changed; Refresh before staging."
                if status.state == "stale"
                else "Git status failed. Retry Refresh."
            )
            self.set_current_status(f"Status: {token} — {detail}")
        else:
            self.clear_commit_availability()
            self._clear_rows()
            if not authority_available or status.state == "unavailable":
                detail = status.message or "Restore Git, then Refresh."
                self.set_current_status(f"Status: UNAVAILABLE — {detail}")
                self.clear_last_action()
            elif status.state == "stale":
                detail = (
                    status.message or "Session notes changed; Refresh before staging."
                )
                self.set_current_status(f"Status: STALE — {detail}")
            elif status.state == "error":
                detail = status.message or "Git status failed. Retry Refresh."
                self.set_current_status(f"Status: STALE · ERROR — {detail}")

        self._sync_empty_state()
        self._update_actions()

    def render_untrusted(self, repository_path: str) -> None:
        """Render the pre-trust state without implying status was executed."""
        self._trusted = False
        self._trust_available = True
        self._status_ready = False
        self._mutating = False
        self.clear_commit_availability()
        self._clear_rows()
        self.clear_last_action()
        self._set_repository_text(
            f"Repository: {_repository_path_for_display(repository_path)}"
        )
        self.set_current_status(
            "Status: TRUST REQUIRED — Trust this repository to check "
            "current session notes."
        )
        self._sync_empty_state()
        self._update_actions()

    def render_checking(
        self,
        repository_path: str,
        *,
        retain_rows: bool = False,
    ) -> None:
        """Render status checking with only explicitly authorized old rows.

        Args:
            repository_path: Worktree path to display while checking status.
            retain_rows: Whether the caller has proved previously rendered rows
                still belong to the current trusted repository authority.
        """
        self._trusted = True
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
        self.clear_commit_availability()
        if not retain_rows:
            self._clear_rows()
        self._set_repository_text(
            f"Repository: {_repository_path_for_display(repository_path)}"
        )
        self.set_current_status("Status: CHECKING — Checking current session notes…")
        self._sync_empty_state()
        self._update_actions()

    def set_mutating(self, active: bool, detail: str = "") -> None:
        """Render transient mutation state without owning its lifecycle."""
        self._mutating = active
        if active:
            self.set_current_status(
                "Status: UPDATING INDEX" + (f" — {detail}" if detail else "")
            )
        elif detail:
            self.set_current_status(detail)
        self._sync_commit_availability()
        self._update_actions()

    def set_current_status(self, detail: str) -> None:
        """Set repository freshness/recovery without changing last action."""
        self._current_status_text = detail
        self._fit_fixed_regions()

    def set_last_action(self, detail: str) -> None:
        """Set the latest action result independently of current freshness."""
        widget = self.query_one("#file-notes-git-action-status", Static)
        self._last_action_text = detail
        self._fit_fixed_regions()
        widget.display = bool(detail)

    def clear_last_action(self) -> None:
        """Clear an obsolete latest-action presentation."""
        widget = self.query_one("#file-notes-git-action-status", Static)
        self._last_action_text = ""
        self._fit_fixed_regions()
        widget.display = False

    def set_action_status(self, detail: str) -> None:
        """Compatibility alias for the latest workspace-owned action result."""
        self.set_last_action(detail)

    def render_unavailable(self, detail: str) -> None:
        """Render a non-trustable discovery failure with Back as recovery."""
        self._trusted = False
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
        self.clear_commit_availability()
        self._clear_rows()
        self.clear_last_action()
        self._set_repository_text("Repository: unavailable")
        self.set_current_status(f"Status: UNAVAILABLE — {detail}")
        self._sync_empty_state()
        self._update_actions()

    def mark_stale(
        self,
        detail: str = "Session notes changed; Refresh before staging.",
        *,
        retain_rows: bool = False,
        error: bool = False,
    ) -> None:
        """Disable mutations, retaining rows only under caller-proved authority."""
        self._status_ready = False
        self._mutating = False
        self.clear_commit_availability()
        if not retain_rows:
            self._clear_rows()
        token = "STALE · ERROR" if error else "STALE"
        self.set_current_status(f"Status: {token} — {detail}")
        self._sync_empty_state()
        self._update_actions()

    def _clear_rows(self) -> None:
        """Invalidate row presentation before scheduling the real list clear."""
        self._rows = ()
        self._selected_group_id = None
        self._replace_rows(None)

    def _sync_empty_state(self) -> None:
        ready_empty = (
            not self._replacing_rows
            and self._status_ready
            and not self._rows
        )
        self.query_one("#file-notes-git-empty", Static).display = ready_empty
        self.query_one("#file-notes-git-rows", ListView).display = (
            bool(self._rows) and not self._replacing_rows
        )

    def _settle_commit_list_focus(self) -> None:
        """Focus a requested row only after its mounted generation settles."""
        if (
            not self._commit_list_focus_pending
            or self._commit_phase != "list"
            or self._replacing_rows
        ):
            return
        self._commit_list_focus_pending = False
        list_view = self.query_one("#file-notes-git-rows", ListView)
        focus_selector = self._commit_list_focus_selector
        self._commit_list_focus_selector = None
        if focus_selector is not None:
            target = self.query_one(focus_selector)
            if (
                target.can_focus
                and not getattr(target, "disabled", False)
                and all(
                    not isinstance(node, Widget) or node.display
                    for node in target.ancestors_with_self
                )
            ):
                self.screen.set_focus(target, scroll_visible=False)
                self._commit_list_preferred_group_id = None
                return
        group_ids = tuple(row.group_id for row in self._rows)
        preferred_group_id = self._commit_list_preferred_group_id
        target_group_id = (
            preferred_group_id
            if preferred_group_id in group_ids
            else (group_ids[0] if group_ids else None)
        )
        self._commit_list_preferred_group_id = None
        if target_group_id is None:
            self._selected_group_id = None
            self._update_actions()
            self.screen.set_focus(
                self.query_one("#file-notes-git-back", Button),
                scroll_visible=False,
            )
            return
        self._selected_group_id = target_group_id
        list_view.index = group_ids.index(target_group_id)
        self._update_actions()
        self.screen.set_focus(list_view, scroll_visible=False)

    def _replace_rows(self, prior_group_id: int | None) -> None:
        rows = self._rows
        group_ids = tuple(row.group_id for row in rows)
        if prior_group_id in group_ids:
            self._selected_group_id = prior_group_id
        elif group_ids:
            self._selected_group_id = group_ids[0]
        else:
            self._selected_group_id = None
        self._row_render_generation += 1
        generation = self._row_render_generation
        group_id = self._selected_group_id
        self._replacing_rows = True
        self.query_one("#file-notes-git-rows", ListView).display = False
        self.run_worker(
            partial(
                self._render_rows,
                generation,
                group_id,
                rows,
            ),
            name="file-notes-git-render-rows",
            group="file-notes-git-render-rows",
            exclusive=True,
        )

    async def _render_rows(
        self,
        generation: int,
        group_id: int | None,
        rows: tuple[SessionGitRow, ...],
    ) -> None:
        list_view = self.query_one("#file-notes-git-rows", ListView)
        try:
            await list_view.clear()
            if generation != self._row_render_generation:
                return
            await list_view.extend(_SessionGitListItem(row) for row in rows)
            if generation != self._row_render_generation:
                return
            group_ids = tuple(row.group_id for row in rows)
            list_view.index = (
                group_ids.index(group_id)
                if group_id is not None and group_id in group_ids
                else None
            )
            self._selected_group_id = group_id
        finally:
            if generation == self._row_render_generation:
                self._replacing_rows = False
                self._sync_empty_state()
                self._update_actions()
                if self._commit_list_focus_pending:
                    self._settle_commit_list_focus()

    @staticmethod
    def _head_label(head: HeadIdentity | None) -> str:
        if head is None:
            return "HEAD unavailable"
        if head.kind == "detached":
            object_id = head.object_id or "unknown"
            return f"Detached HEAD {object_id[:12]}"
        if head.kind == "unborn":
            return f"Branch: {head.branch or 'unborn'} (unborn)"
        return f"Branch: {head.branch or 'unknown'}"

    def _selected_row(self) -> SessionGitRow | None:
        return next(
            (
                row
                for row in self._rows
                if row.group_id == self._selected_group_id
            ),
            None,
        )

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        focused = self.screen.focused
        back = self.query_one("#file-notes-git-back", Button)
        trust = self.query_one("#file-notes-git-trust", Button)
        refresh = self.query_one("#file-notes-git-refresh", Button)
        stage_selected = self.query_one(
            "#file-notes-git-stage-selected",
            Button,
        )
        unstage_selected = self.query_one(
            "#file-notes-git-unstage-selected",
            Button,
        )
        stage_all = self.query_one("#file-notes-git-stage-all", Button)
        unstage_all = self.query_one("#file-notes-git-unstage-all", Button)
        selected_note = self.query_one(
            "#file-notes-git-selected-note",
            Static,
        )

        trust.display = self._trust_available and not self._trusted
        refresh.display = self._trusted
        refresh.disabled = self._mutating
        selected = self._selected_row()
        if selected is None:
            self._selected_note_text = ""
            self._fit_fixed_regions()
            selected_note.display = False
        else:
            self._selected_note_text = (
                f"Selected note: {_group_path_for_display(selected)}"
            )
            self._fit_fixed_regions()
            selected_note.display = True
        can_mutate = self._status_ready and not self._mutating
        stage_selected.display = (
            self._trusted and selected is not None and selected.stage_action is not None
        )
        stage_selected.disabled = not can_mutate
        if selected is not None and selected.stage_action == "stage_update":
            stage_selected.label = "Stage update"
        else:
            stage_selected.label = "Stage"
        unstage_selected.display = (
            self._trusted and selected is not None and selected.unstage_eligible
        )
        unstage_selected.label = "Unstage"
        unstage_selected.disabled = not can_mutate
        stage_count = sum(row.stage_eligible for row in self._rows)
        unstage_count = sum(row.unstage_eligible for row in self._rows)
        stage_all.label = f"Stage all ({stage_count})"
        unstage_all.label = f"Unstage all ({unstage_count})"
        stage_all.display = self._trusted and bool(self._rows)
        unstage_all.display = self._trusted and bool(self._rows)
        stage_all.disabled = not (can_mutate and stage_count > 0)
        unstage_all.disabled = not (can_mutate and unstage_count > 0)
        self._sync_action_layout(self.content_region.width)
        self._repair_hidden_focus(focused, back, trust, refresh)

    def _repair_hidden_focus(
        self,
        focused: Widget | None,
        back: Button,
        trust: Button,
        refresh: Button,
    ) -> None:
        """Move hidden panel focus to one safe, visible recovery control."""
        if focused is None or self not in focused.ancestors:
            return
        # Authorized rows are only hidden while their new generation mounts.
        if self._replacing_rows and self._rows:
            return
        if all(
            not isinstance(node, Widget) or node.display
            for node in focused.ancestors_with_self
        ):
            return

        if self._trust_available and trust.display:
            target = trust
        elif not self._trusted:
            target = back
        elif refresh.display and not refresh.disabled:
            target = refresh
        else:
            target = back
        self.screen.set_focus(target, scroll_visible=False)

    @on(ListView.Highlighted, "#file-notes-git-rows")
    def _row_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if not self._replacing_rows and isinstance(item, _SessionGitListItem):
            self._selected_group_id = item.row.group_id
            self._update_actions()

    @on(ListView.Highlighted, "#file-notes-git-commit-included-notes")
    def _commit_note_highlighted(self, _event: ListView.Highlighted) -> None:
        self._update_commit_included_selection()

    @on(Input.Changed, "#file-notes-git-commit-subject")
    def _commit_subject_changed(self, _event: Input.Changed) -> None:
        if self._commit_phase != "form":
            return
        self._publish_commit_draft_change()

    @on(TextArea.Changed, "#file-notes-git-commit-body-input")
    def _commit_body_changed(self, _event: TextArea.Changed) -> None:
        if self._commit_phase != "form":
            return
        self._publish_commit_draft_change()

    def _publish_commit_draft_change(self) -> None:
        projection = self._active_commit_draft
        if projection is None:
            return
        subject = self.query_one(
            "#file-notes-git-commit-subject",
            Input,
        ).value
        body = self.query_one(
            "#file-notes-git-commit-body-input",
            TextArea,
        ).text
        if subject == projection.subject and body == projection.body:
            return
        self._active_commit_draft = replace(
            projection,
            subject=subject,
            body=body,
            form_error=None,
            subject_error=None,
            body_error=None,
        )
        form_error = self.query_one(
            "#file-notes-git-commit-form-error",
            Static,
        )
        form_error.update("")
        form_error.display = False
        self._render_commit_form_error(
            "#file-notes-git-commit-subject-error",
            self.query_one("#file-notes-git-commit-subject", Input),
            None,
        )
        self._render_commit_form_error(
            "#file-notes-git-commit-body-error",
            self.query_one("#file-notes-git-commit-body-input", TextArea),
            None,
        )
        self.post_message(
            self.CommitDraftChanged(
                projection.binding_key,
                subject,
                body,
            )
        )

    @on(Button.Pressed, "#file-notes-git-back")
    def _back_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.BackRequested())

    @on(Button.Pressed, "#file-notes-git-refresh")
    def _refresh_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RefreshRequested())

    @on(Button.Pressed, "#file-notes-git-trust")
    def _trust_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.TrustRequested())

    @on(Button.Pressed, "#file-notes-git-stage-selected")
    def _stage_selected_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        row = self._selected_row()
        if row is not None and row.stage_eligible:
            self.post_message(self.StageRequested((row.group_id,)))

    @on(Button.Pressed, "#file-notes-git-unstage-selected")
    def _unstage_selected_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        row = self._selected_row()
        if row is not None and row.unstage_eligible:
            self.post_message(self.UnstageRequested((row.group_id,)))

    @on(Button.Pressed, "#file-notes-git-stage-all")
    def _stage_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        group_ids = self._eligible_group_ids(
            row for row in self._rows if row.stage_eligible
        )
        if group_ids:
            self.post_message(self.StageRequested(group_ids, bulk=True))

    @on(Button.Pressed, "#file-notes-git-unstage-all")
    def _unstage_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        group_ids = self._eligible_group_ids(
            row for row in self._rows if row.unstage_eligible
        )
        if group_ids:
            self.post_message(self.UnstageRequested(group_ids, bulk=True))

    @on(Button.Pressed, "#file-notes-git-commit-staged")
    def _commit_staged_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        projection = self._commit_availability
        if projection is None or projection.staged_note_count == 0:
            return
        focused = self.screen.focused
        list_surface = self.query_one("#file-notes-git-list-surface")
        if (
            focused is not None
            and focused.id is not None
            and focused.can_focus
            and any(
                node is list_surface
                for node in focused.ancestors_with_self
            )
        ):
            self._commit_entry_focus = (
                projection.binding_key,
                f"#{focused.id}",
            )
        else:
            self._commit_entry_focus = None
        self.post_message(
            self.CommitStagedRequested(projection.binding_key)
        )
        active = self._active_commit_draft
        if active is not None and active.binding_key == projection.binding_key:
            projection = replace(
                projection,
                subject=active.subject,
                body=active.body,
            )
        self.render_commit_form(
            replace(
                projection,
                form_error=None,
                subject_error=None,
                body_error=None,
            )
        )

    @on(Button.Pressed, "#file-notes-git-commit-review")
    def _review_commit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        projection = self._active_commit_draft
        if projection is None or self._commit_phase != "form":
            return
        subject = self.query_one(
            "#file-notes-git-commit-subject",
            Input,
        ).value
        body = self.query_one(
            "#file-notes-git-commit-body-input",
            TextArea,
        ).text
        if not subject.strip():
            projection = replace(
                projection,
                subject=subject,
                body=body,
                form_error=None,
                subject_error="Commit subject is required.",
                body_error=None,
            )
            self.render_commit_form(projection)
            return
        projection = replace(
            projection,
            subject=subject,
            body=body,
            form_error=None,
            subject_error=None,
            body_error=None,
        )
        self._active_commit_draft = projection
        self.post_message(
            self.ReviewCommitRequested(
                projection.binding_key,
                subject,
                body,
            )
        )
        self.render_commit_checking()

    @on(Button.Pressed, "#file-notes-git-commit-edit")
    def _edit_commit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._edit_commit_message()

    def _edit_commit_message(self) -> None:
        projection = self._active_commit_draft
        if projection is None:
            return
        self.render_commit_form(projection)
        self.post_message(self.EditCommitMessageRequested())

    @on(Button.Pressed, "#file-notes-git-commit-cancel")
    def _cancel_commit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_commit()

    def _cancel_commit(self) -> None:
        from_phase = self._commit_phase
        if from_phase not in {
            "form",
            "checking",
            "review",
            "confirming",
            "result",
        }:
            return
        if from_phase != "confirming":
            self.return_to_commit_list(
                preferred_group_id=self._selected_group_id,
                restore_entry_focus=True,
            )
        self.post_message(self.CancelCommitRequested(from_phase))

    @on(Button.Pressed, "#file-notes-git-commit-confirm")
    def _confirm_commit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._commit_phase != "review":
            return
        self.post_message(self.ConfirmCommitRequested())
        self.render_commit_confirming()

    @on(Button.Pressed, "#file-notes-git-commit-check-again")
    def _check_commit_again_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if (
            self._commit_phase == "result"
            and self._commit_result is not None
            and self._commit_result.outcome.state == "uncertain"
            and self._commit_result.recovery is not None
        ):
            self.post_message(self.CheckCommitAgainRequested())

    @on(Button.Pressed, "#file-notes-git-commit-included-toggle")
    def _included_notes_toggled(self, event: Button.Pressed) -> None:
        event.stop()
        if self._commit_phase != "review" or not self._commit_notes:
            return
        self._commit_included_expanded = not self._commit_included_expanded
        button = self.query_one(
            "#file-notes-git-commit-included-toggle",
            Button,
        )
        button.label = (
            "Hide included notes"
            if self._commit_included_expanded
            else f"Show included notes ({len(self._commit_notes)})"
        )
        list_view = self.query_one(
            "#file-notes-git-commit-included-notes",
            ListView,
        )
        list_view.display = self._commit_included_expanded
        if self._commit_included_expanded and list_view.index is None:
            list_view.index = 0
        self._update_commit_included_selection()

    @staticmethod
    def _eligible_group_ids(rows: Iterable[SessionGitRow]) -> tuple[int, ...]:
        return tuple(row.group_id for row in rows)

    def action_back_to_files(self) -> None:
        """Apply the phase-specific safe Escape behavior."""
        if self._commit_phase == "list":
            self.post_message(self.BackRequested())
        elif self._commit_phase in {"form", "checking", "confirming"}:
            self._cancel_commit()
        elif self._commit_phase == "review":
            self._edit_commit_message()
        elif (
            self._commit_phase == "result"
            and self._commit_result is not None
            and self._commit_result.outcome.state
            in {"blocked", "failed_unchanged"}
        ):
            self._edit_commit_message()

class SessionGitTrustDialog(ConfirmationDialog):
    """Safe-focus process-only trust prompt for worktree-aware Git commands."""

    def __init__(self, repository_path: str) -> None:
        display_path = _repository_path_for_display(
            repository_path,
            markup=True,
        )
        super().__init__(
            title="Trust Session Git repository?",
            message=(
                f"Repository: {display_path}\n\n"
                "Trust lasts only for this application process. Git status "
                "and staging may execute configured Git filters, including "
                "arbitrary programs with side effects outside Chatbook.\n\n"
                "Continue only if you trust this repository and its Git "
                "configuration."
            ),
            confirm_label="Trust and check status",
            cancel_label="Cancel",
        )

    def on_mount(self) -> None:
        """Put the safe non-executing choice first in the focus order."""
        self.query_one("#cancel-button", Button).focus()
