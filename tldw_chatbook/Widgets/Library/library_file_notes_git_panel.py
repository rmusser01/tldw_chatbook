"""Presentation-only Session Git controls for the File Notes navigator."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from rich.cells import cell_len, split_graphemes
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.message import Message
from textual.widgets import Button, ListItem, ListView, Static

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


def _middle_elide_cells(text: str, width: int) -> str:
    """Middle-elide on grapheme boundaries without exceeding cell width."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width <= 3:
        return "." * width

    graphemes, _ = split_graphemes(text)
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
            "; resolve conflict, then Refresh",
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
            "; restore Git, then Refresh",
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
    #file-notes-git-bulk-actions {
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
    LibraryFileNotesGitPanel.-stack-actions #file-notes-git-bulk-actions {
        layout: vertical;
    }

    LibraryFileNotesGitPanel.-stack-actions Button {
        width: 1fr;
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

    @property
    def selected_group_id(self) -> int | None:
        """Return the stable identity of the currently selected row."""
        return self._selected_group_id

    @property
    def rows(self) -> tuple[SessionGitRow, ...]:
        """Return the immutable row snapshot currently presented to the user."""
        return self._rows

    def compose(self) -> ComposeResult:
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

    def on_mount(self) -> None:
        self._sync_action_layout(self.size.width)
        self._update_actions()
        self.call_after_refresh(self._fit_fixed_regions)

    def on_resize(self, event: Resize) -> None:
        """Recompute mounted copy and actions from real available geometry."""
        self._sync_action_layout(event.size.width)
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
            )
        )
        self.set_class(needs_stack, "-stack-actions")

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
            fitted = text if width <= 0 else _middle_elide_cells(text, width * 2)
            widget.update(fitted)

    def _set_repository_text(self, text: str) -> None:
        self._repository_text = text
        self._fit_fixed_regions()

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
        """Render status checking with only explicitly authorized old rows."""
        self._trusted = True
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
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
        ready_empty = self._status_ready and not self._rows
        self.query_one("#file-notes-git-empty", Static).display = ready_empty
        self.query_one("#file-notes-git-rows", ListView).display = not ready_empty

    def _replace_rows(self, prior_group_id: int | None) -> None:
        group_ids = tuple(row.group_id for row in self._rows)
        if prior_group_id in group_ids:
            self._selected_group_id = prior_group_id
        elif group_ids:
            self._selected_group_id = group_ids[0]
        else:
            self._selected_group_id = None
        self._row_render_generation += 1
        self.run_worker(
            self._render_rows(
                self._row_render_generation,
                self._selected_group_id,
            ),
            name="file-notes-git-render-rows",
            group="file-notes-git-render-rows",
            exclusive=True,
        )

    async def _render_rows(
        self,
        generation: int,
        group_id: int | None,
    ) -> None:
        list_view = self.query_one("#file-notes-git-rows", ListView)
        self._replacing_rows = True
        try:
            await list_view.clear()
            await list_view.extend(_SessionGitListItem(row) for row in self._rows)
            if generation != self._row_render_generation:
                return
            group_ids = tuple(row.group_id for row in self._rows)
            list_view.index = (
                group_ids.index(group_id)
                if group_id is not None and group_id in group_ids
                else None
            )
            self._selected_group_id = group_id
        finally:
            if generation == self._row_render_generation:
                self._replacing_rows = False
                self._update_actions()

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

    @on(ListView.Highlighted, "#file-notes-git-rows")
    def _row_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if not self._replacing_rows and isinstance(item, _SessionGitListItem):
            self._selected_group_id = item.row.group_id
            self._update_actions()

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

    @staticmethod
    def _eligible_group_ids(rows: Iterable[SessionGitRow]) -> tuple[int, ...]:
        return tuple(row.group_id for row in rows)

    def action_back_to_files(self) -> None:
        """Let Escape request the same safe in-screen navigation as Back."""
        self.post_message(self.BackRequested())

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
