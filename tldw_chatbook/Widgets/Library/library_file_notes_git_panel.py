"""Presentation-only Session Git controls for the File Notes navigator."""

from __future__ import annotations

from collections.abc import Iterable

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, ListItem, ListView, Static

from tldw_chatbook.Notes.file_notes_session_owner import (
    HeadIdentity,
    SessionGitRow,
    SessionGitStatus,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

_ROW_STATE_LABELS = {
    "unstaged": "Unstaged",
    "owned": "Staged by Chatbook",
    "owned_newer_edits": "Staged by Chatbook · newer unstaged edits",
    "owned_topology_changed": "Staged by Chatbook · path lineage changed",
    "external_staged": "Staged externally",
    "external_partial": "Partially staged externally",
    "clean": "Clean · currently matches HEAD",
    "ignored": "Ignored",
    "conflict": "Git conflict",
    "unsupported": "Unsupported Git index state",
    "nested_repository": "Nested repository unsupported",
    "unsafe_closure": "Unsafe session path closure",
    "ambiguous_lineage": "Ambiguous session path lineage",
    "unavailable": "Git unavailable",
    "error": "Error",
}


class _SessionGitListItem(ListItem):
    """One selectable row whose action policy remains in ``SessionGitRow``."""

    def __init__(self, row: SessionGitRow) -> None:
        super().__init__(classes="file-notes-git-row")
        self.row = row

    def compose(self) -> ComposeResult:
        label = _ROW_STATE_LABELS[self.row.state]
        reason = (
            ""
            if not self.row.disabled_reason
            else f" — {self.row.disabled_reason}"
        )
        yield Static(
            f"{self.row.group.display_text} · {label}{reason}",
            classes="file-notes-git-row-copy",
            markup=False,
        )


class LibraryFileNotesGitPanel(Vertical):
    """Render immutable Session Git policy and emit typed user intent."""

    BINDINGS = [
        Binding("escape", "back_to_files", "Back to Files", show=False),
    ]

    DEFAULT_CSS = """
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
    #file-notes-git-scope,
    #file-notes-git-complete-state,
    #file-notes-git-action-status {
        height: auto;
        min-height: 1;
    }

    #file-notes-git-repository {
        text-style: bold;
    }

    #file-notes-git-scope,
    #file-notes-git-complete-state,
    #file-notes-git-action-status {
        color: $text-muted;
    }

    #file-notes-git-rows {
        height: 1fr;
        min-height: 3;
        margin: 1 0;
    }

    .file-notes-git-row {
        height: auto;
        min-height: 1;
        padding: 0 1;
    }

    .file-notes-git-row:focus,
    .file-notes-git-row.-highlight {
        outline: heavy $accent;
    }

    .file-notes-git-row-copy {
        height: auto;
        min-height: 1;
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

    LibraryFileNotesGitPanel Button:focus {
        outline: heavy $accent;
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

        def __init__(self, group_ids: tuple[int, ...]) -> None:
            self.group_ids = group_ids
            super().__init__()

    class UnstageRequested(Message):
        """Request Unstage for exact stable session group identities."""

        def __init__(self, group_ids: tuple[int, ...]) -> None:
            self.group_ids = group_ids
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

    @property
    def selected_group_id(self) -> int | None:
        """Return the stable identity of the currently selected row."""
        return self._selected_group_id

    def compose(self) -> ComposeResult:
        with Horizontal(id="file-notes-git-header"):
            yield Button(
                "Back to Files",
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
            "Repository: not checked",
            id="file-notes-git-repository",
            markup=False,
        )
        yield Static(
            "Session paths only",
            id="file-notes-git-scope",
            markup=False,
        )
        yield Static(
            "Stages complete file state (content, deletion, and mode)",
            id="file-notes-git-complete-state",
            markup=False,
        )
        yield ListView(id="file-notes-git-rows")
        with Horizontal(id="file-notes-git-selected-actions"):
            yield Button(
                "Stage selected",
                id="file-notes-git-stage-selected",
                compact=True,
            )
            yield Button(
                "Unstage selected",
                id="file-notes-git-unstage-selected",
                compact=True,
            )
        with Horizontal(id="file-notes-git-bulk-actions"):
            yield Button(
                "Stage All",
                id="file-notes-git-stage-all",
                compact=True,
            )
            yield Button(
                "Unstage All",
                id="file-notes-git-unstage-all",
                compact=True,
            )
        yield Static(
            "Open Session Git to check status.",
            id="file-notes-git-action-status",
            markup=False,
        )

    def on_mount(self) -> None:
        self._update_actions()

    def render_status(self, status: SessionGitStatus) -> None:
        """Render one immutable owner-published status result."""
        prior_group_id = self._selected_group_id
        self._rows = status.rows
        self._trusted = status.repository is not None
        self._trust_available = False
        self._status_ready = status.state == "ready"
        self._mutating = False
        if status.repository is None:
            repository_text = "Repository: unavailable"
        else:
            repository_text = (
                f"Repository: {status.repository.worktree_root}"
                f" · {self._head_label(status.head)}"
            )
        self.query_one("#file-notes-git-repository", Static).update(
            repository_text
        )
        message = status.message or {
            "ready": "Status ready.",
            "stale": "Stale — refresh status before staging.",
            "unavailable": "Git unavailable.",
            "error": "Git status failed; retry.",
        }[status.state]
        self.query_one("#file-notes-git-action-status", Static).update(message)
        self._replace_rows(prior_group_id)
        self._update_actions()

    def render_untrusted(self, repository_path: str) -> None:
        """Render the pre-trust state without implying status was executed."""
        self._trusted = False
        self._trust_available = True
        self._status_ready = False
        self._mutating = False
        self.query_one("#file-notes-git-repository", Static).update(
            f"Repository: {repository_path}"
        )
        self.query_one("#file-notes-git-action-status", Static).update(
            "Trust is required before checking Session Git status."
        )
        self._update_actions()

    def render_checking(self, repository_path: str) -> None:
        """Render transient status-checking presentation state."""
        self._trusted = True
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
        self.query_one("#file-notes-git-repository", Static).update(
            f"Repository: {repository_path}"
        )
        self.query_one("#file-notes-git-action-status", Static).update(
            "Checking Session Git status…"
        )
        self._update_actions()

    def set_mutating(self, active: bool, detail: str = "") -> None:
        """Render transient mutation state without owning its lifecycle."""
        self._mutating = active
        if detail:
            self.query_one("#file-notes-git-action-status", Static).update(detail)
        self._update_actions()

    def set_action_status(self, detail: str) -> None:
        """Render the latest workspace-owned action/result description."""
        self.query_one("#file-notes-git-action-status", Static).update(detail)

    def render_unavailable(self, detail: str) -> None:
        """Render a non-trustable discovery failure with Back as recovery."""
        self._trusted = False
        self._trust_available = False
        self._status_ready = False
        self._mutating = False
        self.query_one("#file-notes-git-repository", Static).update(
            "Repository: unavailable"
        )
        self.query_one("#file-notes-git-action-status", Static).update(detail)
        self._update_actions()

    def mark_stale(self, detail: str = "Session paths changed; refresh status.") -> None:
        """Retain rendered rows while disabling mutations until refresh."""
        self._status_ready = False
        self._mutating = False
        self.query_one("#file-notes-git-action-status", Static).update(
            f"Stale — {detail}"
        )
        self._update_actions()

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

        trust.display = self._trust_available and not self._trusted
        refresh.display = self._trusted
        refresh.disabled = self._mutating
        selected = self._selected_row()
        can_mutate = self._status_ready and not self._mutating
        stage_selected.display = (
            self._trusted
            and selected is not None
            and selected.stage_action is not None
        )
        stage_selected.disabled = not can_mutate
        if selected is not None and selected.stage_action == "stage_update":
            stage_selected.label = "Stage update"
        else:
            stage_selected.label = "Stage selected"
        unstage_selected.display = (
            self._trusted
            and selected is not None
            and selected.unstage_eligible
        )
        unstage_selected.disabled = not can_mutate
        stage_all.display = self._trusted
        unstage_all.display = self._trusted
        stage_all.disabled = not (
            can_mutate and any(row.stage_eligible for row in self._rows)
        )
        unstage_all.disabled = not (
            can_mutate and any(row.unstage_eligible for row in self._rows)
        )

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
            self.post_message(self.StageRequested(group_ids))

    @on(Button.Pressed, "#file-notes-git-unstage-all")
    def _unstage_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        group_ids = self._eligible_group_ids(
            row for row in self._rows if row.unstage_eligible
        )
        if group_ids:
            self.post_message(self.UnstageRequested(group_ids))

    @staticmethod
    def _eligible_group_ids(rows: Iterable[SessionGitRow]) -> tuple[int, ...]:
        return tuple(row.group_id for row in rows)

    def action_back_to_files(self) -> None:
        """Let Escape request the same safe in-screen navigation as Back."""
        self.post_message(self.BackRequested())

class SessionGitTrustDialog(ConfirmationDialog):
    """Safe-focus process-only trust prompt for worktree-aware Git commands."""

    def __init__(self, repository_path: str) -> None:
        super().__init__(
            title="Trust Session Git repository?",
            message=(
                f"Repository: {repository_path}\n\n"
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
