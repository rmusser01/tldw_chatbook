"""Retained disk-backed File Notes workspace for Library."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from rich.cells import cell_len
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Resize
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.worker import Worker
from textual.widgets import Button, Input, ListView, Static, TextArea, Tree

from tldw_chatbook.config import (
    apply_settings_mutation_to_cli_config,
    get_cli_setting,
    get_user_data_dir,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DISABLED_ACTION_MARKER,
    library_disabled_action_label,
)
from tldw_chatbook.Notes.file_notes_git_service import (
    DiscoveryResult,
    GitActionResult,
    GitMutationAdmissionError,
    GitStatusAdmissionError,
    PushExecutionResult,
    PushPreflightResult,
    RetainedCommitOperation,
    RetainedPushOperation,
    coalesce_session_changes,
)
from tldw_chatbook.Notes.file_notes_conflict_compare import (
    ConflictComparison,
    ConflictSide,
    build_conflict_comparison,
)
from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitOutcome,
    CommitRecoveryProjection,
    CommitReviewHandle,
    CommitReviewResult,
)
from tldw_chatbook.Notes.file_notes_git_push import (
    PushAuthorizationProjection,
    PushCandidateProjection,
    PushDestinationPolicyResult,
    PushReviewHandle,
    PushRecoveryProjection,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    FileNotesSessionSnapshot,
    PushCandidateAvailability,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionGitRow,
    SessionGitStatus,
    SessionTransitionKind,
)
from tldw_chatbook.Notes.file_notes_service import (
    LARGE_FILE_EXCERPT_CHARS,
    FileNoteEntry,
    FileNotesService,
    OpenedFileNote,
    OperationResult,
    ReconcileResult,
    ScanResult,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    CommitDraftProjection,
    CommitExecutionProjection,
    CommitPanelReviewProjection,
    CommitResultProjection,
    CommitReviewNoteProjection,
    LibraryFileNotesGitPanel,
    PushDestinationAuthorizationDialog,
    PushEndpointDetailsDialog,
    PushPanelPhase,
    PushPanelResultProjection,
    PushPanelReviewProjection,
    PushResultAction,
    SessionGitTrustDialog,
    _middle_elide_cells,
)

SaveState = Literal["idle", "dirty", "saving", "saved", "conflict", "error"]
_SAVE_STATE_COPY: dict[SaveState, str] = {
    "idle": "Auto-save to local folder: idle",
    "dirty": "Auto-save pending for local folder",
    "saving": "Saving to local folder…",
    "saved": "Saved to local folder",
    "conflict": "Conflict: draft preserved in editor",
    "error": "Save failed: draft preserved in editor",
}
_UNSET = object()
_SESSION_GIT_MUTATION_BUSY = (
    "Git operation in progress; structural actions are busy."
)
FILE_TREE_BATCH_SIZE = 100


@dataclass(frozen=True, slots=True)
class _FolderNodeData:
    """One lazily materialized navigator folder."""

    relative_path: str


@dataclass(frozen=True, slots=True)
class _DeletedFolderData:
    """The virtual Recently deleted folder."""


@dataclass(frozen=True, slots=True)
class _TreePageData:
    """Cursor for one bounded navigator or search-result batch."""

    source: Literal["files", "deleted", "search"]
    folder_key: str
    offset: int


@dataclass(frozen=True, slots=True)
class _FolderItem:
    """One pure folder-index child awaiting Textual materialization."""

    kind: Literal["file", "folder", "deleted"]
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class _GitActionSummaryContext:
    """Immutable presentation counts captured before one bulk action."""

    bulk: bool = False
    skipped: int = 0
    already_staged: int = 0
    clean: int = 0
    blocked: int = 0


@dataclass(frozen=True, slots=True)
class _GitLastAction:
    """One action presentation bound to exact owner authority and changes."""

    binding: SessionBinding
    repository: RepositoryIdentity
    changes: tuple[SequencedSessionChange, ...]
    text: str
    complete: bool = False


@dataclass(frozen=True, slots=True)
class _CommitBindingKey:
    """Exact root-generation and repository identity for one commit draft."""

    binding: SessionBinding
    repository: RepositoryIdentity


@dataclass(frozen=True, slots=True)
class _PushBindingKey:
    """Exact process-only owner candidate identity for push presentation."""

    binding: SessionBinding
    generation: int
    candidate: PushCandidateProjection


@dataclass(frozen=True, slots=True)
class _ReloadConfirmation:
    """Exact editor and disk identities captured before destructive reload."""

    service: FileNotesService
    binding: SessionBinding
    root_generation: int
    session_key: str
    opened: OpenedFileNote
    save_state: Literal["conflict", "error"]
    disk_content_hash: str
    opener_id: str


@dataclass(frozen=True, slots=True)
class _ConflictCompareRequest:
    """Exact editor identity captured before reading the latest disk side."""

    service: FileNotesService
    binding: SessionBinding
    root_generation: int
    session_key: str
    opened: OpenedFileNote
    draft: str


@dataclass(frozen=True, slots=True)
class _CommitDraftState:
    """Literal commit message retained only for one exact binding key."""

    key: _CommitBindingKey
    subject: str = ""
    body: str = ""


@dataclass(slots=True)
class _EditorReadOnlyLease:
    """Idempotent token for one exact editor and binding."""

    token: object
    editor: TextArea
    binding: SessionBinding
    _release_callback: Callable[["_EditorReadOnlyLease"], None]
    _released: bool = False

    def release(self) -> None:
        """Release only this exact reason once."""
        if self._released:
            return
        self._released = True
        self._release_callback(self)


class _SessionGitService(Protocol):
    """UI-facing service contract already implemented by FileNotesGitService."""

    async def discover(self, binding: SessionBinding) -> DiscoveryResult: ...

    async def revalidate_repository(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool: ...

    def retained_status(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[SessionGitStatus] | None: ...

    def start_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
    ) -> asyncio.Task[SessionGitStatus]: ...

    def start_stage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]: ...

    def start_unstage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]: ...

    def start_commit_review(
        self,
        binding: SessionBinding,
        subject: str,
        body: str = "",
    ) -> asyncio.Task[CommitReviewResult]: ...

    def start_commit(
        self,
        binding: SessionBinding,
        handle: CommitReviewHandle,
        *,
        subject: str | None = None,
        body: str = "",
    ) -> asyncio.Task[CommitOutcome]: ...

    def retained_commit_operation(
        self,
        binding: SessionBinding,
    ) -> RetainedCommitOperation | None: ...

    def retained_push_operation(
        self,
        binding: SessionBinding,
    ) -> RetainedPushOperation | None: ...

    def start_push_review(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[PushDestinationPolicyResult]: ...

    def authorize_and_check_push(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushPreflightResult]: ...

    def start_push(
        self,
        binding: SessionBinding,
        handle: PushReviewHandle,
    ) -> asyncio.Task[PushExecutionResult]: ...

    def cancel_push(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> bool: ...

    def authorize_push_recovery(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> bool: ...

    def check_push_again(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushRecoveryProjection]: ...

    def cancel_commit(
        self,
        binding: SessionBinding,
    ) -> bool: ...

    def check_commit_again(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[CommitOutcome]: ...


class FileNotesRootDetailsDialog(SafeModalDismissMixin, ModalScreen[None]):
    """Show the exact linked-root state through a keyboard-readable surface."""

    BINDINGS = [("escape", "request_safe_cancel", "Close")]
    SAFE_MODAL_CONTENT = "#file-notes-root-details-dialog"

    DEFAULT_CSS = """
    FileNotesRootDetailsDialog {
        align: center middle;
    }

    #file-notes-root-details-dialog {
        width: 76;
        max-width: 95%;
        height: 12;
        max-height: 85%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #file-notes-root-details-title {
        height: 1;
        text-style: bold;
    }

    #file-notes-root-details-text {
        height: 1fr;
        min-height: 3;
    }

    #file-notes-root-details-close {
        width: auto;
        height: 1;
        min-height: 1;
        margin-top: 1;
    }
    """

    def __init__(self, detail: str) -> None:
        """Initialize the dialog with the exact linked-root detail.

        Args:
            detail: Full linked-root path or warning text to display.
        """
        super().__init__(id="file-notes-root-details-dialog-screen")
        self._detail = detail

    def compose(self) -> ComposeResult:
        """Compose the read-only detail surface and close control.

        Returns:
            The widgets that make up the dialog.
        """
        with Vertical(id="file-notes-root-details-dialog"):
            yield Static(
                "File Notes folder details",
                id="file-notes-root-details-title",
                markup=False,
            )
            yield TextArea(
                self._detail,
                id="file-notes-root-details-text",
                read_only=True,
                soft_wrap=True,
            )
            yield Button(
                "Close",
                id="file-notes-root-details-close",
                compact=True,
            )

    def on_mount(self) -> None:
        """Focus the detail text when the dialog opens."""
        self.query_one("#file-notes-root-details-text", TextArea).focus()

    @on(Button.Pressed, "#file-notes-root-details-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(None)


class FileNotesConflictCompareDialog(SafeModalDismissMixin, ModalScreen[None]):
    """Show one immutable, bounded Base/Draft/Disk comparison."""

    BINDINGS = [("escape", "request_safe_cancel", "Close")]
    SAFE_MODAL_CONTENT = "#file-notes-conflict-dialog"

    DEFAULT_CSS = """
    FileNotesConflictCompareDialog {
        align: center middle;
    }

    #file-notes-conflict-dialog {
        width: 110;
        max-width: 95%;
        height: 90%;
        min-height: 16;
        max-height: 95%;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }

    #file-notes-conflict-title {
        height: 1;
        color: $warning;
        text-style: bold;
    }

    #file-notes-conflict-path,
    #file-notes-conflict-help {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #file-notes-conflict-summary {
        height: 7;
        min-height: 4;
        margin-top: 1;
    }

    #file-notes-conflict-diff {
        height: 1fr;
        min-height: 4;
        margin-top: 1;
    }

    #file-notes-conflict-close {
        width: auto;
        height: 1;
        min-height: 1;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        relative_path: str,
        comparison: ConflictComparison,
    ) -> None:
        """Initialize an immutable conflict comparison.

        Args:
            relative_path: Literal note path owning the conflict.
            comparison: Precomputed bounded comparison payload.
        """
        super().__init__(id="file-notes-conflict-dialog-screen")
        self._relative_path = relative_path
        self._comparison = comparison

    def compose(self) -> ComposeResult:
        """Compose labeled read-only side identities and diffs.

        Returns:
            The widgets that make up the comparison dialog.
        """
        with Vertical(id="file-notes-conflict-dialog"):
            yield Static(
                "Compare conflict",
                id="file-notes-conflict-title",
                markup=False,
            )
            yield Static(
                self._relative_path,
                id="file-notes-conflict-path",
                markup=False,
            )
            yield Static(
                (
                    "Base is the editor baseline. Draft is the current editor. "
                    "Disk is the latest captured file state. Comparing does not "
                    "resolve the conflict."
                ),
                id="file-notes-conflict-help",
                markup=False,
            )
            summary = TextArea(
                self._comparison.summary_text,
                id="file-notes-conflict-summary",
                read_only=True,
                soft_wrap=True,
            )
            summary.tooltip = "Base, Draft, and Disk identities"
            yield summary
            diff = TextArea(
                self._comparison.diff_text,
                id="file-notes-conflict-diff",
                read_only=True,
                soft_wrap=False,
            )
            diff.tooltip = "Base to Draft and Base to Disk unified comparisons"
            yield diff
            yield Button("Close", id="file-notes-conflict-close", compact=True)

    def on_mount(self) -> None:
        """Focus the comparison output for immediate keyboard reading."""
        self.query_one("#file-notes-conflict-diff", TextArea).focus()

    @on(Button.Pressed, "#file-notes-conflict-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(None)


class LibraryFileNotesWorkspace(Vertical):
    """Browse and edit one disk-authoritative Markdown/text root."""

    class ReloadConfirmationChanged(Message):
        """Announce whether the destructive reload confirmation is active."""

        def __init__(self, active: bool) -> None:
            super().__init__()
            self.active = active

    DEFAULT_CSS = """
    LibraryFileNotesWorkspace {
        height: 1fr;
        min-height: 12;
        min-width: 0;
    }

    #file-notes-root-row {
        height: 1;
        min-height: 1;
        max-height: 1;
    }

    /* Pinned authority remains readable without consuming the editor's
       compact conflict controls. */
    #file-notes-authority {
        width: 100%;
        height: auto;
        min-height: 1;
        max-height: 2;
        color: $text-muted;
        text-wrap: wrap;
        overflow: hidden hidden;
    }

    #file-notes-root-status {
        width: 1fr;
        height: 1;
        min-height: 1;
        max-height: 1;
        color: $text-muted;
        text-wrap: nowrap;
        overflow: hidden hidden;
    }

    #file-notes-root-status.-warning,
    #file-notes-root-status.-offline,
    #file-notes-save-status.-conflict {
        color: $text;
        background: $warning 14%;
        text-style: bold;
    }

    #file-notes-save-status.-error {
        color: $text;
        background: $error 14%;
        text-style: bold;
    }

    /* task-2850: the "no root chosen yet" empty state is a prompt +
       adjacent action, not a status toolbar -- the ``1fr`` above is
       correct once a root is linked (it reserves room for a Details/
       Change pair pinned to the right of a long elided path), but on the
       root-less first paint it shoved "Choose folder..." ~150 columns
       away from "Choose a notes folder.", reading as two unrelated,
       far-apart widgets. Hugging the prompt's own width keeps the two
       adjacent regardless of the workspace's mounted width. */
    #file-notes-root-status.-empty-root {
        width: auto;
    }

    #file-notes-root-details,
    #file-notes-choose-root {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
    }

    #file-notes-root-details {
        display: none;
    }

    #file-notes-body {
        height: 1fr;
        min-height: 8;
    }

    #file-notes-navigator {
        width: 3fr;
        min-width: 24;
        height: 100%;
        padding-right: 1;
        border-right: solid $surface-lighten-1;
    }

    #file-notes-editor-pane {
        width: 7fr;
        min-width: 32;
        height: 100%;
        padding-left: 1;
    }

    #file-notes-search-row,
    #file-notes-path-row,
    #file-notes-search,
    #file-notes-path {
        height: 3;
        min-height: 3;
    }

    .file-notes-field-label {
        height: 3;
        min-height: 3;
        color: $text-muted;
        text-style: bold;
        content-align: left middle;
        padding-right: 1;
        text-wrap: nowrap;
    }

    #file-notes-search-label {
        width: 8;
        min-width: 8;
    }

    #file-notes-path-label {
        width: 17;
        min-width: 17;
    }

    #file-notes-search,
    #file-notes-path {
        width: 1fr;
    }

    #file-notes-tree,
    #file-notes-search-results {
        height: 1fr;
        min-height: 4;
    }

    #file-notes-breadcrumb,
    #file-notes-save-status,
    #file-notes-preview-status,
    #file-notes-action-status {
        height: auto;
        min-height: 1;
    }

    #file-notes-breadcrumb {
        text-style: bold;
    }

    #file-notes-save-status,
    #file-notes-preview-status,
    #file-notes-action-status {
        color: $text-muted;
    }

    #file-notes-preview-status {
        max-height: 3;
        text-wrap: wrap;
        overflow: hidden hidden;
    }

    #file-notes-reload-confirm-copy {
        width: 100%;
        height: auto;
        min-height: 1;
        color: $warning;
        text-style: bold;
        text-wrap: wrap;
    }

    #file-notes-reload-confirm-copy,
    #file-notes-reload-confirm-actions {
        display: none;
    }

    #file-notes-resolution-copy {
        width: 100%;
        height: auto;
        min-height: 1;
        color: $warning;
        text-style: bold;
        text-wrap: wrap;
    }

    #file-notes-resolution-copy,
    #file-notes-resolution-actions {
        display: none;
    }

    LibraryFileNotesWorkspace.-reload-confirming #file-notes-path-row {
        display: none;
    }

    #file-notes-reload-confirm {
        color: $error;
        text-style: bold;
    }

    #file-notes-editor {
        height: 1fr;
        min-height: 5;
    }

    .file-notes-toolbar {
        height: auto;
        min-height: 1;
    }

    .file-notes-toolbar.-empty-actions {
        display: none;
    }

    .file-notes-toolbar Button,
    #file-notes-back,
    #file-notes-session-changes {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
    }

    LibraryFileNotesWorkspace.-prepare-session-wide .file-notes-toolbar {
        display: none;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions .file-notes-toolbar {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        height: auto;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions .file-notes-toolbar Button {
        width: 1fr;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    .file-notes-toolbar.-confirm-delete {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        height: auto;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    .file-notes-toolbar.-confirm-delete Button {
        padding: 0;
    }

    #file-notes-delete-spacer {
        width: 1fr;
        height: 1;
        min-height: 1;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    #file-notes-delete-spacer {
        display: none;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions #file-notes-delete {
        column-span: 2;
        width: 1fr;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions #file-notes-save-copy {
        column-span: 2;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    #file-notes-maintenance-toggle {
        column-span: 2;
    }

    LibraryFileNotesWorkspace.-single-editor-actions .file-notes-toolbar {
        grid-size: 1;
        grid-columns: 1fr;
    }

    LibraryFileNotesWorkspace.-single-editor-actions #file-notes-delete,
    LibraryFileNotesWorkspace.-single-editor-actions #file-notes-save-copy,
    LibraryFileNotesWorkspace.-single-editor-actions
    #file-notes-maintenance-toggle {
        column-span: 1;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    #file-notes-resolution-actions {
        grid-size: 1;
        grid-columns: 1fr;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions
    #file-notes-resolution-actions Button {
        width: 1fr;
        padding: 0;
    }

    LibraryFileNotesWorkspace.-stack-editor-actions #file-notes-editor {
        min-height: 3;
    }

    #file-notes-delete.-confirm-delete {
        color: $error;
        text-style: bold underline;
    }
    """

    def __init__(
        self,
        *,
        root: str | Path | None | object = _UNSET,
        replica: FileNotesReplica | None | object = _UNSET,
        replica_path: str | Path | None = None,
        session_owner: FileNotesSessionOwner | None = None,
        poll_interval: float = 1.5,
        autosave_delay: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Create a retained workspace.

        Args:
            root: Explicit root for tests/embedding. When omitted, the
                canonical persisted root is loaded from CLI config.
            replica: Optional injected replica. When omitted, the standard
                user-data replica is opened off the UI loop.
            replica_path: Optional persistent replica location.
            session_owner: Optional process owner retained across workspaces.
            poll_interval: Reconciliation cadence in seconds.
            autosave_delay: Debounce delay in seconds.
            **kwargs: Textual widget arguments.
        """
        kwargs.setdefault("id", "library-file-notes-workspace")
        super().__init__(**kwargs)
        self._root_seed = root
        self._root = None if root is _UNSET else self._configured_root(root)
        self._replica_seed = replica
        self._replica_path = Path(replica_path) if replica_path is not None else None
        self._replica: FileNotesReplica | None = (
            replica if isinstance(replica, FileNotesReplica) else None
        )
        self._owns_replica = replica is _UNSET
        self._session_owner = (
            FileNotesSessionOwner() if session_owner is None else session_owner
        )
        self._owns_session_owner = session_owner is None
        self._initial_session_binding = self._session_owner.current_binding()
        self._session_binding: SessionBinding | None = None
        self._service: FileNotesService | None = None
        self._runtime_warning = ""
        self._root_status_detail = "Choose a notes folder."
        self._root_status_summary = "Choose a notes folder."

        self._poll_interval = max(0.02, poll_interval)
        self._autosave_delay = max(0.01, autosave_delay)
        self._poll_timer: Timer | None = None
        self._autosave_timer: Timer | None = None
        self._poll_worker: Worker[Any] | None = None
        self._save_worker: Worker[Any] | None = None
        self._git_status_worker: Worker[Any] | None = None
        self._git_action_worker: Worker[Any] | None = None
        self._git_status_task: asyncio.Task[SessionGitStatus] | None = None
        self._git_status_task_binding: SessionBinding | None = None
        self._active = False
        self._refresh_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        self._runtime_lock = Lock()
        self._service_lock = RLock()
        self._root_generation = 0
        self._root_transitioning = False
        self._path_transitioning = False
        self._shutdown = False

        self._entries: dict[str, FileNoteEntry] = {}
        self._deleted_paths: tuple[str, ...] = ()
        self._folder_children: dict[str, tuple[_FolderItem, ...]] = {}
        self._search_paths: tuple[str, ...] = ()
        self._restore_expanded_folders: set[str] = set()
        self._opened: OpenedFileNote | None = None
        self._current_path = ""
        self._selected_deleted_path = ""
        self._session_key = ""
        self._save_state: SaveState = "idle"
        self._save_detail = ""
        self._reload_confirmation: _ReloadConfirmation | None = None
        self._conflict_resolution_active = False
        self._delete_confirmation_path = ""
        self._search_generation = 0
        self._search_query = ""
        self._action_detail = ""
        self._initialized = False
        self._root_offline: bool | None = None
        self._narrow = False
        self._narrow_view: Literal["navigator", "editor"] = "navigator"
        self._navigator_mode: Literal["files", "search", "git"] = "files"
        self._navigator_mode_before_git: Literal["files", "search"] = "files"
        self._editor_action_layout_sync_scheduled = False
        self._editor_action_focus_target: str | None = None
        self._maintenance_expanded = False
        self._git_observed_changes: tuple[SequencedSessionChange, ...] | None = None
        self._git_refresh_timer: Timer | None = None
        self._git_refresh_after_mutation = False
        self._git_last_action: _GitLastAction | None = None
        self._push_availability: PushCandidateAvailability | None = None
        self._push_key: _PushBindingKey | None = None
        self._push_operation: RetainedPushOperation | None = None
        self._push_operation_key: _PushBindingKey | None = None
        self._push_operation_admitted = False
        self._push_service_identity: object | None = None
        self._push_latest_service_operation_id = 0
        self._push_operation_id = 0
        self._push_phase: Literal[
            "idle",
            "checking",
            "pushing",
            "needs_attention",
        ] = "idle"
        self._push_view_phase: PushPanelPhase = "list"
        self._push_authorization_projection: (
            PushAuthorizationProjection | None
        ) = None
        self._push_review_handle: PushReviewHandle | None = None
        self._push_review_projection: PushPanelReviewProjection | None = None
        self._push_result: object | None = None
        self._push_result_projection: PushPanelResultProjection | None = None
        self._push_authorization_dialog_operation_id: int | None = None
        self._push_observer_task: asyncio.Task[None] | None = None
        self._push_settlement_tasks: set[asyncio.Task[None]] = set()
        self._commit_availability: CommitDraftProjection | None = None
        self._commit_draft: _CommitDraftState | None = None
        self._commit_view_phase: Literal[
            "list",
            "form",
            "checking",
            "review",
            "confirming",
            "executing",
            "result",
        ] = "list"
        self._commit_operation_id = 0
        self._commit_operation: RetainedCommitOperation | None = None
        self._commit_review_handle: CommitReviewHandle | None = None
        self._commit_review_projection: CommitPanelReviewProjection | None = None
        self._commit_result_projection: CommitResultProjection | None = None
        self._commit_editor_lease: _EditorReadOnlyLease | None = None
        self._commit_settlement_tasks: set[asyncio.Task[None]] = set()
        self._git_commit_worker: Worker[Any] | None = None
        self._git_commit_child_worker: Worker[Any] | None = None
        self._editor_read_only_leases: dict[object, _EditorReadOnlyLease] = {}
        self._git_panel_widget = LibraryFileNotesGitPanel()
        # The editor itself is retained across parent Library recompositions.
        # Textual calls ``compose`` again when this same workspace object is
        # remounted, so constructing it inside ``compose`` would silently
        # replace the user's draft with a new TextArea instance.
        self._editor_widget = TextArea(
            "",
            id="file-notes-editor",
            read_only=True,
        )

    @staticmethod
    def _configured_root(value: object) -> Path | None:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            return None
        return Path(value).expanduser()

    @staticmethod
    def _canonical_root(value: object) -> Path | None:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            return None
        return Path(value).expanduser().resolve(strict=False)

    @property
    def root(self) -> Path | None:
        """Return the canonical configured root."""
        return self._root

    @property
    def initialized(self) -> bool:
        """Return whether the initial background scan has completed."""
        return self._initialized

    @property
    def entries(self) -> dict[str, FileNoteEntry]:
        """Return a snapshot of visible active files keyed by relative path."""
        return dict(self._entries)

    @property
    def current_path(self) -> str:
        """Return the open relative path, if any."""
        return self._current_path

    @property
    def current_document(self) -> OpenedFileNote | None:
        """Return the exact-byte baseline for the open editor."""
        return self._opened

    @property
    def session_key(self) -> str:
        """Return the current open/reload editing-session key."""
        return self._session_key

    @property
    def save_state(self) -> SaveState:
        """Return the current editor save state."""
        return self._save_state

    @property
    def reload_confirmation_active(self) -> bool:
        """Return whether destructive conflict reload awaits a decision."""
        return self._reload_confirmation is not None

    @property
    def conflict_resolution_active(self) -> bool:
        """Return whether the bounded conflict choices are disclosed."""
        return self._conflict_resolution_active

    @property
    def leave_allowed(self) -> bool:
        """Return whether the retained draft can be left without a flush."""
        binding = self._session_binding
        return (
            not self._root_transitioning
            and not self._path_transitioning
            and not (
                binding is not None
                and self._session_owner.mutation_active(binding)
            )
            and self._save_state not in {"dirty", "saving", "conflict", "error"}
        )

    def acquire_transition(
        self,
        kind: SessionTransitionKind,
    ) -> Callable[[], None] | Literal[False] | None:
        """Synchronously admit an exact-binding screen or source transition."""
        if kind not in {"screen", "source"}:
            raise ValueError(f"Unsupported workspace transition kind: {kind}")
        binding = self._session_binding
        if binding is None:
            return None
        lease = self._session_owner.try_acquire_transition(binding, kind)
        if lease is None:
            return False
        return lease.release

    @property
    def narrow(self) -> bool:
        """Return whether actual mounted width selected narrow navigation."""
        return self._narrow

    @property
    def navigator_visible(self) -> bool:
        """Return whether the navigator pane is currently displayed."""
        return self.query_one("#file-notes-navigator").display

    @property
    def editor_visible(self) -> bool:
        """Return whether the editor pane is currently displayed."""
        return self.query_one("#file-notes-editor-pane").display

    def _path_field_label_copy(self) -> str:
        """Describe the action context currently represented by the path field."""
        if self._conflict_resolution_active:
            return "New note path"
        if self._selected_deleted_path:
            return "Restore path"
        if self._opened is not None:
            return "New / move path"
        return "New path"

    @staticmethod
    def _large_file_preview_copy(opened: OpenedFileNote) -> str:
        """Describe an exact-size preview without implying editable content."""
        return (
            f"Read-only excerpt: first {LARGE_FILE_EXCERPT_CHARS:,} body "
            f"characters. Exact size: {opened.character_count:,} characters · "
            f"{opened.size:,} bytes."
        )

    def _sync_large_file_preview(self) -> None:
        """Render or hide the retained large-file preview disclosure."""
        if not self._active or not self.is_mounted:
            return
        status = self.query_one("#file-notes-preview-status", Static)
        opened = self._opened
        if opened is None or not opened.is_excerpt:
            status.update("")
            status.display = False
            return
        status.update(self._large_file_preview_copy(opened))
        status.display = True

    def compose(self) -> ComposeResult:
        yield Static(
            self._authority_copy(),
            id="file-notes-authority",
            markup=False,
        )
        with Horizontal(id="file-notes-root-row"):
            root_status = Static(
                "Choose a notes folder.",
                id="file-notes-root-status",
                markup=False,
            )
            # task-2850: match ``_update_root_surface``'s class so the very
            # first paint (before ``on_mount`` runs) already keeps the
            # prompt and its button adjacent when no root is configured.
            root_status.set_class(self._root is None, "-empty-root")
            yield root_status
            yield Button(
                "Details",
                id="file-notes-root-details",
                compact=True,
            )
            yield Button(
                "Choose folder…",
                id="file-notes-choose-root",
                compact=True,
            )
        with Horizontal(id="file-notes-body"):
            with Vertical(id="file-notes-navigator"):
                with Horizontal(id="file-notes-search-row"):
                    yield Static(
                        "Search",
                        id="file-notes-search-label",
                        classes="file-notes-field-label",
                        markup=False,
                    )
                    yield Input(
                        placeholder="File contents…",
                        id="file-notes-search",
                        value=self._search_query,
                    )
                yield Tree[object]("Files", id="file-notes-tree")
                search_results = Tree[object](
                    "Search results",
                    id="file-notes-search-results",
                )
                search_results.display = False
                yield search_results
                yield Button(
                    "Review session changes (0)",
                    id="file-notes-session-changes",
                    compact=True,
                )
                yield self._git_panel_widget
            with Vertical(id="file-notes-editor-pane"):
                back = Button(
                    "Back to navigator",
                    id="file-notes-back",
                    compact=True,
                )
                back.display = False
                yield back
                yield Static(
                    "No file selected",
                    id="file-notes-breadcrumb",
                    markup=False,
                )
                yield Static(
                    _SAVE_STATE_COPY["idle"],
                    id="file-notes-save-status",
                    markup=False,
                )
                preview_status = Static(
                    "",
                    id="file-notes-preview-status",
                    markup=False,
                )
                preview_status.display = False
                yield preview_status
                with Horizontal(id="file-notes-path-row"):
                    yield Static(
                        self._path_field_label_copy(),
                        id="file-notes-path-label",
                        classes="file-notes-field-label",
                        markup=False,
                    )
                    yield Input(
                        placeholder="relative/path.md",
                        id="file-notes-path",
                        value=self._selected_deleted_path or self._current_path,
                    )
                yield self._editor_widget
                with Horizontal(
                    id="file-notes-file-actions",
                    classes="file-notes-toolbar",
                ):
                    yield Button("New", id="file-notes-new", compact=True)
                    yield Button("Restore", id="file-notes-restore", compact=True)
                    yield Button(
                        "Compare",
                        id="file-notes-compare",
                        compact=True,
                    )
                    yield Button(
                        "Resolve conflict",
                        id="file-notes-resolve-conflict",
                        compact=True,
                    )
                    yield Button(
                        "Save draft as copy",
                        id="file-notes-save-copy",
                        compact=True,
                    )
                    yield Button(
                        "More file actions",
                        id="file-notes-maintenance-toggle",
                        compact=True,
                    )
                    yield Static("", id="file-notes-delete-spacer")
                    yield Button("Delete", id="file-notes-delete", compact=True)
                with Horizontal(
                    id="file-notes-maintenance-actions",
                    classes="file-notes-toolbar",
                ):
                    yield Button("Move", id="file-notes-move", compact=True)
                    yield Button("Protect", id="file-notes-protect", compact=True)
                    yield Button("Reload", id="file-notes-reload", compact=True)
                    yield Button("Refresh", id="file-notes-refresh", compact=True)
                yield Static(
                    (
                        "Choose a safe next step. No option overwrites the disk "
                        "file."
                    ),
                    id="file-notes-resolution-copy",
                    markup=False,
                )
                with Horizontal(
                    id="file-notes-resolution-actions",
                    classes="file-notes-toolbar",
                ):
                    keep = Button(
                        "Keep editing",
                        id="file-notes-resolution-keep",
                        compact=True,
                    )
                    keep.tooltip = "Close these choices and leave the conflict open"
                    yield keep
                    save_new = Button(
                        "Save draft as new note",
                        id="file-notes-resolution-save-new",
                        compact=True,
                    )
                    save_new.tooltip = (
                        "Write the complete draft to the New note path without "
                        "replacing an existing file"
                    )
                    yield save_new
                    discard = Button(
                        "Discard draft and load disk",
                        id="file-notes-resolution-discard",
                        compact=True,
                    )
                    discard.tooltip = (
                        "Open a separate confirmation before replacing the editor "
                        "with the current disk file"
                    )
                    yield discard
                yield Static(
                    (
                        self._reload_confirmation_copy()
                        if self.reload_confirmation_active
                        else ""
                    ),
                    id="file-notes-reload-confirm-copy",
                    markup=False,
                )
                with Horizontal(
                    id="file-notes-reload-confirm-actions",
                    classes="file-notes-toolbar",
                ):
                    yield Button(
                        "Cancel",
                        id="file-notes-reload-cancel",
                        compact=True,
                    )
                    yield Button(
                        "Discard draft and load disk",
                        id="file-notes-reload-confirm",
                        compact=True,
                    )
                yield Static("", id="file-notes-action-status", markup=False)

    def _authority_copy(self, session_git_count: int | None = None) -> str:
        """Describe disk authority, current work, and the next available action."""
        if self._root is None:
            return "Folder files · No folder selected · Next: Choose folder."
        folder_name = self._root.name or self._root.anchor or str(self._root)
        folder_label = Text(folder_name)
        folder_label.truncate(5, overflow="ellipsis")
        first_line = ["Folder files", f"Folder: {folder_label.plain}"]
        state_copy = ""
        next_action = ""
        if self._root_transitioning:
            state_copy = "Changing folder"
            next_action = "Wait for change."
        elif self._path_transitioning:
            state_copy = "File operation"
            next_action = "Wait for file."
        elif self._root_offline is None:
            state_copy = "Checking"
            next_action = "Wait for check."
        elif self._root_offline is True:
            state_copy = "Offline+Warning" if self._runtime_warning else "Offline"
            next_action = "Reconnect/change."
        elif self._runtime_warning:
            state_copy = "Warning"
            next_action = "Open Details."
        if state_copy:
            first_line.append(state_copy)
        save_copy = {
            "dirty": "Unsaved",
            "saving": "Saving",
            "saved": "Saved",
            "conflict": "Conflict",
            "error": "Save failed",
        }.get(self._save_state, "")
        if save_copy:
            first_line.append(save_copy)
        elif not state_copy:
            first_line.append("Ready")
        if session_git_count is None:
            binding = self._session_binding
            changes = (
                () if binding is None else self._session_owner.snapshot(binding).changes
            )
            session_git_count = len(coalesce_session_changes(changes))
        git_count_copy = "99+" if session_git_count > 99 else str(session_git_count)
        if self._push_phase == "idle":
            change_word = "change" if session_git_count == 1 else "changes"
            git_copy = f"Session Git: {git_count_copy} {change_word}"
        else:
            push_copy = {
                "checking": "Check push",
                "pushing": "Pushing",
                "needs_attention": "Push attention",
            }[self._push_phase]
            git_copy = f"Session Git: {git_count_copy} · {push_copy}"
        if not next_action:
            if self._save_state == "conflict":
                next_action = "Resolve/copy."
            elif self._save_state == "error":
                next_action = "Retry/copy."
            elif self._save_state == "saving":
                next_action = "Wait for save."
            elif self._save_state == "dirty":
                next_action = "Keep editing."
            elif self._push_phase != "idle" or session_git_count:
                next_action = "Review changes."
            elif self._save_state == "saved":
                next_action = "Keep editing."
            else:
                next_action = "Choose/new file."
        return (
            f"{' · '.join(first_line)}\n"
            f"{git_copy} · Next: {next_action}"
        )

    def on_mount(self) -> None:
        """Start background initialization and polling for this mount."""
        if self._shutdown:
            return
        self._active = True
        self._apply_responsive_layout(self.size.width)
        if self._opened is not None:
            self.query_one("#file-notes-breadcrumb", Static).update(
                self._opened.relative_path
            )
        elif self._selected_deleted_path:
            self.query_one("#file-notes-breadcrumb", Static).update(
                f"Recently deleted: {self._selected_deleted_path}"
            )
        self._set_save_state(self._save_state, self._save_detail)
        self._sync_large_file_preview()
        self._set_action_status(self._action_detail)
        self._update_root_surface()
        self._sync_navigator_mode()
        self._rehydrate_git_presentation()
        self._update_controls()
        self.run_worker(
            self._initialize(),
            name="file-notes-initialize",
            group="file-notes-initialize",
            exclusive=True,
        )
        self._poll_timer = self.set_interval(
            self._poll_interval,
            self._start_poll,
            pause=False,
        )

    def on_unmount(self) -> None:
        """Pause timers; Textual cancels node workers during removal."""
        self._active = False
        if self._save_state == "saving":
            self._save_state = "dirty"
            self._save_detail = "save interrupted"
        for timer in (
            self._poll_timer,
            self._autosave_timer,
            self._git_refresh_timer,
        ):
            if timer is not None:
                timer.stop()
        self._poll_timer = None
        self._autosave_timer = None
        self._git_refresh_timer = None
        self._poll_worker = None
        self._save_worker = None
        self._git_status_worker = None
        self._git_action_worker = None
        self._git_commit_worker = None
        self._git_commit_child_worker = None
        self._editor_action_layout_sync_scheduled = False

    async def shutdown(self) -> None:
        """Permanently close this workspace's owned replica once."""
        with self._runtime_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._active = False
        for timer in (
            self._poll_timer,
            self._autosave_timer,
            self._git_refresh_timer,
        ):
            if timer is not None:
                timer.stop()
        self._poll_timer = None
        self._autosave_timer = None
        self._git_refresh_timer = None
        if self._owns_session_owner:
            await asyncio.to_thread(self._session_owner.shutdown)
        elif self._owns_replica:
            await asyncio.to_thread(
                self._session_owner.wait_for_root_commit,
            )
        if self._owns_replica:
            await asyncio.to_thread(self._close_owned_replica)

    def _close_owned_replica(self) -> None:
        with self._runtime_lock:
            replica = self._replica
            if replica is None:
                return
            service = self._service
            if service is not None:
                service.close()
            else:
                with self._service_lock:
                    replica.close()
            self._replica = None
            self._service = None

    def on_resize(self, event: Resize) -> None:
        """Choose wide or narrow panes from the mounted workspace width."""
        self._apply_responsive_layout(event.size.width)
        self.call_after_refresh(self._fit_root_status)

    async def _initialize(self) -> None:
        generation = self._root_generation
        expected_binding = (
            self._session_binding
            if self._session_binding is not None
            else self._initial_session_binding
        )
        previous_service = self._service
        was_initialized = self._initialized
        root, replica, service, warning = await asyncio.to_thread(
            self._build_runtime,
            generation,
            expected_binding,
        )
        if not self._active or generation != self._root_generation:
            return
        resuming = was_initialized and service is previous_service
        self._root = root
        self._replica = replica
        self._service = service
        self._runtime_warning = warning
        self._update_root_surface()
        self._update_controls()
        if service is None:
            self._initialized = True
            self._update_root_surface()
            self._update_controls()
            return
        if resuming:
            result = await asyncio.to_thread(service.reconcile)
            deleted = await self._load_deleted_paths(
                replica=replica,
                service=service,
            )
            if (
                not self._active
                or generation != self._root_generation
                or service is not self._service
            ):
                return
            if not self._apply_reconcile(result, deleted):
                return
            await self._handle_open_external_change(result, resuming=True)
            if (
                self._active
                and generation == self._root_generation
                and service is self._service
                and self._save_state == "dirty"
            ):
                self._arm_autosave()
            return
        result = await asyncio.to_thread(service.scan)
        deleted = await self._load_deleted_paths(replica=replica, service=service)
        if (
            not self._active
            or generation != self._root_generation
            or service is not self._service
        ):
            return
        self._initialized = True
        self._apply_scan(result, deleted)

    def _build_runtime(
        self,
        expected_generation: int,
        _expected_binding: SessionBinding | None,
        *,
        bind_session: bool = True,
    ) -> tuple[
        Path | None,
        FileNotesReplica | None,
        FileNotesService | None,
        str,
    ]:
        with self._runtime_lock:
            if self._shutdown:
                return self._root, self._replica, None, ""
            configured_root = self._root
            if self._root_seed is _UNSET and configured_root is None:
                configured_root = get_cli_setting("file_notes", "root", None)
            root = self._canonical_root(configured_root)
            replica = self._replica
            previous_replica = replica
            warning = ""
            if self._replica_seed is _UNSET and replica is None:
                try:
                    replica_path = self._replica_path or (
                        get_user_data_dir() / "file_notes.sqlite"
                    )
                    replica = FileNotesReplica(replica_path)
                    # Keep the handle even if the awaiting mount is cancelled.
                    # A rapid remount will reuse it instead of opening a second
                    # connection after the background thread completes.
                    self._replica = replica
                except Exception as error:
                    replica = None
                    warning = f"Recovery unavailable: {error}"
            generation_is_current = expected_generation == self._root_generation
            if not bind_session or not generation_is_current:
                return root, replica, self._service, warning

        stable_root = self._session_owner.acquire_stable_root(root)
        if stable_root is None:
            return root, replica, None, warning
        try:
            binding = stable_root.binding
            root = None if binding is None else Path(binding.root_key)
            with self._runtime_lock:
                if self._shutdown:
                    return self._root, self._replica, None, warning
                if expected_generation != self._root_generation:
                    return self._root, self._replica, self._service, warning
                service = self._service
                if root is None:
                    service = None
                elif (
                    service is None
                    or service.root_key != str(root)
                    or previous_replica is not replica
                    or self._session_binding != binding
                ):
                    assert binding is not None
                    service = FileNotesService(
                        root,
                        replica,
                        operation_lock=self._service_lock,
                        session_owner=self._session_owner,
                        session_binding=binding,
                    )
                self._root = root
                self._replica = replica
                self._service = service
                self._session_binding = binding
                self._runtime_warning = warning
                return root, replica, service, warning
        finally:
            stable_root.release()

    async def _commit_root_candidate(
        self,
        root: Path,
        service: FileNotesService,
        result: ScanResult,
        deleted: tuple[str, ...],
        generation: int,
        expected_binding: SessionBinding | None,
        *,
        persist: bool,
    ) -> bool:
        with self._runtime_lock:
            if (
                self._shutdown
                or not self._active
                or generation != self._root_generation
                or service.root_key != str(root)
            ):
                return False
            reservation = self._session_owner.try_reserve_root(
                root,
                expected_binding=expected_binding,
            )
        if reservation is None:
            return False

        cancellation: asyncio.CancelledError | None = None
        persistence_warning = ""
        try:
            if persist:
                persistence_task = asyncio.create_task(
                    asyncio.to_thread(
                        apply_settings_mutation_to_cli_config,
                        {"file_notes": {"root": str(root)}},
                    )
                )
                while not persistence_task.done():
                    try:
                        await asyncio.shield(persistence_task)
                    except asyncio.CancelledError as error:
                        if cancellation is None:
                            cancellation = error
                    except BaseException:
                        break
                try:
                    persistence = persistence_task.result()
                except BaseException as error:
                    if cancellation is not None:
                        raise cancellation from error
                    raise
                if not persistence.file_replaced:
                    if cancellation is not None:
                        raise cancellation
                    return False
                if persistence.failure_phase == "cache_reload":
                    persistence_warning = (
                        "Root saved, but configuration cache reload failed."
                    )

            def publish(binding: SessionBinding) -> None:
                binding_changed = binding != self._session_binding
                if binding_changed:
                    self._clear_git_last_action()
                    self._invalidate_commit_binding(
                        "Selected notes root changed; the commit draft was cleared."
                    )
                    self._git_observed_changes = None
                    self._git_status_task = None
                    self._git_status_task_binding = None
                    if self._active and self.is_mounted:
                        self._git_panel_widget.render_unavailable(
                            "Selected notes root changed. Open Review session "
                            "changes to check the new root."
                        )
                service._bind_session_owner(self._session_owner, binding)
                self._root = root
                self._session_binding = binding
                self._service = service
                self._clear_open_document()
                self._initialized = True
                self._apply_scan(
                    result,
                    deleted,
                    additional_warning=persistence_warning,
                )

            # Atomic file replacement is the point of no return. There is
            # deliberately no await between observing it and synchronously
            # aligning the owner, service, workspace, and scan projection under
            # this reservation.
            with self._runtime_lock:
                reservation.commit(publish)
            if cancellation is not None:
                raise cancellation
            return True
        finally:
            reservation.release()

    async def _load_deleted_paths(
        self,
        *,
        replica: FileNotesReplica | None = None,
        service: FileNotesService | None = None,
    ) -> tuple[str, ...]:
        replica = self._replica if replica is None else replica
        service = self._service if service is None else service
        if replica is None or service is None:
            return ()
        try:
            paths = await asyncio.to_thread(
                replica.list_deleted,
                service.root_key,
            )
        except Exception as error:
            self._runtime_warning = f"Recovery unavailable: {error}"
            return ()
        return tuple(paths)

    def _apply_scan(
        self,
        result: ScanResult,
        deleted: tuple[str, ...],
        *,
        additional_warning: str = "",
    ) -> bool:
        self._adopt_scan_state(
            result,
            deleted,
            additional_warning=additional_warning,
        )
        return self._render_scan_state()

    def _adopt_scan_state(
        self,
        result: ScanResult,
        deleted: tuple[str, ...],
        *,
        additional_warning: str = "",
    ) -> None:
        """Install a scan projection without requiring mounted widgets."""
        self._entries = {entry.relative_path: entry for entry in result.entries}
        self._deleted_paths = deleted
        self._root_offline = result.offline
        self._runtime_warning = "; ".join(
            warning
            for warning in (result.replica_warning, additional_warning)
            if warning
        )

    def _render_projection(
        self,
        *,
        offline: bool | None,
        rebuild_tree: bool,
    ) -> bool:
        """Best-effort render after state adoption across widget teardown."""
        if not self._active or not self.is_mounted or not self.children:
            return False
        try:
            self._update_root_surface(offline=offline)
            if rebuild_tree:
                self._rebuild_tree()
            self._refresh_active_search()
            self._refresh_session_changes()
            self._update_controls()
        except NoMatches:
            # Textual removes descendants before this widget's ``on_unmount``
            # flips ``_active``. A polling/scan worker may therefore pass its
            # lifecycle guard just before the projection nodes disappear.
            return False
        return True

    def _render_scan_state(self) -> bool:
        """Render the installed scan projection when the widget tree is live."""
        return self._render_projection(
            offline=self._root_offline,
            rebuild_tree=True,
        )

    def _apply_reconcile(
        self,
        result: ReconcileResult,
        deleted: tuple[str, ...],
    ) -> bool:
        new_entries = {entry.relative_path: entry for entry in result.entries}
        navigator_changed = (
            new_entries != self._entries or deleted != self._deleted_paths
        )
        self._entries = new_entries
        self._deleted_paths = deleted
        self._root_offline = result.offline
        self._runtime_warning = result.replica_warning or ""
        return self._render_projection(
            offline=result.offline,
            rebuild_tree=navigator_changed,
        )

    def _update_root_surface(self, *, offline: bool | None = None) -> None:
        if not self._active or not self.is_mounted or not self.children:
            return
        try:
            authority = self.query_one("#file-notes-authority", Static)
            status = self.query_one("#file-notes-root-status", Static)
            body = self.query_one("#file-notes-body")
            details = self.query_one("#file-notes-root-details", Button)
            choose = self.query_one("#file-notes-choose-root", Button)
        except NoMatches:
            return
        binding = self._session_binding
        mutation_active = (
            binding is not None
            and self._session_owner.mutation_active(binding)
        )
        choose.disabled = (
            self._root_transitioning
            or self._path_transitioning
            or mutation_active
        )
        if self._root is None:
            self._root_status_detail = "Choose a notes folder."
            self._root_status_summary = self._root_status_detail
            status.tooltip = None
            status.update(self._root_status_summary)
            status.set_class(True, "-empty-root")
            status.set_class(False, "-warning")
            status.set_class(False, "-offline")
            body.display = False
            details.display = False
            choose.label = "Choose folder…"
            choose.display = True
            authority.update(self._authority_copy())
            return
        status.set_class(False, "-empty-root")
        is_offline = self._root_offline if offline is None else offline
        status.set_class(bool(self._runtime_warning), "-warning")
        status.set_class(is_offline is True, "-offline")
        state = (
            "Checking"
            if is_offline is None
            else ("Offline" if is_offline else "Linked")
        )
        detail = f"{state} — {self._root}"
        if self._runtime_warning:
            detail = f"{detail} · {self._runtime_warning}"
        self._root_status_detail = detail
        folder_name = self._root.name or self._root.anchor or str(self._root)
        display_state = (
            "Offline · Warning"
            if is_offline is True and self._runtime_warning
            else ("Warning" if self._runtime_warning else state)
        )
        self._root_status_summary = (
            f"{display_state} · Local folder: {folder_name}"
        )
        status.tooltip = Text(detail)
        status.update(self._root_status_summary)
        body.display = True
        details.display = True
        choose.label = "Change…"
        choose.display = True
        authority.update(self._authority_copy())
        self._apply_responsive_layout(self.size.width)
        self.call_after_refresh(self._fit_root_status)

    def _fit_root_status(self) -> None:
        """Fit the friendly root summary while retaining exact detail."""
        if not self._active or not self.is_mounted or not self.children:
            return
        try:
            status = self.query_one("#file-notes-root-status", Static)
        except NoMatches:
            return
        width = status.content_region.width
        if width > 0:
            status.update(_middle_elide_cells(self._root_status_summary, width))

    def _apply_responsive_layout(self, width: int) -> None:
        if not self._active or not self.is_mounted:
            return
        was_narrow = self._narrow
        self._narrow = width < 80
        if self._narrow and not was_narrow and (
            self._opened is not None or self._selected_deleted_path
        ):
            # task-15790 (bisected to 4202930d6's era, born-green then
            # regressed): `_narrow_view` was only ever set to "editor" when a
            # document was opened WHILE ALREADY NARROW, so opening on a wide
            # terminal and then shrinking routed the narrow shell to the
            # NAVIGATOR -- hiding the pane out from under its own focused
            # editor (the hide blurs it; focus fell to None). Entering narrow
            # with an open document means the user's context is that
            # document. Transition-only on purpose: while ALREADY narrow,
            # Back's explicit navigator choice must keep winning.
            self._narrow_view = "editor"
        navigator = self.query_one("#file-notes-navigator")
        editor = self.query_one("#file-notes-editor-pane")
        back = self.query_one("#file-notes-back", Button)
        if self._narrow:
            navigator.display = self._narrow_view == "navigator"
            editor.display = self._narrow_view == "editor"
            back.display = self._narrow_view == "editor"
        else:
            navigator.display = True
            editor.display = True
            back.display = False
        self._sync_navigator_mode()
        self._schedule_editor_action_layout()

    def _sync_navigator_mode(self) -> None:
        """Show one retained navigator surface without remounting its peers."""
        if not self._active or not self.is_mounted:
            return
        search_row = self.query_one("#file-notes-search-row")
        tree = self.query_one("#file-notes-tree", Tree)
        results = self.query_one("#file-notes-search-results", Tree)
        entry = self.query_one("#file-notes-session-changes", Button)
        panel = self.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        git_visible = self._navigator_mode == "git"
        panel.display = git_visible
        search_row.display = not git_visible
        entry.display = not git_visible
        tree.display = not git_visible and self._navigator_mode == "files"
        results.display = not git_visible and self._navigator_mode == "search"
        self.set_class(
            git_visible and not self._narrow,
            "-prepare-session-wide",
        )

    def _schedule_editor_action_layout(self) -> None:
        """Coalesce editor-action measurements after Textual refreshes layout."""
        if (
            not self._active
            or not self.is_mounted
            or self._editor_action_layout_sync_scheduled
        ):
            return
        self._editor_action_layout_sync_scheduled = True
        self.call_after_refresh(self._sync_editor_action_layout)

    def _set_delete_confirmation(self, relative_path: str = "") -> None:
        """Project one confirmation state into its copy and narrow layout."""
        confirmed = bool(relative_path)
        self._delete_confirmation_path = relative_path
        delete = self.query_one("#file-notes-delete", Button)
        delete.label = "Confirm delete" if confirmed else "Delete"
        delete.set_class(confirmed, "-confirm-delete")
        delete.refresh(layout=True)
        toolbar = delete.parent
        if toolbar is not None:
            toolbar.set_class(confirmed, "-confirm-delete")
            toolbar.refresh(layout=True)
        self._schedule_editor_action_layout()

    @staticmethod
    def _reload_confirmation_copy() -> str:
        """Return complete destructive reload copy."""
        return (
            "Discard the draft in the editor and load the current disk version? "
            "This cannot be undone."
        )

    def _reload_confirmation_is_current(
        self,
        confirmation: _ReloadConfirmation,
    ) -> bool:
        """Validate every retained identity before destructive replacement."""
        return (
            self._active
            and self.is_mounted
            and self._service is confirmation.service
            and self._root_generation == confirmation.root_generation
            and self._session_binding == confirmation.binding
            and self._session_owner.current_binding() == confirmation.binding
            and self._opened is confirmation.opened
            and self._current_path == confirmation.opened.relative_path
            and self._session_key == confirmation.session_key
            and self._save_state == confirmation.save_state
        )

    def _set_reload_confirmation(
        self,
        confirmation: _ReloadConfirmation,
    ) -> None:
        """Show the retained inline decision and focus the safe default."""
        was_active = self.reload_confirmation_active
        self._reload_confirmation = confirmation
        if self._active and self.is_mounted:
            self.query_one("#file-notes-reload-confirm-copy", Static).update(
                self._reload_confirmation_copy()
            )
            self._update_controls()
            cancel = self.query_one("#file-notes-reload-cancel", Button)
            # task-15790: `call_after_refresh` here NEVER FIRED -- it waits
            # for THIS widget's own refresh, and `_update_controls` patches
            # children in place, so no workspace-level refresh ever comes.
            # The cancel-first safety focus therefore never happened: the
            # feature's own test was born red at 1fbd46ec6 and nobody saw
            # it until the full-suite sweep. `call_later` orders on the
            # message queue instead, which needs no repaint to flush; focus
            # placement does not need paint.
            self.call_later(cancel.focus)
        if not was_active:
            self.post_message(self.ReloadConfirmationChanged(True))

    def _dismiss_reload_confirmation(self, *, focus_opener: bool) -> bool:
        """Close a pending reload decision without changing editor content."""
        confirmation = self._reload_confirmation
        if confirmation is None:
            return False
        self._reload_confirmation = None
        if self._active and self.is_mounted:
            self.query_one("#file-notes-reload-confirm-copy", Static).update("")
            self._update_controls()
            if focus_opener:
                opener = self.query_one(f"#{confirmation.opener_id}", Button)
                if opener.display and not opener.disabled:
                    self.call_after_refresh(opener.focus)
        self.post_message(self.ReloadConfirmationChanged(False))
        return True

    def cancel_reload_confirmation(self) -> bool:
        """Cancel destructive reload and restore focus to its opener."""
        return self._dismiss_reload_confirmation(focus_opener=True)

    def _sync_editor_action_layout(self) -> None:
        """Stack current editor actions only when their labels need the space."""
        self._editor_action_layout_sync_scheduled = False
        if not self._active or not self.is_mounted:
            return
        pane = self.query_one("#file-notes-editor-pane")
        if not pane.display:
            return
        available_width = pane.content_region.width
        if available_width <= 0:
            return
        single_column = available_width <= 40
        needs_stack = single_column or any(
            toolbar.display
            and sum(
                cell_len(str(button.label)) + 4
                for button in toolbar.query(Button)
                if button.display
            )
            > available_width
            for toolbar in pane.query(".file-notes-toolbar")
        )
        self.set_class(needs_stack, "-stack-editor-actions")
        self.set_class(single_column, "-single-editor-actions")
        delete = self.query_one("#file-notes-delete", Button)
        self.query_one("#file-notes-delete-spacer", Static).display = (
            delete.display and not needs_stack
        )

    def _rebuild_tree(self) -> None:
        if not self._active or not self.is_mounted:
            return
        tree = self.query_one("#file-notes-tree", Tree)
        expanded_folders: set[str] = set()

        def remember_expanded(node: Any) -> None:
            data = node.data
            if node.is_expanded and isinstance(data, _FolderNodeData):
                expanded_folders.add(data.relative_path)
            for child in node.children:
                remember_expanded(child)

        for child in tree.root.children:
            remember_expanded(child)
        self._folder_children = self._build_folder_index(tuple(self._entries))
        self._restore_expanded_folders = expanded_folders
        root_label = self._root.name if self._root is not None else "Files"
        tree.reset(Text(root_label or "Files"))
        self._append_tree_page(
            tree.root,
            _TreePageData("files", "", 0),
        )
        if self._deleted_paths:
            deleted = tree.root.add(
                Text("Recently deleted"),
                data=_DeletedFolderData(),
            )
            self._populate_folder_node(deleted)
            deleted.expand()
        tree.root.expand()

    @staticmethod
    def _build_folder_index(
        relative_paths: tuple[str, ...],
    ) -> dict[str, tuple[_FolderItem, ...]]:
        """Build a pure path index without constructing Textual nodes."""
        children: dict[str, dict[str, _FolderItem]] = {"": {}}
        for relative_path in sorted(relative_paths):
            parts = PurePosixPath(relative_path).parts
            parent_key = ""
            for part in parts[:-1]:
                key = f"{parent_key}/{part}".lstrip("/")
                children.setdefault(parent_key, {})[part] = _FolderItem(
                    "folder",
                    part,
                    key,
                )
                children.setdefault(key, {})
                parent_key = key
            children.setdefault(parent_key, {})[parts[-1]] = _FolderItem(
                "file",
                parts[-1],
                relative_path,
            )
        return {
            folder: tuple(
                sorted(
                    items.values(),
                    key=lambda item: (
                        item.kind != "folder",
                        item.label.casefold(),
                        item.label,
                    ),
                )
            )
            for folder, items in children.items()
        }

    def _page_items(self, page: _TreePageData) -> tuple[_FolderItem, ...]:
        """Return the immutable source rows represented by one page cursor."""
        if page.source == "files":
            return self._folder_children.get(page.folder_key, ())
        if page.source == "deleted":
            return tuple(
                _FolderItem("deleted", path, path) for path in self._deleted_paths
            )
        return tuple(_FolderItem("file", path, path) for path in self._search_paths)

    def _append_tree_page(
        self,
        parent: Any,
        page: _TreePageData,
        *,
        page_node: Any | None = None,
    ) -> Any | None:
        """Materialize at most one fixed batch before an optional cursor row."""
        items = self._page_items(page)
        end = min(page.offset + FILE_TREE_BATCH_SIZE, len(items))
        last_node: Any | None = None
        for item in items[page.offset:end]:
            if item.kind == "folder":
                last_node = parent.add(
                    Text(item.label),
                    data=_FolderNodeData(item.value),
                    before=page_node,
                    allow_expand=True,
                )
                if item.value in self._restore_expanded_folders:
                    self.call_later(self._restore_folder_expansion, last_node)
            else:
                last_node = parent.add_leaf(
                    Text(item.label),
                    data=(item.kind, item.value),
                    before=page_node,
                )

        remaining = len(items) - end
        if remaining > 0:
            next_page = _TreePageData(page.source, page.folder_key, end)
            label = Text(f"Load more ({remaining:,} remaining)")
            if page_node is None:
                parent.add_leaf(label, data=next_page)
            else:
                page_node.data = next_page
                page_node.set_label(label)
        elif page_node is not None:
            page_node.remove()
        return last_node

    def _restore_folder_expansion(self, node: Any) -> None:
        """Restore one expanded level in its own message-loop callback."""
        if not self._active or not self.is_mounted:
            return
        self._populate_folder_node(node)
        node.expand()

    def _populate_folder_node(self, node: Any) -> None:
        """Populate the first bounded batch for an expanded empty folder."""
        if node.children:
            return
        data = node.data
        if isinstance(data, _FolderNodeData):
            page = _TreePageData("files", data.relative_path, 0)
        elif isinstance(data, _DeletedFolderData):
            page = _TreePageData("deleted", "", 0)
        else:
            return
        self._append_tree_page(node, page)

    def _rebuild_search_results(self, paths: tuple[str, ...]) -> None:
        if not self._active or not self.is_mounted:
            return
        results = self.query_one("#file-notes-search-results", Tree)
        self._search_paths = paths
        results.reset(Text("Search results"))
        self._append_tree_page(
            results.root,
            _TreePageData("search", "", 0),
        )
        results.root.expand()

    def _refresh_active_search(self) -> None:
        if not self._active or not self.is_mounted:
            return
        query = self.query_one("#file-notes-search", Input).value.strip()
        if query:
            self._start_search(query)

    def _refresh_session_changes(self) -> None:
        if not self._active or not self.is_mounted:
            return
        binding = self._session_binding
        changes: tuple[SequencedSessionChange, ...] = ()
        if binding is not None:
            snapshot = self._session_owner.snapshot(binding)
            changes = snapshot.changes
            service = self._session_git_service()
            self._rehydrate_push_state(service, binding, snapshot)
        self._sync_git_last_action()
        count = len(coalesce_session_changes(changes))
        self._render_session_git_label(count)
        prior = self._git_observed_changes
        self._git_observed_changes = changes
        if prior is None or prior == changes or binding is None:
            return
        self._git_panel_widget.mark_stale(
            retain_rows=self._git_can_retain_rows(binding),
        )
        if self._navigator_mode == "git":
            self._schedule_git_refresh()

    def _schedule_git_refresh(self) -> None:
        """Debounce visible mutation-driven refresh requests."""
        if self._navigator_mode != "git":
            return
        if self._git_refresh_timer is not None:
            self._git_refresh_timer.stop()
        self._git_refresh_timer = self.set_timer(
            0.05,
            self._debounced_git_refresh,
        )

    def _debounced_git_refresh(self) -> None:
        self._git_refresh_timer = None
        self._start_git_refresh()

    def _session_git_service(self) -> _SessionGitService | None:
        service = self._session_owner.attached_git_service()
        return None if service is None else cast(_SessionGitService, service)

    def _render_session_git_label(self, count: int | None = None) -> None:
        """Compose the outcome-led Git entry from independent session state."""
        if not self._active or not self.is_mounted:
            return
        if count is None:
            binding = self._session_binding
            changes = (
                ()
                if binding is None
                else self._session_owner.snapshot(binding).changes
            )
            count = len(coalesce_session_changes(changes))
        suffix = {
            "checking": " · Checking push",
            "pushing": " · Pushing",
            "needs_attention": " · Push needs attention",
        }.get(self._push_phase, "")
        try:
            entry = self.query_one("#file-notes-session-changes", Button)
            authority = self.query_one("#file-notes-authority", Static)
        except NoMatches:
            return
        entry.label = f"Review session changes ({count}){suffix}"
        authority.update(self._authority_copy(count))

    def _clear_push_presentation(self) -> None:
        """Retire visible push state without canceling service-owned work."""
        self._push_operation_id += 1
        self._push_availability = None
        self._push_operation = None
        self._push_operation_key = None
        self._push_operation_admitted = False
        self._push_observer_task = None
        self._push_key = None
        self._push_result = None
        self._push_view_phase = "list"
        self._push_authorization_projection = None
        self._push_review_handle = None
        self._push_review_projection = None
        self._push_result_projection = None
        self._push_authorization_dialog_operation_id = None
        self._push_phase = "idle"
        if self._git_panel_widget.is_attached:
            self._git_panel_widget.clear_push_availability()
            self._git_panel_widget.return_to_push_list()
        self._render_session_git_label()

    @staticmethod
    def _push_key_for_availability(
        binding: SessionBinding,
        availability: PushCandidateAvailability | None,
    ) -> _PushBindingKey | None:
        """Build the push key without consulting commit draft or status state."""
        if availability is None:
            return None
        return _PushBindingKey(
            binding,
            availability.generation,
            availability.candidate,
        )

    @classmethod
    def _push_key_for_operation(
        cls,
        operation: RetainedPushOperation,
    ) -> _PushBindingKey:
        """Build the immutable identity carried by one retained operation."""
        key = cls._push_key_for_availability(
            operation.binding,
            operation.candidate,
        )
        assert key is not None
        return key

    def _push_operation_is_current(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
    ) -> bool:
        """Reject callbacks from any other binding, candidate, or operation."""
        return (
            self._push_operation is operation
            and self._push_operation_admitted
            and self._push_operation_id == operation_id
            and self._push_operation_key == key
            and self._push_key == key
            and operation.binding == key.binding
            and key.binding == self._session_binding
            and key.binding == self._session_owner.current_binding()
        )

    @staticmethod
    def _settled_push_phase(result: object) -> Literal[
        "idle", "needs_attention"
    ]:
        """Map typed service outcomes to the persistent indicator."""
        if isinstance(result, PushDestinationPolicyResult):
            return "needs_attention" if result.state != "ready" else "idle"
        if isinstance(result, PushPreflightResult):
            return "needs_attention" if result.state == "blocked" else "idle"
        if isinstance(result, PushExecutionResult):
            return (
                "needs_attention"
                if result.state
                in {"blocked", "failed_no_update_observed", "uncertain"}
                else "idle"
            )
        if isinstance(result, PushRecoveryProjection):
            return "idle" if result.state == "succeeded" else "needs_attention"
        return "needs_attention"

    def _sync_push_availability_projection(self) -> None:
        """Render only the current owner projection at the list-level seam."""
        if not self._active or not self._git_panel_widget.is_attached:
            return
        if self._push_availability is None:
            self._git_panel_widget.clear_push_availability()
        else:
            self._git_panel_widget.render_push_availability(
                self._push_availability
            )

    @staticmethod
    def _push_view_for_operation(
        operation: RetainedPushOperation,
    ) -> PushPanelPhase:
        if operation.kind == "local_proof":
            return "checking_candidate"
        if operation.kind == "recovery":
            return "checking_uncertain"
        if operation.kind == "push" and operation.child_started:
            return "pushing"
        return "checking_remote"

    def _render_current_push_view(self) -> None:
        """Render cached push state only when its Prepare panel is visible."""
        if (
            not self._active
            or not self._git_panel_widget.is_attached
            or self._navigator_mode != "git"
        ):
            return
        operation_id = self._push_operation_id
        phase = self._push_view_phase
        if phase == "list":
            self._git_panel_widget.return_to_push_list()
        elif phase in {
            "checking_candidate",
            "checking_remote",
            "checking_uncertain",
            "pushing",
        }:
            self._git_panel_widget.render_push_progress(
                phase,
                operation_id=operation_id
            )
        elif phase == "review" and self._push_review_projection is not None:
            self._git_panel_widget.render_push_review(
                self._push_review_projection,
                operation_id=operation_id,
            )
        elif phase == "result" and self._push_result_projection is not None:
            self._git_panel_widget.render_push_result(
                self._push_result_projection,
                operation_id=operation_id,
            )

    def _start_push_observer(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
    ) -> None:
        """Attach one process-lifetime observer without transferring ownership."""
        self._push_operation_id += 1
        operation_id = self._push_operation_id
        self._push_operation = operation
        self._push_operation_key = key
        self._push_operation_admitted = True
        self._push_key = key
        self._push_result = None
        self._push_result_projection = None
        self._push_view_phase = self._push_view_for_operation(operation)
        if operation.kind == "local_proof":
            self._push_authorization_projection = None
            self._push_review_handle = None
            self._push_review_projection = None
        elif operation.kind == "preflight":
            self._push_authorization_projection = None
            self._push_review_handle = None
            self._push_review_projection = None
        elif operation.kind == "push":
            self._push_review_handle = None
        self._push_phase = (
            "pushing"
            if operation.kind == "push" and operation.child_started
            else "checking"
        )
        self._render_session_git_label()
        task = asyncio.create_task(
            self._observe_push_operation(operation, key, operation_id)
        )
        self._push_observer_task = task
        self._push_settlement_tasks.add(task)
        task.add_done_callback(self._push_settlement_tasks.discard)
        self._render_current_push_view()

    async def _observe_push_operation(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
    ) -> None:
        """Cache exact push settlement even while its panel is hidden."""
        try:
            if operation.kind == "push" and not operation.settled:
                child_started = await operation.wait_child_started()
                if (
                    child_started
                    and not operation.settled
                    and self._push_operation_is_current(
                        operation,
                        key,
                        operation_id,
                    )
                ):
                    self._push_phase = "pushing"
                    self._push_view_phase = "pushing"
                    self._render_session_git_label()
                    self._render_current_push_view()
            result = await operation.wait()
        except (asyncio.CancelledError, Exception):
            self._publish_push_observer_failure(operation, key, operation_id)
            return
        if not self._push_operation_is_current(operation, key, operation_id):
            return
        self._push_result = result
        self._push_phase = self._settled_push_phase(result)
        snapshot = self._session_owner.snapshot(key.binding)
        self._push_availability = snapshot.push_candidate
        self._sync_push_availability_projection()
        self._render_session_git_label()
        self._publish_push_settlement(
            operation,
            key,
            operation_id,
            result,
            snapshot,
        )

    def _publish_push_observer_failure(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
    ) -> None:
        """Replace progress with sanitized copy for one exact observer fault."""
        if not self._push_operation_is_current(operation, key, operation_id):
            return
        pre_network = operation.kind in {"local_proof", "preflight"}
        self._push_phase = "needs_attention"
        self._set_push_result_projection(
            PushPanelResultProjection(
                title=(
                    "Push check needs attention"
                    if pre_network
                    else "Push outcome needs attention"
                ),
                message=(
                    "Chatbook could not complete the exact push check. "
                    "Review the candidate again."
                    if pre_network
                    else "Chatbook could not observe the exact operation "
                    "result. Inspect the configured destination externally "
                    "before taking further action."
                ),
                action=(
                    "review_again" if pre_network else "back_to_session"
                ),
            )
        )
        self._render_session_git_label()

    def _publish_push_settlement(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
        result: (
            PushDestinationPolicyResult
            | PushPreflightResult
            | PushExecutionResult
            | PushRecoveryProjection
        ),
        snapshot: FileNotesSessionSnapshot,
    ) -> None:
        """Project one already-typed settlement through the existing observer."""
        if not self._push_operation_is_current(operation, key, operation_id):
            return
        if isinstance(result, PushDestinationPolicyResult):
            if result.state == "ready" and result.authorization is not None:
                self._push_authorization_projection = result.authorization
                self._push_view_phase = "checking_candidate"
                self._render_current_push_view()
                self._open_push_authorization(operation, key, operation_id)
                return
            self._set_push_result_projection(
                PushPanelResultProjection(
                    title="Blocked",
                    message=result.message,
                    action="review_again",
                )
            )
            return
        if isinstance(result, PushPreflightResult):
            if (
                result.state == "review"
                and result.handle is not None
                and result.review is not None
                and result.review.candidate == operation.candidate.candidate
            ):
                try:
                    projection = PushPanelReviewProjection(
                        review=result.review,
                        availability=operation.candidate,
                    )
                except ValueError:
                    projection = None
                if projection is not None:
                    self._push_review_handle = result.handle
                    self._push_review_projection = projection
                    self._push_view_phase = "review"
                    self._render_current_push_view()
                    return
            if result.state == "cancelled":
                self._return_push_to_list()
                return
            if result.outcome is not None:
                self._set_push_result_projection(
                    self._push_outcome_result_projection(
                        result.outcome.title,
                        result.outcome.message,
                        result.outcome.state,
                        snapshot,
                    )
                )
                return
            self._set_push_result_projection(
                PushPanelResultProjection(
                    title="Blocked",
                    message=(
                        "The configured destination could not be proved ready "
                        "for this reviewed commit."
                    ),
                    action="review_again",
                )
            )
            return
        if isinstance(result, PushExecutionResult):
            if result.state == "cancelled":
                self._return_push_to_list()
                return
            if result.outcome is None:
                self._set_push_result_projection(
                    PushPanelResultProjection(
                        title="Blocked",
                        message=(
                            "The reviewed push was blocked before Chatbook "
                            "could prove a remote update."
                        ),
                        action="review_again",
                    )
                )
                return
            self._set_push_result_projection(
                self._push_outcome_result_projection(
                    result.outcome.title,
                    result.outcome.message,
                    result.outcome.state,
                    snapshot,
                )
            )
            return
        action_enabled = (
            snapshot.push_recovery_available and result.can_check_again
        )
        self._set_push_result_projection(
            PushPanelResultProjection(
                title=result.title,
                message=result.message,
                action=(
                    "back_to_session"
                    if result.state == "succeeded"
                    else "check_remote_again"
                ),
                action_enabled=(
                    True if result.state == "succeeded" else action_enabled
                ),
                disabled_reason=(
                    None
                    if result.state == "succeeded" or action_enabled
                    else self._push_recovery_not_ready_copy()
                ),
            )
        )

    def _push_outcome_result_projection(
        self,
        title: str,
        message: str,
        state: str,
        snapshot: FileNotesSessionSnapshot,
    ) -> PushPanelResultProjection:
        if state in {"already_published", "succeeded"}:
            return PushPanelResultProjection(
                title=title,
                message=message,
                action="back_to_session",
            )
        if state == "failed_no_update_observed":
            return PushPanelResultProjection(
                title=title,
                message=message,
                action="review_again",
            )
        recovery = snapshot.push_recovery
        available = (
            snapshot.push_recovery_available
            and recovery is not None
            and recovery.can_check_again
        )
        return PushPanelResultProjection(
            title=title,
            message=message,
            action="check_remote_again",
            action_enabled=available,
            disabled_reason=(
                None if available else self._push_recovery_not_ready_copy()
            ),
        )

    def _refresh_push_recovery_readiness(
        self,
        snapshot: FileNotesSessionSnapshot,
    ) -> bool:
        """Reproject only a changed query-readiness bit on owner refresh."""
        projection = self._push_result_projection
        recovery = snapshot.push_recovery
        if projection is None or projection.action != "check_remote_again":
            return False
        available = (
            snapshot.push_recovery_available
            and recovery is not None
            and recovery.can_check_again
        )
        updated = replace(
            projection,
            action_enabled=available,
            disabled_reason=(
                None if available else self._push_recovery_not_ready_copy()
            ),
        )
        if updated == projection:
            return False
        self._set_push_result_projection(updated)
        return True

    @staticmethod
    def _push_recovery_not_ready_copy() -> str:
        return (
            "Owned push descendants are still settling; checking becomes "
            "available after every owned process ends."
        )

    def _set_push_result_projection(
        self,
        projection: PushPanelResultProjection,
    ) -> None:
        self._push_review_handle = None
        self._push_result_projection = projection
        self._push_view_phase = "result"
        self._render_current_push_view()

    def _publish_push_review_admission_failure(self) -> None:
        """Replace a silent local-review start failure with safe copy."""
        if self._push_view_phase == "list":
            self._push_operation_id += 1
            self._push_operation = None
            self._push_operation_key = None
            self._push_operation_admitted = False
            self._push_observer_task = None
            self._push_result = None
        self._push_phase = "needs_attention"
        self._set_push_result_projection(
            PushPanelResultProjection(
                title="Push review unavailable",
                message=(
                    "Chatbook could not start the exact push review. "
                    "Return to the session and try again later."
                ),
                action="back_to_session",
            )
        )
        self._render_session_git_label()

    def _publish_push_recovery_admission_failure(self) -> None:
        """Replace a refused query-only recovery with safe terminal copy."""
        self._push_authorization_projection = None
        self._push_phase = "needs_attention"
        self._set_push_result_projection(
            PushPanelResultProjection(
                title="Remote check unavailable",
                message=(
                    "Chatbook could not start another exact check of the "
                    "configured destination. Inspect it externally before "
                    "taking further action."
                ),
                action="back_to_session",
            )
        )
        self._render_session_git_label()

    def _return_push_to_list(self) -> None:
        self._push_view_phase = "list"
        self._push_authorization_projection = None
        self._push_authorization_dialog_operation_id = None
        self._push_review_handle = None
        self._push_review_projection = None
        self._push_result_projection = None
        if self._git_panel_widget.is_attached:
            self._git_panel_widget.return_to_push_list()

    def _current_push_result_context(
        self,
        operation_id: int,
        action: PushResultAction,
    ) -> tuple[_SessionGitService, RetainedPushOperation, _PushBindingKey] | None:
        """Resolve one exact, enabled result intent to retained authority."""
        operation = self._push_operation
        key = self._push_operation_key
        projection = self._push_result_projection
        service = self._session_git_service()
        if (
            operation is None
            or key is None
            or projection is None
            or service is None
            or self._push_view_phase != "result"
            or operation_id != self._push_operation_id
            or projection.action != action
            or not projection.action_enabled
            or not self._push_operation_is_current(
                operation,
                key,
                operation_id,
            )
        ):
            return None
        return service, operation, key

    def _retained_push_after_start(
        self,
        service: _SessionGitService,
        key: _PushBindingKey,
        expected_kind: Literal[
            "local_proof",
            "preflight",
            "push",
            "recovery",
        ],
    ) -> RetainedPushOperation:
        """Resolve exact service identity after a start method returns."""
        operation = service.retained_push_operation(key.binding)
        if (
            operation is None
            or operation.kind != expected_kind
            or self._push_key_for_operation(operation) != key
        ):
            raise RuntimeError(
                f"retained {expected_kind} push operation is unavailable"
            )
        return operation

    def _adopt_push_operation(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
    ) -> None:
        """Attach the sole workspace observer to service-owned identity."""
        self._push_latest_service_operation_id = max(
            self._push_latest_service_operation_id,
            operation.operation_id,
        )
        self._start_push_observer(operation, key)

    def _open_push_authorization(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
        *,
        recovery: bool = False,
    ) -> None:
        projection = self._push_authorization_projection
        if (
            projection is None
            or self._push_authorization_dialog_operation_id == operation_id
            or not self._push_operation_is_current(operation, key, operation_id)
            or not self._active
            or not self.is_mounted
            or self._navigator_mode != "git"
        ):
            return
        self._push_authorization_dialog_operation_id = operation_id
        self.app.push_screen(
            PushDestinationAuthorizationDialog(
                operation.candidate.candidate,
                projection,
            ),
            callback=partial(
                (
                    self._push_recovery_authorization_resolved
                    if recovery
                    else self._push_authorization_resolved
                ),
                operation,
                key,
                operation_id,
            ),
        )

    def _push_authorization_resolved(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
        authorized: bool | None,
    ) -> None:
        self._push_authorization_dialog_operation_id = None
        if not self._push_operation_is_current(operation, key, operation_id):
            return
        service = self._session_git_service()
        if service is None:
            return
        if authorized is not True:
            if service.cancel_push(key.binding, operation):
                self._return_push_to_list()
            return
        try:
            service.authorize_and_check_push(key.binding, operation)
            next_operation = self._retained_push_after_start(
                service,
                key,
                "preflight",
            )
        except (GitMutationAdmissionError, RuntimeError):
            self._set_push_result_projection(
                PushPanelResultProjection(
                    title="Blocked",
                    message=(
                        "The configured destination could not be checked. "
                        "Review the exact candidate again."
                    ),
                    action="review_again",
                )
            )
            return
        self._adopt_push_operation(next_operation, key)

    def _push_recovery_authorization_resolved(
        self,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
        authorized: bool | None,
    ) -> None:
        """Authorize and query only the exact retained uncertain endpoint."""
        self._push_authorization_dialog_operation_id = None
        if authorized is not True or not self._push_operation_is_current(
            operation,
            key,
            operation_id,
        ):
            return
        service = self._session_git_service()
        if service is None:
            self._publish_push_recovery_admission_failure()
            return
        try:
            accepted = service.authorize_push_recovery(
                key.binding,
                operation,
            )
        except (GitMutationAdmissionError, RuntimeError):
            self._publish_push_recovery_admission_failure()
            return
        if not accepted:
            self._publish_push_recovery_admission_failure()
            return
        self._start_push_recovery_query(
            service,
            operation,
            key,
            operation_id,
        )

    def _start_push_recovery_query(
        self,
        service: _SessionGitService,
        operation: RetainedPushOperation,
        key: _PushBindingKey,
        operation_id: int,
    ) -> None:
        """Start one query-only cycle and adopt only service-retained identity."""
        try:
            service.check_push_again(key.binding, operation)
            next_operation = self._retained_push_after_start(
                service,
                key,
                "recovery",
            )
        except GitMutationAdmissionError as error:
            if error.reason == "authorization_required":
                snapshot = self._session_owner.snapshot(key.binding)
                recovery = snapshot.push_recovery
                if (
                    recovery is not None
                    and snapshot.push_recovery_candidate
                    == operation.candidate
                ):
                    self._push_authorization_projection = (
                        PushAuthorizationProjection(recovery.destination)
                    )
                    self._open_push_authorization(
                        operation,
                        key,
                        operation_id,
                        recovery=True,
                    )
                    return
                self._publish_push_recovery_admission_failure()
                return
            if error.reason == "recovery_not_ready":
                projection = self._push_result_projection
                if projection is not None:
                    self._set_push_result_projection(
                        replace(
                            projection,
                            action_enabled=False,
                            disabled_reason=(
                                self._push_recovery_not_ready_copy()
                            ),
                        )
                    )
                return
            self._publish_push_recovery_admission_failure()
            return
        except RuntimeError:
            self._publish_push_recovery_admission_failure()
            return
        self._push_authorization_projection = None
        self._adopt_push_operation(next_operation, key)

    def _rehydrate_push_state(
        self,
        service: _SessionGitService | None,
        binding: SessionBinding,
        snapshot: FileNotesSessionSnapshot,
    ) -> bool:
        """Reattach to exact retained push state without starting Git work."""
        if (
            binding != self._session_binding
            or binding != self._session_owner.current_binding()
        ):
            self._clear_push_presentation()
            return False
        availability = snapshot.push_candidate
        self._push_availability = availability
        self._sync_push_availability_projection()
        candidate_key = self._push_key_for_availability(binding, availability)
        if service is not self._push_service_identity:
            self._push_operation_id += 1
            self._push_service_identity = service
            self._push_latest_service_operation_id = 0
            self._push_operation = None
            self._push_operation_key = None
            self._push_operation_admitted = False
            self._push_observer_task = None
            self._push_result = None
            self._return_push_to_list()
        retained_push = (
            None if service is None else getattr(service, "retained_push_operation", None)
        )
        operation = None if retained_push is None else retained_push(binding)
        if operation is None:
            if self._push_operation is not None:
                self._push_operation_id += 1
                self._push_operation = None
                self._push_operation_key = None
                self._push_operation_admitted = False
                self._push_observer_task = None
                self._push_result = None
                self._return_push_to_list()
            self._push_key = candidate_key
            self._push_phase = (
                "needs_attention"
                if snapshot.push_recovery is not None
                else "idle"
            )
            self._render_session_git_label()
            return availability is not None or snapshot.push_recovery is not None
        operation_key = self._push_key_for_operation(operation)
        current_operation = self._push_operation
        if (
            current_operation is not None
            and self._push_operation_admitted
            and (
                (candidate_key is not None and candidate_key != self._push_operation_key)
                or (
                    candidate_key is None
                    and current_operation.kind not in {"push", "recovery"}
                )
            )
        ):
            self._push_operation_id += 1
            self._push_operation_admitted = False
            self._push_result = None
            self._push_phase = (
                "needs_attention"
                if snapshot.push_recovery is not None
                else "idle"
            )
            self._push_key = candidate_key
            self._return_push_to_list()
        if operation is self._push_operation:
            if not self._push_operation_admitted:
                self._push_key = candidate_key
                self._push_phase = (
                    "needs_attention"
                    if snapshot.push_recovery is not None
                    else "idle"
                )
                self._return_push_to_list()
                self._render_session_git_label()
                return (
                    availability is not None
                    or snapshot.push_recovery is not None
                )
            if (
                self._push_operation_admitted
                and self._push_operation_key == operation_key
                and self._push_result is None
                and not operation.settled
            ):
                self._push_key = operation_key
                self._push_phase = (
                    "pushing"
                    if operation.kind == "push" and operation.child_started
                    else "checking"
                )
                self._push_view_phase = self._push_view_for_operation(
                    operation
                )
            refreshed = self._refresh_push_recovery_readiness(snapshot)
            if (
                not refreshed
                and self._git_panel_widget.push_phase
                != self._push_view_phase
            ):
                self._render_current_push_view()
            if (
                self._push_authorization_projection is not None
                and self._push_view_phase == "checking_candidate"
                and self._push_operation_key is not None
            ):
                self._open_push_authorization(
                    operation,
                    self._push_operation_key,
                    self._push_operation_id,
                )
            self._render_session_git_label()
            return True
        if operation.operation_id <= self._push_latest_service_operation_id:
            if not self._push_operation_admitted:
                self._push_key = candidate_key
                self._push_phase = (
                    "needs_attention"
                    if snapshot.push_recovery is not None
                    else "idle"
                )
                self._return_push_to_list()
            self._render_session_git_label()
            return True
        self._push_latest_service_operation_id = operation.operation_id
        exact_live_candidate = (
            operation.binding == binding
            and operation.kind != "recovery"
            and operation_key == candidate_key
        )
        exact_recovery = (
            operation.binding == binding
            and operation.kind == "recovery"
            and snapshot.push_recovery is not None
            and operation_key
            == self._push_key_for_availability(
                binding,
                snapshot.push_recovery_candidate,
            )
        )
        if exact_live_candidate or exact_recovery:
            self._start_push_observer(operation, operation_key)
        else:
            self._push_operation_id += 1
            self._push_operation = operation
            self._push_operation_key = operation_key
            self._push_operation_admitted = False
            self._push_observer_task = None
            self._push_key = candidate_key
            self._push_result = None
            self._return_push_to_list()
            self._push_phase = (
                "needs_attention"
                if snapshot.push_recovery is not None
                else "idle"
            )
        self._render_session_git_label()
        return True

    def _restore_dismissed_push_result(
        self,
        binding: SessionBinding,
        snapshot: FileNotesSessionSnapshot,
    ) -> bool:
        """Restore exact cached result presentation without starting work."""
        operation = self._push_operation
        operation_key = self._push_operation_key
        cached_result = self._push_result
        if (
            operation is None
            or operation_key is None
            or not operation.settled
            or self._push_result_projection is not None
            or self._push_view_phase != "list"
            or snapshot.push_recovery is None
            or self._push_key_for_availability(
                binding,
                snapshot.push_recovery_candidate,
            )
            != operation_key
            or not self._push_operation_is_current(
                operation,
                operation_key,
                self._push_operation_id,
            )
        ):
            return False
        exact_result = (
            operation.kind == "push"
            and isinstance(cached_result, PushExecutionResult)
            and cached_result.state == "uncertain"
        ) or (
            operation.kind == "recovery"
            and isinstance(cached_result, PushRecoveryProjection)
            and cached_result == snapshot.push_recovery
        )
        if not exact_result:
            return False
        self._publish_push_settlement(
            operation,
            operation_key,
            self._push_operation_id,
            cached_result,
            snapshot,
        )
        return self._push_view_phase == "result"

    def _git_binding_is_current(self, binding: SessionBinding) -> bool:
        return (
            self._active
            and self._navigator_mode == "git"
            and self._git_binding_matches_session(binding)
        )

    def _git_binding_matches_session(self, binding: SessionBinding) -> bool:
        """Return whether a result still belongs to this retained root."""
        return (
            binding == self._session_binding
            and binding == self._session_owner.current_binding()
        )

    @staticmethod
    def _repository_identity_is_complete(
        repository: RepositoryIdentity | None,
    ) -> bool:
        """Return whether a repository identity is complete enough to key UI."""
        return (
            repository is not None
            and bool(repository.worktree_root)
            and bool(repository.git_dir)
            and bool(repository.git_common_dir)
            and repository.worktree_identity is not None
            and repository.git_dir_identity is not None
            and repository.git_common_dir_identity is not None
        )

    def _commit_key_for_status(
        self,
        binding: SessionBinding,
        status: SessionGitStatus,
    ) -> _CommitBindingKey | None:
        """Return an exact key only for current, complete ready authority."""
        repository = status.repository
        if (
            status.state != "ready"
            or status.binding_generation != binding.generation
            or not self._repository_identity_is_complete(repository)
        ):
            return None
        snapshot = self._session_owner.snapshot(binding)
        if snapshot.trusted_repository != repository or snapshot.git_status != status:
            return None
        assert repository is not None
        return _CommitBindingKey(binding, repository)

    def _sync_commit_availability(
        self,
        binding: SessionBinding,
        status: SessionGitStatus,
    ) -> None:
        """Render exact owned-staging availability without replacing a draft."""
        key = self._commit_key_for_status(binding, status)
        if key is None:
            self._commit_availability = None
            self._git_panel_widget.clear_commit_availability()
            return
        draft = self._commit_draft
        if draft is not None and draft.key != key:
            self._invalidate_commit_binding(
                "Repository changed; the previous commit draft was cleared."
            )
            draft = None
        if draft is None:
            draft = _CommitDraftState(key)
        head = status.head
        branch = (
            head.branch
            if head is not None and head.kind == "attached" and head.branch
            else "detached or unavailable HEAD"
        )
        projection = CommitDraftProjection(
            binding_key=key,
            branch=branch,
            staged_note_count=sum(row.unstage_eligible for row in status.rows),
            subject=draft.subject,
            body=draft.body,
        )
        self._commit_availability = projection
        self._git_panel_widget.render_commit_availability(projection)

    def _invalidate_commit_binding(self, detail: str) -> None:
        """Clear every draft projection that cannot cross repository binding."""
        self._commit_operation_id += 1
        self._commit_availability = None
        self._commit_draft = None
        self._commit_view_phase = "list"
        self._commit_operation = None
        self._commit_review_handle = None
        self._commit_review_projection = None
        self._commit_result_projection = None
        self._release_commit_editor_lease()
        if self._active and self.is_mounted:
            self._git_panel_widget.invalidate_commit_binding()
            self._set_action_status(detail)

    def _clear_commit_draft_after_success(self) -> None:
        """Clear only message/workflow state after a proven commit success."""
        self._commit_draft = None
        self._commit_view_phase = "list"
        self._commit_review_handle = None
        self._commit_review_projection = None
        self._commit_result_projection = None
        binding = self._session_binding
        snapshot = (
            None
            if binding is None
            else self._session_owner.snapshot(binding)
        )
        if binding is not None and snapshot is not None:
            self._rehydrate_push_state(
                self._session_git_service(),
                binding,
                snapshot,
            )
        if not self._active or not self.is_mounted:
            self._commit_availability = None
            return
        self._git_panel_widget.invalidate_commit_binding()
        if binding is None:
            self._commit_availability = None
            return
        assert snapshot is not None
        status = snapshot.git_status
        if status is None:
            self._commit_availability = None
            return
        self._git_panel_widget.render_status(status)
        self._sync_commit_availability(binding, status)

    def _capture_git_action_key(
        self,
        binding: SessionBinding,
    ) -> _GitLastAction | None:
        """Capture exact owner authority immediately around action admission."""
        if binding != self._session_owner.current_binding():
            return None
        snapshot = self._session_owner.snapshot(binding)
        repository = snapshot.trusted_repository
        if not self._repository_identity_is_complete(repository):
            return None
        assert repository is not None
        return _GitLastAction(
            binding=snapshot.binding,
            repository=repository,
            changes=snapshot.changes,
            text="",
        )

    def _git_action_key_is_current(self, action: _GitLastAction) -> bool:
        """Compare one captured key with a fresh immutable owner snapshot."""
        if action.binding != self._session_owner.current_binding():
            return False
        snapshot = self._session_owner.snapshot(action.binding)
        return (
            snapshot.binding == action.binding
            and snapshot.trusted_repository == action.repository
            and snapshot.changes == action.changes
            and self._repository_identity_is_complete(
                snapshot.trusted_repository
            )
        )

    def _clear_git_last_action(self) -> None:
        """Discard obsolete action text in both workspace and mounted panel."""
        self._git_last_action = None
        if self._active and self.is_mounted:
            self._git_panel_widget.clear_last_action()

    def _sync_git_last_action(self) -> bool:
        """Validate and project the retained action against fresh owner state."""
        action = self._git_last_action
        if action is not None and not self._git_action_key_is_current(action):
            action = None
            self._git_last_action = None
        if self._active and self.is_mounted:
            if action is None:
                self._git_panel_widget.clear_last_action()
            else:
                self._git_panel_widget.set_last_action(
                    action.text,
                    complete=action.complete,
                )
        return action is not None

    def _git_can_retain_rows(self, binding: SessionBinding) -> bool:
        """Prove prior rows belong to the same complete trusted authority."""
        if binding != self._session_owner.current_binding():
            return False
        snapshot = self._session_owner.snapshot(binding)
        repository = snapshot.trusted_repository
        status = snapshot.git_status
        return (
            self._repository_identity_is_complete(repository)
            and status is not None
            and status.binding_generation == binding.generation
            and status.repository == repository
        )

    def _ensure_git_status_waiter(
        self,
        task: asyncio.Task[SessionGitStatus],
        binding: SessionBinding,
        *,
        replace: bool = False,
    ) -> None:
        """Attach presentation to one service-owned task without restarting it."""
        if (
            not self._active
            or not self.is_mounted
            or (
                not replace
                and self._git_status_worker is not None
                and not self._git_status_worker.is_finished
            )
        ):
            return
        self._git_status_worker = self.run_worker(
            self._render_git_status(task, binding),
            name="file-notes-git-status",
            group="file-notes-git-status",
            exclusive=True,
        )

    def _rehydrate_commit_presentation(
        self,
        service: _SessionGitService,
        binding: SessionBinding,
    ) -> bool:
        """Reattach UI observers without restarting process-owned commit work."""
        draft = self._commit_draft
        if draft is None or draft.key.binding != binding:
            return False
        key = draft.key
        if not self._commit_key_is_current(key):
            if (
                self._session_owner.snapshot(binding).trusted_repository
                is None
            ):
                return False
            self._invalidate_commit_binding(
                "Repository changed; the previous commit draft was cleared."
            )
            return False
        if (
            self._commit_view_phase == "review"
            and self._commit_review_handle is not None
            and self._commit_review_projection is not None
        ):
            self._git_panel_widget.render_commit_review(self._commit_review_projection)
            return True
        if (
            self._commit_view_phase == "result"
            and self._commit_result_projection is not None
        ):
            self._git_panel_widget.render_commit_result(self._commit_result_projection)
            return True
        if self._commit_view_phase == "form":
            projection = self._current_commit_draft_projection()
            if projection is not None:
                self._git_panel_widget.render_commit_form(projection)
                return True
        if self._commit_view_phase == "list":
            return False

        operation = service.retained_commit_operation(binding)
        if operation is None:
            return False
        if self._commit_operation is not operation:
            operation_id = self._begin_commit_operation(
                operation,
                "checking" if operation.kind != "commit" else "confirming",
            )
        else:
            operation_id = self._commit_operation_id
        if operation.kind == "review":
            self._commit_view_phase = "checking"
            self._git_panel_widget.render_commit_checking()
            self._git_commit_worker = self.run_worker(
                self._observe_commit_review(operation, key, operation_id),
                name="file-notes-git-commit-review",
                group="file-notes-git-commit",
                exclusive=True,
            )
            return True
        if operation.kind == "commit":
            if operation.child_started and not operation.settled:
                review = self._commit_review_projection
                count = 0 if review is None else review.review.included_note_count
                self._commit_view_phase = "executing"
                self._git_panel_widget.render_commit_executing(
                    CommitExecutionProjection(count)
                )
            else:
                self._commit_view_phase = "confirming"
                self._git_panel_widget.render_commit_confirming()
            self._git_commit_child_worker = self.run_worker(
                self._observe_commit_child_start(operation, key, operation_id),
                name="file-notes-git-commit-child",
                group="file-notes-git-commit-child",
                exclusive=True,
            )
            self._git_commit_worker = self.run_worker(
                self._observe_commit_outcome(operation, key, operation_id),
                name="file-notes-git-commit-outcome",
                group="file-notes-git-commit",
                exclusive=True,
            )
            return True
        self._commit_view_phase = "executing"
        self._git_panel_widget.render_commit_recovery_checking()
        self._git_commit_worker = self.run_worker(
            self._observe_commit_outcome(operation, key, operation_id),
            name="file-notes-git-commit-recovery",
            group="file-notes-git-commit",
            exclusive=True,
        )
        return True

    def _rehydrate_git_presentation(
        self,
        *,
        restore_dismissed_push_result: bool = False,
    ) -> bool:
        """Render retained owner/task state without starting hidden Git work."""
        if not self._active or not self.is_mounted:
            return False
        binding = self._session_binding
        if binding is None or not self._git_binding_matches_session(binding):
            self._clear_git_last_action()
            self._clear_push_presentation()
            return False
        snapshot = self._session_owner.snapshot(binding)
        service = self._session_git_service()
        push_retained = self._rehydrate_push_state(service, binding, snapshot)
        if self._navigator_mode != "git":
            return push_retained
        if restore_dismissed_push_result:
            self._restore_dismissed_push_result(binding, snapshot)
        if self._push_view_phase != "list":
            self._render_current_push_view()
            return True
        self._sync_git_last_action()
        if service is not None and self._rehydrate_commit_presentation(
            service, binding
        ):
            return True
        if self._session_owner.mutation_active(binding):
            if snapshot.git_status is not None:
                self._git_panel_widget.render_status(
                    snapshot.git_status,
                    retain_rows=self._git_can_retain_rows(binding),
                )
            self._git_panel_widget.set_mutating(
                True,
                "Git mutation in progress…",
            )
            self._git_refresh_after_mutation = True
            return True
        retained_task = (
            None if service is None else service.retained_status(binding)
        )
        if retained_task is not None:
            self._git_status_task = retained_task
            self._git_status_task_binding = binding
        task = self._git_status_task
        if (
            task is not None
            and self._git_status_task_binding == binding
            and not task.done()
        ):
            repository = snapshot.trusted_repository
            if repository is not None:
                self._git_panel_widget.render_checking(
                    repository.worktree_root,
                    retain_rows=self._git_can_retain_rows(binding),
                )
            self._ensure_git_status_waiter(task, binding)
            return True
        if snapshot.git_status is not None:
            self._git_panel_widget.render_status(
                snapshot.git_status,
                retain_rows=self._git_can_retain_rows(binding),
            )
            self._sync_commit_availability(binding, snapshot.git_status)
            self._sync_git_last_action()
            return True
        if (
            task is not None
            and self._git_status_task_binding == binding
            and task.done()
        ):
            self._ensure_git_status_waiter(task, binding)
            return True
        return False

    @staticmethod
    def _git_discovery_failure_detail(discovery: DiscoveryResult) -> str:
        """Project one discovery failure to reason plus feasible recovery."""
        if discovery.state == "not_repository":
            return (
                "This notes folder is not in a Git worktree. "
                "Notes remain fully usable."
            )

        defaults = {
            "unavailable": "Git is unavailable",
            "unsupported": "Git repository compatibility is unsupported",
            "unsafe_root": "Selected File Notes root is unsafe",
        }
        recoveries = {
            "unavailable": (
                "Install or restore Git, then open Review session changes again."
            ),
            "unsupported": (
                "Resolve Git compatibility outside Chatbook, then open Review "
                "session changes again."
            ),
            "unsafe_root": (
                "Select or fix a safe notes root, then open Review session "
                "changes again."
            ),
        }
        reason = (
            discovery.message
            or defaults.get(
                discovery.state,
                "Git repository discovery is unavailable",
            )
        ).strip()
        if reason and reason[-1] not in ".!?":
            reason += "."
        recovery = recoveries.get(
            discovery.state,
            "Open Review session changes again.",
        )
        return f"{reason} {recovery}"

    async def _open_session_git(self, *, force_prompt: bool = False) -> None:
        binding = self._session_binding
        service = self._session_git_service()
        if binding is None or service is None:
            self._clear_git_last_action()
            self._git_panel_widget.render_unavailable(
                "Git is unavailable for the selected File Notes root. "
                "Restore Git, then open Review session changes again."
            )
            return
        discovery = await service.discover(binding)
        if not self._git_binding_is_current(binding):
            return
        repository = discovery.repository
        if discovery.state != "ready" or repository is None:
            self._clear_git_last_action()
            snapshot = self._session_owner.snapshot(binding)
            if discovery.state == "unavailable":
                service.cancel_commit(binding)
            self._session_owner.clear_trust_if_matches(
                binding,
                snapshot.trusted_repository,
            )
            if discovery.state == "unavailable":
                self._commit_operation_id += 1
                self._commit_availability = None
                self._commit_view_phase = "list"
                self._commit_operation = None
                self._commit_review_handle = None
                self._commit_review_projection = None
                self._commit_result_projection = None
                self._release_commit_editor_lease()
                self._git_panel_widget.invalidate_commit_binding()
            else:
                self._invalidate_commit_binding(
                    "Repository check failed; the commit draft was cleared."
                )
            self._git_panel_widget.render_unavailable(
                self._git_discovery_failure_detail(discovery)
            )
            if discovery.state == "unavailable":
                self._set_action_status(
                    "Git discovery is temporarily unavailable; "
                    "the commit draft was preserved."
                )
            return
        snapshot = self._session_owner.snapshot(binding)
        needs_trust = (
            force_prompt or snapshot.trusted_repository != repository
        )
        if needs_trust:
            self._clear_git_last_action()
            self._git_panel_widget.render_untrusted(repository.worktree_root)
            accepted = await self.app.push_screen_wait(
                SessionGitTrustDialog(repository.worktree_root)
            )
            if not accepted or not self._git_binding_is_current(binding):
                return
            if not await service.revalidate_repository(binding, repository):
                if self._git_binding_is_current(binding):
                    self._clear_git_last_action()
                    self._git_panel_widget.render_untrusted(
                        repository.worktree_root
                    )
                    self._git_panel_widget.set_current_status(
                        "Status: TRUST REQUIRED — Repository identity changed; "
                        "retry Trust and check status.",
                        complete=True,
                    )
                return
            if not self._session_owner.publish_trust(binding, repository):
                self._clear_git_last_action()
                return
            snapshot = self._session_owner.snapshot(binding)
        if (
            self._git_refresh_after_mutation
            and not self._session_owner.mutation_active(binding)
        ):
            self._git_refresh_after_mutation = False
            self._start_git_refresh()
            self.call_after_refresh(self._focus_session_git_panel)
            return
        if self._rehydrate_git_presentation(
            restore_dismissed_push_result=True,
        ):
            self.call_after_refresh(self._focus_session_git_panel)
            return
        self._start_git_refresh()

    def _start_git_refresh(self) -> None:
        """Synchronously admit visible status and delegate retained awaiting."""
        if self._navigator_mode != "git":
            return
        binding = self._session_binding
        service = self._session_git_service()
        if binding is None or service is None:
            self._clear_git_last_action()
            self._git_panel_widget.render_unavailable(
                "Git status is unavailable. Restore Git, then open Review "
                "session changes again."
            )
            return
        if self._session_owner.mutation_active(binding):
            self._git_refresh_after_mutation = True
            self._git_panel_widget.mark_stale(
                "Git mutation in progress; refresh will follow.",
                retain_rows=self._git_can_retain_rows(binding),
            )
            return
        snapshot = self._session_owner.snapshot(binding)
        repository = snapshot.trusted_repository
        if repository is None:
            self._clear_git_last_action()
            self._git_panel_widget.render_unavailable(
                "Repository trust is unavailable. Return to the navigator, "
                "open Review session changes again, and trust the repository."
            )
            return
        self._git_panel_widget.render_checking(
            repository.worktree_root,
            retain_rows=self._git_can_retain_rows(binding),
        )
        try:
            task = service.start_status(binding, snapshot.changes)
        except GitStatusAdmissionError as error:
            if error.reason == "mutation_active":
                self._git_refresh_after_mutation = True
            self._git_panel_widget.mark_stale(
                f"{error}. Retry Refresh.",
                retain_rows=self._git_can_retain_rows(binding),
            )
            return
        self._git_status_task = task
        self._git_status_task_binding = binding
        self._ensure_git_status_waiter(task, binding, replace=True)

    async def _render_git_status(
        self,
        task: asyncio.Task[SessionGitStatus],
        binding: SessionBinding,
    ) -> None:
        try:
            status = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._git_binding_is_current(binding):
                self._sync_git_last_action()
                self._git_panel_widget.mark_stale(
                    f"Git status failed: {error}. Retry Refresh.",
                    retain_rows=self._git_can_retain_rows(binding),
                    error=True,
                )
                if self._git_status_task is task:
                    self._git_status_task = None
                    self._git_status_task_binding = None
            return
        self._sync_git_last_action()
        if not self._git_binding_is_current(binding):
            return
        snapshot = self._session_owner.snapshot(binding)
        if self._git_status_task is task:
            self._git_status_task = None
            self._git_status_task_binding = None
        if snapshot.trusted_repository is None:
            self._clear_git_last_action()
            self._git_panel_widget.render_unavailable(
                "Repository trust changed while checking status. Return to "
                "the navigator, open Review session changes again, and trust "
                "the current repository."
            )
            return
        if status == snapshot.git_status:
            self._git_panel_widget.render_status(
                status,
                retain_rows=self._git_can_retain_rows(binding),
            )
            self._sync_commit_availability(binding, status)
            self._sync_git_last_action()
            return
        if snapshot.git_status is not None:
            self._git_panel_widget.render_status(
                snapshot.git_status,
                retain_rows=self._git_can_retain_rows(binding),
            )
            self._sync_commit_availability(binding, snapshot.git_status)
            self._sync_git_last_action()
            return
        if (
            snapshot.trusted_repository is not None
            and not self._rehydrate_git_presentation()
        ):
            self._start_git_refresh()

    def _set_save_state(self, state: SaveState, detail: str = "") -> None:
        self._save_state = state
        self._save_detail = detail
        if state != "conflict":
            self._conflict_resolution_active = False
        if self._active and self.is_mounted:
            label = _SAVE_STATE_COPY[state]
            if detail:
                label = f"{label}; {detail}"
            status = self.query_one("#file-notes-save-status", Static)
            status.set_class(state == "conflict", "-conflict")
            status.set_class(state == "error", "-error")
            status.update(label)
            self.query_one("#file-notes-authority", Static).update(
                self._authority_copy()
            )
            self._update_controls()

    def _set_action_status(self, text: str) -> None:
        self._action_detail = text
        if self._active and self.is_mounted:
            self.query_one("#file-notes-action-status", Static).update(text)

    def _acquire_editor_read_only(
        self,
        binding: SessionBinding,
    ) -> _EditorReadOnlyLease | None:
        """Acquire one exact token without owning other read-only state."""
        editor = self._editor_widget
        if (
            binding != self._session_binding
            or binding != self._session_owner.current_binding()
        ):
            return None
        token = object()
        lease = _EditorReadOnlyLease(
            token,
            editor,
            binding,
            self._release_editor_read_only,
        )
        self._editor_read_only_leases[token] = lease
        self._sync_editor_read_only()
        return lease

    def _release_editor_read_only(
        self,
        lease: _EditorReadOnlyLease,
    ) -> None:
        """Remove only a still-registered exact lease."""
        if self._editor_read_only_leases.get(lease.token) is not lease:
            return
        self._editor_read_only_leases.pop(lease.token, None)
        self._sync_editor_read_only()

    def _sync_editor_read_only(self) -> None:
        """Combine document, transition, and tokenized read-only reasons."""
        editor = self._editor_widget
        binding = self._session_binding
        leased = any(
            not lease._released and lease.editor is editor and lease.binding == binding
            for lease in self._editor_read_only_leases.values()
        )
        editor.read_only = (
            self._root_transitioning
            or self._path_transitioning
            or self._opened is None
            or not self._opened.editable
            or self.reload_confirmation_active
            or leased
        )

    async def _settle_commit_editor(
        self,
        lease: _EditorReadOnlyLease,
    ) -> bool:
        """Settle autosave while the exact editor is already read-only."""
        try:
            flushed = await self.flush_pending_work()
        except BaseException:
            if self._commit_editor_lease is lease:
                self._commit_editor_lease = None
            lease.release()
            raise
        unresolved = self._save_state in {
            "dirty",
            "saving",
            "conflict",
            "error",
        }
        if (
            flushed
            and not unresolved
            and lease.binding == self._session_binding
            and lease.editor is self._editor_widget
        ):
            return True
        state = self._save_state
        lease.release()
        if state == "conflict":
            detail = "Save conflict must be resolved before reviewing the commit."
        elif state == "error":
            detail = "Fix the save error before reviewing the commit."
        else:
            detail = "Save the note before reviewing the commit."
        self._set_action_status(detail)
        return False

    def _release_commit_editor_lease(self) -> None:
        """Release the guarded-flow reason without changing other reasons."""
        lease = self._commit_editor_lease
        self._commit_editor_lease = None
        if lease is not None:
            lease.release()

    def _commit_key_is_current(self, key: _CommitBindingKey) -> bool:
        """Validate the exact root and repository without requiring status."""
        if (
            key.binding != self._session_binding
            or key.binding != self._session_owner.current_binding()
        ):
            return False
        return (
            self._session_owner.snapshot(key.binding).trusted_repository
            == key.repository
        )

    def _active_commit_key(self) -> _CommitBindingKey | None:
        draft = self._commit_draft
        if draft is None or not self._commit_key_is_current(draft.key):
            return None
        return draft.key

    def _current_commit_draft_projection(
        self,
        *,
        form_error: str | None = None,
        subject_error: str | None = None,
        body_error: str | None = None,
    ) -> CommitDraftProjection | None:
        """Project the retained draft with optional bounded recovery copy."""
        draft = self._commit_draft
        availability = self._commit_availability
        if (
            draft is None
            or availability is None
            or availability.binding_key != draft.key
            or not self._commit_key_is_current(draft.key)
        ):
            return None
        return replace(
            availability,
            subject=draft.subject,
            body=draft.body,
            form_error=form_error,
            subject_error=subject_error,
            body_error=body_error,
        )

    def _begin_commit_operation(
        self,
        operation: RetainedCommitOperation,
        phase: Literal["checking", "confirming"],
    ) -> int:
        """Install one service observation and return its monotonic UI ID."""
        self._commit_operation_id += 1
        self._commit_operation = operation
        self._commit_view_phase = phase
        return self._commit_operation_id

    def _commit_operation_is_current(
        self,
        operation: RetainedCommitOperation,
        key: _CommitBindingKey,
        operation_id: int,
    ) -> bool:
        """Reject stale renderers while leaving service settlement untouched."""
        return (
            self._commit_operation is operation
            and self._commit_operation_id == operation_id
            and operation.binding == key.binding
            and self._commit_key_is_current(key)
        )

    def _attach_commit_lease_settlement(
        self,
        operation: RetainedCommitOperation,
        key: _CommitBindingKey,
        lease: _EditorReadOnlyLease,
        *,
        hold_ready_review: bool,
    ) -> None:
        """Retain lease finalization outside disposable Textual workers."""

        async def finalize() -> None:
            try:
                result = await operation.wait()
            except BaseException:
                result = None
            if (
                hold_ready_review
                and isinstance(result, CommitReviewResult)
                and result.state == "ready"
                and self._commit_operation is operation
                and self._commit_key_is_current(key)
            ):
                return
            if self._commit_editor_lease is lease:
                self._commit_editor_lease = None
            lease.release()

        task = asyncio.create_task(finalize())
        self._commit_settlement_tasks.add(task)
        task.add_done_callback(self._commit_settlement_tasks.discard)

    def _update_controls(self) -> None:
        if not self._active or not self.is_mounted:
            return
        self.query_one("#file-notes-path-label", Static).update(
            self._path_field_label_copy()
        )
        transitioning = self._root_transitioning or self._path_transitioning
        binding = self._session_binding
        mutation_active = (
            binding is not None
            and self._session_owner.mutation_active(binding)
        )
        focused = self.app.focused
        if (
            (transitioning or mutation_active)
            and isinstance(focused, Button)
            and focused.id
            in {
                "file-notes-new",
                "file-notes-move",
                "file-notes-delete",
                "file-notes-restore",
                "file-notes-compare",
                "file-notes-resolve-conflict",
                "file-notes-resolution-keep",
                "file-notes-resolution-save-new",
                "file-notes-resolution-discard",
                "file-notes-protect",
                "file-notes-reload",
                "file-notes-save-copy",
                "file-notes-refresh",
                "file-notes-maintenance-toggle",
            }
        ):
            self._editor_action_focus_target = focused.id
        structurally_available = not transitioning and not mutation_active
        has_service = (
            self._service is not None
            and self._initialized
            and structurally_available
        )
        has_document = self._opened is not None and not transitioning
        has_deleted = bool(self._selected_deleted_path) and not transitioning
        self.query_one("#file-notes-new", Button).disabled = not has_service
        for selector in ("move", "delete", "reload"):
            self.query_one(
                f"#file-notes-{selector}", Button
            ).disabled = not (
                has_document and structurally_available
            )
        self.query_one("#file-notes-protect", Button).disabled = not (
            has_document and structurally_available
        )
        self.query_one("#file-notes-compare", Button).disabled = not (
            has_document
            and structurally_available
            and self._save_state == "conflict"
        )
        self.query_one("#file-notes-resolve-conflict", Button).disabled = not (
            has_document
            and structurally_available
            and self._save_state == "conflict"
        )
        for selector in (
            "resolution-keep",
            "resolution-save-new",
            "resolution-discard",
        ):
            self.query_one(f"#file-notes-{selector}", Button).disabled = not (
                structurally_available
                and self._save_state == "conflict"
                and self._conflict_resolution_active
            )
        copy_button = self.query_one("#file-notes-save-copy", Button)
        exact_export = self._opened is not None and self._opened.is_excerpt
        copy_label = "Export exact copy" if exact_export else "Save draft as copy"
        disabled_prefix = f"{LIBRARY_DISABLED_ACTION_MARKER} "
        if str(copy_button.label).removeprefix(disabled_prefix) != copy_label:
            copy_button.label = copy_label
            copy_button.refresh(layout=True)
            if copy_button.parent is not None:
                copy_button.parent.refresh(layout=True)
        copy_button.disabled = (
            not has_document
            or not structurally_available
            or (
                not exact_export
                and self._save_state not in {"dirty", "conflict", "error"}
            )
        )
        self.query_one("#file-notes-restore", Button).disabled = (
            not has_service or not has_deleted or not structurally_available
        )
        self.query_one("#file-notes-refresh", Button).disabled = (
            self._service is None or not structurally_available
        )
        self.query_one("#file-notes-maintenance-toggle", Button).disabled = (
            self._service is None or not structurally_available
        )
        protect = self.query_one("#file-notes-protect", Button)
        protect_label = (
            "Unprotect"
            if self._opened is not None and self._opened.protected
            else "Protect"
        )
        if str(protect.label) != protect_label:
            protect.label = protect_label
            protect.refresh(layout=True)
            if protect.parent is not None:
                protect.parent.refresh(layout=True)
        reload_button = self.query_one("#file-notes-reload", Button)
        reload_label = (
            "Discard draft and reload"
            if self._save_state in {"conflict", "error"}
            else "Reload"
        )
        if str(reload_button.label) != reload_label:
            reload_button.label = reload_label
            reload_button.refresh(layout=True)
            if reload_button.parent is not None:
                reload_button.parent.refresh(layout=True)
        self._sync_editor_action_visibility()
        self.query_one("#file-notes-reload-cancel", Button).disabled = (
            not self.reload_confirmation_active
        )
        self.query_one("#file-notes-reload-confirm", Button).disabled = (
            not self.reload_confirmation_active or not structurally_available
        )
        busy_reason = ""
        if transitioning:
            busy_reason = (
                "File operation in progress; editor actions are temporarily "
                "unavailable. Wait for it to finish."
            )
        elif mutation_active:
            busy_reason = (
                "Git operation in progress; editor actions are temporarily "
                "unavailable. Wait for it to finish."
            )
        self.query_one("#file-notes-action-status", Static).update(
            busy_reason or self._action_detail
        )
        self._sync_editor_action_disabled_presentation()
        self._schedule_editor_action_layout()
        self.query_one("#file-notes-search", Input).disabled = transitioning
        self.query_one("#file-notes-path", Input).disabled = (
            transitioning or mutation_active
        )
        self.query_one("#file-notes-tree", Tree).disabled = (
            transitioning or mutation_active
        )
        self.query_one("#file-notes-search-results", Tree).disabled = (
            transitioning or mutation_active
        )
        self._sync_editor_read_only()
        self._git_panel_widget.set_mutating(mutation_active)

    def _sync_editor_action_disabled_presentation(self) -> None:
        """Keep every disabled editor action readable and visibly inert."""
        prefix = f"{LIBRARY_DISABLED_ACTION_MARKER} "
        for button in self.query(".file-notes-toolbar Button"):
            label = str(button.label)
            base_label = label.removeprefix(prefix)
            rendered_label = library_disabled_action_label(
                base_label,
                button.disabled,
            )
            if label != rendered_label:
                button.label = rendered_label

    def _sync_editor_action_visibility(self) -> None:
        """Disclose only editor actions relevant to the retained state."""
        confirming_reload = self.reload_confirmation_active
        resolving_conflict = (
            self._conflict_resolution_active
            and self._save_state == "conflict"
            and self._opened is not None
        )
        self.set_class(confirming_reload, "-reload-confirming")
        self.set_class(resolving_conflict, "-resolving-conflict")
        has_service = self._service is not None
        has_document = self._opened is not None
        has_deleted = has_service and bool(self._selected_deleted_path)
        visibility = {
            "file-notes-new": has_service and not resolving_conflict,
            "file-notes-move": has_document,
            "file-notes-delete": has_document and not resolving_conflict,
            "file-notes-restore": has_deleted,
            "file-notes-compare": (
                has_document and self._save_state == "conflict"
            ),
            "file-notes-resolve-conflict": (
                has_document
                and self._save_state == "conflict"
                and not resolving_conflict
            ),
            "file-notes-protect": has_document,
            "file-notes-reload": has_document,
            "file-notes-save-copy": (
                has_document
                and (
                    (self._opened is not None and self._opened.is_excerpt)
                    or self._save_state in {"dirty", "error"}
                )
            ),
            "file-notes-refresh": has_service,
        }
        maintenance_ids = {
            "file-notes-move",
            "file-notes-protect",
            "file-notes-reload",
            "file-notes-refresh",
        }
        maintenance_available = any(
            visibility[action_id] for action_id in maintenance_ids
        )
        visibility["file-notes-maintenance-toggle"] = (
            maintenance_available and not resolving_conflict
        )
        focused = self.app.focused
        for action_id, displayed in visibility.items():
            if action_id in maintenance_ids:
                displayed = displayed and self._maintenance_expanded
            displayed = displayed and not confirming_reload
            button = self.query_one(f"#{action_id}", Button)
            if button is focused and not displayed:
                self._editor_action_focus_target = action_id
            button.display = displayed
        self.query_one("#file-notes-delete-spacer", Static).display = (
            visibility["file-notes-delete"]
            and not self.has_class("-stack-editor-actions")
        )

        maintenance_toggle = self.query_one(
            "#file-notes-maintenance-toggle", Button
        )
        maintenance_toggle.label = (
            "Hide file actions"
            if self._maintenance_expanded
            else "More file actions"
        )
        maintenance = self.query_one("#file-notes-maintenance-actions")
        maintenance.display = (
            maintenance_available
            and self._maintenance_expanded
            and not resolving_conflict
            and not confirming_reload
        )
        resolution_copy = self.query_one(
            "#file-notes-resolution-copy",
            Static,
        )
        resolution_actions = self.query_one("#file-notes-resolution-actions")
        show_resolution = resolving_conflict and not confirming_reload
        resolution_copy.display = show_resolution
        resolution_actions.display = show_resolution
        for button in resolution_actions.query(Button):
            button.display = show_resolution
        confirmation_copy = self.query_one(
            "#file-notes-reload-confirm-copy",
            Static,
        )
        confirmation_actions = self.query_one(
            "#file-notes-reload-confirm-actions"
        )
        confirmation_copy.display = confirming_reload
        confirmation_actions.display = confirming_reload
        for button in confirmation_actions.query(Button):
            button.display = confirming_reload
            button.disabled = not confirming_reload

        for toolbar in self.query(".file-notes-toolbar"):
            toolbar.set_class(
                not any(button.display for button in toolbar.query(Button)),
                "-empty-actions",
            )

        if self._editor_action_focus_target is not None:
            for selector in dict.fromkeys(
                (
                    f"#{self._editor_action_focus_target}",
                    "#file-notes-restore",
                    "#file-notes-new",
                    "#file-notes-resolve-conflict",
                    "#file-notes-compare",
                    "#file-notes-maintenance-toggle",
                    "#file-notes-refresh",
                )
            ):
                candidate = self.query_one(selector, Button)
                if candidate.display and not candidate.disabled:
                    focused = self.app.focused
                    if (
                        focused is None
                        or focused.id == self._editor_action_focus_target
                    ):
                        candidate.focus()
                    self._editor_action_focus_target = None
                    break

    @contextmanager
    def _hold_path_transition(
        self,
    ) -> Iterator[tuple[FileNotesService, int] | None]:
        service = self._service
        binding = self._session_binding
        if (
            not self._active
            or self._root_transitioning
            or self._path_transitioning
            or service is None
            or binding is None
        ):
            yield None
            return
        lease = self._session_owner.try_acquire_transition(binding, "path")
        if lease is None:
            if self._session_owner.mutation_active(binding):
                self._set_action_status(_SESSION_GIT_MUTATION_BUSY)
            yield None
            return
        self._path_transitioning = True
        self._update_root_surface()
        self._update_controls()
        try:
            yield service, self._root_generation
        finally:
            self._path_transitioning = False
            lease.release()
            self._update_root_surface()
            self._update_controls()

    def _path_result_is_stale(
        self,
        service: FileNotesService,
        generation: int,
    ) -> bool:
        return (
            not self._active
            or not self.is_mounted
            or not self.query("#file-notes-root-status")
            or not self.query("#file-notes-editor")
            or generation != self._root_generation
            or service is not self._service
        )

    async def set_root(self, path: str | Path, *, persist: bool = True) -> bool:
        """Adopt one canonical root after the common draft leave guard."""
        if not self._active or self._path_transitioning or self._shutdown:
            return False
        if not await self.flush_pending_work():
            return False
        if not self._active or self._path_transitioning or self._shutdown:
            return False
        expected_binding = (
            self._session_binding
            if self._session_binding is not None
            else self._initial_session_binding
        )
        root_lease = (
            None
            if expected_binding is None
            else self._session_owner.try_acquire_transition(
                expected_binding,
                "root",
            )
        )
        if expected_binding is not None and root_lease is None:
            if self._session_owner.mutation_active(expected_binding):
                self._set_action_status(
                    "Git operation in progress; root change is busy."
                )
            return False
        self._root_generation += 1
        generation = self._root_generation
        self._root_transitioning = True
        self._update_root_surface()
        self._update_controls()
        try:
            canonical = await asyncio.to_thread(self._canonical_root, path)
            if canonical is None:
                return False
            if self._replica_seed is _UNSET and self._replica is None:
                _, replica, _, warning = await asyncio.to_thread(
                    self._build_runtime,
                    generation,
                    expected_binding,
                    bind_session=False,
                )
                if not self._active or generation != self._root_generation:
                    return False
                self._replica = replica
                self._runtime_warning = warning
            service = await asyncio.to_thread(
                FileNotesService,
                canonical,
                self._replica,
                operation_lock=self._service_lock,
            )
            if not self._active or generation != self._root_generation:
                return False
            result = await asyncio.to_thread(service.scan)
            deleted = await self._load_deleted_paths(
                replica=self._replica,
                service=service,
            )
            if not self._active or generation != self._root_generation:
                return False
            if not await self._commit_root_candidate(
                canonical,
                service,
                result,
                deleted,
                generation,
                expected_binding,
                persist=persist,
            ):
                return False
            return True
        finally:
            if root_lease is not None:
                root_lease.release()
            if generation == self._root_generation:
                self._root_transitioning = False
                self._update_root_surface()
                self._update_controls()

    async def open_path(self, relative_path: str) -> bool:
        """Flush the old draft, then open a disk file into the same editor."""
        service = self._service
        generation = self._root_generation
        if (
            not self._active
            or self._root_transitioning
            or self._path_transitioning
            or self._shutdown
            or service is None
        ):
            return False
        if self._opened is not None and not await self.flush_pending_work():
            return False
        if (
            not self._active
            or self._root_transitioning
            or self._path_transitioning
            or generation != self._root_generation
            or service is not self._service
        ):
            return False
        with self._hold_path_transition() as transition:
            if transition is None:
                return False
            try:
                opened = await asyncio.to_thread(
                    service.open_file,
                    relative_path,
                )
            except Exception as error:
                self._set_action_status(f"Open failed: {error}")
                return False
            if (
                not self._active
                or generation != self._root_generation
                or service is not self._service
            ):
                return False
            self._apply_opened_document(opened)
            return True

    def _apply_opened_document(self, opened: OpenedFileNote) -> None:
        if not self._active:
            return
        self._dismiss_reload_confirmation(focus_opener=False)
        self._opened = opened
        self._current_path = opened.relative_path
        self._selected_deleted_path = ""
        self._session_key = uuid4().hex
        self._set_delete_confirmation()
        editor = self.query_one("#file-notes-editor", TextArea)
        with editor.prevent(TextArea.Changed):
            editor.load_text(opened.body)
        self._sync_editor_read_only()
        self._sync_large_file_preview()
        self.query_one("#file-notes-path", Input).value = opened.relative_path
        self.query_one("#file-notes-breadcrumb", Static).update(opened.relative_path)
        if opened.editable:
            self._set_save_state("saved")
        else:
            self._set_save_state(
                "saved",
                f"read only: {opened.read_only_reason or 'unsupported content'}",
            )
        self._set_action_status(opened.replica_warning or "")
        if self._narrow:
            self._narrow_view = "editor"
            self._apply_responsive_layout(self.size.width)
        self._update_controls()

    def _clear_open_document(self, *, keep_restore_path: bool = False) -> None:
        self._dismiss_reload_confirmation(focus_opener=False)
        self._opened = None
        self._current_path = ""
        self._session_key = ""
        self._delete_confirmation_path = ""
        if not keep_restore_path:
            self._selected_deleted_path = ""
        if not self._active or not self.is_mounted:
            self._save_state = "idle"
            self._save_detail = ""
            return
        editor = self.query_one("#file-notes-editor", TextArea)
        with editor.prevent(TextArea.Changed):
            editor.load_text("")
        self._sync_editor_read_only()
        self._sync_large_file_preview()
        if not keep_restore_path:
            self.query_one("#file-notes-path", Input).value = ""
            self.query_one("#file-notes-breadcrumb", Static).update("No file selected")
        self._set_delete_confirmation()
        self._set_save_state("idle")
        self._update_controls()

    def select_deleted(self, relative_path: str) -> bool:
        """Select one persistent tombstone for the Restore action."""
        if not self._active or relative_path not in self._deleted_paths:
            return False
        self._selected_deleted_path = relative_path
        self._clear_open_document(keep_restore_path=True)
        self.query_one("#file-notes-path", Input).value = relative_path
        self.query_one("#file-notes-breadcrumb", Static).update(
            f"Recently deleted: {relative_path}"
        )
        self._set_action_status("Ready to restore.")
        if self._narrow:
            self._narrow_view = "editor"
            self._apply_responsive_layout(self.size.width)
        self._update_controls()
        return True

    async def refresh_files(self) -> bool:
        """Run one reconciliation off the UI loop and update retained widgets."""
        service = self._service
        generation = self._root_generation
        if service is None or self._path_transitioning:
            return False
        async with self._refresh_lock:
            if (
                not self._active
                or not self.is_mounted
                or not self.children
                or self._path_transitioning
                or generation != self._root_generation
                or service is not self._service
            ):
                return False
            try:
                result = await asyncio.to_thread(service.reconcile)
                deleted = await self._load_deleted_paths(
                    replica=self._replica,
                    service=service,
                )
            except Exception as error:
                self._set_action_status(f"Refresh failed: {error}")
                return False
            if self._path_result_is_stale(service, generation) or (
                not self.children or self._path_transitioning
            ):
                return False
            if not self._apply_reconcile(result, deleted):
                return False
            await self._handle_open_external_change(result)
        return True

    async def _handle_open_external_change(
        self,
        result: ReconcileResult,
        *,
        resuming: bool = False,
    ) -> None:
        opened = self._opened
        if opened is None:
            return
        path = opened.relative_path
        if path in result.deleted:
            self._set_save_state("conflict", "file deleted on disk")
            return
        changed = path in result.modified
        if resuming and not changed:
            entry = self._entries.get(path)
            changed = entry is not None and entry.content_hash != opened.content_hash
        if not changed:
            return
        state = self._save_state
        if state in {"conflict", "error"} or (
            state in {"dirty", "saving"} and not resuming
        ):
            self._set_save_state("conflict", "file changed on disk")
            return
        service = self._service
        generation = self._root_generation
        assert service is not None
        try:
            reloaded = await asyncio.to_thread(service.open_file, path)
        except Exception as error:
            self._set_save_state("error", f"reload failed: {error}")
            return
        if self._path_result_is_stale(service, generation):
            return
        if self._opened is not opened:
            return
        if self._save_state != state:
            if self._save_state in {"dirty", "saving", "conflict", "error"}:
                self._set_save_state("conflict", "file changed on disk")
            return
        if resuming and state in {"dirty", "saving"}:
            editor = self.query_one("#file-notes-editor", TextArea)
            if reloaded.body != editor.text:
                self._set_save_state("conflict", "file changed on disk")
                return
            self._opened = reloaded
            self._set_save_state("saved")
            self._set_action_status(reloaded.replica_warning or "")
            return
        self._apply_opened_document(reloaded)

    def _start_poll(self) -> None:
        if (
            not self._active
            or self._root_transitioning
            or self._path_transitioning
            or self._service is None
            or (self._poll_worker is not None and not self._poll_worker.is_finished)
        ):
            return
        self._poll_worker = self.run_worker(
            self.refresh_files(),
            name="file-notes-poll",
            group="file-notes-poll",
            exclusive=True,
        )

    def _arm_autosave(self) -> None:
        if self._autosave_timer is not None:
            self._autosave_timer.stop()
        self._autosave_timer = self.set_timer(
            self._autosave_delay,
            self._start_autosave,
        )

    def _start_autosave(self) -> None:
        self._autosave_timer = None
        if (
            not self._active
            or self._save_state != "dirty"
            or (
                self._save_worker is not None
                and not self._save_worker.is_finished
            )
        ):
            return
        self._save_worker = self.run_worker(
            self._save_draft(),
            name="file-notes-autosave",
            group="file-notes-save",
            exclusive=False,
        )

    async def _save_draft(self) -> bool:
        async with self._save_lock:
            opened = self._opened
            service = self._service
            if opened is None or service is None:
                return True
            if self._save_state in {"conflict", "error"}:
                return False
            editor = self.query_one("#file-notes-editor", TextArea)
            body = editor.text
            self._set_save_state("saving")
            try:
                result = await asyncio.to_thread(
                    service.save_file,
                    opened,
                    body,
                    session_key=self._session_key,
                )
            except Exception as error:
                self._set_save_state("error", str(error))
                return False
            if not self._active:
                return False
            if result.status == "ok" and result.content_hash is not None:
                self._opened = replace(
                    opened,
                    body=body,
                    content_hash=result.content_hash,
                )
                if editor.text == body:
                    self._set_save_state("saved")
                else:
                    self._set_save_state("dirty")
                    self._arm_autosave()
                self._set_action_status(result.replica_warning or "")
                self._refresh_session_changes()
                return True
            if result.status in {"conflict", "missing"}:
                self._set_save_state("conflict", result.message or result.status)
            else:
                self._set_save_state("error", result.message or result.status)
            self._set_action_status(result.replica_warning or "")
            return False

    def _mutation_blocks_flush(self, binding: SessionBinding | None) -> bool:
        """Explain an exact-binding mutation refusal without changing admission."""
        if binding is None or not self._session_owner.mutation_active(binding):
            return False
        self._set_action_status(_SESSION_GIT_MUTATION_BUSY)
        return True

    async def flush_pending_work(self) -> bool:
        """Flush a pending autosave; unresolved draft states veto leaving."""
        binding = self._session_binding
        if self._root_transitioning or self._path_transitioning:
            return False
        if self._mutation_blocks_flush(binding):
            return False
        if not self._active:
            return self.leave_allowed
        if self._autosave_timer is not None:
            self._autosave_timer.stop()
            self._autosave_timer = None
        worker = self._save_worker
        if worker is not None and not worker.is_finished:
            try:
                await worker.wait()
            except Exception:
                pass
        if self._save_state in {"conflict", "error"}:
            return False
        if self._save_state == "dirty":
            await self._save_draft()
        binding = self._session_binding
        if self._mutation_blocks_flush(binding):
            return False
        return self.leave_allowed

    async def _rescan_after_action(self) -> bool:
        service = self._service
        generation = self._root_generation
        if not self._active or service is None:
            return False
        result = await asyncio.to_thread(service.scan)
        deleted = await self._load_deleted_paths(
            replica=self._replica,
            service=service,
        )
        if self._path_result_is_stale(service, generation):
            return False
        return self._apply_scan(result, deleted)

    def _operation_error(self, action: str, result: OperationResult) -> None:
        detail = (
            "destination already exists"
            if result.status == "exists"
            else result.message or result.status
        )
        self._set_action_status(f"{action} failed: {detail}")
        if self._opened is not None and result.status in {"conflict", "missing"}:
            self._set_save_state("conflict", detail)
        elif self._save_state in {"dirty", "saving"}:
            self._set_save_state("error", detail)

    def _validated_path_input(self, action: str) -> str | None:
        raw_path = self.query_one("#file-notes-path", Input).value
        relative_path = raw_path.strip()
        valid_text = validate_text_input(
            raw_path,
            max_length=4096,
            allow_html=True,
        )
        if not relative_path or not valid_text:
            self._set_action_status(f"{action} failed: unsupported path text.")
            return None
        return relative_path

    async def _complete_path_action(
        self,
        action: str,
        relative_path: str,
        operation: Callable[..., OperationResult],
        *args: object,
    ) -> None:
        with self._hold_path_transition() as transition:
            if transition is None:
                return
            service, generation = transition
            result = await asyncio.to_thread(operation, *args)
            if self._path_result_is_stale(service, generation):
                return
            if not result.succeeded:
                self._operation_error(action, result)
                return
            if not await self._rescan_after_action():
                return
            try:
                opened = await asyncio.to_thread(service.open_file, relative_path)
            except Exception as error:
                self._set_action_status(f"Open failed: {error}")
                return
            if self._path_result_is_stale(service, generation):
                return
            self._apply_opened_document(opened)

    @on(TextArea.Changed, "#file-notes-editor")
    def _editor_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        if (
            self._root_transitioning
            or self._path_transitioning
            or self._opened is None
            or not self._opened.editable
        ):
            return
        self._set_delete_confirmation()
        self._set_save_state("dirty")
        self._arm_autosave()

    @on(Input.Changed, "#file-notes-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        query = event.value.strip()
        self._search_query = event.value
        results = self.query_one("#file-notes-search-results", Tree)
        if not query:
            self._search_generation += 1
            if self._navigator_mode == "git":
                self._navigator_mode_before_git = "files"
            else:
                self._navigator_mode = "files"
            self._sync_navigator_mode()
            self._search_paths = ()
            results.reset(Text("Search results"))
            return
        if self._navigator_mode == "git":
            self._navigator_mode_before_git = "search"
        else:
            self._navigator_mode = "search"
        self._sync_navigator_mode()
        self._start_search(query)

    def _start_search(self, query: str) -> None:
        self._search_generation += 1
        generation = self._search_generation
        self.run_worker(
            self._run_search(query, generation),
            name="file-notes-search",
            group="file-notes-search",
            exclusive=True,
        )

    async def _run_search(self, query: str, generation: int) -> None:
        if self._replica is not None and self._service is not None:
            try:
                paths = await asyncio.to_thread(
                    self._replica.search,
                    self._service.root_key,
                    query,
                )
            except Exception:
                paths = []
        else:
            folded = query.casefold()
            paths = [path for path in self._entries if folded in path.casefold()]
        if generation != self._search_generation or not self._active:
            return
        self._rebuild_search_results(tuple(paths))

    @on(Tree.NodeExpanded, "#file-notes-tree")
    def _tree_node_expanded(self, event: Tree.NodeExpanded[object]) -> None:
        """Materialize one folder level only when the user expands it."""
        self._populate_folder_node(event.node)

    @on(Tree.NodeSelected)
    async def _tree_node_selected(self, event: Tree.NodeSelected[object]) -> None:
        data = event.node.data
        if isinstance(data, _TreePageData):
            event.stop()
            parent = event.node.parent
            if parent is None:
                return
            last_node = self._append_tree_page(
                parent,
                data,
                page_node=event.node,
            )
            if event.node.parent is None and last_node is not None:
                tree = self.query_one(
                    (
                        "#file-notes-search-results"
                        if data.source == "search"
                        else "#file-notes-tree"
                    ),
                    Tree,
                )
                tree.move_cursor(last_node)
            return
        if not isinstance(data, tuple) or len(data) != 2:
            return
        kind, relative_path = data
        if kind == "file":
            event.stop()
            await self.open_path(relative_path)
        elif kind == "deleted":
            event.stop()
            if self._opened is not None and not await self.flush_pending_work():
                return
            self.select_deleted(relative_path)

    @on(Button.Pressed, "#file-notes-choose-root")
    async def _choose_root(self, event: Button.Pressed) -> None:
        event.stop()
        location = (
            self._root
            if self._root is not None and self._root_offline is False
            else Path.home()
        )
        await self.app.push_screen(
            SelectDirectory(location, title="Choose File Notes Folder"),
            callback=self._root_selected,
        )

    @on(Button.Pressed, "#file-notes-root-details")
    def _show_root_details(self, event: Button.Pressed) -> None:
        """Open the exact root state without depending on pointer hover."""
        event.stop()
        self.app.push_screen(FileNotesRootDetailsDialog(self._root_status_detail))

    def _root_selected(self, path: Path | None) -> None:
        if path is None or not self._active:
            return
        self.run_worker(
            self.set_root(path),
            name="file-notes-root-change",
            group="file-notes-root-change",
            exclusive=True,
        )

    @on(Button.Pressed, "#file-notes-back")
    def _back_to_navigator(self, event: Button.Pressed) -> None:
        event.stop()
        self._narrow_view = "navigator"
        self._apply_responsive_layout(self.size.width)

    @on(Button.Pressed, "#file-notes-session-changes")
    def _session_git_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry_owned_focus = event.button.has_focus
        if self._navigator_mode != "git":
            self._navigator_mode_before_git = (
                "search"
                if self._navigator_mode == "search"
                else "files"
            )
        self._navigator_mode = "git"
        self._sync_navigator_mode()
        if entry_owned_focus:
            self.screen.set_focus(
                self.query_one("#file-notes-git-back", Button),
                scroll_visible=False,
            )
        self.call_after_refresh(self._focus_session_git_panel)
        self.run_worker(
            self._open_session_git(),
            name="file-notes-git-open",
            group="file-notes-git-open",
            exclusive=True,
        )

    def _focus_session_git_panel(self, retries_remaining: int = 8) -> None:
        """Move focus off the hidden entry into the visible Git surface."""
        if (
            not self._active
            or not self.is_mounted
            or self._navigator_mode != "git"
            or self._git_panel_widget.commit_phase != "list"
        ):
            return
        entry = self.query_one("#file-notes-session-changes", Button)
        back = self.query_one("#file-notes-git-back", Button)
        rows = self.query_one("#file-notes-git-rows", ListView)
        focused = self.app.focused
        if not (
            focused is entry
            or focused is back
            or focused is rows
            or (focused is not None and rows in focused.ancestors)
        ):
            return
        if self._git_panel_widget.rows:
            if rows.display:
                self.screen.set_focus(rows, scroll_visible=False)
            elif retries_remaining:
                self.call_after_refresh(
                    self._focus_session_git_panel,
                    retries_remaining - 1,
                )
            else:
                self.screen.set_focus(back, scroll_visible=False)
            return
        self.screen.set_focus(back, scroll_visible=False)

    @on(LibraryFileNotesGitPanel.BackRequested)
    def _session_git_back(
        self,
        event: LibraryFileNotesGitPanel.BackRequested,
    ) -> None:
        event.stop()
        self._navigator_mode = self._navigator_mode_before_git
        self._sync_navigator_mode()
        self.call_after_refresh(
            self.query_one("#file-notes-session-changes", Button).focus
        )

    @on(LibraryFileNotesGitPanel.RefreshRequested)
    def _session_git_refresh(
        self,
        event: LibraryFileNotesGitPanel.RefreshRequested,
    ) -> None:
        event.stop()
        self._start_git_refresh()

    @on(LibraryFileNotesGitPanel.TrustRequested)
    def _session_git_trust(
        self,
        event: LibraryFileNotesGitPanel.TrustRequested,
    ) -> None:
        event.stop()
        self.run_worker(
            self._open_session_git(force_prompt=True),
            name="file-notes-git-trust",
            group="file-notes-git-open",
            exclusive=True,
        )

    @on(LibraryFileNotesGitPanel.ReviewPushRequested)
    def _session_git_review_push(
        self,
        event: LibraryFileNotesGitPanel.ReviewPushRequested,
    ) -> None:
        """Start local proof and adopt only the service-retained operation."""
        event.stop()
        binding = self._session_binding
        availability = self._push_availability
        service = self._session_git_service()
        key = (
            None
            if binding is None or availability is None
            else self._push_key_for_availability(binding, availability)
        )
        if (
            binding is None
            or availability is None
            or event.availability != availability
            or service is None
            or key is None
            or key != self._push_key
        ):
            return
        try:
            service.start_push_review(binding)
            operation = self._retained_push_after_start(
                service,
                key,
                "local_proof",
            )
        except (GitMutationAdmissionError, RuntimeError):
            self._publish_push_review_admission_failure()
            return
        self._adopt_push_operation(operation, key)

    @on(LibraryFileNotesGitPanel.PushOperationRequested)
    def _session_git_push_operation(
        self,
        event: LibraryFileNotesGitPanel.PushOperationRequested,
    ) -> None:
        """Dispatch one typed action for an exact push operation ID."""
        event.stop()
        handler = {
            "endpoint_details": self._session_git_push_endpoint_details,
            "back_from_review": self._session_git_back_from_push_review,
            "push_reviewed_commit": self._session_git_push_reviewed_commit,
            "cancel_check": self._session_git_cancel_push_check,
            "back_to_files": self._session_git_back_to_files_push_continues,
            "back_to_session": self._session_git_back_to_push_session,
            "review_again": self._session_git_review_push_again,
            "check_remote_again": self._session_git_check_remote_again,
        }[event.action]
        handler(event.operation_id)

    def _session_git_push_endpoint_details(self, operation_id: int) -> None:
        """Open exact review details only for the current operation ID."""
        projection = self._push_review_projection
        if (
            projection is None
            or self._push_view_phase != "review"
            or operation_id != self._push_operation_id
        ):
            return
        self.app.push_screen(
            PushEndpointDetailsDialog(
                projection.review.destination,
                restore_focus=False,
            ),
            callback=partial(
                self._restore_push_review_details_focus,
                operation_id,
            ),
        )

    def _restore_push_review_details_focus(
        self,
        operation_id: int,
        _result: None,
    ) -> None:
        self._git_panel_widget.restore_push_focus(
            "review",
            operation_id,
            "#file-notes-git-push-review-details",
        )

    def _session_git_back_from_push_review(self, operation_id: int) -> None:
        """Discard only the exact ready review selected by the user."""
        operation = self._push_operation
        key = self._push_operation_key
        service = self._session_git_service()
        if (
            operation is None
            or key is None
            or service is None
            or self._push_view_phase != "review"
            or operation_id != self._push_operation_id
            or not self._push_operation_is_current(
                operation,
                key,
                operation_id,
            )
            or not service.cancel_push(key.binding, operation)
        ):
            return
        self._push_operation_id += 1
        self._push_operation = None
        self._push_operation_key = None
        self._push_operation_admitted = False
        self._push_observer_task = None
        self._push_key = self._push_key_for_availability(
            key.binding,
            self._push_availability,
        )
        self._return_push_to_list()

    def _session_git_push_reviewed_commit(self, operation_id: int) -> None:
        """Consume one workspace-held review and adopt the retained push."""
        operation = self._push_operation
        key = self._push_operation_key
        handle = self._push_review_handle
        service = self._session_git_service()
        if (
            operation is None
            or key is None
            or handle is None
            or service is None
            or self._push_view_phase != "review"
            or operation_id != self._push_operation_id
            or not self._push_operation_is_current(
                operation,
                key,
                operation_id,
            )
        ):
            return
        try:
            service.start_push(key.binding, handle)
            next_operation = self._retained_push_after_start(
                service,
                key,
                "push",
            )
        except (GitMutationAdmissionError, RuntimeError):
            self._set_push_result_projection(
                PushPanelResultProjection(
                    title="Blocked",
                    message=(
                        "The reviewed push could not start. Review the exact "
                        "candidate again."
                    ),
                    action="review_again",
                )
            )
            return
        self._push_review_handle = None
        self._adopt_push_operation(next_operation, key)

    def _session_git_cancel_push_check(self, operation_id: int) -> None:
        """Request cancellation only before the exact push child starts."""
        operation = self._push_operation
        key = self._push_operation_key
        service = self._session_git_service()
        if (
            operation is None
            or key is None
            or service is None
            or self._push_view_phase
            not in {"checking_candidate", "checking_remote"}
            or operation_id != self._push_operation_id
            or not self._push_operation_is_current(
                operation,
                key,
                operation_id,
            )
        ):
            return
        accepted = service.cancel_push(key.binding, operation)
        if accepted:
            self._push_operation_id += 1
            self._push_operation_admitted = False
            self._push_result = None
            self._push_phase = "idle"
            self._return_push_to_list()
            self._render_session_git_label()
        elif not accepted and operation.child_started:
            self._push_phase = "pushing"
            self._push_view_phase = "pushing"
            self._render_session_git_label()
            self._render_current_push_view()

    def _session_git_back_to_files_push_continues(
        self,
        operation_id: int,
    ) -> None:
        """Hide an admitted push while retaining the one existing observer."""
        operation = self._push_operation
        key = self._push_operation_key
        if (
            operation is None
            or key is None
            or self._push_view_phase not in {"checking_uncertain", "pushing"}
            or operation_id != self._push_operation_id
            or (
                operation.kind != "recovery"
                and (operation.kind != "push" or not operation.child_started)
            )
            or not self._push_operation_is_current(
                operation,
                key,
                operation_id,
            )
        ):
            return
        self._navigator_mode = self._navigator_mode_before_git
        self._sync_navigator_mode()
        self.call_after_refresh(
            self.query_one("#file-notes-session-changes", Button).focus
        )

    def _session_git_back_to_push_session(self, operation_id: int) -> None:
        """Leave one proven result without disturbing retained service state."""
        if (
            self._push_view_phase == "result"
            and operation_id == self._push_operation_id
            and self._push_result_projection is not None
        ):
            self._return_push_to_list()

    def _session_git_review_push_again(self, operation_id: int) -> None:
        """Begin fresh local proof after one definite failed attempt."""
        context = self._current_push_result_context(
            operation_id,
            "review_again",
        )
        availability = self._push_availability
        if context is None or availability is None:
            return
        service, _operation, key = context
        if self._push_key_for_availability(key.binding, availability) != key:
            return
        try:
            service.start_push_review(key.binding)
            next_operation = self._retained_push_after_start(
                service,
                key,
                "local_proof",
            )
        except (GitMutationAdmissionError, RuntimeError):
            self._publish_push_review_admission_failure()
            return
        self._adopt_push_operation(next_operation, key)

    def _session_git_check_remote_again(self, operation_id: int) -> None:
        """Query only the retained uncertain destination; never push again."""
        context = self._current_push_result_context(
            operation_id,
            "check_remote_again",
        )
        if context is None:
            return
        service, operation, key = context
        self._start_push_recovery_query(
            service,
            operation,
            key,
            operation_id,
        )

    @on(LibraryFileNotesGitPanel.CommitStagedRequested)
    def _session_git_commit_staged(
        self,
        event: LibraryFileNotesGitPanel.CommitStagedRequested,
    ) -> None:
        """Adopt only the exact current availability projection as a draft."""
        event.stop()
        projection = self._commit_availability
        if (
            projection is None
            or event.binding_key != projection.binding_key
            or not isinstance(event.binding_key, _CommitBindingKey)
            or event.binding_key.binding != self._session_binding
        ):
            self._git_panel_widget.invalidate_commit_binding()
            self._set_action_status(
                "Commit availability changed; Refresh and try again."
            )
            return
        draft = self._commit_draft
        if draft is None or draft.key != event.binding_key:
            self._commit_draft = _CommitDraftState(
                event.binding_key,
                projection.subject,
                projection.body,
            )
        self._commit_view_phase = "form"

    @on(LibraryFileNotesGitPanel.CommitDraftChanged)
    def _session_git_commit_draft_changed(
        self,
        event: LibraryFileNotesGitPanel.CommitDraftChanged,
    ) -> None:
        """Retain literal message edits only for their exact active key."""
        event.stop()
        draft = self._commit_draft
        if (
            draft is None
            or event.binding_key != draft.key
            or draft.key.binding != self._session_binding
        ):
            return
        self._commit_draft = replace(
            draft,
            subject=event.subject,
            body=event.body,
        )
        projection = self._commit_availability
        if projection is not None and projection.binding_key == draft.key:
            self._commit_availability = replace(
                projection,
                subject=event.subject,
                body=event.body,
            )

    @on(LibraryFileNotesGitPanel.ReviewCommitRequested)
    async def _session_git_review_commit(
        self,
        event: LibraryFileNotesGitPanel.ReviewCommitRequested,
    ) -> None:
        """Settle the editor, then start one retained service review."""
        event.stop()
        draft = self._commit_draft
        service = self._session_git_service()
        if (
            draft is None
            or service is None
            or event.binding_key != draft.key
            or event.subject != draft.subject
            or event.body != draft.body
            or not self._commit_key_is_current(draft.key)
        ):
            self._git_panel_widget.return_to_commit_list()
            self._set_action_status(
                "Commit draft changed before review; reopen it and try again."
            )
            return
        self._release_commit_editor_lease()
        lease = self._acquire_editor_read_only(draft.key.binding)
        if lease is None:
            self._git_panel_widget.return_to_commit_list()
            self._set_action_status(
                "Commit binding changed before review; Refresh and try again."
            )
            return
        self._commit_editor_lease = lease
        if not await self._settle_commit_editor(lease):
            if self._commit_editor_lease is lease:
                self._commit_editor_lease = None
            projection = self._current_commit_draft_projection(
                form_error=self._action_detail,
            )
            if projection is not None:
                self._commit_view_phase = "form"
                self._git_panel_widget.render_commit_form(projection)
            return
        if not self._commit_key_is_current(draft.key):
            self._release_commit_editor_lease()
            self._invalidate_commit_binding(
                "Repository changed while saving; the commit draft was cleared."
            )
            return
        try:
            service.start_commit_review(
                draft.key.binding,
                draft.subject,
                draft.body,
            )
            operation = service.retained_commit_operation(draft.key.binding)
            if operation is None or operation.kind != "review":
                raise RuntimeError("retained commit review is unavailable")
        except (GitMutationAdmissionError, RuntimeError) as error:
            self._release_commit_editor_lease()
            projection = self._current_commit_draft_projection(
                form_error=str(error),
            )
            if projection is not None:
                self._commit_view_phase = "form"
                self._git_panel_widget.render_commit_form(projection)
            self._set_action_status(f"Commit review blocked: {error}")
            return
        operation_id = self._begin_commit_operation(operation, "checking")
        self._commit_review_handle = None
        self._commit_review_projection = None
        self._commit_result_projection = None
        self._attach_commit_lease_settlement(
            operation,
            draft.key,
            lease,
            hold_ready_review=True,
        )
        self._set_action_status("Checking commit…")
        self._git_commit_worker = self.run_worker(
            self._observe_commit_review(
                operation,
                draft.key,
                operation_id,
            ),
            name="file-notes-git-commit-review",
            group="file-notes-git-commit",
            exclusive=True,
        )

    async def _observe_commit_review(
        self,
        operation: RetainedCommitOperation,
        key: _CommitBindingKey,
        operation_id: int,
    ) -> None:
        """Render one retained review settlement only while it is current."""
        try:
            result = await operation.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._commit_operation_is_current(operation, key, operation_id):
                self._release_commit_editor_lease()
                self._commit_view_phase = "form"
                projection = self._current_commit_draft_projection(
                    form_error=f"Commit review failed: {error}",
                )
                if projection is not None:
                    self._git_panel_widget.render_commit_form(projection)
            return
        if not isinstance(
            result, CommitReviewResult
        ) or not self._commit_operation_is_current(
            operation,
            key,
            operation_id,
        ):
            return
        if (
            result.state == "ready"
            and result.handle is not None
            and result.projection is not None
        ):
            projection = self._build_commit_review_projection(
                key,
                result.projection,
            )
            if projection is None:
                service = self._session_git_service()
                if service is not None:
                    service.cancel_commit(key.binding)
                self._release_commit_editor_lease()
                self._commit_view_phase = "form"
                draft_projection = self._current_commit_draft_projection(
                    form_error=("Session notes changed; Refresh and review again."),
                )
                if draft_projection is not None:
                    self._git_panel_widget.render_commit_form(draft_projection)
                return
            self._commit_review_handle = result.handle
            self._commit_review_projection = projection
            self._commit_view_phase = "review"
            self._git_panel_widget.render_commit_review(projection)
            self._set_action_status("Commit review ready.")
            return
        self._commit_review_handle = None
        self._commit_review_projection = None
        if result.state == "cancelled":
            self._commit_view_phase = "list"
            self._git_panel_widget.return_to_commit_list(
                restore_entry_focus=True,
            )
            self._set_action_status("Commit review cancelled.")
            return
        self._commit_view_phase = "form"
        detail = result.message or "Commit review was blocked."
        projection = self._current_commit_draft_projection(
            form_error=detail,
        )
        if projection is not None:
            self._git_panel_widget.render_commit_form(projection)
        self._set_action_status(detail)

    def _build_commit_review_projection(
        self,
        key: _CommitBindingKey,
        review,
    ) -> CommitPanelReviewProjection | None:
        """Render the service's immutable, Git-proven included-note facts."""
        if not self._commit_key_is_current(key):
            return None
        try:
            return CommitPanelReviewProjection(
                review,
                tuple(
                    CommitReviewNoteProjection(note)
                    for note in review.included_notes
                ),
            )
        except ValueError:
            return None

    @on(LibraryFileNotesGitPanel.EditCommitMessageRequested)
    def _session_git_edit_commit_message(
        self,
        event: LibraryFileNotesGitPanel.EditCommitMessageRequested,
    ) -> None:
        """Consume any ready review and restore editor writability."""
        event.stop()
        key = self._active_commit_key()
        service = self._session_git_service()
        if key is not None and service is not None:
            service.cancel_commit(key.binding)
        self._commit_review_handle = None
        self._commit_review_projection = None
        self._commit_result_projection = None
        self._commit_view_phase = "form"
        self._release_commit_editor_lease()

    @on(LibraryFileNotesGitPanel.ConfirmCommitRequested)
    def _session_git_confirm_commit(
        self,
        event: LibraryFileNotesGitPanel.ConfirmCommitRequested,
    ) -> None:
        """Start exact revalidation and observe the child-start boundary."""
        event.stop()
        key = self._active_commit_key()
        draft = self._commit_draft
        handle = self._commit_review_handle
        review = self._commit_review_projection
        lease = self._commit_editor_lease
        service = self._session_git_service()
        if (
            key is None
            or draft is None
            or handle is None
            or review is None
            or lease is None
            or service is None
        ):
            self._release_commit_editor_lease()
            self._commit_view_phase = "form"
            projection = self._current_commit_draft_projection(
                form_error="Commit review expired; review again.",
            )
            if projection is not None:
                self._git_panel_widget.render_commit_form(projection)
            return
        try:
            service.start_commit(
                key.binding,
                handle,
                subject=draft.subject,
                body=draft.body,
            )
            operation = service.retained_commit_operation(key.binding)
            if operation is None or operation.kind != "commit":
                raise RuntimeError("retained commit operation is unavailable")
        except (GitMutationAdmissionError, RuntimeError) as error:
            self._release_commit_editor_lease()
            self._commit_review_handle = None
            self._commit_review_projection = None
            self._commit_view_phase = "form"
            projection = self._current_commit_draft_projection(
                form_error=str(error),
            )
            if projection is not None:
                self._git_panel_widget.render_commit_form(projection)
            return
        self._commit_review_handle = None
        operation_id = self._begin_commit_operation(operation, "confirming")
        self._attach_commit_lease_settlement(
            operation,
            key,
            lease,
            hold_ready_review=False,
        )
        self._set_action_status("Checking commit before branch update…")
        self._git_commit_child_worker = self.run_worker(
            self._observe_commit_child_start(operation, key, operation_id),
            name="file-notes-git-commit-child",
            group="file-notes-git-commit-child",
            exclusive=True,
        )
        self._git_commit_worker = self.run_worker(
            self._observe_commit_outcome(operation, key, operation_id),
            name="file-notes-git-commit-outcome",
            group="file-notes-git-commit",
            exclusive=True,
        )

    async def _observe_commit_child_start(
        self,
        operation: RetainedCommitOperation,
        key: _CommitBindingKey,
        operation_id: int,
    ) -> None:
        """Render execution only after the exact service-owned transition."""
        try:
            child_started = await operation.wait_child_started()
        except asyncio.CancelledError:
            raise
        if (
            not child_started
            or operation.settled
            or not self._commit_operation_is_current(
                operation,
                key,
                operation_id,
            )
        ):
            return
        review = self._commit_review_projection
        count = 0 if review is None else review.review.included_note_count
        self._commit_view_phase = "executing"
        self._git_panel_widget.render_commit_executing(CommitExecutionProjection(count))
        self._set_action_status(f"Committing {count} session notes…")

    async def _observe_commit_outcome(
        self,
        operation: RetainedCommitOperation,
        key: _CommitBindingKey,
        operation_id: int,
    ) -> None:
        """Render one typed terminal/recoverable outcome with stale guards."""
        try:
            result = await operation.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._commit_operation_is_current(operation, key, operation_id):
                self._commit_operation_id += 1
                self._commit_operation = None
                self._commit_review_handle = None
                self._commit_review_projection = None
                self._commit_result_projection = None
                self._release_commit_editor_lease()
                self._commit_view_phase = "form"
                detail = (
                    "Commit result could not be observed. "
                    "Inspect Git before trying again."
                )
                projection = self._current_commit_draft_projection(
                    form_error=detail,
                )
                if projection is not None:
                    self._git_panel_widget.render_commit_form(projection)
                self._set_action_status(detail)
            return
        if not isinstance(
            result, CommitOutcome
        ) or not self._commit_operation_is_current(
            operation,
            key,
            operation_id,
        ):
            return
        self._commit_review_handle = None
        if result.state == "cancelled":
            self._commit_review_projection = None
            self._commit_result_projection = None
            self._commit_view_phase = "list"
            self._git_panel_widget.return_to_commit_list(
                restore_entry_focus=True,
            )
            self._set_action_status("Commit cancelled before branch update.")
            return
        if result.state == "blocked":
            self._commit_review_projection = None
            self._commit_result_projection = None
            self._commit_view_phase = "form"
            projection = self._current_commit_draft_projection(
                form_error=result.message,
            )
            if projection is not None:
                self._git_panel_widget.render_commit_form(projection)
            self._set_action_status(result.message)
            return
        if result.state == "succeeded":
            self._clear_commit_draft_after_success()
            snapshot = self._session_owner.snapshot(key.binding)
            if snapshot.trusted_repository == key.repository:
                note_label = (
                    "note" if result.committed_note_count == 1 else "notes"
                )
                self._git_last_action = _GitLastAction(
                    binding=key.binding,
                    repository=key.repository,
                    changes=snapshot.changes,
                    text=(
                        f"Committed {result.committed_note_count} session "
                        f"{note_label}; unrelated changes untouched."
                    ),
                )
                self._sync_git_last_action()
            self._git_panel_widget.return_to_commit_list()
            self._set_action_status(result.message)
            return
        recovery = None
        if result.state == "uncertain":
            recovery = self._session_owner.snapshot(key.binding).commit_recovery
            if recovery is None:
                recovery = CommitRecoveryProjection(result.message, False)
        projection = CommitResultProjection(result, recovery)
        self._commit_result_projection = projection
        self._commit_view_phase = "result"
        self._git_panel_widget.render_commit_result(projection)
        self._set_action_status(result.message)

    @on(LibraryFileNotesGitPanel.CancelCommitRequested)
    def _session_git_cancel_draft(
        self,
        event: LibraryFileNotesGitPanel.CancelCommitRequested,
    ) -> None:
        """Cancel only service-owned pre-child work and preserve the draft."""
        event.stop()
        phase = event.from_phase
        if phase == "form":
            self._commit_view_phase = "list"
            return
        key = self._active_commit_key()
        service = self._session_git_service()
        if phase in {"checking", "review", "confirming"}:
            operation = self._commit_operation
            accepted = (
                key is not None
                and service is not None
                and service.cancel_commit(key.binding)
            )
            if accepted:
                if operation is not None and not operation.settled:
                    self._set_action_status(
                        "Cancelling after Git finishes its current check…"
                    )
                    return
                self._commit_operation_id += 1
                self._commit_operation = None
                self._commit_review_handle = None
                self._commit_review_projection = None
                self._commit_result_projection = None
                self._release_commit_editor_lease()
                self._commit_view_phase = "list"
                if phase == "confirming":
                    self._git_panel_widget.return_to_commit_list(
                        restore_entry_focus=True,
                    )
                self._set_action_status(
                    "Commit cancelled before branch update."
                    if phase == "confirming"
                    else "Commit review cancelled."
                )
                return
            if (
                phase == "confirming"
                and operation is not None
                and operation.child_started
            ):
                review = self._commit_review_projection
                count = 0 if review is None else review.review.included_note_count
                self._commit_view_phase = "executing"
                self._git_panel_widget.render_commit_executing(
                    CommitExecutionProjection(count)
                )
            return
        if phase == "result":
            self._commit_view_phase = "list"

    @on(LibraryFileNotesGitPanel.CheckCommitAgainRequested)
    async def _session_git_check_commit_again(
        self,
        event: LibraryFileNotesGitPanel.CheckCommitAgainRequested,
    ) -> None:
        """Inspect one retained uncertain attempt without starting a commit."""
        event.stop()
        key = self._active_commit_key()
        service = self._session_git_service()
        current_result = self._commit_result_projection
        if (
            key is None
            or service is None
            or current_result is None
            or current_result.outcome.state != "uncertain"
            or current_result.recovery is None
        ):
            return
        self._release_commit_editor_lease()
        lease = self._acquire_editor_read_only(key.binding)
        if lease is None:
            return
        self._commit_editor_lease = lease
        if not await self._settle_commit_editor(lease):
            if self._commit_editor_lease is lease:
                self._commit_editor_lease = None
            self._git_panel_widget.render_commit_result(current_result)
            return
        try:
            service.check_commit_again(key.binding)
            operation = service.retained_commit_operation(key.binding)
            if operation is None or operation.kind != "recovery":
                raise RuntimeError("retained commit recovery is unavailable")
        except (GitMutationAdmissionError, RuntimeError) as error:
            self._release_commit_editor_lease()
            recovery = CommitRecoveryProjection(str(error), False)
            projection = CommitResultProjection(
                current_result.outcome,
                recovery,
            )
            self._commit_result_projection = projection
            self._git_panel_widget.render_commit_result(projection)
            return
        operation_id = self._begin_commit_operation(operation, "checking")
        self._commit_view_phase = "executing"
        self._attach_commit_lease_settlement(
            operation,
            key,
            lease,
            hold_ready_review=False,
        )
        self._git_panel_widget.render_commit_recovery_checking()
        self._set_action_status("Checking the retained commit attempt…")
        self._git_commit_worker = self.run_worker(
            self._observe_commit_outcome(operation, key, operation_id),
            name="file-notes-git-commit-recovery",
            group="file-notes-git-commit",
            exclusive=True,
        )

    @on(LibraryFileNotesGitPanel.StageRequested)
    async def _session_git_stage(
        self,
        event: LibraryFileNotesGitPanel.StageRequested,
    ) -> None:
        event.stop()
        await self._start_git_action("stage", event.group_ids, bulk=event.bulk)

    @on(LibraryFileNotesGitPanel.UnstageRequested)
    async def _session_git_unstage(
        self,
        event: LibraryFileNotesGitPanel.UnstageRequested,
    ) -> None:
        event.stop()
        await self._start_git_action("unstage", event.group_ids, bulk=event.bulk)

    async def _start_git_action(
        self,
        action: Literal["stage", "unstage"],
        group_ids: tuple[int, ...],
        *,
        bulk: bool,
    ) -> None:
        """Flush as required, then synchronously admit one retained action."""
        binding = self._session_binding
        service = self._session_git_service()
        if binding is None or service is None or self._navigator_mode != "git":
            return
        summary_context = self._git_action_summary_context(
            action,
            group_ids,
            bulk=bulk,
        )
        pending_save = (
            self._save_state in {"dirty", "saving"}
            or self._autosave_timer is not None
            or (
                self._save_worker is not None
                and not self._save_worker.is_finished
            )
        )
        if (action == "stage" or pending_save) and not await self.flush_pending_work():
            gerund = "staging" if action == "stage" else "unstaging"
            if self._save_state == "conflict":
                detail = (
                    f"Save conflict must be resolved before {gerund}. "
                    "Return to the editor."
                )
            elif self._save_state == "error":
                detail = (
                    f"Fix the save error before {gerund}. "
                    "Return to the editor."
                )
            else:
                detail = (
                    f"Save the note before {gerund}. Return to the editor."
                )
            self._git_panel_widget.set_current_status(
                f"Status: CURRENT · BLOCKED — {detail}",
                complete=True,
            )
            return
        if (
            not self._git_binding_is_current(binding)
            or self._root_transitioning
            or self._path_transitioning
            or self._save_state in {"dirty", "saving", "conflict", "error"}
        ):
            self._git_panel_widget.set_current_status(
                "Status: CURRENT · BLOCKED — File Notes changed before "
                f"{action.title()}. Return to the editor, finish the save or "
                "transition, then Refresh.",
                complete=True,
            )
            return
        action_key = self._capture_git_action_key(binding)
        if action_key is None:
            self._git_panel_widget.set_current_status(
                "Status: STALE · BLOCKED — Repository or session authority "
                "changed before the action. Return to the navigator, open "
                "Review session changes again, then Refresh.",
                complete=True,
            )
            return
        try:
            task = (
                service.start_stage(binding, group_ids)
                if action == "stage"
                else service.start_unstage(binding, group_ids)
            )
        except GitMutationAdmissionError as error:
            self._git_panel_widget.set_current_status(
                f"Status: CURRENT · BLOCKED — {action.title()} could not "
                f"start: {error}. Finish the active File Notes action, then "
                "Refresh.",
                complete=True,
            )
            return
        action_key_after_admission = self._capture_git_action_key(binding)
        if action_key_after_admission != action_key:
            action_key = None
        self._clear_git_last_action()
        self._git_status_task = None
        self._git_status_task_binding = None
        self._git_panel_widget.set_mutating(
            True,
            f"{action.title()} in progress…",
        )
        self._update_root_surface()
        self._update_controls()
        self._git_action_worker = self.run_worker(
            self._render_git_action(
                task,
                binding,
                summary_context,
                action_key,
            ),
            name=f"file-notes-git-{action}",
            group="file-notes-git-action",
            exclusive=True,
        )

    async def _render_git_action(
        self,
        task: asyncio.Task[GitActionResult],
        binding: SessionBinding,
        summary_context: _GitActionSummaryContext,
        action_key: _GitLastAction | None,
    ) -> None:
        result: GitActionResult | None = None
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            detail = (
                f"Git action failed: {error}. Inspect the repository "
                "index outside Chatbook, then Refresh."
            )
            if (
                action_key is not None
                and self._git_action_key_is_current(action_key)
            ):
                self._git_last_action = replace(
                    action_key,
                    text=f"Last action: FAILED — {detail}",
                    complete=True,
                )
                if self._git_binding_is_current(binding):
                    self._sync_git_last_action()
            if self._git_binding_is_current(binding):
                self._git_panel_widget.mark_stale(
                    detail,
                    retain_rows=self._git_can_retain_rows(binding),
                    error=True,
                )
        else:
            if action_key is not None:
                summary = self._git_action_summary(
                    result,
                    summary_context,
                    action_key,
                )
                if (
                    summary is not None
                    and self._git_action_key_is_current(action_key)
                ):
                    self._git_last_action = replace(
                        action_key,
                        text=(
                            f"Last action: {self._git_action_label(result)} — "
                            f"{summary}"
                        ),
                        complete=result.state != "success",
                    )
                    if self._git_binding_is_current(binding):
                        self._sync_git_last_action()
        finally:
            binding_changed = (
                binding != self._session_binding
                or binding != self._session_owner.current_binding()
            )
            if not binding_changed:
                if self._session_owner.mutation_active(binding):
                    self._git_refresh_after_mutation = True
                else:
                    self._update_root_surface()
                    self._update_controls()
                    if self._navigator_mode == "git" and self._active:
                        self._git_refresh_after_mutation = False
                        self._start_git_refresh()
                    else:
                        self._git_refresh_after_mutation = True
                        if self._active and self.is_mounted:
                            self._git_panel_widget.mark_stale(
                                "Git action finished while Review session changes "
                                "was hidden. Open it again to Refresh.",
                                retain_rows=self._git_can_retain_rows(binding),
                            )

    def _git_action_summary_context(
        self,
        action: Literal["stage", "unstage"],
        group_ids: tuple[int, ...],
        *,
        bulk: bool,
    ) -> _GitActionSummaryContext:
        """Classify excluded displayed rows without re-deciding eligibility."""
        if not bulk:
            return _GitActionSummaryContext()
        requested = frozenset(group_ids)
        excluded: tuple[SessionGitRow, ...] = tuple(
            row
            for row in self._git_panel_widget.rows
            if row.group_id not in requested
        )
        clean = sum(row.state == "clean" for row in excluded)
        if action == "stage":
            already_staged = sum(row.state == "owned" for row in excluded)
            return _GitActionSummaryContext(
                bulk=True,
                already_staged=already_staged,
                clean=clean,
                blocked=len(excluded) - clean - already_staged,
            )
        skipped = sum(row.state == "unstaged" for row in excluded)
        return _GitActionSummaryContext(
            bulk=True,
            skipped=skipped,
            clean=clean,
            blocked=len(excluded) - clean - skipped,
        )

    def _git_action_summary(
        self,
        result: GitActionResult,
        context: _GitActionSummaryContext,
        action_key: _GitLastAction,
    ) -> str | None:
        """Render one current checked result without overstating its proof."""
        if not self._git_action_key_is_current(action_key):
            return None
        affected_group_ids = (
            result.staged_group_ids
            if result.action == "stage"
            else result.unstaged_group_ids
        )
        affected = len(tuple(dict.fromkeys(affected_group_ids)))
        counts: list[str] = []
        if (
            context.bulk
            and result.action == "stage"
            and context.already_staged
        ):
            counts.append(f"already staged {context.already_staged}")
        if context.bulk and result.action == "unstage" and context.skipped:
            counts.append(f"skipped {context.skipped}")
        clean = len(result.clean_group_ids) + context.clean
        blocked = len(result.blocked_group_ids) + context.blocked
        if clean:
            counts.append(f"clean {clean}")
        if blocked:
            counts.append(f"blocked {blocked}")
        counts_text = (
            f"Counts: {'; '.join(counts)}." if counts else ""
        )

        message = (result.message or "").strip()
        if message and message[-1] not in ".!?":
            message += "."

        if result.state == "success" and affected:
            note = "session note" if affected == 1 else "session notes"
            if result.action == "stage":
                core = (
                    f"{affected} {note} staged; Chatbook targeted only "
                    "eligible session paths."
                )
            else:
                entry = "entry" if affected == 1 else "entries"
                core = (
                    f"{affected} {note} unstaged; Chatbook restored only its "
                    f"owned session {entry}."
                )
            return " ".join(
                part for part in (core, message, counts_text) if part
            )

        if result.state == "success":
            past = "staged" if result.action == "stage" else "unstaged"
            return " ".join(
                part
                for part in (
                    f"No session notes {past}.",
                    message,
                    counts_text,
                    "Review current eligibility, then Refresh.",
                )
                if part
            )

        if result.state == "blocked" and not affected and clean and not blocked:
            return " ".join(
                part
                for part in (
                    message,
                    counts_text,
                    "No eligible note changes remain; Refresh status.",
                )
                if part
            )

        fallback = {
            "blocked": f"{result.action.title()} was blocked.",
            "stale": f"{result.action.title()} status became stale.",
            "error": f"{result.action.title()} failed.",
            "uncertain": f"{result.action.title()} outcome is uncertain.",
        }[result.state]
        recovery = {
            "blocked": (
                "Resolve the reported Git state outside Chatbook, then Refresh."
            ),
            "stale": (
                "Review the changed repository or session state, then Refresh."
            ),
            "error": "Fix the reported Git error outside Chatbook, then Refresh.",
            "uncertain": (
                "Inspect the repository index outside Chatbook, then Refresh."
            ),
        }[result.state]
        return " ".join(
            part for part in (message or fallback, counts_text, recovery) if part
        )

    @staticmethod
    def _git_action_label(result: GitActionResult) -> str:
        """Return the semantic token for one checked action result."""
        affected = (
            result.staged_group_ids
            if result.action == "stage"
            else result.unstaged_group_ids
        )
        if result.state == "success" and affected:
            return "STAGED" if result.action == "stage" else "UNSTAGED"
        if result.state == "success":
            return "NO CHANGE"
        return {
            "blocked": "BLOCKED",
            "stale": "STALE",
            "error": "FAILED",
            "uncertain": "UNCERTAIN",
        }[result.state]

    @on(Button.Pressed, "#file-notes-new")
    async def _new_file(self, event: Button.Pressed) -> None:
        event.stop()
        if not await self.flush_pending_work():
            return
        service = self._service
        if service is None:
            return
        destination = self._validated_path_input("Create")
        if destination is None:
            return
        await self._complete_path_action(
            "Create",
            destination,
            service.create_file,
            destination,
        )

    @on(Button.Pressed, "#file-notes-move")
    async def _move_file(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        if opened is None or not await self.flush_pending_work():
            return
        service = self._service
        if service is None:
            return
        destination = self._validated_path_input("Move")
        if destination is None:
            return
        await self._complete_path_action(
            "Move",
            destination,
            service.move_file,
            opened.relative_path,
            destination,
        )

    @on(Button.Pressed, "#file-notes-maintenance-toggle")
    def _toggle_maintenance_actions(self) -> None:
        """Reveal or hide secondary file operations without moving focus."""
        self._maintenance_expanded = not self._maintenance_expanded
        toggle = self.query_one("#file-notes-maintenance-toggle", Button)
        self._sync_editor_action_visibility()
        self._schedule_editor_action_layout()
        self.call_after_refresh(toggle.focus)

    def _set_conflict_resolution(
        self,
        active: bool,
        *,
        focus_opener: bool = False,
    ) -> None:
        """Project the bounded conflict choices without resolving any side."""
        self._conflict_resolution_active = bool(
            active
            and self._opened is not None
            and self._save_state == "conflict"
        )
        self._update_controls()
        if not self._active or not self.is_mounted:
            return
        if self._conflict_resolution_active:
            keep = self.query_one("#file-notes-resolution-keep", Button)
            self.screen.set_focus(keep)
            self.call_after_refresh(partial(self.screen.set_focus, keep))
        elif focus_opener:
            opener = self.query_one("#file-notes-resolve-conflict", Button)
            if opener.display and not opener.disabled:
                self.screen.set_focus(opener)
                self.call_after_refresh(partial(self.screen.set_focus, opener))

    @on(Button.Pressed, "#file-notes-resolve-conflict")
    def _open_conflict_resolution(self, event: Button.Pressed) -> None:
        """Disclose the safe conflict choices and focus their safe default."""
        event.stop()
        self._set_conflict_resolution(True)

    @on(Button.Pressed, "#file-notes-resolution-keep")
    def _keep_editing_conflict(self, event: Button.Pressed) -> None:
        """Close the choices while preserving every conflict side unchanged."""
        event.stop()
        self._set_conflict_resolution(False, focus_opener=True)

    @on(Button.Pressed, "#file-notes-delete")
    async def _delete_file(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        if self._service is None or opened is None:
            return
        if self._delete_confirmation_path != opened.relative_path:
            self._set_delete_confirmation(opened.relative_path)
            self._set_action_status("Activate Delete again to confirm.")
            return
        if not await self.flush_pending_work():
            return
        opened = self._opened
        if opened is None:
            return
        with self._hold_path_transition() as transition:
            if transition is None:
                return
            service, generation = transition
            result = await asyncio.to_thread(
                service.delete_file,
                opened.relative_path,
                expected_hash=opened.content_hash,
            )
            if (
                self._path_result_is_stale(service, generation)
                or self._opened is not opened
            ):
                return
            if not result.succeeded:
                self._operation_error("Delete", result)
                return
            deleted_path = opened.relative_path
            self._selected_deleted_path = deleted_path
            self._clear_open_document(keep_restore_path=True)
            self.query_one("#file-notes-path", Input).value = deleted_path
            self.query_one("#file-notes-breadcrumb", Static).update(
                f"Recently deleted: {deleted_path}"
            )
            if not await self._rescan_after_action():
                return
            self._set_action_status("Deleted. Restore remains available.")
            self._update_controls()

    @on(Button.Pressed, "#file-notes-restore")
    async def _restore_file(self, event: Button.Pressed) -> None:
        event.stop()
        service = self._service
        if service is None:
            return
        relative_path = self._selected_deleted_path
        if not relative_path:
            relative_path = self._validated_path_input("Restore")
            if relative_path is None:
                return
        await self._complete_path_action(
            "Restore",
            relative_path,
            service.restore_file,
            relative_path,
        )

    @on(Button.Pressed, "#file-notes-protect")
    async def _toggle_protect(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        service = self._service
        generation = self._root_generation
        if service is None or opened is None:
            return
        target = not opened.protected
        operation = (
            service.protect_path if target else service.unprotect_path
        )
        result = await asyncio.to_thread(operation, opened.relative_path)
        if self._path_result_is_stale(service, generation):
            return
        current = self._opened
        if current is None or current.relative_path != opened.relative_path:
            return
        if not result.succeeded:
            self._operation_error("Protect" if target else "Unprotect", result)
            return
        self._opened = replace(current, protected=target)
        self._set_action_status("Protected." if target else "Unprotected.")
        self._update_controls()

    @on(Button.Pressed, "#file-notes-reload")
    async def _reload_file(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        service = self._service
        if service is None or opened is None:
            return
        if self._save_state in {"conflict", "error"}:
            binding = self._session_binding
            session_key = self._session_key
            state = self._save_state
            if (
                binding is None
                or binding != self._session_owner.current_binding()
                or state not in {"conflict", "error"}
            ):
                self._set_action_status(
                    "Reload stopped: the active File Notes root changed. "
                    "Draft preserved; review the current file before trying again."
                )
                return
            confirmation: _ReloadConfirmation | None = None
            with self._hold_path_transition() as transition:
                if transition is None:
                    return
                service, generation = transition
                try:
                    disk_snapshot = await asyncio.to_thread(
                        service.open_file,
                        opened.relative_path,
                    )
                except FileNotFoundError:
                    self._set_action_status(
                        f"Reload stopped: {opened.relative_path} is no longer "
                        "available on disk. Draft preserved; restore the file or "
                        "save the draft as a copy."
                    )
                    return
                except (OSError, ValueError) as error:
                    self._set_action_status(
                        f"Reload stopped: {opened.relative_path} could not be read "
                        f"from disk ({error}). Draft preserved; check the folder "
                        "and try again."
                    )
                    return
                confirmation = _ReloadConfirmation(
                    service=service,
                    binding=binding,
                    root_generation=generation,
                    session_key=session_key,
                    opened=opened,
                    save_state=state,
                    disk_content_hash=disk_snapshot.content_hash,
                    opener_id=event.button.id or "file-notes-reload",
                )
                if not self._reload_confirmation_is_current(confirmation):
                    self._set_action_status(
                        "Reload stopped: the active root, file, or editing session "
                        "changed. Draft preserved; review the current file before "
                        "trying again."
                    )
                    return
            self._set_reload_confirmation(confirmation)
            return
        if self._save_state == "dirty" and not await self.flush_pending_work():
            return
        opened = self._opened
        if opened is None:
            return
        with self._hold_path_transition() as transition:
            if transition is None:
                return
            service, generation = transition
            try:
                reloaded = await asyncio.to_thread(
                    service.open_file,
                    opened.relative_path,
                )
            except Exception as error:
                self._set_save_state("error", f"reload failed: {error}")
                return
            if (
                self._path_result_is_stale(service, generation)
                or self._opened is not opened
            ):
                return
            self._apply_opened_document(reloaded)

    def _conflict_compare_request_is_current(
        self,
        request: _ConflictCompareRequest,
    ) -> bool:
        """Validate the exact editor identity before publishing comparison."""
        if (
            not self._active
            or not self.is_mounted
            or self._service is not request.service
            or self._root_generation != request.root_generation
            or self._session_binding != request.binding
            or self._session_owner.current_binding() != request.binding
            or self._opened is not request.opened
            or self._current_path != request.opened.relative_path
            or self._session_key != request.session_key
            or self._save_state != "conflict"
        ):
            return False
        return self.query_one("#file-notes-editor", TextArea).text == request.draft

    @staticmethod
    def _build_conflict_comparison(
        opened: OpenedFileNote,
        draft: str,
        disk: OpenedFileNote | None,
        disk_error: str = "",
    ) -> ConflictComparison:
        """Build the immutable comparison away from the UI loop.

        Args:
            opened: Exact editor baseline retained by the current session.
            draft: Exact current editor body.
            disk: Latest readable disk snapshot, when present.
            disk_error: Bounded read failure detail when disk is unreadable.

        Returns:
            A bounded display payload for the comparison modal.
        """
        disk_side = (
            ConflictSide.from_text("Disk", disk.body)
            if disk is not None
            else (
                ConflictSide.unreadable(disk_error)
                if disk_error
                else ConflictSide.absent()
            )
        )
        return build_conflict_comparison(
            ConflictSide.from_text("Base", opened.body),
            ConflictSide.from_text("Draft", draft),
            disk_side,
        )

    @on(Button.Pressed, "#file-notes-compare")
    async def _compare_conflict(self, event: Button.Pressed) -> None:
        """Capture and show Base, Draft, and latest Disk without resolving."""
        event.stop()
        opened = self._opened
        service = self._service
        binding = self._session_binding
        if (
            opened is None
            or service is None
            or binding is None
            or self._save_state != "conflict"
        ):
            return
        request = _ConflictCompareRequest(
            service=service,
            binding=binding,
            root_generation=self._root_generation,
            session_key=self._session_key,
            opened=opened,
            draft=self.query_one("#file-notes-editor", TextArea).text,
        )
        disk: OpenedFileNote | None = None
        disk_error = ""
        try:
            disk = await asyncio.to_thread(
                service.open_file,
                opened.relative_path,
            )
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            disk_error = str(error)
        if not self._conflict_compare_request_is_current(request):
            self._set_action_status(
                "Compare stopped: the active root, file, draft, or editing "
                "session changed. Draft preserved; open Compare again."
            )
            return
        comparison = await asyncio.to_thread(
            self._build_conflict_comparison,
            opened,
            request.draft,
            disk,
            disk_error,
        )
        if not self._conflict_compare_request_is_current(request):
            self._set_action_status(
                "Compare stopped: the active root, file, draft, or editing "
                "session changed. Draft preserved; open Compare again."
            )
            return
        await self.app.push_screen(
            FileNotesConflictCompareDialog(
                opened.relative_path,
                comparison,
            )
        )

    @on(Button.Pressed, "#file-notes-reload-cancel")
    def _cancel_reload(self, event: Button.Pressed) -> None:
        """Preserve the draft and return focus to the destructive opener."""
        event.stop()
        self.cancel_reload_confirmation()

    @on(Button.Pressed, "#file-notes-resolution-discard")
    async def _discard_conflict_draft(self, event: Button.Pressed) -> None:
        """Route the destructive choice through the existing confirmation."""
        await self._reload_file(event)

    @on(Button.Pressed, "#file-notes-reload-confirm")
    async def _confirm_reload(self, event: Button.Pressed) -> None:
        """Revalidate every identity before intentionally loading disk bytes."""
        event.stop()
        confirmation = self._reload_confirmation
        if confirmation is None:
            return
        if not self._reload_confirmation_is_current(confirmation):
            self._dismiss_reload_confirmation(focus_opener=True)
            self._set_action_status(
                "Reload stopped: the active root, file, or editing session changed. "
                "Draft preserved; review the current file before trying again."
            )
            return
        with self._hold_path_transition() as transition:
            if transition is None:
                self._set_action_status(
                    "Reload is temporarily unavailable. Draft preserved; cancel or "
                    "try again after the active file operation finishes."
                )
                return
            service, generation = transition
            if (
                service is not confirmation.service
                or generation != confirmation.root_generation
            ):
                self._dismiss_reload_confirmation(focus_opener=True)
                self._set_action_status(
                    "Reload stopped: the active File Notes root changed. Draft "
                    "preserved; review the current file before trying again."
                )
                return
            try:
                reloaded = await asyncio.to_thread(
                    service.open_file,
                    confirmation.opened.relative_path,
                )
            except FileNotFoundError:
                if self._reload_confirmation is confirmation:
                    self._dismiss_reload_confirmation(focus_opener=True)
                    self._set_action_status(
                        f"Reload stopped: {confirmation.opened.relative_path} is no "
                        "longer available on disk. Draft preserved; restore the file "
                        "or save the draft as a copy."
                    )
                return
            except (OSError, ValueError) as error:
                if self._reload_confirmation is confirmation:
                    self._dismiss_reload_confirmation(focus_opener=True)
                    self._set_action_status(
                        f"Reload stopped: {confirmation.opened.relative_path} could "
                        f"not be read from disk ({error}). Draft preserved; check "
                        "the folder and try again."
                    )
                return
            if self._reload_confirmation is not confirmation:
                return
            if (
                self._path_result_is_stale(service, generation)
                or not self._reload_confirmation_is_current(confirmation)
            ):
                self._dismiss_reload_confirmation(focus_opener=True)
                self._set_action_status(
                    "Reload stopped: the active root, file, or editing session "
                    "changed. Draft preserved; review the current file before "
                    "trying again."
                )
                return
            if reloaded.content_hash != confirmation.disk_content_hash:
                self._dismiss_reload_confirmation(focus_opener=True)
                self._set_action_status(
                    f"Reload stopped: {confirmation.opened.relative_path} changed "
                    "again on disk. Draft preserved; activate Discard draft and "
                    "reload again to review the latest disk version."
                )
                return
            self._dismiss_reload_confirmation(focus_opener=False)
            self._apply_opened_document(reloaded)

    async def _save_editor_copy(self, action: str) -> bool:
        """Run the shared exact, validated, no-clobber editor-copy path."""
        opened = self._opened
        service = self._service
        if service is None or opened is None:
            return False
        destination = self._validated_path_input(action)
        if destination is None:
            return False
        if opened.is_excerpt:
            await self._complete_path_action(
                action,
                destination,
                service.export_exact_file,
                opened,
                destination,
            )
        else:
            body = self.query_one("#file-notes-editor", TextArea).text
            await self._complete_path_action(
                action,
                destination,
                service.save_copy,
                opened,
                body,
                destination,
            )
        return (
            self._opened is not None
            and self._current_path == destination
            and self._save_state == "saved"
        )

    @on(Button.Pressed, "#file-notes-save-copy")
    async def _save_copy(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        if opened is None:
            return
        action = "Export exact copy" if opened.is_excerpt else "Save draft as copy"
        await self._save_editor_copy(action)

    @on(Button.Pressed, "#file-notes-resolution-save-new")
    async def _save_conflict_draft_as_new_note(
        self,
        event: Button.Pressed,
    ) -> None:
        """Save the retained draft through the existing no-clobber copy path."""
        event.stop()
        if await self._save_editor_copy("Save draft as new note"):
            editor = self.query_one("#file-notes-editor", TextArea)
            self.call_after_refresh(editor.focus)

    @on(Button.Pressed, "#file-notes-refresh")
    async def _refresh_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.refresh_files()
