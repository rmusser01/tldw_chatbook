"""Retained disk-backed File Notes workspace for Library."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.timer import Timer
from textual.worker import Worker
from textual.widgets import Button, Input, ListView, Static, TextArea, Tree

from tldw_chatbook.config import (
    apply_settings_mutation_to_cli_config,
    get_cli_setting,
    get_user_data_dir,
)
from tldw_chatbook.Notes.file_notes_git_service import (
    DiscoveryResult,
    GitActionResult,
    GitMutationAdmissionError,
    GitStatusAdmissionError,
    coalesce_session_changes,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionGitRow,
    SessionGitStatus,
    SessionTransitionKind,
)
from tldw_chatbook.Notes.file_notes_service import (
    FileNoteEntry,
    FileNotesService,
    OpenedFileNote,
    OperationResult,
    ReconcileResult,
    ScanResult,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    LibraryFileNotesGitPanel,
    SessionGitTrustDialog,
)

SaveState = Literal["idle", "dirty", "saving", "saved", "conflict", "error"]
_UNSET = object()
_SESSION_GIT_MUTATION_BUSY = (
    "Session Git mutation in progress; structural actions are busy."
)
_TreeData = tuple[Literal["file", "folder", "deleted"], str]


@dataclass(frozen=True, slots=True)
class _GitActionSummaryContext:
    """Immutable presentation counts captured before one bulk action."""

    bulk: bool = False
    skipped: int = 0
    already_staged: int = 0
    clean: int = 0
    blocked: int = 0


class _SessionGitService(Protocol):
    """UI-facing service contract already implemented by FileNotesGitService."""

    async def discover(self, binding: SessionBinding) -> DiscoveryResult: ...

    async def revalidate_repository(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool: ...

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


class LibraryFileNotesWorkspace(Vertical):
    """Browse and edit one disk-authoritative Markdown/text root."""

    DEFAULT_CSS = """
    LibraryFileNotesWorkspace {
        height: 1fr;
        min-height: 12;
        min-width: 0;
    }

    #file-notes-root-row {
        height: auto;
        min-height: 1;
    }

    #file-notes-root-status {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }

    #file-notes-choose-root {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
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

    #file-notes-search,
    #file-notes-path {
        height: 3;
        min-height: 3;
    }

    #file-notes-tree,
    #file-notes-search-results {
        height: 1fr;
        min-height: 4;
    }

    #file-notes-breadcrumb,
    #file-notes-save-status,
    #file-notes-action-status {
        height: auto;
        min-height: 1;
    }

    #file-notes-breadcrumb {
        text-style: bold;
    }

    #file-notes-save-status,
    #file-notes-action-status {
        color: $text-muted;
    }

    #file-notes-editor {
        height: 1fr;
        min-height: 5;
    }

    .file-notes-toolbar {
        height: auto;
        min-height: 1;
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
        self._opened: OpenedFileNote | None = None
        self._current_path = ""
        self._selected_deleted_path = ""
        self._session_key = ""
        self._save_state: SaveState = "idle"
        self._save_detail = ""
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
        self._git_observed_changes: tuple[SequencedSessionChange, ...] | None = None
        self._git_refresh_timer: Timer | None = None
        self._git_refresh_after_mutation = False
        self._git_action_detail = ""
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

    def compose(self) -> ComposeResult:
        with Horizontal(id="file-notes-root-row"):
            yield Static(
                "Choose a notes folder.",
                id="file-notes-root-status",
                markup=False,
            )
            yield Button(
                "Choose folder…",
                id="file-notes-choose-root",
                compact=True,
            )
        with Horizontal(id="file-notes-body"):
            with Vertical(id="file-notes-navigator"):
                yield Input(
                    placeholder="Search file contents…",
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
                    "Session Git (0)",
                    id="file-notes-session-changes",
                    compact=True,
                )
                yield self._git_panel_widget
            with Vertical(id="file-notes-editor-pane"):
                back = Button(
                    "‹ Navigator",
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
                yield Static("Idle", id="file-notes-save-status", markup=False)
                yield Input(
                    placeholder="relative/path.md",
                    id="file-notes-path",
                    value=self._selected_deleted_path or self._current_path,
                )
                yield self._editor_widget
                with Horizontal(classes="file-notes-toolbar"):
                    yield Button("New", id="file-notes-new", compact=True)
                    yield Button("Move", id="file-notes-move", compact=True)
                    yield Button("Delete", id="file-notes-delete", compact=True)
                    yield Button("Restore", id="file-notes-restore", compact=True)
                    yield Button("Protect", id="file-notes-protect", compact=True)
                with Horizontal(classes="file-notes-toolbar"):
                    yield Button("Reload", id="file-notes-reload", compact=True)
                    yield Button(
                        "Save Copy",
                        id="file-notes-save-copy",
                        compact=True,
                    )
                    yield Button("Refresh", id="file-notes-refresh", compact=True)
                yield Static("", id="file-notes-action-status", markup=False)

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
            self._apply_reconcile(result, deleted)
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
                service._bind_session_owner(self._session_owner, binding)
                self._root = root
                self._session_binding = binding
                self._service = service
                self._clear_open_document()
                self._initialized = True
                if persistence_warning:
                    self._runtime_warning = persistence_warning
                self._apply_scan(result, deleted)

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
    ) -> None:
        self._adopt_scan_state(result, deleted)
        self._render_scan_state()

    def _adopt_scan_state(
        self,
        result: ScanResult,
        deleted: tuple[str, ...],
    ) -> None:
        """Install a scan projection without requiring mounted widgets."""
        self._entries = {entry.relative_path: entry for entry in result.entries}
        self._deleted_paths = deleted
        self._root_offline = result.offline
        if result.replica_warning:
            self._runtime_warning = result.replica_warning

    def _render_scan_state(self) -> None:
        """Render the installed scan projection when the widget tree is live."""
        if not self._active or not self.is_mounted or not self.children:
            return
        self._update_root_surface(offline=self._root_offline)
        self._rebuild_tree()
        self._refresh_active_search()
        self._refresh_session_changes()
        self._update_controls()

    def _apply_reconcile(
        self,
        result: ReconcileResult,
        deleted: tuple[str, ...],
    ) -> None:
        new_entries = {entry.relative_path: entry for entry in result.entries}
        navigator_changed = (
            new_entries != self._entries or deleted != self._deleted_paths
        )
        self._entries = new_entries
        self._deleted_paths = deleted
        self._root_offline = result.offline
        if result.replica_warning:
            self._runtime_warning = result.replica_warning
        self._update_root_surface(offline=result.offline)
        if navigator_changed:
            self._rebuild_tree()
        self._refresh_active_search()
        self._refresh_session_changes()
        self._update_controls()

    def _update_root_surface(self, *, offline: bool | None = None) -> None:
        if not self._active or not self.is_mounted or not self.children:
            return
        status = self.query_one("#file-notes-root-status", Static)
        body = self.query_one("#file-notes-body")
        choose = self.query_one("#file-notes-choose-root", Button)
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
            status.update("Choose a notes folder.")
            body.display = False
            choose.display = True
            return
        is_offline = self._root_offline if offline is None else offline
        state = (
            "Checking"
            if is_offline is None
            else ("Offline" if is_offline else "Linked")
        )
        detail = f"{state} — {self._root}"
        if self._runtime_warning:
            detail = f"{detail} · {self._runtime_warning}"
        status.update(detail)
        body.display = True
        choose.display = True
        self._apply_responsive_layout(self.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        if not self._active or not self.is_mounted:
            return
        self._narrow = width < 80
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

    def _sync_navigator_mode(self) -> None:
        """Show one retained navigator surface without remounting its peers."""
        if not self._active or not self.is_mounted:
            return
        search = self.query_one("#file-notes-search", Input)
        tree = self.query_one("#file-notes-tree", Tree)
        results = self.query_one("#file-notes-search-results", Tree)
        entry = self.query_one("#file-notes-session-changes", Button)
        panel = self.query_one(
            "#file-notes-git-panel",
            LibraryFileNotesGitPanel,
        )
        git_visible = self._navigator_mode == "git"
        panel.display = git_visible
        search.display = not git_visible
        entry.display = not git_visible
        tree.display = not git_visible and self._navigator_mode == "files"
        results.display = not git_visible and self._navigator_mode == "search"

    def _rebuild_tree(self) -> None:
        if not self._active or not self.is_mounted:
            return
        tree = self.query_one("#file-notes-tree", Tree)
        expanded_folders: set[str] = set()

        def remember_expanded(node: Any) -> None:
            data = node.data
            if (
                node.is_expanded
                and isinstance(data, tuple)
                and len(data) == 2
                and data[0] == "folder"
            ):
                expanded_folders.add(data[1])
            for child in node.children:
                remember_expanded(child)

        for child in tree.root.children:
            remember_expanded(child)
        root_label = self._root.name if self._root is not None else "Files"
        tree.reset(Text(root_label or "Files"))
        folder_nodes: dict[str, Any] = {"": tree.root}
        for relative_path in sorted(self._entries):
            parts = PurePosixPath(relative_path).parts
            parent_key = ""
            parent = tree.root
            for part in parts[:-1]:
                key = f"{parent_key}/{part}".lstrip("/")
                node = folder_nodes.get(key)
                if node is None:
                    node = parent.add(
                        Text(part),
                        data=("folder", key),
                        expand=key in expanded_folders,
                    )
                    folder_nodes[key] = node
                parent = node
                parent_key = key
            parent.add_leaf(
                Text(parts[-1]),
                data=("file", relative_path),
            )
        if self._deleted_paths:
            deleted = tree.root.add(
                Text("Recently deleted"),
                data=("folder", "recently-deleted"),
                expand=True,
            )
            for relative_path in self._deleted_paths:
                deleted.add_leaf(
                    Text(relative_path),
                    data=("deleted", relative_path),
                )
        tree.root.expand()

    def _rebuild_search_results(self, paths: tuple[str, ...]) -> None:
        if not self._active or not self.is_mounted:
            return
        results = self.query_one("#file-notes-search-results", Tree)
        results.reset(Text("Search results"))
        for path in paths:
            results.root.add_leaf(Text(path), data=("file", path))
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
            changes = self._session_owner.snapshot(binding).changes
        count = len(coalesce_session_changes(changes))
        self.query_one("#file-notes-session-changes", Button).label = (
            f"Session Git ({count})"
        )
        prior = self._git_observed_changes
        self._git_observed_changes = changes
        if prior is None or prior == changes or binding is None:
            return
        self._session_owner.clear_status(binding)
        self._git_panel_widget.mark_stale()
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

    def _rehydrate_git_presentation(self) -> bool:
        """Render retained owner/task state without starting hidden Git work."""
        if (
            not self._active
            or not self.is_mounted
            or self._navigator_mode != "git"
        ):
            return False
        binding = self._session_binding
        if binding is None or not self._git_binding_matches_session(binding):
            return False
        snapshot = self._session_owner.snapshot(binding)
        if self._session_owner.mutation_active(binding):
            if snapshot.git_status is not None:
                self._git_panel_widget.render_status(snapshot.git_status)
            self._git_panel_widget.set_mutating(
                True,
                self._git_action_detail or "Git mutation in progress…",
            )
            self._git_refresh_after_mutation = True
            return True
        task = self._git_status_task
        if (
            task is not None
            and self._git_status_task_binding == binding
            and not task.done()
        ):
            repository = snapshot.trusted_repository
            if repository is not None:
                self._git_panel_widget.render_checking(
                    repository.worktree_root
                )
            self._ensure_git_status_waiter(task, binding)
            return True
        if snapshot.git_status is not None:
            self._git_panel_widget.render_status(snapshot.git_status)
            if self._git_action_detail:
                self._git_panel_widget.set_action_status(
                    self._git_action_detail
                )
            return True
        if (
            task is not None
            and self._git_status_task_binding == binding
            and task.done()
        ):
            self._ensure_git_status_waiter(task, binding)
            return True
        return False

    async def _open_session_git(self, *, force_prompt: bool = False) -> None:
        binding = self._session_binding
        service = self._session_git_service()
        if binding is None or service is None:
            self._git_panel_widget.set_action_status(
                "Git is unavailable for the selected File Notes root."
            )
            return
        discovery = await service.discover(binding)
        if not self._git_binding_is_current(binding):
            return
        repository = discovery.repository
        if discovery.state != "ready" or repository is None:
            self._git_panel_widget.render_unavailable(
                discovery.message or "Git repository status is unavailable."
            )
            return
        snapshot = self._session_owner.snapshot(binding)
        needs_trust = (
            force_prompt or snapshot.trusted_repository != repository
        )
        if needs_trust:
            self._git_panel_widget.render_untrusted(repository.worktree_root)
            accepted = await self.app.push_screen_wait(
                SessionGitTrustDialog(repository.worktree_root)
            )
            if not accepted or not self._git_binding_is_current(binding):
                return
            if not await service.revalidate_repository(binding, repository):
                if self._git_binding_is_current(binding):
                    self._git_panel_widget.render_untrusted(
                        repository.worktree_root
                    )
                    self._git_panel_widget.set_action_status(
                        "Repository identity changed; trust was not granted."
                    )
                return
            if not self._session_owner.publish_trust(binding, repository):
                return
            snapshot = self._session_owner.snapshot(binding)
        if self._rehydrate_git_presentation():
            return
        self._start_git_refresh()

    def _start_git_refresh(self) -> None:
        """Synchronously admit visible status and delegate retained awaiting."""
        if self._navigator_mode != "git":
            return
        binding = self._session_binding
        service = self._session_git_service()
        if binding is None or service is None:
            return
        if self._session_owner.mutation_active(binding):
            self._git_refresh_after_mutation = True
            self._git_panel_widget.mark_stale(
                "Git mutation in progress; refresh will follow."
            )
            return
        snapshot = self._session_owner.snapshot(binding)
        repository = snapshot.trusted_repository
        if repository is None:
            self._git_panel_widget.set_action_status(
                "Trust is required before checking Session Git status."
            )
            return
        self._git_panel_widget.render_checking(repository.worktree_root)
        try:
            task = service.start_status(binding, snapshot.changes)
        except GitStatusAdmissionError as error:
            if error.reason == "mutation_active":
                self._git_refresh_after_mutation = True
            self._git_panel_widget.mark_stale(str(error))
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
                self._git_panel_widget.mark_stale(f"Git status failed: {error}")
                if self._git_status_task is task:
                    self._git_status_task = None
                    self._git_status_task_binding = None
            return
        if (
            self._git_binding_is_current(binding)
            and status == self._session_owner.snapshot(binding).git_status
        ):
            self._git_panel_widget.render_status(status)
            if self._git_action_detail:
                self._git_panel_widget.set_action_status(
                    self._git_action_detail
                )
            if self._git_status_task is task:
                self._git_status_task = None
                self._git_status_task_binding = None

    def _set_save_state(self, state: SaveState, detail: str = "") -> None:
        self._save_state = state
        self._save_detail = detail
        if self._active and self.is_mounted:
            label = state.capitalize()
            if detail:
                label = f"{label} — {detail}"
            self.query_one("#file-notes-save-status", Static).update(label)
            self._update_controls()

    def _set_action_status(self, text: str) -> None:
        self._action_detail = text
        if self._active and self.is_mounted:
            self.query_one("#file-notes-action-status", Static).update(text)

    def _update_controls(self) -> None:
        if not self._active or not self.is_mounted:
            return
        transitioning = self._root_transitioning or self._path_transitioning
        binding = self._session_binding
        mutation_active = (
            binding is not None
            and self._session_owner.mutation_active(binding)
        )
        structurally_available = not transitioning and not mutation_active
        has_service = self._service is not None and structurally_available
        has_document = self._opened is not None and not transitioning
        has_deleted = bool(self._selected_deleted_path) and not transitioning
        self.query_one("#file-notes-new", Button).disabled = not has_service
        for selector in ("move", "delete", "reload"):
            self.query_one(
                f"#file-notes-{selector}", Button
            ).disabled = not (
                has_document and structurally_available
            )
        self.query_one("#file-notes-protect", Button).disabled = not has_document
        self.query_one("#file-notes-save-copy", Button).disabled = (
            not has_document
            or not structurally_available
            or self._save_state not in {"dirty", "conflict", "error"}
        )
        self.query_one("#file-notes-restore", Button).disabled = (
            not has_service or not has_deleted or not structurally_available
        )
        self.query_one("#file-notes-refresh", Button).disabled = (
            self._service is None or transitioning
        )
        protect = self.query_one("#file-notes-protect", Button)
        protect.label = (
            "Unprotect"
            if self._opened is not None and self._opened.protected
            else "Protect"
        )
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
        editor = self.query_one("#file-notes-editor", TextArea)
        editor.read_only = transitioning or not (
            self._opened is not None and self._opened.editable
        )
        self._git_panel_widget.set_mutating(mutation_active)

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
                    "Session Git mutation in progress; root change is busy."
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
        self._opened = opened
        self._current_path = opened.relative_path
        self._selected_deleted_path = ""
        self._session_key = uuid4().hex
        self._delete_confirmation_path = ""
        editor = self.query_one("#file-notes-editor", TextArea)
        with editor.prevent(TextArea.Changed):
            editor.load_text(opened.body)
        editor.read_only = not opened.editable
        self.query_one("#file-notes-path", Input).value = opened.relative_path
        self.query_one("#file-notes-breadcrumb", Static).update(opened.relative_path)
        self.query_one("#file-notes-delete", Button).label = "Delete"
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
        editor.read_only = True
        if not keep_restore_path:
            self.query_one("#file-notes-path", Input).value = ""
            self.query_one("#file-notes-breadcrumb", Static).update("No file selected")
        self.query_one("#file-notes-delete", Button).label = "Delete"
        self._set_save_state("idle")
        self._update_controls()

    def select_deleted(self, relative_path: str) -> bool:
        """Select one persistent tombstone for the Restore action."""
        if not self._active or relative_path not in self._deleted_paths:
            return False
        self._clear_open_document()
        self._selected_deleted_path = relative_path
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
            if (
                not self._active
                or not self.is_mounted
                or not self.children
                or self._path_transitioning
                or generation != self._root_generation
                or service is not self._service
            ):
                return False
            self._apply_reconcile(result, deleted)
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
        self._apply_scan(result, deleted)
        return True

    def _operation_error(self, action: str, result: OperationResult) -> None:
        detail = result.message or result.status
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
        self._delete_confirmation_path = ""
        self.query_one("#file-notes-delete", Button).label = "Delete"
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

    @on(Tree.NodeSelected)
    async def _tree_node_selected(self, event: Tree.NodeSelected[object]) -> None:
        data = event.node.data
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
        if self._navigator_mode != "git":
            self._navigator_mode_before_git = (
                "search"
                if self.query_one("#file-notes-search", Input).value.strip()
                else "files"
            )
        self._navigator_mode = "git"
        self._sync_navigator_mode()
        self.call_after_refresh(self._focus_session_git_panel)
        self.run_worker(
            self._open_session_git(),
            name="file-notes-git-open",
            group="file-notes-git-open",
            exclusive=True,
        )

    def _focus_session_git_panel(self) -> None:
        """Move focus off the hidden entry into the visible Git surface."""
        if (
            not self._active
            or not self.is_mounted
            or self._navigator_mode != "git"
        ):
            return
        rows = self.query_one("#file-notes-git-rows", ListView)
        if self._git_panel_widget.rows and rows.display:
            rows.focus()
            return
        self.query_one("#file-notes-git-back", Button).focus()

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
            self._git_panel_widget.set_action_status(
                "Stage blocked: settle the File Notes draft first."
                if action == "stage"
                else "Unstage blocked: settle the File Notes draft first."
            )
            return
        if (
            not self._git_binding_is_current(binding)
            or self._root_transitioning
            or self._path_transitioning
            or self._save_state in {"dirty", "saving", "conflict", "error"}
        ):
            self._git_panel_widget.set_action_status(
                f"{action.title()} blocked: File Notes state changed."
            )
            return
        try:
            task = (
                service.start_stage(binding, group_ids)
                if action == "stage"
                else service.start_unstage(binding, group_ids)
            )
        except GitMutationAdmissionError as error:
            self._git_panel_widget.set_action_status(
                f"{action.title()} blocked: {error}"
            )
            return
        self._git_status_task = None
        self._git_status_task_binding = None
        self._git_panel_widget.set_mutating(
            True,
            f"{action.title()} in progress…",
        )
        self._update_root_surface()
        self._update_controls()
        self._git_action_worker = self.run_worker(
            self._render_git_action(task, binding, summary_context),
            name=f"file-notes-git-{action}",
            group="file-notes-git-action",
            exclusive=True,
        )

    async def _render_git_action(
        self,
        task: asyncio.Task[GitActionResult],
        binding: SessionBinding,
        summary_context: _GitActionSummaryContext,
    ) -> None:
        result: GitActionResult | None = None
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._git_binding_matches_session(binding):
                self._git_action_detail = f"Git action failed: {error}"
                if self._git_binding_is_current(binding):
                    self._git_panel_widget.set_action_status(
                        self._git_action_detail
                    )
        else:
            if self._git_binding_matches_session(binding):
                self._git_action_detail = self._git_action_summary(
                    result,
                    summary_context,
                )
                if self._git_binding_is_current(binding):
                    self._git_panel_widget.set_action_status(
                        self._git_action_detail
                    )
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
                                "Git action finished while Session Git was hidden."
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

    @staticmethod
    def _git_action_summary(
        result: GitActionResult,
        context: _GitActionSummaryContext,
    ) -> str:
        """Render selected counts or complete pre-action bulk counts."""
        if result.message:
            return result.message
        verb = (
            ("Staged" if result.action == "stage" else "Unstaged")
            if result.state == "success"
            else f"{result.action.title()} {result.state}"
        )
        changed = (
            len(result.staged_group_ids)
            if result.action == "stage"
            else len(result.unstaged_group_ids)
        )
        parts = [f"{verb} {changed}" if result.state == "success" else verb]
        if context.bulk and result.action == "stage":
            parts.append(f"already staged {context.already_staged}")
        if context.bulk and result.action == "unstage":
            parts.append(f"skipped {context.skipped}")
        parts.extend(
            (
                f"clean {len(result.clean_group_ids) + context.clean}",
                f"blocked {len(result.blocked_group_ids) + context.blocked}",
            )
        )
        return " · ".join(parts)

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

    @on(Button.Pressed, "#file-notes-delete")
    async def _delete_file(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        if self._service is None or opened is None:
            return
        if self._delete_confirmation_path != opened.relative_path:
            self._delete_confirmation_path = opened.relative_path
            event.button.label = "Confirm delete"
            self._set_action_status("Click Delete again to confirm.")
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
            self._clear_open_document()
            self._selected_deleted_path = deleted_path
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
        generation = self._root_generation
        if service is None or opened is None:
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

    @on(Button.Pressed, "#file-notes-save-copy")
    async def _save_copy(self, event: Button.Pressed) -> None:
        event.stop()
        opened = self._opened
        service = self._service
        if service is None or opened is None:
            return
        destination = self._validated_path_input("Save Copy")
        if destination is None:
            return
        body = self.query_one("#file-notes-editor", TextArea).text
        await self._complete_path_action(
            "Save Copy",
            destination,
            service.save_copy,
            opened,
            body,
            destination,
        )

    @on(Button.Pressed, "#file-notes-refresh")
    async def _refresh_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.refresh_files()
