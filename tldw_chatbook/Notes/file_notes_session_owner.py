"""Process-memory ownership for one File Notes root session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock, RLock
from types import MappingProxyType
from typing import Literal, Protocol

SessionChangeAction = Literal[
    "created",
    "modified",
    "moved",
    "deleted",
    "restored",
]
SessionTransitionKind = Literal["root", "path", "source", "screen"]
HeadKind = Literal["attached", "detached", "unborn"]
SessionGitRowState = Literal[
    "unstaged",
    "owned",
    "owned_newer_edits",
    "owned_topology_changed",
    "external_staged",
    "external_partial",
    "clean",
    "ignored",
    "conflict",
    "unsupported",
    "nested_repository",
    "unsafe_closure",
    "ambiguous_lineage",
    "unavailable",
    "error",
]
SessionGitStageAction = Literal["stage", "stage_update"]
SessionGitStatusState = Literal["ready", "stale", "unavailable", "error"]
SessionChangeTopology = tuple[
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    str,
]
_TRANSITION_KINDS = frozenset({"root", "path", "source", "screen"})


@dataclass(frozen=True, slots=True)
class SessionChange:
    """One successful filesystem mutation initiated by Chatbook."""

    action: SessionChangeAction
    relative_path: str
    destination_path: str | None = None


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """Canonical selected root paired with its owner generation."""

    root_key: str
    generation: int


@dataclass(frozen=True, slots=True)
class SequencedSessionChange:
    """One session change in process-wide publication order."""

    sequence: int
    change: SessionChange


def _sanitize_display_path(path: str) -> str:
    replacements = {
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
    }
    display: list[str] = []
    for character in path:
        codepoint = ord(character)
        if character in replacements:
            display.append(replacements[character])
        elif codepoint < 32 or 127 <= codepoint <= 159:
            display.append(f"\\x{codepoint:02x}")
        elif 0xDC80 <= codepoint <= 0xDCFF:
            display.append(f"\\x{codepoint - 0xDC00:02x}")
        else:
            display.append(character)
    return "".join(display)


@dataclass(frozen=True, slots=True)
class FileSystemIdentity:
    """Platform-stable identity for one repository filesystem location."""

    device: int | None
    inode: int | None


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    """Canonical worktree and Git-directory identity trusted for one root."""

    worktree_root: str
    git_dir: str
    git_common_dir: str
    worktree_identity: FileSystemIdentity
    git_dir_identity: FileSystemIdentity
    git_common_dir_identity: FileSystemIdentity


@dataclass(frozen=True, slots=True)
class HeadIdentity:
    """Exact attached, detached, or explicit unborn HEAD identity."""

    kind: HeadKind
    object_id: str | None
    branch: str | None

    @classmethod
    def attached(cls, branch: str, object_id: str) -> HeadIdentity:
        """Build an attached branch identity."""
        return cls("attached", object_id, branch)

    @classmethod
    def detached(cls, object_id: str) -> HeadIdentity:
        """Build a detached HEAD identity."""
        return cls("detached", object_id, None)

    @classmethod
    def unborn(cls, branch: str) -> HeadIdentity:
        """Build an explicit unborn branch identity."""
        return cls("unborn", None, branch)


@dataclass(frozen=True, slots=True)
class SessionChangeGroup:
    """One stable, inseparable lineage of session-authored paths."""

    group_id: int
    endpoints: tuple[str, ...]
    source_path: str
    destination_path: str | None
    current_path: str
    latest_action: SessionChangeAction
    latest_sequence: int
    move_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoints", tuple(self.endpoints))
        object.__setattr__(
            self,
            "move_edges",
            tuple(tuple(edge) for edge in self.move_edges),
        )

    @property
    def topology_signature(self) -> SessionChangeTopology:
        """Return exact ordered move topology and active endpoint identity."""
        return self.endpoints, self.move_edges, self.current_path

    @property
    def display_text(self) -> str:
        """Return a control-character-safe label without changing raw paths."""
        source = _sanitize_display_path(self.source_path)
        if self.destination_path is None:
            return source
        return f"{source} → {_sanitize_display_path(self.destination_path)}"


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One exact Git index entry and its nondefault semantic flags."""

    path: str
    mode: str
    object_id: str
    stage: int = 0
    semantic_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_flags",
            tuple(sorted(set(self.semantic_flags))),
        )


@dataclass(frozen=True, slots=True)
class IndexBaseline:
    """Exact pre-Stage stage-0 entry, or an explicit absence."""

    entry: IndexEntry | None


@dataclass(frozen=True, slots=True)
class StagingOwnership:
    """Evidence needed to reverse exactly one Chatbook Stage result."""

    repository: RepositoryIdentity
    head: HeadIdentity
    approved_endpoint_topology: tuple[str, ...]
    approved_move_edges: tuple[tuple[str, str], ...]
    approved_current_path: str
    original_baselines: Mapping[str, IndexBaseline]
    post_stage_entries: Mapping[str, IndexEntry | None]

    def __post_init__(self) -> None:
        baselines = dict(self.original_baselines)
        post_stage_entries = dict(self.post_stage_entries)
        for path, baseline in baselines.items():
            if baseline.entry is not None and baseline.entry.path != path:
                raise ValueError("Index baseline path does not match its key")
        for path, entry in post_stage_entries.items():
            if entry is not None and entry.path != path:
                raise ValueError("Post-Stage index entry path does not match its key")
        object.__setattr__(
            self,
            "approved_endpoint_topology",
            tuple(self.approved_endpoint_topology),
        )
        object.__setattr__(
            self,
            "approved_move_edges",
            tuple(tuple(edge) for edge in self.approved_move_edges),
        )
        object.__setattr__(
            self,
            "original_baselines",
            MappingProxyType(baselines),
        )
        object.__setattr__(
            self,
            "post_stage_entries",
            MappingProxyType(post_stage_entries),
        )

    @property
    def topology_signature(self) -> SessionChangeTopology:
        """Return the exact topology approved by the last Stage."""
        return (
            self.approved_endpoint_topology,
            self.approved_move_edges,
            self.approved_current_path,
        )


@dataclass(frozen=True, slots=True)
class SessionGitRow:
    """Frozen presentation and action policy for one session group."""

    group: SessionChangeGroup
    state: SessionGitRowState
    stage_action: SessionGitStageAction | None = None
    unstage_eligible: bool = False
    disabled_reason: str | None = None

    @property
    def group_id(self) -> int:
        """Return the stable earliest-sequence group identity."""
        return self.group.group_id

    @property
    def stage_eligible(self) -> bool:
        """Return whether this row participates in Stage actions."""
        return self.stage_action is not None


@dataclass(frozen=True, slots=True)
class SessionGitStatus:
    """One generation-checked immutable session Git status result."""

    binding_generation: int
    status_generation: int
    state: SessionGitStatusState
    rows: tuple[SessionGitRow, ...] = ()
    repository: RepositoryIdentity | None = None
    head: HeadIdentity | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True, slots=True)
class FileNotesSessionSnapshot:
    """Immutable state for one requested root generation."""

    binding: SessionBinding
    changes: tuple[SequencedSessionChange, ...] = ()
    trusted_repository: RepositoryIdentity | None = None
    git_status: SessionGitStatus | None = None
    staging_ownership: Mapping[int, StagingOwnership] = field(
        default_factory=lambda: MappingProxyType({})
    )


class FileNotesGitServiceLifecycle(Protocol):
    """Narrow lifecycle boundary for an owner-attached Git service."""

    def shutdown(self) -> None:
        """Stop accepting work and settle retained service work."""


@dataclass(frozen=True, slots=True)
class SessionTransitionLease:
    """Idempotently releasable root/path/source/screen transition admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    kind: SessionTransitionKind

    def release(self) -> None:
        """Release this transition admission once."""
        self._owner._release_transition(self._token)


@dataclass(frozen=True, slots=True)
class GitMutationLease:
    """Idempotently releasable Git mutation admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def release(self) -> None:
        """Release this mutation admission once."""
        self._owner._release_mutation(self._token)


@dataclass(frozen=True, slots=True)
class GitStatusLease:
    """Idempotently releasable Git status admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def release(self) -> None:
        """Release this status admission once."""
        self._owner._release_status(self._token)


@dataclass(frozen=True, slots=True)
class RootCommitReservation:
    """Fail-fast ownership of one validated root commit."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    _root_key: str = field(repr=False, compare=False)

    def commit(
        self,
        publish: Callable[[SessionBinding], None],
    ) -> SessionBinding:
        """Select the reserved root and synchronously publish its binding."""
        return self._owner._commit_root_reservation(
            self._token,
            self._root_key,
            publish,
        )

    def release(self) -> None:
        """Release this reservation once."""
        self._owner._release_root_reservation(self._token)


@dataclass(slots=True)
class StableRootAccess:
    """Worker-thread access to one root binding while root changes are paused."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    binding: SessionBinding | None
    _released: bool = field(default=False, repr=False, compare=False)

    def release(self) -> None:
        """Release stable-root access once."""
        if self._released:
            return
        self._released = True
        self._owner._root_commit_lock.release()


class FileNotesSessionOwner:
    """Own root-scoped File Notes session state for one application process."""

    __slots__ = (
        "_binding",
        "_changes",
        "_generation",
        "_git_service",
        "_git_status",
        "_lock",
        "_mutation_token",
        "_next_sequence",
        "_root_commit_lock",
        "_root_commit_state",
        "_root_commit_token",
        "_shutdown",
        "_shutdown_condition",
        "_shutdown_error",
        "_shutdown_state",
        "_staging_ownership",
        "_status_generation",
        "_status_token",
        "_trusted_repository",
        "_transition_tokens",
    )

    def __init__(self) -> None:
        self._lock = RLock()
        self._root_commit_lock = Lock()
        self._root_commit_token: object | None = None
        self._root_commit_state: Literal["idle", "reserved", "committed"] = "idle"
        self._binding: SessionBinding | None = None
        self._generation = 0
        self._changes: list[SequencedSessionChange] = []
        self._next_sequence = 1
        self._trusted_repository: RepositoryIdentity | None = None
        self._git_status: SessionGitStatus | None = None
        self._staging_ownership: dict[int, StagingOwnership] = {}
        self._status_generation = 0
        self._transition_tokens: set[object] = set()
        self._mutation_token: object | None = None
        self._status_token: object | None = None
        self._shutdown = False
        self._shutdown_condition = Condition(self._lock)
        self._shutdown_error: BaseException | None = None
        self._shutdown_state: Literal["open", "closing", "closed", "failed"] = "open"
        self._git_service: FileNotesGitServiceLifecycle | None = None

    def select_root(self, root: str | Path) -> SessionBinding:
        """Select one canonical root, resetting state only when it changes.

        Args:
            root: Filesystem root to bind for this process session.

        Returns:
            The current immutable root-generation binding.

        Raises:
            RuntimeError: If the owner has shut down or another root commit is
                in progress.
        """
        root_key = str(Path(root).expanduser().resolve(strict=False))
        if not self._root_commit_lock.acquire(blocking=False):
            raise RuntimeError("File Notes root commit is in progress")
        try:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("File Notes session owner is shut down")
                return self._select_root_locked(root_key)
        finally:
            self._root_commit_lock.release()

    def current_binding(self) -> SessionBinding | None:
        """Return the currently selected immutable root binding, if any."""
        with self._lock:
            return self._binding

    def try_select_root(
        self,
        root: str | Path,
        *,
        expected_binding: SessionBinding | None,
    ) -> SessionBinding | None:
        """Select a root only if shared owner state still matches expectation.

        A caller may join an identical root selected directly from the same
        expected generation. Older same-root ABA bindings and every unexpected
        different root are rejected.
        """
        root_key = str(Path(root).expanduser().resolve(strict=False))
        if not self._root_commit_lock.acquire(blocking=False):
            return None
        try:
            with self._lock:
                if self._shutdown:
                    return None
                if not self._root_selection_matches_locked(
                    root_key,
                    expected_binding,
                ):
                    return None
                return self._select_root_locked(root_key)
        finally:
            self._root_commit_lock.release()

    def wait_for_root_commit(self) -> None:
        """Block a worker thread until the active root reservation settles."""
        with self._root_commit_lock:
            return

    def acquire_stable_root(
        self,
        configured_root: str | Path | None,
    ) -> StableRootAccess | None:
        """Block a worker thread and hold one authoritative root binding.

        An existing owner binding always wins. The configured candidate is
        selected only when the owner is empty. Callers must release the
        returned access after synchronously publishing dependent runtime state.
        """
        root_key = (
            None
            if configured_root is None
            else str(Path(configured_root).expanduser().resolve(strict=False))
        )
        self._root_commit_lock.acquire()
        access: StableRootAccess | None = None
        try:
            with self._lock:
                if self._shutdown:
                    return None
                binding = self._binding
                if binding is None and root_key is not None:
                    binding = self._select_root_locked(root_key)
                access = StableRootAccess(self, binding)
                return access
        finally:
            if access is None:
                self._root_commit_lock.release()

    def try_reserve_root(
        self,
        root: str | Path,
        *,
        expected_binding: SessionBinding | None,
    ) -> RootCommitReservation | None:
        """Try to reserve one validated root commit without blocking."""
        root_key = str(Path(root).expanduser().resolve(strict=False))
        if not self._root_commit_lock.acquire(blocking=False):
            return None
        token = object()
        try:
            with self._lock:
                if self._shutdown or not self._root_selection_matches_locked(
                    root_key,
                    expected_binding,
                ):
                    return None
                self._root_commit_token = token
                self._root_commit_state = "reserved"
                return RootCommitReservation(self, token, root_key)
        finally:
            if self._root_commit_token is not token:
                self._root_commit_lock.release()

    def record_change(
        self,
        binding: SessionBinding,
        change: SessionChange,
    ) -> bool:
        """Publish one change if its root generation is still current."""
        with self._lock:
            if self._shutdown or binding != self._binding:
                return False
            self._changes.append(
                SequencedSessionChange(
                    sequence=self._next_sequence,
                    change=change,
                )
            )
            self._next_sequence += 1
            return True

    def snapshot(self, binding: SessionBinding) -> FileNotesSessionSnapshot:
        """Return an immutable snapshot without exposing another generation."""
        with self._lock:
            if binding != self._binding:
                return FileNotesSessionSnapshot(binding=binding)
            return FileNotesSessionSnapshot(
                binding=binding,
                changes=tuple(self._changes),
                trusted_repository=self._trusted_repository,
                git_status=self._git_status,
                staging_ownership=MappingProxyType(
                    dict(self._staging_ownership)
                ),
            )

    def publish_trust(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        """Publish process-only repository trust for the current binding."""
        with self._lock:
            if self._shutdown or binding != self._binding:
                return False
            if (
                self._trusted_repository is not None
                and self._trusted_repository != repository
            ):
                self._clear_git_status_locked()
                self._staging_ownership.clear()
            self._trusted_repository = repository
            return True

    def clear_trust(self, binding: SessionBinding) -> bool:
        """Clear trust and all state whose authority depends on that trust."""
        with self._lock:
            if binding != self._binding:
                return False
            self._trusted_repository = None
            self._clear_git_status_locked()
            self._staging_ownership.clear()
            return True

    def next_status_generation(
        self,
        binding: SessionBinding,
    ) -> int | None:
        """Return the next monotonic status generation for this binding."""
        with self._lock:
            if self._shutdown or binding != self._binding:
                return None
            return self._status_generation + 1

    def publish_status(
        self,
        binding: SessionBinding,
        status: SessionGitStatus,
    ) -> bool:
        """Publish a newer status only for the current root generation."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or status.binding_generation != binding.generation
                or status.status_generation <= self._status_generation
            ):
                return False
            self._status_generation = status.status_generation
            self._git_status = status
            return True

    def clear_status(self, binding: SessionBinding) -> bool:
        """Clear status and invalidate already-started older publications."""
        with self._lock:
            if binding != self._binding:
                return False
            self._clear_git_status_locked()
            return True

    def publish_ownership(
        self,
        binding: SessionBinding,
        ownership: Mapping[int, StagingOwnership],
    ) -> bool:
        """Replace staging ownership for the current binding atomically."""
        with self._lock:
            if self._shutdown or binding != self._binding:
                return False
            self._staging_ownership = dict(ownership)
            return True

    def clear_ownership(self, binding: SessionBinding) -> bool:
        """Clear every staging ownership record for the current binding."""
        with self._lock:
            if binding != self._binding:
                return False
            self._staging_ownership.clear()
            return True

    def try_acquire_transition(
        self,
        binding: SessionBinding,
        kind: SessionTransitionKind,
    ) -> SessionTransitionLease | None:
        """Try to admit a transition without awaiting."""
        if kind not in _TRANSITION_KINDS:
            raise ValueError(f"Unsupported File Notes transition kind: {kind}")
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._mutation_token is not None
            ):
                return None
            token = object()
            self._transition_tokens.add(token)
            return SessionTransitionLease(self, token, kind)

    def try_acquire_mutation(
        self,
        binding: SessionBinding,
    ) -> GitMutationLease | None:
        """Try to admit one mutation without awaiting."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._transition_tokens
                or self._mutation_token is not None
            ):
                return None
            token = object()
            self._mutation_token = token
            return GitMutationLease(self, token)

    def try_acquire_status(
        self,
        binding: SessionBinding,
    ) -> GitStatusLease | None:
        """Try to admit one status operation without awaiting."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._mutation_token is not None
                or self._status_token is not None
            ):
                return None
            token = object()
            self._status_token = token
            return GitStatusLease(self, token)

    def attach_git_service(self, service: FileNotesGitServiceLifecycle) -> None:
        """Attach the one optional service whose lifecycle this owner controls."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("File Notes session owner is shut down")
            if self._git_service is not None:
                raise RuntimeError("A File Notes Git service is already attached")
            self._git_service = service

    def shutdown(self) -> None:
        """Seal admission and shut down the attached service exactly once."""
        with self._shutdown_condition:
            while self._shutdown_state == "closing":
                self._shutdown_condition.wait()
            if self._shutdown_state == "closed":
                return
            if self._shutdown_state == "failed":
                assert self._shutdown_error is not None
                raise self._shutdown_error
            self._shutdown = True
            self._shutdown_state = "closing"
        with self._root_commit_lock:
            with self._lock:
                service = self._git_service
        try:
            if service is not None:
                service.shutdown()
        except BaseException as error:
            with self._shutdown_condition:
                self._shutdown_error = error
                self._shutdown_state = "failed"
                self._shutdown_condition.notify_all()
            raise
        with self._shutdown_condition:
            self._git_service = None
            self._shutdown_state = "closed"
            self._shutdown_condition.notify_all()

    def _release_transition(self, token: object) -> None:
        with self._lock:
            self._transition_tokens.discard(token)

    def _release_mutation(self, token: object) -> None:
        with self._lock:
            if self._mutation_token is token:
                self._mutation_token = None

    def _release_status(self, token: object) -> None:
        with self._lock:
            if self._status_token is token:
                self._status_token = None

    def _commit_root_reservation(
        self,
        token: object,
        root_key: str,
        publish: Callable[[SessionBinding], None],
    ) -> SessionBinding:
        with self._lock:
            if (
                self._root_commit_token is not token
                or self._root_commit_state != "reserved"
            ):
                raise RuntimeError("File Notes root reservation is not active")
            binding = self._select_root_locked(root_key)
            self._root_commit_state = "committed"
        publish(binding)
        return binding

    def _release_root_reservation(self, token: object) -> None:
        release_lock = False
        with self._lock:
            if self._root_commit_token is token:
                self._root_commit_token = None
                self._root_commit_state = "idle"
                release_lock = True
        if release_lock:
            self._root_commit_lock.release()

    def _select_root_locked(self, root_key: str) -> SessionBinding:
        if self._binding is not None and self._binding.root_key == root_key:
            return self._binding
        self._generation += 1
        self._binding = SessionBinding(root_key, self._generation)
        self._changes.clear()
        self._next_sequence = 1
        self._trusted_repository = None
        self._clear_git_status_locked()
        self._staging_ownership.clear()
        return self._binding

    def _clear_git_status_locked(self) -> None:
        self._status_generation += 1
        self._git_status = None

    def _root_selection_matches_locked(
        self,
        root_key: str,
        expected_binding: SessionBinding | None,
    ) -> bool:
        current = self._binding
        if current == expected_binding:
            return True
        if current is None or current.root_key != root_key:
            return False
        if expected_binding is None:
            return current.generation == 1
        return current.generation == expected_binding.generation + 1
