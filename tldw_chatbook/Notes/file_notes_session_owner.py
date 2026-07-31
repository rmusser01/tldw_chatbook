"""Process-memory ownership for one File Notes root session."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock, RLock
from types import MappingProxyType
from typing import Literal, Protocol

from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitRecoveryProjection,
    CommitReviewChangeType,
)
from tldw_chatbook.Notes.file_notes_git_push import (
    PushCandidateProjection,
    PushContractError,
    PushIncludedNote,
)

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
GitStatusAdmissionReason = Literal[
    "mutation_active",
    "status_active",
    "stale_binding",
    "shutdown",
]
GitMutationAdmissionReason = Literal[
    "mutation_active",
    "recovery_required",
    "transition_active",
    "stale_binding",
    "shutdown",
]
CommitPublicationState = Literal[
    "succeeded",
    "failed_unchanged",
    "uncertain",
]
CommitRecoveryAdmissionReason = Literal[
    "invalid_capability",
    "mutation_active",
    "ownership_active",
    "stale_binding",
    "shutdown",
    "transition_active",
]
_TRANSITION_KINDS = frozenset({"root", "path", "source", "screen"})
_PUSH_CHANGE_TYPE_ORDER: tuple[CommitReviewChangeType, ...] = (
    "New",
    "Modified",
    "Deleted",
    "Moved",
)


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
    sequence_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoints", tuple(self.endpoints))
        object.__setattr__(
            self,
            "move_edges",
            tuple(tuple(edge) for edge in self.move_edges),
        )
        object.__setattr__(self, "sequence_ids", tuple(self.sequence_ids))

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


@dataclass(slots=True)
class _GroupBuilder:
    group_id: int
    endpoints: list[str]
    source_path: str
    destination_path: str | None
    current_path: str
    latest_action: SessionChangeAction
    latest_sequence: int
    move_edges: list[tuple[str, str]]
    sequence_ids: list[int]

    def add_endpoint(self, path: str) -> None:
        if path not in self.endpoints:
            self.endpoints.append(path)

    def freeze(self) -> SessionChangeGroup:
        return SessionChangeGroup(
            group_id=self.group_id,
            endpoints=tuple(self.endpoints),
            source_path=self.source_path,
            destination_path=self.destination_path,
            current_path=self.current_path,
            latest_action=self.latest_action,
            latest_sequence=self.latest_sequence,
            move_edges=tuple(self.move_edges),
            sequence_ids=tuple(self.sequence_ids),
        )


def coalesce_session_changes(
    changes: Sequence[SequencedSessionChange],
) -> tuple[SessionChangeGroup, ...]:
    """Coalesce session changes using each lineage's active path.

    Args:
        changes: Sequenced changes. Input may be unordered; changes are
            processed in ascending sequence order.

    Returns:
        Coalesced lineage groups in earliest-sequence order.

    Raises:
        ValueError: If a moved change has no destination path.
    """
    active_paths: dict[str, _GroupBuilder] = {}
    builders: list[_GroupBuilder] = []

    for sequenced in sorted(changes, key=lambda item: item.sequence):
        change = sequenced.change
        path = change.relative_path
        builder = active_paths.get(path)
        if builder is None:
            builder = _GroupBuilder(
                group_id=sequenced.sequence,
                endpoints=[path],
                source_path=path,
                destination_path=None,
                current_path=path,
                latest_action=change.action,
                latest_sequence=sequenced.sequence,
                move_edges=[],
                sequence_ids=[],
            )
            builders.append(builder)

        if change.action == "moved":
            destination = change.destination_path
            if destination is None:
                raise ValueError("A moved session change requires a destination")
            if active_paths.get(path) is builder:
                del active_paths[path]
            builder.add_endpoint(path)
            builder.add_endpoint(destination)
            builder.move_edges.append((path, destination))
            builder.destination_path = destination
            builder.current_path = destination
            active_paths[destination] = builder
        else:
            builder.add_endpoint(path)
            builder.current_path = path
            active_paths[path] = builder

        builder.latest_action = change.action
        builder.latest_sequence = sequenced.sequence
        builder.sequence_ids.append(sequenced.sequence)

    return tuple(builder.freeze() for builder in builders)


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


def _status_authority_facts(
    status: SessionGitStatus | None,
) -> object:
    """Return security-relevant status facts without presentation versions."""
    if status is None:
        return None
    return (
        status.state,
        status.repository,
        status.head,
        tuple(
            (
                row.group,
                row.state,
                row.stage_action,
                row.unstage_eligible,
            )
            for row in status.rows
        ),
    )


@dataclass(frozen=True, slots=True)
class PushCandidateAvailability:
    """Sanitized availability of one exact process-memory push candidate."""

    generation: int
    candidate: PushCandidateProjection
    change_types: tuple[CommitReviewChangeType, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "change_types", tuple(self.change_types))

    @property
    def change_counts(self) -> tuple[tuple[CommitReviewChangeType, int], ...]:
        """Return nonzero included-note counts in stable display order."""
        return tuple(
            (change_type, self.change_types.count(change_type))
            for change_type in _PUSH_CHANGE_TYPE_ORDER
            if change_type in self.change_types
        )


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
    git_authority_generation: int = 0
    commit_recovery: CommitRecoveryProjection | None = None
    push_candidate: PushCandidateAvailability | None = None
    push_candidate_generation: int = 0


class FileNotesGitServiceLifecycle(Protocol):
    """Narrow lifecycle boundary for an owner-attached Git service."""

    def shutdown(self) -> Awaitable[object] | None:
        """Stop accepting work and return retained service settlement."""


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
    _binding: SessionBinding = field(repr=False, compare=False)

    def release(self) -> None:
        """Release this mutation admission once."""
        self._owner._release_mutation(self._token)


@dataclass(frozen=True, slots=True)
class CommitAuthorityCapture:
    """Exact session-owned facts authorized by one active mutation lease."""

    binding: SessionBinding
    authority_generation: int
    repository_trust_generation: int
    repository: RepositoryIdentity
    head: HeadIdentity
    ownership: Mapping[int, StagingOwnership]
    group_sequence_ids: Mapping[int, tuple[int, ...]]
    _guarded_commit_identity: object = field(repr=False, compare=False)
    _mutation_token: object = field(repr=False, compare=False)
    _quarantine_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.head.kind != "attached"
            or self.head.object_id is None
            or self.head.branch is None
            or not self.head.branch.startswith("refs/heads/")
        ):
            raise ValueError("Commit authority requires an attached branch")
        object.__setattr__(
            self,
            "ownership",
            MappingProxyType(dict(self.ownership)),
        )
        object.__setattr__(
            self,
            "group_sequence_ids",
            MappingProxyType(
                {
                    group_id: tuple(sequence_ids)
                    for group_id, sequence_ids in self.group_sequence_ids.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PushCandidateSeed:
    """Immutable reviewed provenance carried through local commit recovery."""

    binding: SessionBinding
    repository: RepositoryIdentity
    repository_trust_generation: int
    parent_head: HeadIdentity
    subject: str
    included_notes: tuple[PushIncludedNote, ...]
    change_types: tuple[CommitReviewChangeType, ...]
    _guarded_commit_identity: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        notes = tuple(self.included_notes)
        change_types = tuple(self.change_types)
        if (
            not notes
            or len(notes) != len(change_types)
            or len({note.group_id for note in notes}) != len(notes)
            or any(
                change_type not in _PUSH_CHANGE_TYPE_ORDER
                for change_type in change_types
            )
        ):
            raise ValueError("Push candidate seed provenance is invalid")
        object.__setattr__(self, "included_notes", notes)
        object.__setattr__(self, "change_types", change_types)

    @classmethod
    def from_commit_capture(
        cls,
        capture: CommitAuthorityCapture,
        *,
        subject: str,
        included_notes: Sequence[PushIncludedNote],
        change_types: Sequence[CommitReviewChangeType],
    ) -> PushCandidateSeed:
        """Copy the blob-free reviewed facts needed after group retirement."""
        return cls(
            binding=capture.binding,
            repository=capture.repository,
            repository_trust_generation=capture.repository_trust_generation,
            parent_head=capture.head,
            subject=subject,
            included_notes=tuple(included_notes),
            change_types=tuple(change_types),
            _guarded_commit_identity=capture._guarded_commit_identity,
        )


@dataclass(frozen=True, slots=True)
class PushCandidateCapture:
    """Private exact authority for one validated local push candidate."""

    binding: SessionBinding
    repository: RepositoryIdentity
    repository_trust_generation: int
    candidate_generation: int
    candidate: PushCandidateProjection
    change_types: tuple[CommitReviewChangeType, ...]
    sole_parent_oid: str
    _guarded_commit_identity: object = field(repr=False, compare=False)
    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "change_types", tuple(self.change_types))

    @property
    def selected_root_generation(self) -> int:
        """Return the exact selected-root generation bound to this authority."""
        return self.binding.generation


@dataclass(frozen=True, slots=True)
class CommitPublication:
    """One exact owner-side terminal or recoverable commit transition."""

    state: CommitPublicationState
    new_head: HeadIdentity | None = None
    retired_sequence_ids: tuple[int, ...] = ()
    divergent_sequence_ids: tuple[int, ...] = ()
    refreshed_status: SessionGitStatus | None = None
    recovery_projection: CommitRecoveryProjection | None = None
    candidate_seed: PushCandidateSeed | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retired_sequence_ids",
            tuple(self.retired_sequence_ids),
        )
        object.__setattr__(
            self,
            "divergent_sequence_ids",
            tuple(self.divergent_sequence_ids),
        )


@dataclass(frozen=True, slots=True)
class CommitRecoveryCapability:
    """Opaque process-memory authority for one quarantined commit attempt."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class CommitPublicationResult:
    """Whether one atomic owner publication was accepted."""

    published: bool
    recovery_capability: CommitRecoveryCapability | None = None


@dataclass(frozen=True, slots=True)
class GitStatusLease:
    """Idempotently releasable Git status admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    invalidation_generation: int

    def release(self) -> None:
        """Release this status admission once."""
        self._owner._release_status(self._token)


@dataclass(frozen=True, slots=True)
class GitStatusAdmission:
    """Atomic status admission with a typed refusal reason."""

    lease: GitStatusLease | None = None
    reason: GitStatusAdmissionReason | None = None
    invalidation_generation: int | None = None

    def __post_init__(self) -> None:
        if (self.lease is None) == (self.reason is None):
            raise ValueError("Status admission requires exactly one outcome")
        if self.lease is not None:
            if (
                self.invalidation_generation
                != self.lease.invalidation_generation
            ):
                raise ValueError(
                    "Successful status admission requires its generation"
                )
        elif (self.reason == "status_active") != (
            self.invalidation_generation is not None
        ):
            raise ValueError(
                "Only active-status refusal carries a generation"
            )


@dataclass(frozen=True, slots=True)
class GitMutationAdmission:
    """Atomic mutation admission with a typed refusal reason."""

    lease: GitMutationLease | None = None
    reason: GitMutationAdmissionReason | None = None

    def __post_init__(self) -> None:
        if (self.lease is None) == (self.reason is None):
            raise ValueError("Mutation admission requires exactly one outcome")


@dataclass(frozen=True, slots=True)
class CommitRecoveryAdmission:
    """Exclusive mutation admission for one exact quarantine capability."""

    lease: GitMutationLease | None = None
    capture: CommitAuthorityCapture | None = None
    reason: CommitRecoveryAdmissionReason | None = None

    def __post_init__(self) -> None:
        admitted = self.lease is not None and self.capture is not None
        refused = (
            self.lease is None and self.capture is None and self.reason is not None
        )
        if admitted == refused or (admitted and self.reason is not None):
            raise ValueError("Recovery admission requires exactly one outcome")


@dataclass(frozen=True, slots=True)
class _CommitQuarantine:
    """Private exact captured ownership retained after uncertainty."""

    token: object
    capture: CommitAuthorityCapture
    projection: CommitRecoveryProjection


@dataclass(frozen=True, slots=True)
class _PushCandidate:
    """One private exact process-memory push candidate and its authority."""

    token: object = field(repr=False, compare=False)
    generation: int
    binding: SessionBinding
    repository: RepositoryIdentity
    repository_trust_generation: int
    _guarded_commit_identity: object = field(repr=False, compare=False)
    candidate: PushCandidateProjection
    change_types: tuple[CommitReviewChangeType, ...]
    sole_parent_oid: str

    @property
    def availability(self) -> PushCandidateAvailability:
        """Return the only candidate facts exposed in owner snapshots."""
        return PushCandidateAvailability(
            generation=self.generation,
            candidate=self.candidate,
            change_types=self.change_types,
        )


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
        "_commit_publication_closed",
        "_commit_quarantine",
        "_generation",
        "_git_authority_generation",
        "_git_service",
        "_git_shutdown_settlement",
        "_git_shutdown_settlement_future",
        "_git_status",
        "_lock",
        "_mutation_token",
        "_next_sequence",
        "_root_commit_lock",
        "_root_commit_state",
        "_root_commit_token",
        "_push_candidate",
        "_push_candidate_generation",
        "_repository_trust_generation",
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
        self._git_authority_generation = 0
        self._repository_trust_generation = 0
        self._push_candidate_generation = 0
        self._push_candidate: _PushCandidate | None = None
        self._changes: list[SequencedSessionChange] = []
        self._next_sequence = 1
        self._trusted_repository: RepositoryIdentity | None = None
        self._git_status: SessionGitStatus | None = None
        self._staging_ownership: dict[int, StagingOwnership] = {}
        self._commit_publication_closed = False
        self._commit_quarantine: _CommitQuarantine | None = None
        self._status_generation = 0
        self._transition_tokens: set[object] = set()
        self._mutation_token: object | None = None
        self._status_token: object | None = None
        self._shutdown = False
        self._shutdown_condition = Condition(self._lock)
        self._shutdown_error: BaseException | None = None
        self._shutdown_state: Literal["open", "closing", "closed", "failed"] = "open"
        self._git_service: FileNotesGitServiceLifecycle | None = None
        self._git_shutdown_settlement: Awaitable[object] | None = None
        self._git_shutdown_settlement_future: asyncio.Future[object] | None = None

    def select_root(self, root: str | Path) -> SessionBinding:
        """Select one canonical root, resetting state only when it changes.

        Args:
            root: Filesystem root to bind for this process session.

        Returns:
            The current immutable root-generation binding.

        Raises:
            RuntimeError: If the owner has shut down or another root commit is
                in progress, or a Git mutation protects a different root.
        """
        root_key = str(Path(root).expanduser().resolve(strict=False))
        if not self._root_commit_lock.acquire(blocking=False):
            raise RuntimeError("File Notes root commit is in progress")
        try:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("File Notes session owner is shut down")
                if self._root_change_is_blocked_locked(root_key):
                    raise RuntimeError("File Notes Git mutation is in progress")
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
                if self._root_change_is_blocked_locked(root_key):
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
                if (
                    self._shutdown
                    or self._mutation_token is not None
                    or not self._root_selection_matches_locked(
                        root_key,
                        expected_binding,
                    )
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
            self._clear_git_status_locked(invalidate_authority=False)
            self._invalidate_git_authority_locked()
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
                git_authority_generation=self._git_authority_generation,
                commit_recovery=(
                    None
                    if self._commit_quarantine is None
                    else self._commit_quarantine.projection
                ),
                push_candidate=(
                    None
                    if self._push_candidate is None
                    else self._push_candidate.availability
                ),
                push_candidate_generation=self._push_candidate_generation,
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
            if self._trusted_repository == repository:
                return True
            self._revoke_push_candidate_locked()
            if (
                self._trusted_repository is not None
                and self._trusted_repository != repository
            ):
                self._clear_git_status_locked(invalidate_authority=False)
                self._staging_ownership.clear()
            if (
                self._commit_quarantine is not None
                and self._commit_quarantine.capture.repository != repository
            ):
                self._commit_quarantine = None
            self._trusted_repository = repository
            self._repository_trust_generation += 1
            self._invalidate_git_authority_locked()
            return True

    def clear_trust(self, binding: SessionBinding) -> bool:
        """Clear trust and all state whose authority depends on that trust."""
        return self.clear_trust_if_matches(binding)

    def clear_trust_if_matches(
        self,
        binding: SessionBinding,
        expected_repository: RepositoryIdentity | None = None,
    ) -> bool:
        """Compare-and-clear trust without erasing a newer repository grant."""
        with self._lock:
            if binding != self._binding:
                return False
            if (
                expected_repository is not None
                and self._trusted_repository != expected_repository
            ):
                return False
            changed = (
                self._trusted_repository is not None
                or self._git_status is not None
                or bool(self._staging_ownership)
                or self._push_candidate is not None
            )
            trust_changed = self._trusted_repository is not None
            self._trusted_repository = None
            self._clear_git_status_locked(invalidate_authority=False)
            self._staging_ownership.clear()
            self._revoke_push_candidate_locked()
            if trust_changed:
                self._repository_trust_generation += 1
            if changed:
                self._invalidate_git_authority_locked()
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
        *,
        invalidation_generation: int | None = None,
    ) -> bool:
        """Publish a newer status only for the current root generation."""
        with self._lock:
            if (
                self._shutdown
                or self._mutation_token is not None
                or binding != self._binding
                or status.binding_generation != binding.generation
                or self._trusted_repository is None
                or status.repository != self._trusted_repository
                or status.status_generation <= self._status_generation
                or (
                    invalidation_generation is not None
                    and invalidation_generation != self._status_generation
                )
            ):
                return False
            previous_facts = _status_authority_facts(self._git_status)
            self._status_generation = status.status_generation
            self._git_status = status
            self._revoke_push_candidate_for_head_mismatch_locked(status)
            if _status_authority_facts(status) != previous_facts:
                self._invalidate_git_authority_locked()
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
        *,
        group_sequence_ids: Mapping[int, Collection[int]] | None = None,
    ) -> bool:
        """Replace staging ownership for the current binding atomically."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._commit_quarantine is not None
            ):
                return False
            replacement = dict(ownership)
            if not self._supplied_ownership_sequences_match_locked(
                replacement,
                group_sequence_ids,
            ):
                return False
            if replacement != self._staging_ownership:
                self._staging_ownership = replacement
                self._invalidate_git_authority_locked()
            return True

    def publish_stage_result(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
        ownership: Mapping[int, StagingOwnership],
        *,
        group_sequence_ids: Mapping[int, Collection[int]] | None = None,
    ) -> bool:
        """Atomically publish checked ownership and make prior status stale."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or repository != self._trusted_repository
                or self._commit_quarantine is not None
            ):
                return False
            replacement = dict(ownership)
            if not self._supplied_ownership_sequences_match_locked(
                replacement,
                group_sequence_ids,
            ):
                return False
            changed = (
                replacement != self._staging_ownership
                or self._git_status is not None
            )
            self._staging_ownership = replacement
            self._clear_git_status_locked(invalidate_authority=False)
            if changed:
                self._invalidate_git_authority_locked()
            return True

    def publish_unstage_result(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
        expected: Mapping[int, StagingOwnership],
        group_ids: Collection[int],
    ) -> bool:
        """Atomically remove checked ownership and make prior status stale."""
        selected = tuple(dict.fromkeys(group_ids))
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or repository != self._trusted_repository
                or self._commit_quarantine is not None
                or any(
                    self._staging_ownership.get(group_id)
                    != expected.get(group_id)
                    for group_id in selected
                )
            ):
                return False
            changed = (
                any(group_id in self._staging_ownership for group_id in selected)
                or self._git_status is not None
            )
            for group_id in selected:
                self._staging_ownership.pop(group_id, None)
            self._clear_git_status_locked(invalidate_authority=False)
            if changed:
                self._invalidate_git_authority_locked()
            return True

    def clear_ownership(self, binding: SessionBinding) -> bool:
        """Clear every staging ownership record for the current binding."""
        with self._lock:
            if binding != self._binding:
                return False
            if self._staging_ownership:
                self._staging_ownership.clear()
                self._invalidate_git_authority_locked()
            return True

    def capture_commit_authority(
        self,
        lease: GitMutationLease,
        *,
        binding: SessionBinding,
        authority_generation: int,
        repository: RepositoryIdentity,
        head: HeadIdentity,
        group_sequence_ids: Mapping[int, Collection[int]],
        _guarded_commit_identity: object | None = None,
    ) -> CommitAuthorityCapture | None:
        """Capture exact commit authority under one active mutation lease."""
        sequence_ids = {
            group_id: tuple(group_sequences)
            for group_id, group_sequences in group_sequence_ids.items()
        }
        with self._lock:
            if (
                self._shutdown
                or self._commit_quarantine is not None
                or self._status_token is not None
                or not self._lease_is_active_locked(lease)
                or binding != self._binding
                or authority_generation != self._git_authority_generation
                or repository != self._trusted_repository
                or not self._commit_capture_facts_match_locked(
                    repository,
                    head,
                    self._staging_ownership,
                    sequence_ids,
                )
            ):
                return None
            return CommitAuthorityCapture(
                binding=binding,
                authority_generation=authority_generation,
                repository_trust_generation=self._repository_trust_generation,
                repository=repository,
                head=head,
                ownership=self._staging_ownership,
                group_sequence_ids=sequence_ids,
                _guarded_commit_identity=(
                    object()
                    if _guarded_commit_identity is None
                    else _guarded_commit_identity
                ),
                _mutation_token=lease._token,
            )

    def capture_push_candidate(
        self,
        binding: SessionBinding,
        *,
        candidate_generation: int,
        repository: RepositoryIdentity,
        head: HeadIdentity,
        sole_parent_oid: str,
    ) -> PushCandidateCapture | None:
        """Validate exact local lineage and capture one private candidate.

        The caller supplies fresh local-only repository, attached-HEAD, and
        sole-parent proof. A mismatch revokes only the exact candidate
        generation that was checked; stale observations cannot affect a newer
        candidate.
        """
        with self._lock:
            candidate = self._push_candidate
            if (
                candidate is None
                or candidate.generation != candidate_generation
                or binding != self._binding
                or candidate.binding != binding
            ):
                return None
            owner_facts_match = (
                not self._shutdown
                and self._trusted_repository == candidate.repository
                and self._repository_trust_generation
                == candidate.repository_trust_generation
            )
            lineage_matches = (
                repository == candidate.repository
                and head.kind == "attached"
                and head.branch == candidate.candidate.local_branch_ref
                and head.object_id == candidate.candidate.candidate_oid
                and sole_parent_oid == candidate.sole_parent_oid
                and sole_parent_oid == candidate.candidate.parent_oid
            )
            if not owner_facts_match or not lineage_matches:
                self._revoke_push_candidate_locked()
                return None
            return PushCandidateCapture(
                binding=candidate.binding,
                repository=candidate.repository,
                repository_trust_generation=(
                    candidate.repository_trust_generation
                ),
                candidate_generation=candidate.generation,
                candidate=candidate.candidate,
                change_types=candidate.change_types,
                sole_parent_oid=candidate.sole_parent_oid,
                _guarded_commit_identity=candidate._guarded_commit_identity,
                _owner=self,
                _token=candidate.token,
            )

    def clear_push_candidate(self, capture: PushCandidateCapture) -> bool:
        """Compare-and-clear only one exact private candidate capability."""
        with self._lock:
            candidate = self._push_candidate
            if (
                candidate is None
                or capture._owner is not self
                or capture._token is not candidate.token
                or capture.binding != candidate.binding
                or capture.repository != candidate.repository
                or capture.repository_trust_generation
                != candidate.repository_trust_generation
                or capture.candidate_generation != candidate.generation
                or capture.candidate != candidate.candidate
                or capture.sole_parent_oid != candidate.sole_parent_oid
                or capture._guarded_commit_identity
                is not candidate._guarded_commit_identity
            ):
                return False
            self._revoke_push_candidate_locked()
            return True

    def publish_commit_outcome(
        self,
        lease: GitMutationLease,
        capture: CommitAuthorityCapture,
        publication: CommitPublication,
    ) -> CommitPublicationResult:
        """Atomically publish one exact terminal or quarantined commit result."""
        with self._lock:
            exact_match = self._commit_publication_matches_locked(
                lease,
                capture,
            )
            uncertainty_fallback = (
                publication.state == "uncertain"
                and self._commit_uncertainty_fallback_matches_locked(
                    lease,
                    capture,
                )
            )
            if not exact_match and not uncertainty_fallback:
                return CommitPublicationResult(published=False)
            if not self._commit_publication_value_is_valid_locked(
                capture,
                publication,
            ):
                return CommitPublicationResult(published=False)
            push_candidate: _PushCandidate | None = None
            if publication.state == "succeeded":
                push_candidate = self._prepare_push_candidate_locked(
                    capture,
                    publication,
                )
                if push_candidate is None:
                    return CommitPublicationResult(published=False)

            recovery_capability: CommitRecoveryCapability | None = None
            recovering = capture._quarantine_token is not None
            if publication.state == "succeeded":
                retired = frozenset(publication.retired_sequence_ids)
                self._changes = [
                    change for change in self._changes if change.sequence not in retired
                ]
                self._staging_ownership.clear()
                self._commit_quarantine = None
                self._publish_commit_status_locked(publication.refreshed_status)
                assert push_candidate is not None
                self._push_candidate_generation = push_candidate.generation
                self._push_candidate = push_candidate
                if self._shutdown:
                    self._revoke_push_candidate_locked()
            elif publication.state == "failed_unchanged":
                if recovering:
                    if self._staging_ownership:
                        return CommitPublicationResult(published=False)
                    self._staging_ownership = dict(capture.ownership)
                    self._commit_quarantine = None
                if publication.refreshed_status is not None:
                    self._publish_commit_status_locked(publication.refreshed_status)
            else:
                assert publication.recovery_projection is not None
                self._revoke_push_candidate_locked()
                if recovering:
                    assert self._commit_quarantine is not None
                    quarantine_token = self._commit_quarantine.token
                    quarantine_capture = self._commit_quarantine.capture
                else:
                    quarantine_token = object()
                    quarantine_capture = capture
                self._staging_ownership.clear()
                self._clear_git_status_locked(invalidate_authority=False)
                self._commit_quarantine = _CommitQuarantine(
                    token=quarantine_token,
                    capture=quarantine_capture,
                    projection=publication.recovery_projection,
                )
                recovery_capability = CommitRecoveryCapability(
                    self,
                    quarantine_token,
                )

            self._invalidate_git_authority_locked()
            return CommitPublicationResult(
                published=True,
                recovery_capability=recovery_capability,
            )

    def admit_commit_recovery(
        self,
        binding: SessionBinding,
        capability: CommitRecoveryCapability,
    ) -> CommitRecoveryAdmission:
        """Admit fresh proof for one exact quarantined commit attempt."""
        with self._lock:
            if self._shutdown:
                return CommitRecoveryAdmission(reason="shutdown")
            if binding != self._binding:
                return CommitRecoveryAdmission(reason="stale_binding")
            quarantine = self._commit_quarantine
            if (
                quarantine is None
                or capability._owner is not self
                or capability._token is not quarantine.token
                or quarantine.capture.binding != binding
                or self._trusted_repository != quarantine.capture.repository
                or not self._captured_sequences_are_present_locked(
                    quarantine.capture.group_sequence_ids
                )
            ):
                return CommitRecoveryAdmission(
                    reason="invalid_capability",
                )
            if self._staging_ownership:
                return CommitRecoveryAdmission(reason="ownership_active")
            if self._transition_tokens:
                return CommitRecoveryAdmission(reason="transition_active")
            if self._root_commit_token is not None:
                return CommitRecoveryAdmission(reason="transition_active")
            if self._mutation_token is not None:
                return CommitRecoveryAdmission(reason="mutation_active")

            token = object()
            self._mutation_token = token
            lease = GitMutationLease(self, token, binding)
            capture = CommitAuthorityCapture(
                binding=binding,
                authority_generation=self._git_authority_generation,
                repository_trust_generation=self._repository_trust_generation,
                repository=quarantine.capture.repository,
                head=quarantine.capture.head,
                ownership=quarantine.capture.ownership,
                group_sequence_ids=quarantine.capture.group_sequence_ids,
                _guarded_commit_identity=(
                    quarantine.capture._guarded_commit_identity
                ),
                _mutation_token=token,
                _quarantine_token=quarantine.token,
            )
            return CommitRecoveryAdmission(
                lease=lease,
                capture=capture,
            )

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
            self._invalidate_git_authority_locked()
            return SessionTransitionLease(self, token, kind)

    def try_acquire_mutation(
        self,
        binding: SessionBinding,
    ) -> GitMutationLease | None:
        """Try to admit one mutation without awaiting."""
        return self.admit_mutation(binding).lease

    def mutation_active(self, binding: SessionBinding) -> bool:
        """Return whether the current exact binding owns a Git mutation."""
        with self._lock:
            return binding == self._binding and self._mutation_token is not None

    def admit_mutation(
        self,
        binding: SessionBinding,
    ) -> GitMutationAdmission:
        """Atomically admit a mutation or identify the current refusal."""
        with self._lock:
            if self._shutdown:
                return GitMutationAdmission(reason="shutdown")
            if binding != self._binding:
                return GitMutationAdmission(reason="stale_binding")
            if self._commit_quarantine is not None:
                return GitMutationAdmission(reason="recovery_required")
            if self._transition_tokens:
                return GitMutationAdmission(reason="transition_active")
            if self._root_commit_token is not None:
                return GitMutationAdmission(reason="transition_active")
            if self._mutation_token is not None:
                return GitMutationAdmission(reason="mutation_active")
            token = object()
            self._mutation_token = token
            return GitMutationAdmission(
                lease=GitMutationLease(self, token, binding),
            )

    def try_acquire_status(
        self,
        binding: SessionBinding,
    ) -> GitStatusLease | None:
        """Try to admit one status operation without awaiting."""
        return self.admit_status(binding).lease

    def admit_status(
        self,
        binding: SessionBinding,
    ) -> GitStatusAdmission:
        """Atomically admit status or identify the current refusal."""
        with self._lock:
            if self._shutdown:
                return GitStatusAdmission(reason="shutdown")
            if binding != self._binding:
                return GitStatusAdmission(reason="stale_binding")
            if self._mutation_token is not None:
                return GitStatusAdmission(reason="mutation_active")
            if self._status_token is not None:
                return GitStatusAdmission(
                    reason="status_active",
                    invalidation_generation=self._status_generation,
                )
            token = object()
            self._status_token = token
            generation = self._status_generation
            lease = GitStatusLease(
                self,
                token,
                generation,
            )
            return GitStatusAdmission(
                lease=lease,
                invalidation_generation=generation,
            )

    def attach_git_service(self, service: FileNotesGitServiceLifecycle) -> None:
        """Attach the one optional service whose lifecycle this owner controls."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("File Notes session owner is shut down")
            if self._git_service is not None:
                raise RuntimeError("A File Notes Git service is already attached")
            self._git_service = service

    def attached_git_service(self) -> FileNotesGitServiceLifecycle | None:
        """Return the one process-owned Git service, if configured."""
        with self._lock:
            return self._git_service

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
            self._revoke_push_candidate_locked()
            self._commit_publication_closed = False
            self._git_shutdown_settlement_future = None
            self._shutdown_state = "closing"
        with self._root_commit_lock:
            with self._lock:
                service = self._git_service
        try:
            if service is not None:
                settlement = service.shutdown()
                with self._lock:
                    self._git_shutdown_settlement = settlement
        except BaseException as error:
            with self._shutdown_condition:
                if getattr(error, "retryable_shutdown", False):
                    self._shutdown = False
                    self._commit_publication_closed = False
                    self._shutdown_error = None
                    self._git_shutdown_settlement = None
                    self._git_shutdown_settlement_future = None
                    self._shutdown_state = "open"
                else:
                    self._commit_publication_closed = True
                    self._shutdown_error = error
                    self._shutdown_state = "failed"
                self._shutdown_condition.notify_all()
            raise
        with self._shutdown_condition:
            self._git_service = None
            if self._git_shutdown_settlement is None:
                self._commit_publication_closed = True
                self._discard_commit_quarantine_locked()
            self._shutdown_state = "closed"
            self._shutdown_condition.notify_all()

    async def shutdown_async(self) -> None:
        """Shut down on the current loop, then await retained settlement."""
        self.shutdown()
        await self.settle_git_shutdown()

    async def settle_git_shutdown(self) -> None:
        """Await the retained attached-service settlement, if any."""
        with self._lock:
            settlement = self._git_shutdown_settlement
            if settlement is None:
                if self._shutdown:
                    self._commit_publication_closed = True
                    self._discard_commit_quarantine_locked()
                return
            settlement_future = self._git_shutdown_settlement_future
            if settlement_future is None:
                settlement_future = asyncio.ensure_future(
                    self._settle_git_shutdown_once(settlement)
                )
                self._git_shutdown_settlement_future = settlement_future
                settlement_future.add_done_callback(
                    self._retrieve_git_shutdown_settlement,
                )
        await asyncio.shield(settlement_future)

    async def _settle_git_shutdown_once(
        self,
        settlement: Awaitable[object],
    ) -> object:
        try:
            return await settlement
        finally:
            with self._lock:
                self._commit_publication_closed = True
                self._discard_commit_quarantine_locked()

    @staticmethod
    def _retrieve_git_shutdown_settlement(
        settlement: asyncio.Future[object],
    ) -> None:
        if not settlement.cancelled():
            settlement.exception()

    def _lease_is_active_locked(self, lease: GitMutationLease) -> bool:
        return (
            lease._owner is self
            and lease._token is self._mutation_token
            and lease._binding == self._binding
        )

    def _current_ownership_sequences_locked(
        self,
        ownership: Mapping[int, StagingOwnership],
    ) -> dict[int, tuple[int, ...]] | None:
        groups = {
            group.group_id: group
            for group in coalesce_session_changes(self._changes)
        }
        if any(group_id not in groups for group_id in ownership):
            return None
        return {
            group_id: groups[group_id].sequence_ids
            for group_id in ownership
        }

    def _supplied_ownership_sequences_match_locked(
        self,
        ownership: Mapping[int, StagingOwnership],
        group_sequence_ids: Mapping[int, Collection[int]] | None,
    ) -> bool:
        if group_sequence_ids is None:
            return True
        supplied = {
            group_id: tuple(sequence_ids)
            for group_id, sequence_ids in group_sequence_ids.items()
        }
        return supplied == self._current_ownership_sequences_locked(ownership)

    def _commit_capture_facts_match_locked(
        self,
        repository: RepositoryIdentity,
        head: HeadIdentity,
        ownership: Mapping[int, StagingOwnership],
        group_sequence_ids: Mapping[int, tuple[int, ...]],
    ) -> bool:
        current_sequence_ids = self._current_ownership_sequences_locked(
            ownership
        )
        return not (
            head.kind != "attached"
            or head.object_id is None
            or head.branch is None
            or not head.branch.startswith("refs/heads/")
            or not ownership
            or current_sequence_ids is None
            or dict(group_sequence_ids) != current_sequence_ids
            or any(
                item.repository != repository or item.head != head
                for item in ownership.values()
            )
        )

    def _captured_sequences_are_present_locked(
        self,
        group_sequence_ids: Mapping[int, tuple[int, ...]],
    ) -> bool:
        current = {change.sequence for change in self._changes}
        return all(
            sequence in current
            for sequence_ids in group_sequence_ids.values()
            for sequence in sequence_ids
        )

    def _commit_publication_matches_locked(
        self,
        lease: GitMutationLease,
        capture: CommitAuthorityCapture,
    ) -> bool:
        if (
            self._commit_publication_closed
            or not self._lease_is_active_locked(lease)
            or capture._mutation_token is not lease._token
            or capture.binding != self._binding
            or capture.authority_generation != self._git_authority_generation
            or capture.repository_trust_generation
            != self._repository_trust_generation
            or capture.repository != self._trusted_repository
        ):
            return False

        quarantine_token = capture._quarantine_token
        if quarantine_token is None:
            return (
                self._commit_quarantine is None
                and dict(capture.ownership) == self._staging_ownership
                and self._commit_capture_facts_match_locked(
                    capture.repository,
                    capture.head,
                    capture.ownership,
                    capture.group_sequence_ids,
                )
            )

        quarantine = self._commit_quarantine
        original = None if quarantine is None else quarantine.capture
        return (
            quarantine is not None
            and quarantine.token is quarantine_token
            and not self._staging_ownership
            and original is not None
            and original.binding == capture.binding
            and original.repository == capture.repository
            and original.head == capture.head
            and dict(original.ownership) == dict(capture.ownership)
            and dict(original.group_sequence_ids) == dict(capture.group_sequence_ids)
            and self._captured_sequences_are_present_locked(capture.group_sequence_ids)
        )

    def _commit_uncertainty_fallback_matches_locked(
        self,
        lease: GitMutationLease,
        capture: CommitAuthorityCapture,
    ) -> bool:
        """Permit only fail-closed quarantine for one exact active attempt."""
        return (
            not self._commit_publication_closed
            and self._lease_is_active_locked(lease)
            and capture._mutation_token is lease._token
            and capture.binding == self._binding
            and capture._quarantine_token is None
            and self._commit_quarantine is None
        )

    def _commit_publication_value_is_valid_locked(
        self,
        capture: CommitAuthorityCapture,
        publication: CommitPublication,
    ) -> bool:
        if publication.state == "succeeded":
            new_head = publication.new_head
            seed = publication.candidate_seed
            if (
                new_head is None
                or new_head.kind != "attached"
                or new_head.branch != capture.head.branch
                or new_head.object_id is None
                or new_head.object_id == capture.head.object_id
                or publication.recovery_projection is not None
                or seed is None
                or not self._push_candidate_seed_matches_capture_locked(
                    seed,
                    capture,
                )
            ):
                return False
            retired = publication.retired_sequence_ids
            divergent = publication.divergent_sequence_ids
            retired_set = frozenset(retired)
            divergent_set = frozenset(divergent)
            captured_set = frozenset(
                sequence
                for sequences in capture.group_sequence_ids.values()
                for sequence in sequences
            )
            if (
                len(retired) != len(retired_set)
                or len(divergent) != len(divergent_set)
                or retired_set.intersection(divergent_set)
                or retired_set.union(divergent_set) != captured_set
            ):
                return False
            for sequences in capture.group_sequence_ids.values():
                group_sequences = frozenset(sequences)
                if group_sequences != group_sequences.intersection(
                    retired_set
                ) and group_sequences != group_sequences.intersection(divergent_set):
                    return False
            return self._commit_status_matches_locked(
                publication.refreshed_status,
                capture.binding,
                capture.repository,
                new_head,
            )

        if publication.state == "failed_unchanged":
            return (
                publication.new_head is None
                and not publication.retired_sequence_ids
                and not publication.divergent_sequence_ids
                and publication.recovery_projection is None
                and publication.candidate_seed is None
                and self._commit_status_matches_locked(
                    publication.refreshed_status,
                    capture.binding,
                    capture.repository,
                    capture.head,
                )
            )

        if publication.state == "uncertain":
            return (
                publication.new_head is None
                and not publication.retired_sequence_ids
                and not publication.divergent_sequence_ids
                and publication.refreshed_status is None
                and publication.recovery_projection is not None
                and publication.candidate_seed is None
            )
        return False

    def _push_candidate_seed_matches_capture_locked(
        self,
        seed: PushCandidateSeed,
        capture: CommitAuthorityCapture,
    ) -> bool:
        included_group_ids = tuple(note.group_id for note in seed.included_notes)
        return (
            seed._guarded_commit_identity is capture._guarded_commit_identity
            and seed.binding == capture.binding
            and seed.repository == capture.repository
            and seed.repository_trust_generation
            == capture.repository_trust_generation
            and seed.parent_head == capture.head
            and all(
                group_id in capture.group_sequence_ids
                for group_id in included_group_ids
            )
        )

    def _prepare_push_candidate_locked(
        self,
        capture: CommitAuthorityCapture,
        publication: CommitPublication,
    ) -> _PushCandidate | None:
        seed = publication.candidate_seed
        new_head = publication.new_head
        if seed is None or new_head is None or new_head.object_id is None:
            return None
        parent_oid = capture.head.object_id
        branch = capture.head.branch
        if parent_oid is None or branch is None:
            return None
        try:
            projection = PushCandidateProjection(
                local_branch_ref=branch,
                parent_oid=parent_oid,
                candidate_oid=new_head.object_id,
                subject=seed.subject,
                included_notes=seed.included_notes,
            )
        except PushContractError:
            return None
        return _PushCandidate(
            token=object(),
            generation=self._push_candidate_generation + 1,
            binding=capture.binding,
            repository=capture.repository,
            repository_trust_generation=capture.repository_trust_generation,
            _guarded_commit_identity=seed._guarded_commit_identity,
            candidate=projection,
            change_types=seed.change_types,
            sole_parent_oid=parent_oid,
        )

    def _commit_status_matches_locked(
        self,
        status: SessionGitStatus | None,
        binding: SessionBinding,
        repository: RepositoryIdentity,
        head: HeadIdentity,
    ) -> bool:
        return status is None or (
            status.binding_generation == binding.generation
            and status.status_generation > self._status_generation
            and status.repository == repository
            and status.head == head
        )

    def _publish_commit_status_locked(
        self,
        status: SessionGitStatus | None,
    ) -> None:
        if status is None:
            self._clear_git_status_locked(invalidate_authority=False)
            return
        self._status_generation = status.status_generation
        self._git_status = status

    def _revoke_push_candidate_for_head_mismatch_locked(
        self,
        status: SessionGitStatus,
    ) -> None:
        candidate = self._push_candidate
        head = status.head
        if candidate is None or head is None:
            return
        if (
            status.repository == candidate.repository
            and (
                head.kind != "attached"
                or head.branch != candidate.candidate.local_branch_ref
                or head.object_id != candidate.candidate.candidate_oid
            )
        ):
            self._revoke_push_candidate_locked()

    def _revoke_push_candidate_locked(self) -> None:
        if self._push_candidate is None:
            return
        self._push_candidate = None
        self._push_candidate_generation += 1

    def _invalidate_git_authority_locked(self) -> None:
        self._git_authority_generation += 1

    def _discard_commit_quarantine_locked(self) -> None:
        if self._commit_quarantine is None:
            return
        self._commit_quarantine = None
        self._invalidate_git_authority_locked()

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
            if self._mutation_token is not None:
                raise RuntimeError(
                    "File Notes root commit is blocked by a Git mutation"
                )
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
        self._revoke_push_candidate_locked()
        self._trusted_repository = None
        self._repository_trust_generation += 1
        self._clear_git_status_locked(invalidate_authority=False)
        self._staging_ownership.clear()
        self._commit_quarantine = None
        self._invalidate_git_authority_locked()
        return self._binding

    def _root_change_is_blocked_locked(self, root_key: str) -> bool:
        return (
            self._mutation_token is not None
            and self._binding is not None
            and self._binding.root_key != root_key
        )

    def _clear_git_status_locked(
        self,
        *,
        invalidate_authority: bool = True,
    ) -> None:
        changed = self._git_status is not None
        self._status_generation += 1
        self._git_status = None
        if changed and invalidate_authority:
            self._invalidate_git_authority_locked()

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
