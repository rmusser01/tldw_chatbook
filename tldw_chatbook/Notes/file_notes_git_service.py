"""Pure session grouping and Git status policy for File Notes."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Generator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitContractError,
    CommitIncludedNote,
    CommitOutcome,
    CommitRecoveryProjection,
    CommitReviewChangeType,
    CommitReviewHandle,
    CommitReviewProjection,
    CommitReviewResult,
    GitIdentity,
    RawCommitObject,
    RawStagedDeltaEntry,
    normalize_commit_message,
    parse_git_identity,
    parse_raw_commit_object,
    parse_raw_staged_delta,
)
from tldw_chatbook.Notes.file_notes_git_push import (
    PushAuthorizationHandle,
    PushContractError,
    PushDestinationProjection,
    PushDestinationPolicyResult,
    PushIncludedNote,
    PushOutcomeProjection,
    PushOutcomeState,
    PushRecoveryHandle,
    PushRecoveryProjection,
    PushReviewHandle,
    PushReviewProjection,
    RemoteRefObservation,
    TransportAdmission,
    _GitConfigFact,
    _ResolvedPushConfiguration,
    parse_ls_remote_refs,
    parse_push_porcelain,
    push_outcome_copy,
    push_recovery_copy,
    _push_destination_policy_result,
    _resolve_push_configuration,
)
from tldw_chatbook.Notes.file_notes_git_network import (
    NetworkConfigAuthorization,
    NetworkContextFactory,
    NetworkContextError,
    NetworkContextLease,
    NetworkGitExecutionContext,
    SourceObjectDirectoryAuthorization,
    _authorize_network_config_snapshot,
    _authorize_source_object_directory,
)
from tldw_chatbook.Notes.git_process_containment import (
    AsyncChildProcess,
    OwnedProcessTree,
    ProcessTreeAdmissionError,
    ProcessTreeControl,
    ProcessTreeController,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitAuthorityCapture,
    CommitPublication,
    CommitRecoveryCapability,
    FileNotesSessionOwner,
    FileSystemIdentity,
    GitMutationLease,
    GitStatusLease,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    PushCandidateAvailability,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionChangeGroup,
    SessionChangeTopology,
    SessionGitStatus,
    SessionGitRow,
    StagingOwnership,
    PushCandidateSeed,
    _DestinationPolicyCapture,
    _PushCandidateCapture,
    _PushRecoveryCapture,
    _PushReviewCapture,
    coalesce_session_changes,
)

PorcelainKind = Literal[
    "ordinary",
    "rename",
    "unmerged",
    "untracked",
    "ignored",
    "nested_repository",
    "unavailable",
    "error",
]
GitArg = str | bytes
_SettlementValue = TypeVar("_SettlementValue")
DiscoveryState = Literal[
    "ready",
    "not_repository",
    "unavailable",
    "unsupported",
    "unsafe_root",
]
HeadReadFailureKind = Literal["unavailable", "error"]
GitActionState = Literal["success", "blocked", "stale", "error", "uncertain"]
CommitOperationKind = Literal["review", "commit", "recovery"]
PushPreflightState = Literal[
    "review",
    "already_published",
    "blocked",
    "cancelled",
]
PushExecutionState = Literal[
    "blocked",
    "cancelled",
    "already_published",
    "succeeded",
    "failed_no_update_observed",
    "uncertain",
]
RetainedGitChildState = Literal[
    "alive",
    "natural",
    "stop_requested",
    "forced_stop",
    "contained_uncertain",
    "uncertain",
]

_REDIRECTING_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
        "GIT_REPLACE_REF_BASE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_EXEC_PATH",
        "GIT_PREFIX",
        "GIT_CONFIG",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_GLOB_PATHSPECS",
        "GIT_NOGLOB_PATHSPECS",
        "GIT_LITERAL_PATHSPECS",
        "GIT_ICASE_PATHSPECS",
    }
)
_DYNAMIC_CONFIG_ENVIRONMENT_PREFIXES = (
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)
_COMMIT_ONLY_REMOVED_ENVIRONMENT = frozenset(
    {
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_DATE",
        "GIT_EDITOR",
        "GIT_SEQUENCE_EDITOR",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "EDITOR",
        "VISUAL",
        "GIT_OPTIONAL_LOCKS",
    }
)
_LOCAL_PUSH_PROOF_REMOVED_ENVIRONMENT = frozenset(
    {
        *_REDIRECTING_GIT_ENVIRONMENT,
        *_COMMIT_ONLY_REMOVED_ENVIRONMENT,
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_SSH_VARIANT",
        "GIT_PROXY_COMMAND",
        "GIT_HTTP_PROXY_AUTHMETHOD",
        "GIT_ATTR_NOSYSTEM",
        "GIT_ATTR_SOURCE",
        "GCM_INTERACTIVE",
        "SSH_AUTH_SOCK",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)
_LOCAL_PUSH_CONFIG_PATTERN = (
    r"^(branch\..*\.(remote|merge|pushremote)|remote\.pushdefault|"
    r"remote\..*\.(url|pushurl|mirror|push|pushoption|receivepack|vcs|proxy)|"
    r"push\.pushoption|url\..*\.(pushinsteadof|insteadof)|http\..*|"
    r"credential\..*(helper|usehttppath)|core\.sshcommand|"
    r"ssh\.variant|filter\.lfs\..*|"
    r"include\.path|includeif\..*\.path)$"
)
_LOCAL_PUSH_STDOUT_LIMIT_BYTES = 32 * 1024 * 1024
_LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES = 64 * 1024 * 1024
_PUSH_REMOTE_OUTPUT_LIMIT_BYTES = 64 * 1024
_DEFAULT_PUSH_QUERY_TIMEOUT_SECONDS = 30.0
_DEFAULT_PUSH_TIMEOUT_SECONDS = 60.0
_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS = 0.2
DEFAULT_GIT_STDERR_LIMIT_BYTES = 4096
DEFAULT_COMMIT_PROOF_STDOUT_LIMIT_BYTES = 32 * 1024 * 1024
DEFAULT_COMMIT_PROOF_STDERR_LIMIT_BYTES = DEFAULT_GIT_STDERR_LIMIT_BYTES
_GIT_STREAM_CHUNK_BYTES = 64 * 1024
_DISABLED_GIT_BOOLEAN_VALUES = frozenset({b"false", b"no", b"off", b"0"})
_UNCERTAIN_COMMIT_MESSAGE = (
    "Commit may have succeeded. Git actions are disabled until the repository "
    "is checked. Run git status and git log -1, then choose Check again."
)


_RETAINED_CHILD_TOKEN_SECRET = object()


class RetainedGitChildToken:
    """Opaque identity for exactly one still-running uncertain Git child."""

    __slots__ = ()

    def __new__(
        cls,
        secret: object | None = None,
    ) -> RetainedGitChildToken:
        if secret is not _RETAINED_CHILD_TOKEN_SECRET:
            raise TypeError("Retained Git child tokens are created by the runner")
        return super().__new__(cls)


@dataclass(frozen=True, slots=True)
class RetainedGitChildSettlement:
    """Non-sealing observation of one exact retained Git child."""

    state: RetainedGitChildState
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    stop_requested: bool = False
    force_stopped: bool = False
    output_overflow: bool = False
    owned_process_tree: bool = False
    containment_proved: bool = False


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Byte-preserving result from one direct Git child process."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    termination_uncertain: bool = False
    retained_child: RetainedGitChildToken | None = None
    stop_requested: bool = False
    force_stopped: bool = False
    output_overflow: bool = False
    owned_process_tree: bool = False
    containment_proved: bool = False


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Machine-safe repository discovery without granting process trust."""

    state: DiscoveryState
    repository: RepositoryIdentity | None = None
    head: HeadIdentity | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class GitActionResult:
    """Checked result of one retained Git index action."""

    action: Literal["stage", "unstage"]
    state: GitActionState
    requested_group_ids: tuple[int, ...]
    staged_group_ids: tuple[int, ...] = ()
    unstaged_group_ids: tuple[int, ...] = ()
    clean_group_ids: tuple[int, ...] = ()
    blocked_group_ids: tuple[int, ...] = ()
    message: str | None = None


@dataclass(frozen=True, slots=True)
class _StageInspection:
    """Fresh repository facts used by one exact Stage pre/postflight."""

    repository: RepositoryIdentity
    head: HeadIdentity
    groups: tuple[SessionChangeGroup, ...]
    repository_groups: Mapping[int, SessionChangeGroup]
    rows: Mapping[int, SessionGitRow]
    index_sequence: tuple[IndexEntry, ...]
    index_entries: Mapping[str, IndexEntry]
    status_records: tuple[PorcelainRecord, ...]


@dataclass(frozen=True, slots=True)
class _RawGitInspection:
    """Shared fresh HEAD/index/status facts for status and Stage."""

    head: HeadIdentity
    index_entries: tuple[IndexEntry, ...]
    status_records: tuple[PorcelainRecord, ...]


@dataclass(frozen=True, slots=True)
class _RawGitInspectionFailure:
    """Typed shared inspection refusal without UI/action policy."""

    state: Literal["stale", "unavailable", "error", "uncertain"]
    message: str
    head: HeadIdentity | None = None
    revoke_ownership: bool = False


@dataclass(frozen=True, slots=True)
class _CompleteCommitProof:
    """Path-free result of one complete logical-index review proof."""

    expected_tree: str
    index_signature: str
    delta_signature: str
    included_group_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CommitReviewSnapshot:
    """Private immutable single-use evidence retained for confirmation."""

    capture: CommitAuthorityCapture
    proof: _CompleteCommitProof
    message: bytes
    author: GitIdentity
    committer: GitIdentity
    candidate_seed: PushCandidateSeed = field(repr=False)


@dataclass(frozen=True, slots=True)
class _CommitConfirmation:
    """Fresh lease-bound authority after one exact revalidation."""

    capture: CommitAuthorityCapture


@dataclass(frozen=True, slots=True)
class _CommitPostflight:
    """Path-free immediate branch/index proof after one commit child."""

    repository_matches: bool
    local_state_supported: bool
    head: HeadIdentity | None
    index_signature: str | None
    delta_signature: str | None
    tree_object_id: str | None
    raw_commit: RawCommitObject | None


@dataclass(frozen=True, slots=True)
class _CommitRecoveryProof:
    """Path-free immutable facts needed to classify one uncertain commit."""

    binding: SessionBinding
    repository: RepositoryIdentity
    old_head: HeadIdentity
    complete_proof: _CompleteCommitProof
    message: bytes
    author: GitIdentity
    committer: GitIdentity
    candidate_seed: PushCandidateSeed = field(repr=False)


@dataclass(frozen=True, slots=True)
class _UncertainCommitEvidence:
    """Exact process-memory proof retained for one uncertain commit."""

    proof: _CommitRecoveryProof
    recovery_capability: CommitRecoveryCapability | None
    retained_child: RetainedGitChildToken | None
    hooks_directory: Path | None
    mutation_lease: GitMutationLease | None
    termination_known: bool = False
    known_normal_returncode: int | None = None


@dataclass(frozen=True, slots=True)
class _OrphanedCommitLifecycle:
    """Opaque resources retained after commit proof loses its binding."""

    retained_child: RetainedGitChildToken | None
    hooks_directory: Path | None
    mutation_lease: GitMutationLease | None
    termination_known: bool = False


@dataclass(frozen=True, slots=True)
class _HeadReadFailure:
    """Typed failure that cannot be confused with a semantic HEAD state."""

    kind: HeadReadFailureKind
    message: str


@dataclass(frozen=True, slots=True)
class _PushDestinationPolicySnapshot:
    """Service-private exact local proof retained for authorization."""

    owner_capture: _DestinationPolicyCapture
    candidate_capture: _PushCandidateCapture
    configuration: _ResolvedPushConfiguration
    network_configuration: NetworkConfigAuthorization
    source_objects: SourceObjectDirectoryAuthorization
    git_exec_path: str = field(repr=False)
    candidate_tree_oid: str
    included_paths_fingerprint: str


@dataclass(frozen=True, slots=True)
class PushPreflightResult:
    """Sanitized result of one retained authorized remote preflight."""

    state: PushPreflightState
    handle: PushReviewHandle | None = None
    review: PushReviewProjection | None = None
    outcome: PushOutcomeProjection | None = None

    def __post_init__(self) -> None:
        ready = self.state == "review"
        already_published = self.state == "already_published"
        if (
            self.state
            not in {"review", "already_published", "blocked", "cancelled"}
            or ready != (type(self.handle) is PushReviewHandle)
            or ready != (type(self.review) is PushReviewProjection)
            or already_published
            != (type(self.outcome) is PushOutcomeProjection)
        ):
            raise ValueError("Invalid push preflight result")


@dataclass(frozen=True, slots=True)
class PushExecutionResult:
    """Sanitized terminal result of one exact confirmed push attempt."""

    state: PushExecutionState
    outcome: PushOutcomeProjection | None = None

    def __post_init__(self) -> None:
        outcome_state = (
            self.outcome.state
            if type(self.outcome) is PushOutcomeProjection
            else None
        )
        if (
            self.state not in {
                "blocked",
                "cancelled",
                "already_published",
                "succeeded",
                "failed_no_update_observed",
                "uncertain",
            }
            or (self.state in {"blocked", "cancelled"})
            != (self.outcome is None)
            or (
                self.state not in {"blocked", "cancelled"}
                and outcome_state != self.state
            )
        ):
            raise ValueError("Invalid push execution result")


@dataclass(frozen=True, slots=True)
class _PushReviewSnapshot:
    """Service-private context retained by one owner-issued review."""

    binding: SessionBinding
    operation_id: object = field(repr=False)
    policy: _PushDestinationPolicySnapshot = field(repr=False)
    authorization: PushAuthorizationHandle = field(repr=False)
    context: NetworkGitExecutionContext = field(repr=False)
    review_lease: NetworkContextLease = field(repr=False)
    command_policy_fingerprint: str = field(repr=False)
    projection: PushReviewProjection
    consumed_review: _PushReviewCapture | None = field(
        default=None,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class _UnsettledPushPreflight:
    """Fail-closed ownership retained after bounded controller failure."""

    binding: SessionBinding
    retained_child: RetainedGitChildToken = field(repr=False)
    mutation_lease: GitMutationLease = field(repr=False)
    context: NetworkGitExecutionContext = field(repr=False)
    active_lease: NetworkContextLease = field(repr=False)
    authorization: PushAuthorizationHandle = field(repr=False)


@dataclass(frozen=True, slots=True)
class _UncertainPushEvidence:
    """Frozen query-only evidence retained after an uncertain transmission."""

    binding: SessionBinding
    operation_id: int
    recovery_capture: _PushRecoveryCapture = field(repr=False)
    recovery_handle: PushRecoveryHandle | None = field(repr=False)
    candidate_capture: _PushCandidateCapture = field(repr=False)
    endpoint: object = field(repr=False)
    destination: PushDestinationProjection
    parent_oid: str
    candidate_oid: str
    repository_trust_generation: int
    destination_policy_generation: int
    destination_authorization_epoch: int
    context: NetworkGitExecutionContext = field(repr=False)
    context_lease: NetworkContextLease = field(repr=False)
    mutation_lease: GitMutationLease = field(repr=False)
    descendants_terminal: bool
    retained_child: RetainedGitChildToken | None = field(
        default=None,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class _PrivateProofEntry:
    """Retained identity and contents of one allowlisted proof component."""

    identity: FileSystemIdentity
    kind: Literal["directory", "file"]
    mode: int
    owner: int
    size: int
    mtime_ns: int
    ctime_ns: int
    content_digest: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _ExternalProofDirectory:
    """Stable metadata for one read-only directory outside the proof root."""

    identity: FileSystemIdentity
    mode: int
    owner: int
    group: int


class _PrivatePushProofDirectory:
    """Owner-only proof tree with fail-closed exact cleanup.

    The safe-parent and ``0700`` root policy excludes other principals from
    substitution; processes with the same effective UID, and root, are trusted.
    """

    __slots__ = (
        "root",
        "_parent_identity",
        "_repository_device",
        "_entries",
        "_external_directories",
        "_index_path",
        "_closed",
    )

    def __init__(
        self,
        *,
        root: Path,
        parent_identity: FileSystemIdentity,
        repository_device: int,
    ) -> None:
        self.root = root
        self._parent_identity = parent_identity
        self._repository_device = repository_device
        self._entries: dict[Path, _PrivateProofEntry] = {
            root: self._capture_entry(root, "directory")
        }
        self._external_directories: dict[
            Path,
            _ExternalProofDirectory,
        ] = {}
        self._index_path: Path | None = None
        self._closed = False

    def __enter__(self) -> _PrivatePushProofDirectory:
        return self

    def __exit__(self, *_exception: object) -> None:
        if not self._closed:
            self.cleanup()

    def create_directory(self, relative_path: str) -> Path:
        """Create and retain one owner-only directory."""
        path = self._new_path(relative_path)
        if not self.validate():
            raise OSError("Private proof tree changed before directory creation")
        try:
            path.mkdir(mode=0o700)
            path.chmod(0o700)
            self._entries[path] = self._capture_entry(path, "directory")
        except OSError:
            _remove_private_hooks_directory(path)
            raise
        if not self.validate():
            raise OSError("Private proof directory identity changed")
        return path

    def create_file(self, relative_path: str, payload: bytes) -> Path:
        """Create and retain one owner-only file exclusively without symlinks."""
        if len(payload) > _LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES:
            raise OSError("Private proof seed exceeds the bounded file limit")
        path = self._new_path(relative_path)
        if not self.validate():
            raise OSError("Private proof tree changed before file creation")
        file_descriptor: int | None = None
        created_identity: FileSystemIdentity | None = None
        try:
            file_descriptor = os.open(
                path,
                (
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC
                ),
                0o600,
            )
            created_identity = _identity_from_stat_result(
                os.fstat(file_descriptor)
            )
            os.fchmod(file_descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(file_descriptor, view)
                if written <= 0:
                    raise OSError("Private proof seed write did not progress")
                view = view[written:]
        except OSError:
            if file_descriptor is not None:
                os.close(file_descriptor)
            self._unlink_exact_file(path, created_identity)
            raise
        os.close(file_descriptor)
        self._entries[path] = self._capture_entry(path, "file")
        if not self.validate():
            raise OSError("Private proof file identity changed")
        return path

    def reserve_index(self, relative_path: str) -> Path:
        """Reserve the only path Git may create in the private proof tree."""
        if self._index_path is not None:
            raise OSError("Private proof index is already reserved")
        path = self._new_path(relative_path)
        self._index_path = path
        return path

    def capture_index(self) -> bool:
        """Pin and seal the exact index emitted by one settled read-tree."""
        path = self._index_path
        if (
            self._closed
            or path is None
            or path in self._entries
        ):
            return False
        file_descriptor: int | None = None
        safe_identity: FileSystemIdentity | None = None
        try:
            file_descriptor = os.open(
                path,
                (
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC
                ),
            )
            metadata = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_dev != self._repository_device
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_size < 0
            ):
                raise OSError("Private proof index identity is unsafe")
            safe_identity = _identity_from_stat_result(metadata)
            if metadata.st_size > _LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES:
                raise OSError("Private proof index exceeds the bounded limit")
            os.fchmod(file_descriptor, 0o600)
            metadata = os.fstat(file_descriptor)
            digest = _bounded_file_descriptor_digest(
                file_descriptor,
                metadata.st_size,
            )
            entry = self._entry_from_metadata(metadata, "file", digest)
            path_metadata = path.stat(follow_symlinks=False)
            if (
                path.resolve(strict=True) != path
                or self._entry_from_metadata(
                    path_metadata,
                    "file",
                    digest,
                )
                != entry
            ):
                raise OSError("Private proof index path changed")
        except (OSError, RuntimeError):
            if file_descriptor is not None:
                os.close(file_descriptor)
            self._unlink_exact_file(path, safe_identity)
            return False
        os.close(file_descriptor)
        self._entries[path] = entry
        return self.validate()

    def track_external_directory(
        self,
        path: Path,
        identity: FileSystemIdentity,
    ) -> bool:
        """Pin one canonical object directory accessed read-only by Git."""
        try:
            snapshot = self._capture_external_directory(path)
        except (OSError, RuntimeError):
            return False
        if snapshot.identity != identity:
            return False
        self._external_directories[path] = snapshot
        return True

    def validate(self) -> bool:
        """Validate parent, topology, identities, and retained metadata."""
        if self._closed:
            return False
        try:
            return (
                self._private_tree_matches()
                and all(
                    self._external_directory_matches(path, snapshot)
                    for path, snapshot
                    in self._external_directories.items()
                )
            )
        except (KeyError, OSError, RuntimeError):
            return False

    def cleanup(self) -> bool:
        """Remove only the fully validated allowlisted tree, never recursively."""
        if self._closed:
            return False
        if not self._private_tree_matches():
            self._closed = True
            return False
        try:
            ordered = sorted(
                self._entries.items(),
                key=lambda item: len(item[0].parts),
                reverse=True,
            )
            for path, entry in ordered:
                if entry.kind != "file":
                    continue
                if not self._entry_matches(path, entry):
                    raise OSError("Private proof file changed before cleanup")
                path.unlink()
            for path, entry in ordered:
                if entry.kind != "directory":
                    continue
                if (
                    not self._entry_matches(path, entry)
                    or any(path.iterdir())
                ):
                    raise OSError(
                        "Private proof directory changed before cleanup"
                    )
                path.rmdir()
            if _path_present(self.root):
                raise OSError("Private proof root survived cleanup")
            if not self._parent_matches():
                raise OSError("Private proof parent changed during cleanup")
        except OSError:
            self._closed = True
            return False
        self._closed = True
        return True

    def _private_tree_matches(self) -> bool:
        try:
            return (
                self._parent_matches()
                and self.root.resolve(strict=True) == self.root
                and all(
                    self._entry_matches(path, entry)
                    for path, entry in self._entries.items()
                )
                and self._shape_matches()
            )
        except (KeyError, OSError, RuntimeError):
            return False

    def _new_path(
        self,
        relative_path: str,
    ) -> Path:
        if self._closed or not relative_path:
            raise OSError("Private proof directory is closed")
        candidate = Path(relative_path)
        parts = candidate.parts
        path = self.root.joinpath(*parts)
        if (
            candidate.is_absolute()
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
            or path in self._entries
            or path == self._index_path
            or path.parent not in self._entries
            or self._entries[path.parent].kind != "directory"
            or _path_present(path)
        ):
            raise OSError("Private proof path is invalid or already reserved")
        return path

    def _capture_entry(
        self,
        path: Path,
        kind: Literal["directory", "file"],
    ) -> _PrivateProofEntry:
        metadata = path.stat(follow_symlinks=False)
        if path.resolve(strict=True) != path:
            raise OSError("Private proof component metadata is unsafe")
        digest: bytes | None = None
        if kind == "file":
            self._entry_from_metadata(metadata, kind, b"")
            file_descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                descriptor_metadata = os.fstat(file_descriptor)
                if (
                    _identity_from_stat_result(descriptor_metadata)
                    != _identity_from_stat_result(metadata)
                ):
                    raise OSError("Private proof file identity changed")
                digest = _bounded_file_descriptor_digest(
                    file_descriptor,
                    descriptor_metadata.st_size,
                )
            finally:
                os.close(file_descriptor)
            if digest is None:
                raise OSError("Private proof file digest is unavailable")
        return self._entry_from_metadata(metadata, kind, digest)

    def _entry_from_metadata(
        self,
        metadata: os.stat_result,
        kind: Literal["directory", "file"],
        digest: bytes | None,
    ) -> _PrivateProofEntry:
        required_mode = 0o700 if kind == "directory" else 0o600
        if (
            metadata.st_dev != self._repository_device
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != required_mode
            or (
                kind == "directory"
                and not stat.S_ISDIR(metadata.st_mode)
            )
            or (
                kind == "file"
                and (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_size < 0
                    or metadata.st_size
                    > _LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES
                    or digest is None
                )
            )
        ):
            raise OSError("Private proof component metadata is unsafe")
        return _PrivateProofEntry(
            _identity_from_stat_result(metadata),
            kind,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_uid,
            metadata.st_size if kind == "file" else 0,
            metadata.st_mtime_ns if kind == "file" else 0,
            metadata.st_ctime_ns if kind == "file" else 0,
            digest,
        )

    def _entry_matches(
        self,
        path: Path,
        entry: _PrivateProofEntry,
    ) -> bool:
        try:
            metadata = path.stat(follow_symlinks=False)
            return (
                path.resolve(strict=True) == path
                and self._entry_from_metadata(
                    metadata,
                    entry.kind,
                    entry.content_digest,
                )
                == entry
            )
        except (OSError, RuntimeError):
            return False

    def _shape_matches(self) -> bool:
        expected: dict[Path, set[str]] = {
            path: set()
            for path, entry in self._entries.items()
            if entry.kind == "directory"
        }
        for path in self._entries:
            if path != self.root:
                expected[path.parent].add(path.name)
        for path, names in expected.items():
            found: set[str] = set()
            with os.scandir(path) as children:
                for child in children:
                    if child.name not in names:
                        return False
                    found.add(child.name)
            if found != names:
                return False
        return True

    @staticmethod
    def _capture_external_directory(
        path: Path,
    ) -> _ExternalProofDirectory:
        metadata = path.stat(follow_symlinks=False)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid not in {0, os.geteuid()}
            or mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise OSError("External proof directory metadata is unsafe")
        return _ExternalProofDirectory(
            _identity_from_stat_result(metadata),
            mode,
            metadata.st_uid,
            metadata.st_gid,
        )

    @staticmethod
    def _external_directory_matches(
        path: Path,
        snapshot: _ExternalProofDirectory,
    ) -> bool:
        return (
            _PrivatePushProofDirectory._capture_external_directory(path)
            == snapshot
        )

    @staticmethod
    def _unlink_exact_file(
        path: Path,
        identity: FileSystemIdentity | None,
    ) -> None:
        if identity is None:
            return
        try:
            metadata = path.stat(follow_symlinks=False)
            if (
                _identity_from_stat_result(metadata) == identity
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and metadata.st_nlink == 1
            ):
                path.unlink()
        except OSError:
            pass

    def _parent_matches(self) -> bool:
        parent = self.root.parent
        try:
            path_metadata = parent.stat(follow_symlinks=False)
        except (OSError, RuntimeError):
            return False
        return (
            parent.resolve(strict=True) == parent
            and _identity_from_stat_result(path_metadata)
            == self._parent_identity
            and stat.S_ISDIR(path_metadata.st_mode)
            and _hooks_parent_is_safe(
                parent,
                self._repository_device,
            )
        )


GitStatusAdmissionReason = Literal[
    "untrusted",
    "mutation_active",
    "stale_binding",
    "shutdown",
    "status_active",
]
GitMutationAdmissionReason = Literal[
    "authorization_required",
    "invalid_capability",
    "untrusted",
    "mutation_active",
    "recovery_not_ready",
    "recovery_required",
    "transition_active",
    "stale_binding",
    "shutdown",
]


class GitStatusAdmissionError(RuntimeError):
    """Typed synchronous refusal before any worktree-aware child starts."""

    def __init__(
        self,
        reason: GitStatusAdmissionReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class GitMutationAdmissionError(RuntimeError):
    """Typed synchronous refusal before a retained mutation task exists."""

    def __init__(
        self,
        reason: GitMutationAdmissionReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class GitShutdownAffinityError(RuntimeError):
    """Retryable refusal when active shutdown starts off its owning loop."""

    retryable_shutdown = True


class GitRunCancelled(asyncio.CancelledError):
    """Caller cancellation with either a retained child or terminal result."""

    def __init__(
        self,
        *,
        retained_child: RetainedGitChildToken | None = None,
        result: GitCommandResult | None = None,
    ) -> None:
        if (retained_child is None) == (result is None):
            raise ValueError(
                "Cancellation requires exactly one retained child or result"
            )
        super().__init__(
            (
                "Git command waiter cancelled; child retained"
                if retained_child is not None
                else "Git command waiter cancelled after child settlement"
            )
        )
        self.retained_child = retained_child
        self.result = result


class GitRunCancellationRejected(GitRunCancelled):
    """Caller cancellation rejected because child admission is active."""

    cancellation_rejected = True

    def __init__(self, retained_child: RetainedGitChildToken) -> None:
        super().__init__(retained_child=retained_child)
        self.args = (
            "Git command cancellation rejected; admitted child remains active",
        )


class _PushContainmentUnproved(RuntimeError):
    """Internal transfer of an unproved child into fail-closed ownership."""

    def __init__(self, retained_child: RetainedGitChildToken) -> None:
        super().__init__("Push preflight containment remains unproved")
        self.retained_child = retained_child


class GitProcessRunner(Protocol):
    """Injectable direct-argv child-process boundary."""

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
        owned_process_tree: bool = False,
    ) -> GitCommandResult:
        """Run one command without accepting a shell option."""

    def shutdown(self) -> Awaitable[bool] | None:
        """Seal admission and return retained finite child settlement."""

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Read one exact retained child without changing its lifecycle."""

    def claim_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Protect exact terminal evidence from global shutdown cleanup."""

    async def settle_retained_child(
        self,
        token: RetainedGitChildToken,
        *,
        timeout: float = 0.0,
    ) -> RetainedGitChildSettlement:
        """Wait boundedly for one exact retained child without stopping it."""

    def release_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Release terminal evidence; refuse live or uncertain children."""


@dataclass(frozen=True, slots=True)
class _ImmediateSettlement(Generic[_SettlementValue]):
    """Reusable awaitable for shutdown paths with no asynchronous work."""

    value: _SettlementValue

    def __await__(self) -> Generator[None, None, _SettlementValue]:
        if False:
            yield None
        return self.value


@dataclass(frozen=True, slots=True)
class _RetainedSettlement(Generic[_SettlementValue]):
    """Cancellation-safe reusable view of an internally retained task."""

    task: asyncio.Task[_SettlementValue]

    def __await__(self) -> Generator[Any, None, _SettlementValue]:
        return asyncio.shield(self.task).__await__()


@dataclass(frozen=True, slots=True)
class RetainedCommitOperation:
    """Cancellation-safe observation of one exact guarded commit operation."""

    binding: SessionBinding
    kind: CommitOperationKind
    _settlement: asyncio.Task[CommitReviewResult | CommitOutcome]
    _child_started_signal: asyncio.Future[bool] | None = None

    @property
    def settled(self) -> bool:
        """Return whether the retained service operation has settled."""
        return self._settlement.done()

    @property
    def child_started(self) -> bool:
        """Return whether the exact branch-mutating child has started."""
        signal = self._child_started_signal
        return (
            signal is not None
            and signal.done()
            and not signal.cancelled()
            and signal.result()
        )

    async def wait(self) -> CommitReviewResult | CommitOutcome:
        """Await settlement without allowing an observer to cancel it."""
        return await asyncio.shield(self._settlement)

    async def wait_child_started(self) -> bool:
        """Await the exact child-start decision without owning its future."""
        signal = self._child_started_signal
        if signal is None:
            return False
        return await asyncio.shield(signal)


@dataclass(frozen=True, slots=True, eq=False)
class RetainedPushOperation:
    """Cancellation-safe observation of one exact guarded-push phase."""

    binding: SessionBinding
    operation_id: int
    kind: Literal["local_proof", "preflight", "push", "recovery"]
    candidate: PushCandidateAvailability
    _settlement: asyncio.Task[
        PushDestinationPolicyResult
        | PushPreflightResult
        | PushExecutionResult
        | PushRecoveryProjection
    ] = field(repr=False)
    _child_started_signal: asyncio.Future[bool] | None = field(
        default=None,
        repr=False,
    )

    @property
    def settled(self) -> bool:
        """Return whether the retained service operation has settled."""
        return self._settlement.done()

    async def wait(
        self,
    ) -> (
        PushDestinationPolicyResult
        | PushPreflightResult
        | PushExecutionResult
        | PushRecoveryProjection
    ):
        """Await settlement without allowing an observer to cancel it."""
        return await asyncio.shield(self._settlement)

    @property
    def child_started(self) -> bool:
        """Return whether the exact mutating child actually started."""
        signal = self._child_started_signal
        return (
            signal is not None
            and signal.done()
            and not signal.cancelled()
            and signal.result()
        )

    async def wait_child_started(self) -> bool:
        """Await the exact push-child admission decision."""
        signal = self._child_started_signal
        if signal is None:
            return False
        return await asyncio.shield(signal)


@dataclass(slots=True)
class _RetainedChildRecord:
    """One operation record spanning spawn, child, and terminal settlement."""

    token: RetainedGitChildToken
    ready: asyncio.Event
    admission_outcome: asyncio.Future[bool]
    process: AsyncChildProcess | None = None
    process_tree: OwnedProcessTree | None = None
    owned_process_tree: bool = False
    containment_proved: bool = False
    communication: asyncio.Task[tuple[bytes, bytes]] | None = None
    stop_requested: bool = False
    force_stopped: bool = False
    timed_out: bool = False
    exposed: bool = False
    claimed: bool = False
    released: bool = False
    output_overflow: bool = False
    admission_state: Literal[
        "queued",
        "admitting",
        "started",
        "not_started",
    ] = "queued"
    terminate_sent: bool = False
    kill_sent: bool = False
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    owned_task: asyncio.Task[GitCommandResult] | None = None
    settlement: RetainedGitChildSettlement | None = None


def build_git_environment(
    ambient: Mapping[str, str] | None = None,
    *,
    for_status: bool = False,
    stable_locale: bool = False,
) -> dict[str, str]:
    """Build a sanitized environment for a direct Git child process.

    Args:
        ambient: Source environment. Defaults to the current process
            environment.
        for_status: Whether to disable optional Git index locks for a
            read-only status command.
        stable_locale: Whether to force the stable ``C`` locale.

    Returns:
        A copied environment without Git redirection or dynamic-config
        variables and with interactive prompting disabled.
    """
    source = os.environ if ambient is None else ambient
    environment = {
        key: value
        for key, value in source.items()
        if (
            key not in _REDIRECTING_GIT_ENVIRONMENT
            and key != "GIT_CONFIG_COUNT"
            and not key.startswith(_DYNAMIC_CONFIG_ENVIRONMENT_PREFIXES)
        )
    }
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if stable_locale:
        environment["LC_ALL"] = "C"
    if for_status:
        environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def build_local_push_proof_environment(
    ambient: Mapping[str, str] | None = None,
    *,
    index_file: str | None = None,
) -> dict[str, str]:
    """Build the isolated environment for local-only push policy proof.

    Args:
        ambient: Source environment. Defaults to the current process.
        index_file: Optional service-owned temporary index for tree attributes.

    Returns:
        A copied environment stripped of repository, config, transport,
        credential, proxy, editor, pager, and optional-write redirects.
    """
    source = os.environ if ambient is None else ambient
    environment = {
        key: value
        for key, value in source.items()
        if (
            key not in _LOCAL_PUSH_PROOF_REMOVED_ENVIRONMENT
            and key != "GIT_CONFIG_COUNT"
            and not key.startswith(_DYNAMIC_CONFIG_ENVIRONMENT_PREFIXES)
        )
    }
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "",
            "PAGER": "",
            "GIT_EDITOR": "",
            "GIT_SEQUENCE_EDITOR": "",
            "EDITOR": "",
            "VISUAL": "",
        }
    )
    if index_file is not None:
        if not index_file or "\0" in index_file:
            raise ValueError("Invalid local push proof index")
        environment["GIT_INDEX_FILE"] = index_file
    return environment


def build_commit_environment(
    ambient: Mapping[str, str] | None = None,
    *,
    author: GitIdentity | None = None,
    committer: GitIdentity | None = None,
    read_only: bool = False,
) -> dict[str, str]:
    """Build the isolated environment used only by guarded commit work."""
    source = os.environ if ambient is None else ambient
    environment = build_git_environment(
        {
            key: value
            for key, value in source.items()
            if key not in _COMMIT_ONLY_REMOVED_ENVIRONMENT
        },
        stable_locale=True,
    )
    environment.update(
        {
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
            "GIT_ASKPASS": "true",
            "SSH_ASKPASS": "true",
            "EDITOR": "true",
            "VISUAL": "true",
        }
    )
    if read_only:
        environment["GIT_OPTIONAL_LOCKS"] = "0"
    if (author is None) != (committer is None):
        raise ValueError("Commit identities must be bound together")
    if author is not None and committer is not None:
        environment.update(
            {
                "GIT_AUTHOR_NAME": author.name,
                "GIT_AUTHOR_EMAIL": author.email,
                "GIT_COMMITTER_NAME": committer.name,
                "GIT_COMMITTER_EMAIL": committer.email,
            }
        )
    return environment


def build_commit_argv(
    git_executable: str,
    hooks_directory: str,
) -> tuple[GitArg, ...]:
    """Build the exact reviewed branch child without starting it."""
    return (
        git_executable,
        "--no-replace-objects",
        "-c",
        f"core.hooksPath={hooks_directory}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "i18n.commitEncoding=UTF-8",
        "commit",
        "--no-gpg-sign",
        "--cleanup=verbatim",
        "-F",
        "-",
    )


def build_commit_stdin(subject: str, body: str = "") -> bytes:
    """Return the exact validated bytes later supplied to the commit child."""
    return normalize_commit_message(subject, body)


def build_commit_index_argv(git_executable: str) -> tuple[GitArg, ...]:
    """Build the complete semantic-index command used only by commit proof."""
    return (
        git_executable,
        "--no-replace-objects",
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "-z",
        "--stage",
        "-v",
        "--",
    )


def build_commit_delta_argv(
    git_executable: str,
    old_head: str,
) -> tuple[GitArg, ...]:
    """Build the complete staged-delta command against one captured HEAD."""
    return (
        git_executable,
        "--no-replace-objects",
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "diff.renames=false",
        "diff-index",
        "--cached",
        "--raw",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        old_head,
        "--",
    )


def build_commit_worktree_argv(
    git_executable: str,
    repository_paths: Sequence[bytes],
) -> tuple[GitArg, ...]:
    """Build one complete owned-path worktree freshness inspection."""
    return (
        git_executable,
        "--no-replace-objects",
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "status.renames=false",
        "-c",
        "diff.renames=false",
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--no-renames",
        "--",
        *repository_paths,
    )


def complete_commit_delta_matches_ownership(
    delta: Sequence[RawStagedDeltaEntry],
    ownership_entries: Mapping[
        int,
        tuple[Mapping[str, IndexEntry | None], Mapping[str, IndexEntry | None]],
    ],
) -> bool:
    """Compare one complete raw staged delta to the exact owned union."""
    expected = _expected_owned_delta(ownership_entries)
    if not expected or len(delta) != len(expected):
        return False
    actual: dict[bytes, tuple[str, str, str, str, str]] = {}
    for entry in delta:
        if entry.path in actual:
            return False
        actual[entry.path] = (
            entry.old_mode,
            entry.new_mode,
            entry.old_object_id,
            entry.new_object_id,
            entry.status,
        )
    return actual == expected


def build_status_argv(
    git_executable: str,
    repository_paths: Sequence[bytes],
) -> tuple[GitArg, ...]:
    """Build the literal, NUL-safe, no-renames session status command."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "status.renames=false",
        "-c",
        "diff.renames=false",
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--no-renames",
        "--",
        *repository_paths,
    )


def build_index_argv(git_executable: str) -> tuple[GitArg, ...]:
    """Build one complete, NUL-safe semantic index read."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "-z",
        "--stage",
        "-v",
        "--",
    )


def build_stage_argv(
    git_executable: str,
    repository_paths: Sequence[bytes],
) -> tuple[GitArg, ...]:
    """Build one exact fail-fast literal path-scoped Stage command."""
    return (
        git_executable,
        "--literal-pathspecs",
        "-c",
        "add.ignoreErrors=false",
        "add",
        "--all",
        "--",
        *repository_paths,
    )


def build_unstage_argv(git_executable: str) -> tuple[GitArg, ...]:
    """Build one exact index-only saved-baseline Unstage command."""
    return (
        git_executable,
        "update-index",
        "-z",
        "--index-info",
    )


def build_update_index_payload(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> bytes:
    """Build exact conflict removals followed by saved baseline records."""
    conflict_records, baseline_records = _update_index_records(
        ownership,
        current_index_entries,
    )
    return b"".join((*conflict_records, *baseline_records))


def _update_index_records(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    object_id_lengths = {
        len(entry.object_id)
        for entry in (
            *(
                baseline.entry
                for baseline in ownership.original_baselines.values()
                if baseline.entry is not None
            ),
            *(
                entry
                for entry in ownership.post_stage_entries.values()
                if entry is not None
            ),
        )
    }
    if len(object_id_lengths) != 1 or not object_id_lengths.issubset({40, 64}):
        raise ValueError("Unstage ownership has an invalid object ID width")
    zero_oid = b"0" * object_id_lengths.pop()

    conflict_paths = sorted(
        path
        for path in compute_unstage_closure(
            ownership.original_baselines,
            current_index_entries,
        )
        if path not in ownership.original_baselines
    )
    conflict_records = tuple(
        b"0 " + zero_oid + b"\t" + _index_info_path_bytes(path) + b"\0"
        for path in conflict_paths
    )
    baseline_records: list[bytes] = []
    for path, baseline in sorted(ownership.original_baselines.items()):
        raw_path = _index_info_path_bytes(path)
        entry = baseline.entry
        if entry is None:
            baseline_records.append(
                b"0 " + zero_oid + b"\t" + raw_path + b"\0"
            )
            continue
        if (
            entry.path != path
            or entry.stage != 0
            or entry.semantic_flags
            or len(entry.mode) != 6
            or any(character not in "01234567" for character in entry.mode)
            or len(entry.object_id) != len(zero_oid)
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in entry.object_id
            )
        ):
            raise ValueError("Unstage baseline is not an exact stage-0 entry")
        baseline_records.append(
            entry.mode.encode("ascii")
            + b" "
            + entry.object_id.lower().encode("ascii")
            + b" 0\t"
            + raw_path
            + b"\0"
        )
    return conflict_records, tuple(baseline_records)


def _index_info_path_bytes(path: str) -> bytes:
    if (
        not path
        or path.startswith("/")
        or "\0" in path
        or any(component in {"", ".", ".."} for component in path.split("/"))
    ):
        raise ValueError("Unstage path is unsafe")
    return os.fsencode(path)


def _build_combined_update_index_payload(
    ownership: Sequence[StagingOwnership],
    current_index_entries: Mapping[str, IndexEntry],
) -> bytes:
    conflict_records: list[bytes] = []
    baseline_records: list[bytes] = []
    seen_conflicts: set[bytes] = set()
    seen_baselines: set[str] = set()
    for item in ownership:
        conflicts, baselines = _update_index_records(item, current_index_entries)
        for record in conflicts:
            if record not in seen_conflicts:
                seen_conflicts.add(record)
                conflict_records.append(record)
        for path in item.original_baselines:
            if path in seen_baselines:
                raise ValueError("Unstage groups overlap in the Git index")
            seen_baselines.add(path)
        baseline_records.extend(baselines)
    return b"".join((*sorted(conflict_records), *sorted(baseline_records)))


def _baseline_matches_index(
    ownership: StagingOwnership,
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    if any(
        current_index_entries.get(path) != baseline.entry
        for path, baseline in ownership.original_baselines.items()
    ):
        return False
    inserted_baseline_paths = tuple(
        path
        for path, baseline in ownership.original_baselines.items()
        if baseline.entry is not None
    )
    return all(
        current_index_entries.get(path) is None
        for path in ownership.post_stage_entries
        if (
            path not in ownership.original_baselines
            and any(
                _paths_overlap(path, baseline_path)
                for baseline_path in inserted_baseline_paths
            )
        )
    )


def sanitize_git_stderr(
    payload: bytes,
    *,
    limit: int = DEFAULT_GIT_STDERR_LIMIT_BYTES,
) -> str:
    """Bound diagnostics and make terminal control bytes visible."""
    if limit <= 0:
        return ""
    decoded = os.fsdecode(payload)
    parts: list[str] = []
    size = 0
    replacements = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
    for character in decoded:
        codepoint = ord(character)
        if character in replacements:
            piece = replacements[character]
        elif codepoint < 32 or 127 <= codepoint <= 159:
            piece = f"\\x{codepoint:02x}"
        elif 0xDC80 <= codepoint <= 0xDCFF:
            piece = f"\\x{codepoint - 0xDC00:02x}"
        else:
            piece = character
        encoded = piece.encode("utf-8", "surrogateescape")
        if size + len(encoded) > limit:
            break
        parts.append(piece)
        size += len(encoded)
    return "".join(parts)


class AsyncGitProcessRunner:
    """Own direct-argv asyncio Git children without a shell boundary."""

    def __init__(
        self,
        *,
        terminate_timeout: float = 0.25,
        kill_timeout: float = 0.25,
        stderr_limit: int = DEFAULT_GIT_STDERR_LIMIT_BYTES,
        process_tree_controller: ProcessTreeControl | None = None,
    ) -> None:
        self._sealed = False
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._stderr_limit = stderr_limit
        self._process_tree_controller = (
            process_tree_controller or ProcessTreeController()
        )
        self._processes: set[AsyncChildProcess] = set()
        self._run_tasks: set[asyncio.Task[object]] = set()
        self._retained_children: dict[
            RetainedGitChildToken,
            _RetainedChildRecord,
        ] = {}
        self._shutdown_event: asyncio.Event | None = None
        self._shutdown_settlement: Awaitable[bool] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
        owned_process_tree: bool = False,
    ) -> GitCommandResult:
        """Execute one direct child and preserve all standard streams as bytes."""
        if any(
            limit is not None and limit < 0
            for limit in (stdout_limit, stderr_limit)
        ):
            raise ValueError("Git output limits cannot be negative")
        if self._sealed:
            return GitCommandResult(127, b"", b"Git runner is shut down")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._shutdown_event = asyncio.Event()
        elif self._loop is not loop:
            raise RuntimeError("Git runner cannot span multiple event loops")
        assert self._shutdown_event is not None
        token = RetainedGitChildToken(_RETAINED_CHILD_TOKEN_SECRET)
        record = _RetainedChildRecord(
            token,
            asyncio.Event(),
            loop.create_future(),
            owned_process_tree=owned_process_tree,
        )
        self._retained_children[token] = record
        run_task = loop.create_task(
            self._run_owned_command(
                record,
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                on_spawn=on_spawn,
            )
        )
        record.owned_task = run_task
        self._run_tasks.add(run_task)
        run_task.add_done_callback(
            lambda task: self._run_task_completed(task, record)
        )
        try:
            result = await asyncio.shield(run_task)
            if result.retained_child is None:
                self._release_unexposed_record(record)
            else:
                record.exposed = True
            return result
        except asyncio.CancelledError:
            if cancel_before_spawn and record.owned_process_tree:
                if record.admission_state == "queued":
                    run_task.cancel()
                    await self._await_cancelled_run_task(run_task)
                    self._discard_record(record)
                    raise GitRunCancelled(
                        result=GitCommandResult(1, b"", b"")
                    ) from None
                started = await self._await_admission_outcome(record)
                if started and not run_task.done():
                    record.exposed = True
                    raise GitRunCancellationRejected(token) from None
                try:
                    result = await self._await_cancelled_run_task(run_task)
                except BaseException:
                    result = GitCommandResult(1, b"", b"")
                if result is None:
                    result = GitCommandResult(1, b"", b"")
                if started and result.retained_child is not None:
                    record.exposed = True
                    raise GitRunCancellationRejected(
                        result.retained_child
                    ) from None
                if result.retained_child is None:
                    self._release_unexposed_record(record)
                else:
                    record.exposed = True
                raise GitRunCancelled(result=result) from None
            if cancel_before_spawn and record.process is None:
                if not record.owned_process_tree:
                    run_task.cancel()
                    await asyncio.gather(run_task, return_exceptions=True)
                    if record.process is None:
                        self._discard_record(record)
                        raise GitRunCancelled(
                            result=GitCommandResult(1, b"", b"")
                        ) from None
            if record.communication is not None:
                settlement = self._read_record(record)
                if settlement.state not in {"alive", "uncertain"}:
                    result = self._result_from_settlement(
                        settlement,
                        timed_out=record.timed_out,
                        stop_requested=record.stop_requested,
                        force_stopped=record.force_stopped,
                    )
                    self._discard_record(record)
                    raise GitRunCancelled(result=result) from None
            record.exposed = True
            raise GitRunCancelled(retained_child=token) from None
        except BaseException:
            if record.process is None:
                self._retained_children.pop(token, None)
            else:
                settlement = self._read_record(record)
                if settlement.state in {"alive", "uncertain"}:
                    record.exposed = True
                else:
                    self._discard_record(record)
            raise

    async def _run_owned_command(
        self,
        record: _RetainedChildRecord,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None,
        timeout: float | None,
        stdout_limit: int | None,
        stderr_limit: int | None,
        on_spawn: Callable[[], None] | None,
    ) -> GitCommandResult:
        """Run one child in a runner-owned task immune to caller cancellation."""
        assert self._shutdown_event is not None
        if record.owned_process_tree:
            record.admission_state = "admitting"
        shutdown_waiter: asyncio.Task[bool] | None = None
        callback_error: BaseException | None = None
        try:
            admission_failed = False
            if record.owned_process_tree:
                try:
                    process_tree = await self._process_tree_controller.spawn(
                        *argv,
                        cwd=cwd,
                        environment=environment,
                        stdin=stdin is not None,
                    )
                except ProcessTreeAdmissionError as error:
                    process_tree = error.tree
                    admission_failed = True
                record.process_tree = process_tree
                process = process_tree.process
                record.process = process
                if admission_failed:
                    record.admission_state = "not_started"
                    self._publish_admission_outcome(
                        record,
                        started=False,
                    )
                else:
                    if on_spawn is not None:
                        try:
                            on_spawn()
                        except BaseException as error:
                            callback_error = error
                    record.admission_state = "started"
                    self._publish_admission_outcome(
                        record,
                        started=True,
                    )
            else:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=cwd,
                    env=dict(environment),
                    stdin=(
                        asyncio.subprocess.PIPE
                        if stdin is not None
                        else asyncio.subprocess.DEVNULL
                    ),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            self._processes.add(process)
            record.process = process
            if stdout_limit is None and stderr_limit is None:
                communication = asyncio.create_task(
                    self._communicate_unbounded(process, stdin)
                )
            else:
                communication = asyncio.create_task(
                    self._communicate_bounded(
                        record,
                        process,
                        stdin,
                        stdout_limit=stdout_limit,
                        stderr_limit=stderr_limit,
                    )
                )
            record.communication = communication
            record.ready.set()
            if admission_failed:
                terminated = await self._stop_record(record)
                if terminated and not communication.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(communication),
                            timeout=self._kill_timeout,
                        )
                    except TimeoutError:
                        terminated = False
                settlement = self._read_record(record)
                if terminated and settlement.state not in {"alive", "uncertain"}:
                    return self._result_from_settlement(
                        settlement,
                        stop_requested=record.stop_requested,
                        force_stopped=record.force_stopped,
                    )
                return self._uncertain_result(
                    record,
                    fallback_stderr=(
                        b"Git process containment admission failed"
                    ),
                )
            if not record.owned_process_tree and on_spawn is not None:
                try:
                    on_spawn()
                except BaseException as error:
                    callback_error = error
            if callback_error is not None:
                terminated = await self._stop_record(record)
                if terminated and not communication.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(communication),
                            timeout=self._kill_timeout,
                        )
                    except TimeoutError:
                        terminated = False
                settlement = self._read_record(record)
                if terminated and settlement.state not in {"alive", "uncertain"}:
                    raise callback_error
                record.exposed = True
                return self._uncertain_result(
                    record,
                    fallback_stderr=str(callback_error).encode(
                        "utf-8",
                        "backslashreplace",
                    ),
                )
            if not record.owned_process_tree:
                self._publish_admission_outcome(record, started=True)
            shutdown_waiter = asyncio.create_task(
                self._shutdown_event.wait()
            )
            done, _ = await asyncio.wait(
                {communication, shutdown_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if communication in done:
                if record.owned_process_tree:
                    await self._refresh_containment_proof(record, timeout=0.0)
                    if not record.containment_proved:
                        terminated = await self._stop_record(record)
                        settlement = self._read_record(record)
                        if (
                            terminated
                            and settlement.state not in {"alive", "uncertain"}
                        ):
                            return self._result_from_settlement(settlement)
                        return self._uncertain_result(record)
                settlement = self._read_record(record)
                if settlement.state == "natural":
                    return self._result_from_settlement(settlement)
                return self._uncertain_result(record)

            shutdown_requested = shutdown_waiter in done
            record.timed_out = not shutdown_requested
            terminated = await self._stop_record(record)
            if terminated:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communication),
                        timeout=self._kill_timeout,
                    )
                except TimeoutError:
                    terminated = False
            settlement = self._read_record(record)
            if terminated and settlement.state not in {"alive", "uncertain"}:
                return self._result_from_settlement(
                    settlement,
                    timed_out=not shutdown_requested,
                    stop_requested=record.stop_requested,
                    force_stopped=record.force_stopped,
                )
            return self._uncertain_result(
                record,
                timed_out=not shutdown_requested,
                fallback_stderr=(
                    b"Git command stopped during shutdown"
                    if shutdown_requested
                    else b"Git command timed out"
                ),
            )
        except asyncio.CancelledError:
            if record.process is not None:
                await self._stop_record(record)
                record.exposed = True
            raise
        except BaseException:
            if record.process is not None:
                terminated = await self._stop_record(record)
                communication = record.communication
                if terminated and communication is not None and not communication.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(communication),
                            timeout=self._kill_timeout,
                        )
                    except TimeoutError:
                        pass
                if self._read_record(record).state in {"alive", "uncertain"}:
                    record.exposed = True
            raise
        finally:
            if record.process is None:
                record.ready.set()
            if (
                record.owned_process_tree
                and record.admission_state != "started"
            ):
                record.admission_state = "not_started"
            self._publish_admission_outcome(
                record,
                started=(
                    record.process is not None
                    if not record.owned_process_tree
                    else record.admission_state == "started"
                ),
            )
            if shutdown_waiter is not None and not shutdown_waiter.done():
                shutdown_waiter.cancel()
                await asyncio.gather(
                    shutdown_waiter,
                    return_exceptions=True,
                )

    @staticmethod
    async def _await_admission_outcome(
        record: _RetainedChildRecord,
    ) -> bool:
        """Wait through repeated caller cancellation for admission publication."""
        while not record.admission_outcome.done():
            try:
                await asyncio.shield(record.admission_outcome)
            except asyncio.CancelledError:
                continue
        return record.admission_outcome.result()

    @staticmethod
    async def _await_cancelled_run_task(
        task: asyncio.Task[GitCommandResult],
    ) -> GitCommandResult | None:
        """Retrieve terminal no-child settlement despite repeated cancellation."""
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            return task.result()
        except asyncio.CancelledError:
            return None

    @staticmethod
    def _publish_admission_outcome(
        record: _RetainedChildRecord,
        *,
        started: bool,
    ) -> None:
        if not record.admission_outcome.done():
            record.admission_outcome.set_result(started)

    @staticmethod
    async def _communicate_unbounded(
        process: AsyncChildProcess,
        stdin: bytes | None,
    ) -> tuple[bytes, bytes]:
        """Start child I/O only after the synchronous spawn callback boundary."""
        return await process.communicate(stdin)

    async def _communicate_bounded(
        self,
        record: _RetainedChildRecord,
        process: AsyncChildProcess,
        stdin: bytes | None,
        *,
        stdout_limit: int | None,
        stderr_limit: int | None,
    ) -> tuple[bytes, bytes]:
        """Continuously drain both streams while retaining bounded evidence."""
        stdout_stream = process.stdout
        stderr_stream = process.stderr
        assert stdout_stream is not None
        assert stderr_stream is not None
        stdout_reader = asyncio.create_task(
            self._read_bounded_stream(stdout_stream, stdout_limit)
        )
        stderr_reader = asyncio.create_task(
            self._read_bounded_stream(stderr_stream, stderr_limit)
        )
        if stdin is not None:
            stdin_writer = asyncio.create_task(
                self._write_process_stdin(process, stdin)
            )
            (stdout, stdout_overflow), (
                stderr,
                stderr_overflow,
            ), _ = await asyncio.gather(
                stdout_reader,
                stderr_reader,
                stdin_writer,
            )
        else:
            (stdout, stdout_overflow), (
                stderr,
                stderr_overflow,
            ) = await asyncio.gather(stdout_reader, stderr_reader)
        await process.wait()
        record.output_overflow = stdout_overflow or stderr_overflow
        return stdout, stderr

    @staticmethod
    async def _read_bounded_stream(
        stream: asyncio.StreamReader,
        limit: int | None,
    ) -> tuple[bytes, bool]:
        retained = bytearray()
        overflow = False
        while chunk := await stream.read(_GIT_STREAM_CHUNK_BYTES):
            if limit is None:
                retained.extend(chunk)
                continue
            remaining = max(0, limit - len(retained))
            retained.extend(chunk[:remaining])
            overflow = overflow or len(chunk) > remaining
        return bytes(retained), overflow

    @staticmethod
    async def _write_process_stdin(
        process: AsyncChildProcess,
        payload: bytes,
    ) -> None:
        stream = process.stdin
        assert stream is not None
        try:
            stream.write(payload)
            await stream.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.close()

    def _run_task_completed(
        self,
        task: asyncio.Task[GitCommandResult],
        record: _RetainedChildRecord,
    ) -> None:
        """Release one owned task and retrieve any otherwise-orphaned error."""
        self._run_tasks.discard(task)
        if record.owned_task is task:
            record.owned_task = None
        if not task.cancelled():
            task.exception()
        if record.released:
            self._clear_record(record)

    def shutdown(self) -> Awaitable[bool]:
        """Seal admissions and return retained finite cleanup settlement."""
        if self._shutdown_settlement is not None:
            return self._shutdown_settlement
        running_loop: asyncio.AbstractEventLoop | None = None
        if self._run_tasks or self._processes:
            assert self._loop is not None
            assert self._shutdown_event is not None
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError as error:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                ) from error
            if running_loop is not self._loop:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                )
        self._sealed = True
        if not self._run_tasks and not self._processes:
            self._release_terminal_records()
            self._shutdown_settlement = _ImmediateSettlement(True)
            return self._shutdown_settlement
        assert running_loop is not None
        assert self._shutdown_event is not None
        retained_nonowned_before_shutdown = tuple(
            record.token
            for record in self._retained_children.values()
            if (
                not record.owned_process_tree
                and record.exposed
                and record.process is not None
                and (
                    record.owned_task is None
                    or record.owned_task.done()
                )
            )
        )
        self._shutdown_event.set()
        settlement = _RetainedSettlement(
            running_loop.create_task(
                self._settle_shutdown(retained_nonowned_before_shutdown)
            )
        )
        self._shutdown_settlement = settlement
        return settlement

    async def _bounded_process_wait(
        self,
        process: AsyncChildProcess,
        timeout: float,
    ) -> bool:
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def _stop_record(
        self,
        record: _RetainedChildRecord,
    ) -> bool:
        """Request bounded termination and then force-stop one child."""
        async with record.stop_lock:
            process = record.process
            assert process is not None
            if record.owned_process_tree:
                tree = record.process_tree
                assert tree is not None
                if await self._refresh_containment_proof(record, timeout=0.0):
                    return True
                if not record.terminate_sent:
                    record.stop_requested = True
                    record.terminate_sent = True
                    try:
                        self._process_tree_controller.terminate(tree)
                    except Exception:
                        pass
                    if await self._refresh_containment_proof(
                        record,
                        timeout=self._terminate_timeout,
                    ):
                        return True
                if not record.kill_sent:
                    record.force_stopped = True
                    record.kill_sent = True
                    try:
                        self._process_tree_controller.kill(tree)
                    except Exception:
                        pass
                    return await self._refresh_containment_proof(
                        record,
                        timeout=self._kill_timeout,
                    )
                return await self._refresh_containment_proof(
                    record,
                    timeout=0.0,
                )
            if self._read_record(record).state not in {"alive", "uncertain"}:
                return True
            record.stop_requested = True
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except OSError:
                pass
            terminated = await self._bounded_process_wait(
                process,
                self._terminate_timeout,
            )
            if terminated:
                return True
            record.force_stopped = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError:
                pass
            return await self._bounded_process_wait(
                process,
                self._kill_timeout,
            )

    async def _refresh_containment_proof(
        self,
        record: _RetainedChildRecord,
        *,
        timeout: float,
    ) -> bool:
        """Publish proof only after the retained native container is empty."""
        if not record.owned_process_tree:
            return False
        if record.containment_proved:
            return True
        tree = record.process_tree
        assert tree is not None
        try:
            record.containment_proved = bool(
                await self._process_tree_controller.wait(
                    tree,
                    timeout=max(0.0, timeout),
                )
            )
        except Exception:
            record.containment_proved = False
        return record.containment_proved

    async def _settle_shutdown(
        self,
        retained_nonowned_before_shutdown: tuple[
            RetainedGitChildToken,
            ...,
        ],
    ) -> bool:
        current = asyncio.current_task()
        run_tasks = tuple(
            task
            for task in self._run_tasks
            if task is not current
        )
        run_failed = False
        pending_tasks: set[asyncio.Task[object]] = set()
        if run_tasks:
            done_tasks, pending_tasks = await asyncio.wait(
                run_tasks,
                timeout=(
                    self._terminate_timeout
                    + (2 * self._kill_timeout)
                ),
            )
            for task in done_tasks:
                self._run_tasks.discard(task)
                if task.cancelled():
                    run_failed = True
                    continue
                try:
                    task.result()
                except BaseException:
                    run_failed = True
        all_children_settled = True
        retained_nonowned = set(retained_nonowned_before_shutdown)
        for record in tuple(self._retained_children.values()):
            if (
                not record.owned_process_tree
                and record.token not in retained_nonowned
            ):
                continue
            if record.released or record.process is None:
                if not record.released:
                    all_children_settled = False
                continue
            settlement = self._read_record(record)
            if settlement.state not in {"alive", "uncertain"}:
                continue
            terminated = await self._stop_record(record)
            communication = record.communication
            assert communication is not None
            if terminated and not communication.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communication),
                        timeout=self._kill_timeout,
                    )
                except TimeoutError:
                    terminated = False
            settlement = self._read_record(record)
            if terminated and settlement.state not in {"alive", "uncertain"}:
                if not record.exposed:
                    self._release_unexposed_record(record)
            else:
                record.exposed = True
                all_children_settled = False
        self._release_terminal_records()
        if self._processes:
            all_children_settled = False
        return (
            all_children_settled
            and not pending_tasks
            and not self._processes
            and not self._run_tasks
            and not run_failed
        )

    def read_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Read one exact retained child without sealing or stopping it."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        if record.communication is None:
            return RetainedGitChildSettlement("uncertain")
        return self._read_record(record)

    def claim_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Reserve one exact token for explicit settlement and release."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        if record.released:
            return False
        record.claimed = True
        return True

    async def settle_retained_child(
        self,
        token: RetainedGitChildToken,
        *,
        timeout: float = 0.0,
    ) -> RetainedGitChildSettlement:
        """Wait boundedly for one exact retained child, without stopping it."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        if record.communication is None and not record.ready.is_set() and timeout > 0:
            try:
                await asyncio.wait_for(
                    record.ready.wait(),
                    timeout=timeout,
                )
            except TimeoutError:
                return RetainedGitChildSettlement("uncertain")
        if record.communication is None:
            return RetainedGitChildSettlement("uncertain")
        if record.owned_process_tree and not record.containment_proved:
            await self._refresh_containment_proof(record, timeout=0.0)
        settlement = self._read_record(record)
        remaining = deadline - loop.time()
        if (
            settlement.state not in {"alive", "uncertain"}
            or remaining <= 0
        ):
            return settlement
        try:
            await asyncio.wait_for(
                asyncio.shield(record.communication),
                timeout=remaining,
            )
        except TimeoutError:
            pass
        remaining = max(0.0, deadline - loop.time())
        if record.owned_process_tree and not record.containment_proved:
            await self._refresh_containment_proof(record, timeout=remaining)
        return self._read_record(record)

    def release_retained_child(
        self,
        token: RetainedGitChildToken,
    ) -> bool:
        """Release terminal evidence; refuse live or uncertain children."""
        record = self._record_for_token(token)
        self._require_retained_child_loop(record)
        if record.communication is None:
            return False
        settlement = self._read_record(record)
        if settlement.state in {"alive", "uncertain"}:
            return False
        self._discard_record(record)
        return True

    def _record_for_token(
        self,
        token: RetainedGitChildToken,
    ) -> _RetainedChildRecord:
        if not isinstance(token, RetainedGitChildToken):
            raise ValueError("Unknown retained Git child token")
        try:
            return self._retained_children[token]
        except KeyError:
            raise ValueError("Unknown retained Git child token") from None

    def _require_retained_child_loop(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        if record.settlement is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise GitShutdownAffinityError(
                "A live retained Git child must be read from its event loop"
            ) from error
        if loop is not self._loop:
            raise GitShutdownAffinityError(
                "A live retained Git child must be read from its event loop"
            )

    def _read_record(
        self,
        record: _RetainedChildRecord,
    ) -> RetainedGitChildSettlement:
        if record.settlement is not None:
            return record.settlement
        communication = record.communication
        process = record.process
        assert communication is not None
        assert process is not None
        if not communication.done():
            return RetainedGitChildSettlement(
                "alive",
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
                output_overflow=record.output_overflow,
                owned_process_tree=record.owned_process_tree,
                containment_proved=record.containment_proved,
            )
        try:
            stdout, stderr = communication.result()
        except BaseException as error:
            returncode = process.returncode
            if (
                record.owned_process_tree
                and record.containment_proved
                and returncode is not None
            ):
                settlement = RetainedGitChildSettlement(
                    "contained_uncertain",
                    returncode,
                    b"",
                    self._bounded_stderr(
                        str(error).encode("utf-8", "backslashreplace")
                    ),
                    record.stop_requested,
                    record.force_stopped,
                    record.output_overflow,
                    True,
                    True,
                )
                record.settlement = settlement
                self._processes.discard(process)
                return settlement
            return RetainedGitChildSettlement(
                "uncertain",
                returncode,
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
                output_overflow=record.output_overflow,
                owned_process_tree=record.owned_process_tree,
                containment_proved=record.containment_proved,
            )
        returncode = process.returncode
        if record.owned_process_tree and not record.containment_proved:
            return RetainedGitChildSettlement(
                "uncertain",
                returncode,
                stdout,
                self._bounded_stderr(stderr),
                record.stop_requested,
                record.force_stopped,
                record.output_overflow,
                True,
                False,
            )
        if returncode is None:
            return RetainedGitChildSettlement(
                "uncertain" if record.stop_requested else "alive",
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
                output_overflow=record.output_overflow,
                owned_process_tree=record.owned_process_tree,
                containment_proved=record.containment_proved,
            )
        if returncode >= 0 or not record.stop_requested:
            state: RetainedGitChildState = "natural"
        elif record.force_stopped:
            state = "forced_stop"
        else:
            state = "stop_requested"
        settlement = RetainedGitChildSettlement(
            state,
            returncode,
            stdout,
            self._bounded_stderr(stderr),
            record.stop_requested,
            record.force_stopped,
            record.output_overflow,
            record.owned_process_tree,
            record.containment_proved,
        )
        record.settlement = settlement
        self._processes.discard(process)
        return settlement

    def _uncertain_result(
        self,
        record: _RetainedChildRecord,
        *,
        timed_out: bool = False,
        fallback_stderr: bytes = b"",
    ) -> GitCommandResult:
        settlement = self._read_record(record)
        if settlement.state not in {"alive", "uncertain"}:
            return self._result_from_settlement(
                settlement,
                timed_out=timed_out,
                stop_requested=record.stop_requested,
                force_stopped=record.force_stopped,
            )
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr or self._bounded_stderr(fallback_stderr),
            timed_out=timed_out,
            termination_uncertain=True,
            retained_child=record.token,
            stop_requested=record.stop_requested,
            force_stopped=record.force_stopped,
            output_overflow=record.output_overflow,
            owned_process_tree=record.owned_process_tree,
            containment_proved=record.containment_proved,
        )

    def _result_from_settlement(
        self,
        settlement: RetainedGitChildSettlement,
        *,
        timed_out: bool = False,
        stop_requested: bool = False,
        force_stopped: bool = False,
    ) -> GitCommandResult:
        stopped = stop_requested or settlement.stop_requested or settlement.state in {
            "stop_requested",
            "forced_stop",
        }
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr,
            timed_out=timed_out,
            termination_uncertain=(
                stopped or settlement.state == "contained_uncertain"
            ),
            stop_requested=stopped,
            force_stopped=(
                force_stopped
                or settlement.force_stopped
                or settlement.state == "forced_stop"
            ),
            output_overflow=settlement.output_overflow,
            owned_process_tree=settlement.owned_process_tree,
            containment_proved=settlement.containment_proved,
        )

    def _release_unexposed_record(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        if record.exposed:
            return
        self._discard_record(record)

    def _release_terminal_records(self) -> None:
        """Discard only exact terminal records, retaining uncertainty."""
        for record in tuple(self._retained_children.values()):
            if record.communication is None or record.claimed:
                continue
            settlement = self._read_record(record)
            if settlement.state not in {"alive", "uncertain"}:
                self._discard_record(record)

    def _discard_record(
        self,
        record: _RetainedChildRecord,
    ) -> None:
        """Remove one terminal/unexposed record and its heavy references."""
        if record.process is not None:
            self._processes.discard(record.process)
        if record.process_tree is not None and record.containment_proved:
            self._process_tree_controller.close(record.process_tree)
        self._retained_children.pop(record.token, None)
        record.released = True
        if record.owned_task is None or record.owned_task.done():
            self._clear_record(record)

    @staticmethod
    def _clear_record(record: _RetainedChildRecord) -> None:
        """Drop heavy terminal references after the owned task is finished."""
        record.process = None
        record.process_tree = None
        record.communication = None
        record.owned_task = None
        record.settlement = None

    def _bounded_stderr(self, stderr: bytes) -> bytes:
        return sanitize_git_stderr(
            stderr,
            limit=self._stderr_limit,
        ).encode("utf-8", "surrogateescape")


class FileNotesGitService:
    """Trusted, process-owned Git projection for one File Notes owner."""

    def __init__(
        self,
        owner: FileNotesSessionOwner,
        *,
        runner: GitProcessRunner | None = None,
        git_executable: str | None = None,
        environment: Mapping[str, str] | None = None,
        transport_admission: TransportAdmission | None = None,
        network_context_factory: NetworkContextFactory | None = None,
        discovery_timeout: float = 3.0,
        status_timeout: float = 5.0,
        push_query_timeout: float = _DEFAULT_PUSH_QUERY_TIMEOUT_SECONDS,
        push_timeout: float = _DEFAULT_PUSH_TIMEOUT_SECONDS,
        before_push_spawn: (
            Callable[[], Awaitable[None] | None] | None
        ) = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._owner = owner
        self._runner = runner or AsyncGitProcessRunner()
        self._environment = dict(
            os.environ if environment is None else environment
        )
        self._git_executable = (
            git_executable
            if git_executable is not None
            else shutil.which("git", path=self._environment.get("PATH"))
        )
        self._discovery_timeout = discovery_timeout
        self._status_timeout = status_timeout
        self._transport_admission = (
            TransportAdmission()
            if transport_admission is None
            else transport_admission
        )
        if type(self._transport_admission) is not TransportAdmission:
            raise TypeError("Invalid guarded-push transport admission")
        if (
            network_context_factory is not None
            and type(network_context_factory) is not NetworkContextFactory
        ):
            raise TypeError("Invalid guarded-push network context factory")
        self._validate_positive_timeout(
            "push_query_timeout",
            push_query_timeout,
        )
        self._validate_positive_timeout("push_timeout", push_timeout)
        if before_push_spawn is not None and not callable(before_push_spawn):
            raise TypeError("before_push_spawn must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._network_context_factory = network_context_factory
        self._push_query_timeout = push_query_timeout
        self._push_timeout = push_timeout
        self._before_push_spawn = before_push_spawn
        self._clock = clock
        self._push_destination_policy: (
            _PushDestinationPolicySnapshot | None
        ) = None
        self._push_authorization: PushAuthorizationHandle | None = None
        self._push_local_proof_cycle: (
            asyncio.Task[PushDestinationPolicyResult] | None
        ) = None
        self._push_local_proof_waiter: (
            asyncio.Task[PushDestinationPolicyResult] | None
        ) = None
        self._push_preflight_cycle: asyncio.Task[PushPreflightResult] | None = None
        self._push_preflight_waiter: asyncio.Task[PushPreflightResult] | None = None
        self._push_cycle: asyncio.Task[PushExecutionResult] | None = None
        self._push_waiter: asyncio.Task[PushExecutionResult] | None = None
        self._push_recovery_cycle: (
            asyncio.Task[PushRecoveryProjection] | None
        ) = None
        self._push_recovery_waiter: (
            asyncio.Task[PushRecoveryProjection] | None
        ) = None
        self._push_recovery_settlement_cycle: asyncio.Task[None] | None = None
        self._push_child_started = False
        self._push_child_started_signal: asyncio.Future[bool] | None = None
        self._push_quarantine_signal: asyncio.Future[None] | None = None
        self._push_quarantine_requested = False
        self._push_review_snapshots: dict[
            PushReviewHandle,
            _PushReviewSnapshot,
        ] = {}
        self._pending_push_contexts: set[NetworkGitExecutionContext] = set()
        self._unsettled_push_preflight: _UnsettledPushPreflight | None = None
        self._push_operation_generation = 0
        self._retained_push_operation: RetainedPushOperation | None = None
        self._uncertain_push: _UncertainPushEvidence | None = None
        self._sealed = False
        self._status_cycle: asyncio.Task[SessionGitStatus] | None = None
        self._status_cycle_binding: SessionBinding | None = None
        self._status_waiter: asyncio.Task[SessionGitStatus] | None = None
        self._pending_status: tuple[
            SessionBinding,
            tuple[SequencedSessionChange, ...],
            RepositoryIdentity,
            int,
            int,
        ] | None = None
        self._rerun_available = False
        self._status_dirty = False
        self._status_request_generation = 0
        self._action_cycle: asyncio.Task[GitActionResult] | None = None
        self._action_waiter: asyncio.Task[GitActionResult] | None = None
        self._commit_review_cycle: asyncio.Task[CommitReviewResult] | None = None
        self._commit_review_waiter: asyncio.Task[CommitReviewResult] | None = None
        self._commit_review_snapshots: dict[object, _CommitReviewSnapshot] = {}
        self._commit_cycle: asyncio.Task[CommitOutcome] | None = None
        self._commit_waiter: asyncio.Task[CommitOutcome] | None = None
        self._commit_recovery_cycle: asyncio.Task[CommitOutcome] | None = None
        self._commit_recovery_waiter: asyncio.Task[CommitOutcome] | None = None
        self._commit_child_started = False
        self._commit_child_started_signal: asyncio.Future[bool] | None = None
        self._retained_commit_operation: RetainedCommitOperation | None = None
        self._uncertain_commit: _UncertainCommitEvidence | None = None
        self._orphaned_commit: _OrphanedCommitLifecycle | None = None
        self._pending_hooks_cleanup: set[Path] = set()
        self._shutdown_runner_confirmed: bool | None = None
        self._shutdown_settlement: Awaitable[None] | None = None

    @staticmethod
    def _validate_positive_timeout(name: str, value: object) -> None:
        """Reject nonnumeric, nonfinite, and nonpositive network bounds."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    async def discover(
        self,
        binding: SessionBinding,
    ) -> DiscoveryResult:
        """Discover repository/HEAD identity without inspecting worktree state."""
        if self._sealed:
            return DiscoveryResult(
                "unavailable",
                message="File Notes Git service is shut down",
            )
        self._observe_commit_rebinding()
        if binding != self._owner.current_binding():
            return DiscoveryResult(
                "unavailable",
                message="File Notes root binding is stale",
            )
        root = self._safe_root(binding)
        if root is None:
            return DiscoveryResult(
                "unsafe_root",
                message="Selected File Notes root is not a safe directory",
            )
        if self._git_executable is None:
            return DiscoveryResult(
                "unavailable",
                message="Git is not installed",
            )

        inside = await self._run_discovery(
            root,
            ("rev-parse", "--is-inside-work-tree"),
        )
        if inside is None:
            return DiscoveryResult(
                "unavailable",
                message="Git repository discovery failed",
            )
        if inside.returncode != 0:
            diagnostic = sanitize_git_stderr(inside.stderr)
            normalized_diagnostic = diagnostic.lower()
            safety_markers = (
                "dubious ownership",
                "safe.directory",
                "permission denied",
                "operation not permitted",
                "access is denied",
            )
            if (
                "not a git repository" in normalized_diagnostic
                and not any(
                    marker in normalized_diagnostic
                    for marker in safety_markers
                )
            ):
                return DiscoveryResult(
                    "not_repository",
                    message=(
                        "Selected File Notes root is not in a Git worktree"
                    ),
                )
            message = "Git refused repository discovery"
            if diagnostic:
                message = f"{message}: {diagnostic}"
            return DiscoveryResult(
                "unsafe_root",
                message=message,
            )
        if inside.stdout != b"true\n":
            return DiscoveryResult(
                "not_repository",
                message="Selected File Notes root is not in a Git worktree",
            )

        resolved_paths = await self._read_repository_paths(root)
        if resolved_paths is None:
            return DiscoveryResult(
                "unsupported",
                message="Git returned an unsupported repository mapping",
            )
        worktree_root, git_dir, git_common_dir = resolved_paths
        try:
            root.relative_to(worktree_root)
        except ValueError:
            return DiscoveryResult(
                "unsafe_root",
                message="Selected File Notes root is outside the Git worktree",
            )

        repository = RepositoryIdentity(
            worktree_root=str(worktree_root),
            git_dir=str(git_dir),
            git_common_dir=str(git_common_dir),
            worktree_identity=_filesystem_identity(worktree_root),
            git_dir_identity=_filesystem_identity(git_dir),
            git_common_dir_identity=_filesystem_identity(git_common_dir),
        )
        head_result = await self._read_head(root)
        if isinstance(head_result, _HeadReadFailure):
            return DiscoveryResult(
                (
                    "unavailable"
                    if head_result.kind == "unavailable"
                    else "unsupported"
                ),
                message=head_result.message,
            )
        return DiscoveryResult(
            "ready",
            repository=repository,
            head=head_result,
        )

    async def revalidate_repository(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        """Rediscover repository mapping, then restat its trusted identities."""
        self._observe_commit_rebinding()
        valid = False
        root = self._safe_root(binding)
        if (
            not self._sealed
            and root is not None
            and self._git_executable is not None
            and binding == self._owner.current_binding()
        ):
            resolved_paths = await self._read_repository_paths(root)
            valid = (
                not self._sealed
                and binding == self._owner.current_binding()
                and resolved_paths is not None
                and tuple(str(path) for path in resolved_paths)
                == (
                    repository.worktree_root,
                    repository.git_dir,
                    repository.git_common_dir,
                )
                and self._repository_identity_matches(binding, repository)
            )
        if not valid:
            self._owner.clear_trust_if_matches(binding, repository)
        return valid

    async def review_push_destination(
        self,
        binding: SessionBinding,
    ) -> PushDestinationPolicyResult:
        """Run bounded local-only candidate, config, transport, and LFS proof."""
        self._observe_push_rebinding()
        previous = self._push_destination_policy
        if previous is not None:
            self._owner._revoke_destination_policy(previous.owner_capture)
        self._push_destination_policy = None
        self._push_authorization = None
        policy, state = await self._prove_push_destination(binding)
        if policy is None:
            return _push_destination_policy_result(state)
        self._push_destination_policy = policy
        return _push_destination_policy_result(
            "ready",
            policy.configuration.transport.destination,
        )

    def start_push_review(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[PushDestinationPolicyResult]:
        """Admit and retain the local-only guarded-push proof phase."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        self._observe_push_rebinding()
        if (
            self._push_local_proof_cycle is not None
            and not self._push_local_proof_cycle.done()
        ) or (
            self._push_preflight_cycle is not None
            and not self._push_preflight_cycle.done()
        ) or self._push_review_snapshots or self._pending_push_contexts:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A guarded push review is already active",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            raise GitMutationAdmissionError(
                admission.reason or "mutation_active",
                "File Notes Git mutation admission was refused",
            )
        candidate = self._owner.snapshot(binding).push_candidate
        if candidate is None:
            lease.release()
            raise GitMutationAdmissionError(
                "stale_binding",
                "Guarded push candidate is stale",
            )
        self._push_operation_generation += 1
        operation_generation = self._push_operation_generation
        cycle: asyncio.Task[PushDestinationPolicyResult] | None = None
        try:
            loop = asyncio.get_running_loop()
            cycle = loop.create_task(
                self._run_push_local_proof_cycle(binding, lease)
            )
            waiter = loop.create_task(
                self._shield_push_local_proof_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._push_local_proof_cycle = cycle
        self._push_local_proof_waiter = waiter
        self._retained_push_operation = RetainedPushOperation(
            binding,
            operation_generation,
            "local_proof",
            candidate,
            cycle,
        )
        cycle.add_done_callback(self._push_local_proof_cycle_completed)
        return waiter

    async def _run_push_local_proof_cycle(
        self,
        binding: SessionBinding,
        lease: GitMutationLease,
    ) -> PushDestinationPolicyResult:
        try:
            return await self.review_push_destination(binding)
        except asyncio.CancelledError:
            return _push_destination_policy_result("blocked")
        finally:
            lease.release()

    async def _shield_push_local_proof_cycle(
        self,
        cycle: asyncio.Task[PushDestinationPolicyResult],
    ) -> PushDestinationPolicyResult:
        return await asyncio.shield(cycle)

    def _push_local_proof_cycle_completed(
        self,
        cycle: asyncio.Task[PushDestinationPolicyResult],
    ) -> None:
        if self._push_local_proof_cycle is cycle:
            self._push_local_proof_cycle = None
        if not cycle.cancelled():
            cycle.exception()

    def authorize_push_destination(
        self,
        binding: SessionBinding,
    ) -> PushAuthorizationHandle | None:
        """Authorize later contact with the exact locally proved destination."""
        policy = self._push_destination_policy
        if (
            self._sealed
            or policy is None
            or policy.candidate_capture.binding != binding
            or binding != self._owner.current_binding()
        ):
            return None
        authorization = self._owner._authorize_destination_policy(
            policy.owner_capture
        )
        if authorization is None:
            self._push_authorization = None
            return None
        self._push_authorization = authorization
        return authorization

    async def revalidate_push_destination(
        self,
        binding: SessionBinding,
        authorization: PushAuthorizationHandle,
    ) -> bool:
        """Re-run local proof for Confirm without contacting the destination."""
        policy = self._push_destination_policy
        if (
            self._sealed
            or policy is None
            or authorization is not self._push_authorization
            or not self._owner._destination_authorization_matches(
                policy.owner_capture,
                authorization,
            )
        ):
            return False
        refreshed, _state = await self._prove_push_destination(
            binding,
            prior=policy,
        )
        if refreshed is not policy:
            self._owner._revoke_destination_policy(policy.owner_capture)
            self._push_destination_policy = refreshed
            self._push_authorization = None
            return False
        return self._owner._destination_authorization_matches(
            policy.owner_capture,
            authorization,
        )

    def authorize_and_check_push(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushPreflightResult]:
        """Authorize one frozen destination and retain its exact-ref query."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        self._observe_push_rebinding()
        if (
            operation is not self._retained_push_operation
            or operation.binding != binding
            or operation.kind != "local_proof"
            or not operation.settled
        ):
            raise GitMutationAdmissionError(
                "stale_binding",
                "Push local-proof operation is stale",
            )
        policy = self._push_destination_policy
        if (
            policy is None
            or policy.candidate_capture.binding != binding
            or operation.candidate != policy.candidate_capture.availability
            or binding != self._owner.current_binding()
        ):
            raise GitMutationAdmissionError(
                "stale_binding",
                "Push destination proof is stale",
            )
        if (
            self._push_preflight_cycle is not None
            and not self._push_preflight_cycle.done()
        ) or self._push_review_snapshots:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A guarded push review is already retained",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            raise GitMutationAdmissionError(
                admission.reason or "mutation_active",
                "File Notes Git mutation admission was refused",
            )
        self._push_operation_generation += 1
        operation_generation = self._push_operation_generation
        operation_id = object()
        cycle: asyncio.Task[PushPreflightResult] | None = None
        try:
            loop = asyncio.get_running_loop()
            quarantine_signal = loop.create_future()
            cycle = loop.create_task(
                self._run_push_preflight_cycle(
                    binding,
                    policy,
                    operation_id,
                    lease,
                )
            )
            waiter = loop.create_task(
                self._shield_push_preflight_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._push_preflight_cycle = cycle
        self._push_preflight_waiter = waiter
        self._push_quarantine_signal = quarantine_signal
        self._retained_push_operation = RetainedPushOperation(
            binding,
            operation_generation,
            "preflight",
            operation.candidate,
            cycle,
        )
        cycle.add_done_callback(self._push_preflight_cycle_completed)
        return waiter

    async def _shield_push_preflight_cycle(
        self,
        cycle: asyncio.Task[PushPreflightResult],
    ) -> PushPreflightResult:
        return await asyncio.shield(cycle)

    def _push_preflight_cycle_completed(
        self,
        cycle: asyncio.Task[PushPreflightResult],
    ) -> None:
        if self._push_preflight_cycle is cycle:
            self._push_preflight_cycle = None
        quarantine_signal = self._push_quarantine_signal
        if quarantine_signal is not None and not quarantine_signal.done():
            quarantine_signal.cancel()
        self._push_quarantine_signal = None
        if not cycle.cancelled():
            cycle.exception()

    async def _run_push_preflight_cycle(
        self,
        binding: SessionBinding,
        policy: _PushDestinationPolicySnapshot,
        operation_id: object,
        mutation_lease: GitMutationLease,
    ) -> PushPreflightResult:
        context: NetworkGitExecutionContext | None = None
        active_lease: NetworkContextLease | None = None
        authorization: PushAuthorizationHandle | None = None
        review_retained = False
        containment_quarantined = False
        try:
            refreshed, _state = await self._prove_push_destination(
                binding,
                prior=policy,
            )
            if refreshed is not policy:
                self._owner._revoke_destination_policy(policy.owner_capture)
                self._push_destination_policy = refreshed
                self._push_authorization = None
                return PushPreflightResult("blocked")
            if os.name != "posix" or not hasattr(os, "geteuid"):
                return PushPreflightResult("blocked")
            authorization = self.authorize_push_destination(binding)
            if authorization is None:
                return PushPreflightResult("blocked")
            endpoint = policy.configuration.transport.endpoint
            factory = self._network_context_factory
            if endpoint is None:
                return PushPreflightResult("blocked")
            if factory is None:
                factory = NetworkContextFactory(
                    environment=self._environment,
                    git_executable=self._git_executable,
                    git_exec_path=policy.git_exec_path,
                    allow_ssh_agent=True,
                )
            context = factory.create(
                repository=policy.candidate_capture.repository,
                source_objects=policy.source_objects,
                configuration=policy.network_configuration,
                destination=policy.configuration.transport.destination,
                endpoint=endpoint,
            )
            active_lease = context.retain("active")
            settings = context.command_settings()
            argv = context.build_query_argv(endpoint)
            command_policy_fingerprint = self._push_command_policy_fingerprint(
                policy,
                context,
                settings.environment_fingerprint,
            )
            try:
                result = await self._runner.run(
                    argv,
                    cwd=settings.cwd,
                    environment=settings.environment,
                    stdin=settings.stdin,
                    timeout=self._push_query_timeout,
                    stdout_limit=_PUSH_REMOTE_OUTPUT_LIMIT_BYTES,
                    stderr_limit=DEFAULT_GIT_STDERR_LIMIT_BYTES,
                    owned_process_tree=True,
                )
            except GitRunCancelled as cancellation:
                retained_child = cancellation.retained_child
                if retained_child is not None:
                    await self._drain_push_preflight_child(retained_child)
                return PushPreflightResult("cancelled")
            result = await self._settle_push_preflight_result(result)
            if (
                result.returncode != 0
                or result.timed_out
                or result.termination_uncertain
                or result.output_overflow
                or not result.owned_process_tree
                or not result.containment_proved
            ):
                return PushPreflightResult("blocked")
            candidate = policy.candidate_capture.candidate
            observation = parse_ls_remote_refs(
                result.stdout,
                policy.configuration.transport.destination.destination_ref,
                candidate.parent_oid,
                candidate.candidate_oid,
            )
            if observation.state == "candidate":
                if not self._owner.clear_push_candidate(
                    policy.candidate_capture
                ):
                    return PushPreflightResult("blocked")
                return PushPreflightResult(
                    "already_published",
                    outcome=push_outcome_copy("already_published"),
                )
            if observation.state != "parent":
                return PushPreflightResult("blocked")
            review_lease = context.retain("review")
            issued = self._owner._capture_push_review_after_parent_observation(
                policy.owner_capture,
                authorization,
                operation_id=operation_id,
                network_context=context,
                command_policy_fingerprint=command_policy_fingerprint,
                parent_oid=candidate.parent_oid,
            )
            if issued is None:
                review_lease.release()
                return PushPreflightResult("blocked")
            handle, projection = issued
            self._push_review_snapshots[handle] = _PushReviewSnapshot(
                binding=binding,
                operation_id=operation_id,
                policy=policy,
                authorization=authorization,
                context=context,
                review_lease=review_lease,
                command_policy_fingerprint=command_policy_fingerprint,
                projection=projection,
            )
            review_retained = True
            return PushPreflightResult(
                "review",
                handle=handle,
                review=projection,
            )
        except _PushContainmentUnproved as unproved:
            retained_child = unproved.retained_child
            if (
                context is None
                or active_lease is None
                or authorization is None
                or self._unsettled_push_preflight is not None
            ):
                raise RuntimeError(
                    "Push containment ownership transfer failed"
                ) from None
            self._unsettled_push_preflight = _UnsettledPushPreflight(
                binding=binding,
                retained_child=retained_child,
                mutation_lease=mutation_lease,
                context=context,
                active_lease=active_lease,
                authorization=authorization,
            )
            containment_quarantined = True
            quarantine_signal = self._push_quarantine_signal
            if quarantine_signal is not None and not quarantine_signal.done():
                quarantine_signal.set_result(None)
        except asyncio.CancelledError:
            return PushPreflightResult("cancelled")
        except (NetworkContextError, OSError, PushContractError):
            return PushPreflightResult("blocked")
        finally:
            if active_lease is not None and not containment_quarantined:
                active_lease.release()
            if not review_retained and not containment_quarantined:
                if authorization is not None:
                    self._owner._revoke_destination_authorization(
                        authorization
                    )
                if context is not None:
                    self._close_or_retain_push_context(context)
            if not containment_quarantined:
                mutation_lease.release()
        if containment_quarantined:
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

    async def _settle_push_preflight_result(
        self,
        result: GitCommandResult,
    ) -> GitCommandResult:
        """Drain an exposed read-only child before classifying its result."""
        retained_child = result.retained_child
        if retained_child is None:
            return result
        settlement = await self._drain_push_preflight_child(retained_child)
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr,
            timed_out=result.timed_out,
            termination_uncertain=(
                result.termination_uncertain
                or settlement.stop_requested
                or settlement.force_stopped
                or settlement.state == "contained_uncertain"
            ),
            stop_requested=(
                result.stop_requested or settlement.stop_requested
            ),
            force_stopped=(
                result.force_stopped or settlement.force_stopped
            ),
            output_overflow=(
                result.output_overflow or settlement.output_overflow
            ),
            owned_process_tree=settlement.owned_process_tree,
            containment_proved=settlement.containment_proved,
        )

    async def _drain_push_preflight_child(
        self,
        retained_child: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Claim one exact query child until I/O and native tree settle."""
        claim_failed = False
        try:
            claimed = self._runner.claim_retained_child(retained_child)
        except Exception:
            claim_failed = True
            claimed = False
        if claim_failed or claimed is not True:
            raise _PushContainmentUnproved(retained_child) from None
        cancelled = False
        while True:
            if self._sealed and (
                self._shutdown_runner_confirmed is False
                or self._push_quarantine_requested
            ):
                raise _PushContainmentUnproved(retained_child)
            settlement_failed = False
            try:
                settlement = await self._runner.settle_retained_child(
                    retained_child,
                    timeout=0.1,
                )
            except asyncio.CancelledError:
                cancelled = True
                continue
            except Exception:
                settlement_failed = True
                settlement = None
            if settlement_failed:
                raise _PushContainmentUnproved(retained_child) from None
            if self._sealed and (
                self._shutdown_runner_confirmed is False
                or self._push_quarantine_requested
            ):
                raise _PushContainmentUnproved(retained_child)
            if type(settlement) is not RetainedGitChildSettlement:
                raise _PushContainmentUnproved(retained_child)
            settled = (
                settlement.state not in {"alive", "uncertain"}
                and settlement.owned_process_tree
                and settlement.containment_proved
            )
            if settled:
                release_failed = False
                try:
                    released = self._runner.release_retained_child(
                        retained_child
                    )
                except Exception:
                    release_failed = True
                    released = False
                if release_failed or released is not True:
                    raise _PushContainmentUnproved(retained_child) from None
                if cancelled:
                    raise asyncio.CancelledError
                return settlement
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancelled = True
                continue
            except Exception:
                raise _PushContainmentUnproved(retained_child) from None

    def _push_command_policy_fingerprint(
        self,
        policy: _PushDestinationPolicySnapshot,
        context: NetworkGitExecutionContext,
        environment_fingerprint: str,
    ) -> str:
        """Bind exact command facts without retaining endpoint or path text."""
        candidate = policy.candidate_capture.candidate
        digest = hashlib.sha256()
        for value in (
            "file-notes-push-review-v1",
            policy.configuration.configuration_fingerprint,
            policy.network_configuration.copy_fingerprint,
            policy.source_objects.identity_fingerprint,
            policy.source_objects.object_format,
            context.config_copy_fingerprint,
            environment_fingerprint,
            policy.git_exec_path,
            candidate.local_branch_ref,
            candidate.parent_oid,
            candidate.candidate_oid,
            policy.configuration.transport.destination.destination_ref,
            str(self._push_query_timeout),
            str(self._push_timeout),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def start_push(
        self,
        binding: SessionBinding,
        handle: PushReviewHandle,
    ) -> asyncio.Task[PushExecutionResult]:
        """Consume one immutable review and retain its exact push attempt."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        self._observe_push_rebinding()
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if type(handle) is not PushReviewHandle:
            raise GitMutationAdmissionError(
                "invalid_capability",
                "Push review capability is invalid or already consumed",
            )
        snapshot = self._push_review_snapshots.get(handle)
        previous_operation = self._retained_push_operation
        if (
            snapshot is None
            or snapshot.binding != binding
            or previous_operation is None
            or previous_operation.binding != binding
            or previous_operation.kind != "preflight"
            or not previous_operation.settled
            or previous_operation.candidate
            != snapshot.policy.candidate_capture.availability
            or (self._push_cycle is not None and not self._push_cycle.done())
        ):
            raise GitMutationAdmissionError(
                "invalid_capability",
                "Push review capability is invalid or already consumed",
            )
        admission = self._owner.admit_mutation(binding)
        mutation_lease = admission.lease
        if mutation_lease is None:
            raise GitMutationAdmissionError(
                admission.reason or "mutation_active",
                "File Notes Git mutation admission was refused",
            )
        active_lease: NetworkContextLease | None = None
        cycle: asyncio.Task[PushExecutionResult] | None = None
        child_started_signal: asyncio.Future[bool] | None = None
        try:
            active_lease = snapshot.context.retain("active")
            consumed = self._owner._consume_push_review(
                handle,
                operation_id=snapshot.operation_id,
                network_context=snapshot.context,
            )
            if consumed is None:
                raise GitMutationAdmissionError(
                    "invalid_capability",
                    "Push review capability is invalid or already consumed",
                )
            if self._push_review_snapshots.pop(handle, None) is not snapshot:
                raise GitMutationAdmissionError(
                    "invalid_capability",
                    "Push review capability is invalid or already consumed",
                )
            snapshot = replace(snapshot, consumed_review=consumed)
            snapshot.review_lease.release()
            loop = asyncio.get_running_loop()
            child_started_signal = loop.create_future()
            cycle = loop.create_task(
                self._run_exact_push_cycle(
                    binding,
                    snapshot,
                    mutation_lease,
                    active_lease,
                )
            )
            waiter = loop.create_task(self._shield_exact_push_cycle(cycle))
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            if child_started_signal is not None and not child_started_signal.done():
                child_started_signal.set_result(False)
            if active_lease is not None:
                active_lease.release()
            mutation_lease.release()
            if snapshot in self._push_review_snapshots.values():
                self._push_review_snapshots.pop(handle, None)
                snapshot.review_lease.release()
            self._owner._revoke_destination_authorization(
                snapshot.authorization
            )
            self._close_or_retain_push_context(snapshot.context)
            raise
        self._push_operation_generation += 1
        self._push_child_started = False
        self._push_child_started_signal = child_started_signal
        self._push_cycle = cycle
        self._push_waiter = waiter
        self._retained_push_operation = RetainedPushOperation(
            binding,
            self._push_operation_generation,
            "push",
            previous_operation.candidate,
            cycle,
            child_started_signal,
        )
        cycle.add_done_callback(self._push_cycle_completed)
        return waiter

    async def _shield_exact_push_cycle(
        self,
        cycle: asyncio.Task[PushExecutionResult],
    ) -> PushExecutionResult:
        return await asyncio.shield(cycle)

    def _mark_push_child_started(self) -> None:
        """Publish the exact successful-spawn boundary without yielding."""
        self._push_child_started = True
        signal = self._push_child_started_signal
        if signal is not None and not signal.done():
            signal.set_result(True)

    def _push_cycle_completed(
        self,
        cycle: asyncio.Task[PushExecutionResult],
    ) -> None:
        if self._push_cycle is cycle:
            signal = self._push_child_started_signal
            if signal is not None and not signal.done():
                signal.set_result(False)
            self._push_cycle = None
            self._push_waiter = None
            self._push_child_started = False
            self._push_child_started_signal = None
        if not cycle.cancelled():
            cycle.exception()

    async def _run_exact_push_cycle(
        self,
        binding: SessionBinding,
        snapshot: _PushReviewSnapshot,
        mutation_lease: GitMutationLease,
        active_lease: NetworkContextLease,
    ) -> PushExecutionResult:
        context = snapshot.context
        authorization = snapshot.authorization
        candidate = snapshot.policy.candidate_capture.candidate
        endpoint = snapshot.policy.configuration.transport.endpoint
        containment_quarantined = False
        uncertain_retained = False
        try:
            refreshed, _state = await self._prove_push_destination(
                binding,
                prior=snapshot.policy,
            )
            if (
                refreshed is not snapshot.policy
                or endpoint is None
                or not self._owner._destination_authorization_matches(
                    snapshot.policy.owner_capture,
                    authorization,
                )
            ):
                return PushExecutionResult("blocked")
            settings = context.command_settings()
            fingerprint = self._push_command_policy_fingerprint(
                snapshot.policy,
                context,
                settings.environment_fingerprint,
            )
            if fingerprint != snapshot.command_policy_fingerprint:
                return PushExecutionResult("blocked")

            try:
                confirmed = await self._run_timed_push_command(
                    context.build_query_argv(endpoint),
                    settings=settings,
                    timeout=self._push_query_timeout,
                )
            except GitRunCancelled as cancellation:
                if cancellation.retained_child is not None:
                    await self._drain_push_preflight_child(
                        cancellation.retained_child
                    )
                else:
                    assert cancellation.result is not None
                    await self._settle_push_preflight_result(
                        cancellation.result
                    )
                return PushExecutionResult("cancelled")
            confirmed = await self._settle_push_preflight_result(confirmed)
            observation = self._proved_remote_observation(
                confirmed,
                snapshot,
            )
            if observation == "candidate":
                self._owner.clear_push_candidate(
                    snapshot.policy.candidate_capture
                )
                return self._completed_push_result("already_published")
            if observation != "parent":
                return PushExecutionResult("blocked")

            barrier = self._before_push_spawn
            if barrier is not None:
                barrier_result = barrier()
                if inspect.isawaitable(barrier_result):
                    await barrier_result
            push_argv = context.build_push_argv(
                endpoint,
                candidate.parent_oid,
                candidate.candidate_oid,
            )
            try:
                push_result = await self._run_timed_push_command(
                    push_argv,
                    settings=settings,
                    timeout=self._push_timeout,
                    on_spawn=self._mark_push_child_started,
                    cancel_before_spawn=True,
                )
            except GitRunCancellationRejected as cancellation:
                settlement = await self._drain_push_preflight_child(
                    cancellation.retained_child
                )
                push_result = self._push_result_from_settlement(settlement)
            except GitRunCancelled as cancellation:
                if not self._push_child_started:
                    retained = cancellation.retained_child
                    if retained is not None:
                        await self._drain_push_preflight_child(retained)
                    return PushExecutionResult("cancelled")
                push_result = self._push_cancelled_result(cancellation)
            except OSError:
                if not self._push_child_started:
                    return PushExecutionResult("blocked")
                push_result = GitCommandResult(
                    None,
                    b"",
                    b"",
                    termination_uncertain=True,
                    owned_process_tree=True,
                )

            push_result = await self._settle_push_preflight_result(push_result)
            if not self._push_child_started:
                if (
                    not push_result.owned_process_tree
                    or not push_result.containment_proved
                ):
                    self._retain_uncertain_push_evidence(
                        snapshot,
                        mutation_lease,
                        active_lease,
                        descendants_terminal=False,
                    )
                    uncertain_retained = True
                    return self._completed_push_result("uncertain")
                return PushExecutionResult("blocked")
            push_descendants_terminal = self._push_descendants_are_terminal(
                push_result
            )
            if not push_descendants_terminal:
                self._retain_uncertain_push_evidence(
                    snapshot,
                    mutation_lease,
                    active_lease,
                    descendants_terminal=False,
                )
                uncertain_retained = True
                return self._completed_push_result("uncertain")
            try:
                postflight = await self._run_timed_push_command(
                    context.build_query_argv(endpoint),
                    settings=settings,
                    timeout=self._push_query_timeout,
                )
            except GitRunCancelled as cancellation:
                if cancellation.retained_child is not None:
                    settlement = await self._drain_push_preflight_child(
                        cancellation.retained_child
                    )
                    postflight_terminal = (
                        settlement.state not in {"alive", "uncertain"}
                        and settlement.owned_process_tree
                        and settlement.containment_proved
                    )
                else:
                    assert cancellation.result is not None
                    settled_postflight = await self._settle_push_preflight_result(
                        cancellation.result
                    )
                    postflight_terminal = self._push_descendants_are_terminal(
                        settled_postflight
                    )
                self._retain_uncertain_push_evidence(
                    snapshot,
                    mutation_lease,
                    active_lease,
                    descendants_terminal=(
                        push_descendants_terminal and postflight_terminal
                    ),
                )
                uncertain_retained = True
                return self._completed_push_result("uncertain")
            postflight = await self._settle_push_preflight_result(postflight)
            post_observation = self._proved_remote_observation(
                postflight,
                snapshot,
            )
            state = self._classify_exact_push_result(
                push_result,
                post_observation,
                snapshot,
            )
            if state == "succeeded":
                self._owner.clear_push_candidate(
                    snapshot.policy.candidate_capture
                )
            elif state == "uncertain":
                self._retain_uncertain_push_evidence(
                    snapshot,
                    mutation_lease,
                    active_lease,
                    descendants_terminal=(
                        push_descendants_terminal
                        and self._push_descendants_are_terminal(postflight)
                    ),
                )
                uncertain_retained = True
            return self._completed_push_result(state)
        except _PushContainmentUnproved as unproved:
            if self._unsettled_push_preflight is not None:
                raise RuntimeError(
                    "Push containment ownership transfer failed"
                ) from None
            self._unsettled_push_preflight = _UnsettledPushPreflight(
                binding=binding,
                retained_child=unproved.retained_child,
                mutation_lease=mutation_lease,
                context=context,
                active_lease=active_lease,
                authorization=authorization,
            )
            containment_quarantined = True
        except asyncio.CancelledError:
            if not self._push_child_started:
                return PushExecutionResult("cancelled")
            self._retain_uncertain_push_evidence(
                snapshot,
                mutation_lease,
                active_lease,
                descendants_terminal=False,
            )
            uncertain_retained = True
            return self._completed_push_result("uncertain")
        except (NetworkContextError, OSError, PushContractError, ValueError):
            if not self._push_child_started:
                return PushExecutionResult("blocked")
            self._retain_uncertain_push_evidence(
                snapshot,
                mutation_lease,
                active_lease,
                descendants_terminal=False,
            )
            uncertain_retained = True
            return self._completed_push_result("uncertain")
        finally:
            if not containment_quarantined and not uncertain_retained:
                self._owner._revoke_destination_authorization(authorization)
                if self._push_authorization is authorization:
                    self._push_authorization = None
                active_lease.release()
                self._close_or_retain_push_context(context)
                mutation_lease.release()
        if containment_quarantined:
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

    async def _run_timed_push_command(
        self,
        argv: Sequence[GitArg],
        *,
        settings: object,
        timeout: float,
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
    ) -> GitCommandResult:
        """Run one frozen-context command and enforce its injected-clock bound."""
        started = self._read_push_clock()
        result = await self._runner.run(
            argv,
            cwd=settings.cwd,
            environment=settings.environment,
            stdin=settings.stdin,
            timeout=timeout,
            stdout_limit=_PUSH_REMOTE_OUTPUT_LIMIT_BYTES,
            stderr_limit=DEFAULT_GIT_STDERR_LIMIT_BYTES,
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
            owned_process_tree=True,
        )
        elapsed = self._read_push_clock() - started
        if elapsed > timeout and not result.timed_out:
            result = replace(result, timed_out=True)
        return result

    def _read_push_clock(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("clock must return a finite number")
        return float(value)

    @staticmethod
    def _push_result_from_settlement(
        settlement: RetainedGitChildSettlement,
    ) -> GitCommandResult:
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr,
            termination_uncertain=(
                settlement.state == "contained_uncertain"
                or settlement.stop_requested
                or settlement.force_stopped
            ),
            stop_requested=settlement.stop_requested,
            force_stopped=settlement.force_stopped,
            output_overflow=settlement.output_overflow,
            owned_process_tree=settlement.owned_process_tree,
            containment_proved=settlement.containment_proved,
        )

    def _push_cancelled_result(
        self,
        cancellation: GitRunCancelled,
    ) -> GitCommandResult:
        if cancellation.result is not None:
            return cancellation.result
        return GitCommandResult(
            None,
            b"",
            b"",
            termination_uncertain=True,
            retained_child=cancellation.retained_child,
            owned_process_tree=True,
        )

    @staticmethod
    def _proved_network_result(result: GitCommandResult) -> bool:
        return (
            result.returncode is not None
            and result.returncode >= 0
            and not result.timed_out
            and not result.termination_uncertain
            and not result.stop_requested
            and not result.force_stopped
            and not result.output_overflow
            and result.owned_process_tree
            and result.containment_proved
            and result.retained_child is None
        )

    def _proved_remote_observation(
        self,
        result: GitCommandResult,
        snapshot: _PushReviewSnapshot,
    ) -> str | None:
        if not self._proved_network_result(result) or result.returncode != 0:
            return None
        candidate = snapshot.policy.candidate_capture.candidate
        observation = parse_ls_remote_refs(
            result.stdout,
            snapshot.policy.configuration.transport.destination.destination_ref,
            candidate.parent_oid,
            candidate.candidate_oid,
        )
        return observation.state

    def _classify_exact_push_result(
        self,
        result: GitCommandResult,
        post_observation: str | None,
        snapshot: _PushReviewSnapshot,
    ) -> PushOutcomeState:
        if not self._proved_network_result(result):
            return "uncertain"
        candidate = snapshot.policy.candidate_capture.candidate
        parsed = parse_push_porcelain(
            result.stdout,
            candidate.candidate_oid,
            snapshot.policy.configuration.transport.destination.destination_ref,
        )
        if (
            result.returncode == 0
            and parsed.state == "accepted"
            and post_observation == "candidate"
        ):
            return "succeeded"
        if (
            result.returncode != 0
            and parsed.state == "rejected"
            and post_observation == "parent"
        ):
            return "failed_no_update_observed"
        return "uncertain"

    @staticmethod
    def _completed_push_result(state: PushOutcomeState) -> PushExecutionResult:
        return PushExecutionResult(state, push_outcome_copy(state))

    def _retain_uncertain_push_evidence(
        self,
        snapshot: _PushReviewSnapshot,
        mutation_lease: GitMutationLease,
        context_lease: NetworkContextLease,
        *,
        descendants_terminal: bool,
    ) -> _UncertainPushEvidence:
        """Destroy update authority and retain only exact query evidence."""
        endpoint = snapshot.policy.configuration.transport.endpoint
        consumed_review = snapshot.consumed_review
        if endpoint is None or consumed_review is None:
            raise RuntimeError("Exact uncertain push evidence is unavailable")
        projection = push_recovery_copy(
            snapshot.projection.destination,
            RemoteRefObservation("malformed"),
        )
        retained = self._owner._retain_uncertain_push(
            mutation_lease,
            consumed_review,
            projection,
            descendants_terminal=descendants_terminal,
        )
        if retained is None:
            raise RuntimeError("Exact uncertain push evidence was refused")
        recovery_capture, recovery_handle = retained
        if self._push_destination_policy is snapshot.policy:
            self._push_destination_policy = None
        if self._push_authorization is snapshot.authorization:
            self._push_authorization = None
        evidence = _UncertainPushEvidence(
            binding=snapshot.binding,
            operation_id=self._push_operation_generation,
            recovery_capture=recovery_capture,
            recovery_handle=recovery_handle,
            candidate_capture=snapshot.policy.candidate_capture,
            endpoint=endpoint,
            destination=snapshot.projection.destination,
            parent_oid=recovery_capture.parent_oid,
            candidate_oid=recovery_capture.candidate_oid,
            repository_trust_generation=(
                recovery_capture.repository_trust_generation
            ),
            destination_policy_generation=(
                recovery_capture.destination_policy_generation
            ),
            destination_authorization_epoch=(
                recovery_capture.destination_authorization_epoch
            ),
            context=snapshot.context,
            context_lease=context_lease,
            mutation_lease=mutation_lease,
            descendants_terminal=descendants_terminal,
        )
        self._uncertain_push = evidence
        return evidence

    @staticmethod
    def _push_descendants_are_terminal(result: GitCommandResult) -> bool:
        """Return exact local child-tree settlement independent of outcome."""
        return (
            result.retained_child is None
            and result.owned_process_tree
            and result.containment_proved
        )

    def authorize_push_recovery(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> bool:
        """Freshly authorize one query of the retained frozen destination."""
        if self._sealed:
            return False
        evidence = self._uncertain_push
        if (
            evidence is None
            or evidence.binding != binding
            or operation is not self._retained_push_operation
            or operation.binding != evidence.binding
            or operation.operation_id != evidence.operation_id
            or operation.candidate != evidence.candidate_capture.availability
            or not evidence.descendants_terminal
            or (
                self._push_recovery_cycle is not None
                and not self._push_recovery_cycle.done()
            )
        ):
            return False
        handle = self._owner._authorize_push_recovery(
            evidence.recovery_capture
        )
        if handle is None:
            return False
        self._uncertain_push = replace(evidence, recovery_handle=handle)
        return True

    def check_push_again(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> asyncio.Task[PushRecoveryProjection]:
        """Start exactly one query-only check of an uncertain destination."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        evidence = self._uncertain_push
        if (
            evidence is None
            or evidence.binding != binding
            or operation is not self._retained_push_operation
            or operation.binding != evidence.binding
            or operation.operation_id != evidence.operation_id
            or operation.candidate != evidence.candidate_capture.availability
        ):
            raise GitMutationAdmissionError(
                "invalid_capability",
                "No exact recovery operation is available to check",
            )
        if not evidence.descendants_terminal:
            raise GitMutationAdmissionError(
                "recovery_not_ready",
                "Owned push descendants have not settled",
            )
        if (
            self._push_recovery_cycle is not None
            and not self._push_recovery_cycle.done()
        ):
            raise GitMutationAdmissionError(
                "mutation_active",
                "A push recovery query is already active",
            )
        handle = evidence.recovery_handle
        if handle is None or not self._owner._consume_push_recovery(
            evidence.recovery_capture,
            handle,
        ):
            raise GitMutationAdmissionError(
                "authorization_required",
                "Fresh authorization is required for the retained destination",
            )
        prior_evidence = evidence
        next_operation_id = self._push_operation_generation + 1
        evidence = replace(
            evidence,
            operation_id=next_operation_id,
            recovery_handle=None,
        )
        loop = asyncio.get_running_loop()
        try:
            cycle = loop.create_task(self._run_push_recovery_cycle(evidence))
            waiter = loop.create_task(self._shield_push_recovery_cycle(cycle))
        except BaseException:
            replacement = self._owner._authorize_push_recovery(
                evidence.recovery_capture
            )
            self._uncertain_push = replace(
                prior_evidence,
                recovery_handle=replacement,
            )
            raise
        self._uncertain_push = evidence
        self._push_recovery_cycle = cycle
        self._push_recovery_waiter = waiter
        self._push_operation_generation = next_operation_id
        self._retained_push_operation = RetainedPushOperation(
            binding,
            next_operation_id,
            "recovery",
            evidence.candidate_capture.availability,
            cycle,
        )
        cycle.add_done_callback(self._push_recovery_cycle_completed)
        waiter.add_done_callback(self._push_recovery_waiter_completed)
        return waiter

    @staticmethod
    async def _shield_push_recovery_cycle(
        cycle: asyncio.Task[PushRecoveryProjection],
    ) -> PushRecoveryProjection:
        return await asyncio.shield(cycle)

    def _push_recovery_cycle_completed(
        self,
        cycle: asyncio.Task[PushRecoveryProjection],
    ) -> None:
        if self._push_recovery_cycle is cycle:
            self._push_recovery_cycle = None
        if not cycle.cancelled():
            cycle.exception()

    def _push_recovery_waiter_completed(
        self,
        waiter: asyncio.Task[PushRecoveryProjection],
    ) -> None:
        if self._push_recovery_waiter is waiter:
            self._push_recovery_waiter = None
        if not waiter.cancelled():
            waiter.exception()

    def _push_recovery_settlement_completed(
        self,
        cycle: asyncio.Task[None],
    ) -> None:
        if self._push_recovery_settlement_cycle is cycle:
            self._push_recovery_settlement_cycle = None
        if not cycle.cancelled():
            cycle.exception()

    def _start_push_recovery_settlement(
        self,
        evidence: _UncertainPushEvidence,
        projection: PushRecoveryProjection,
    ) -> None:
        """Retain one query-free exact-child settlement continuation."""
        current = self._push_recovery_settlement_cycle
        if current is not None and not current.done():
            return
        cycle = asyncio.get_running_loop().create_task(
            self._settle_push_recovery_child(evidence, projection)
        )
        self._push_recovery_settlement_cycle = cycle
        cycle.add_done_callback(self._push_recovery_settlement_completed)

    async def _settle_push_recovery_child(
        self,
        evidence: _UncertainPushEvidence,
        projection: PushRecoveryProjection,
    ) -> None:
        """Resume only retained child proof; never contact the destination."""
        token = evidence.retained_child
        if token is None:
            return
        while self._uncertain_push is evidence:
            try:
                await self._drain_push_preflight_child(token)
                break
            except _PushContainmentUnproved:
                if self._sealed:
                    return
                await asyncio.sleep(0.01)
        else:
            return
        if self._uncertain_push is not evidence:
            return
        if not self._owner._mark_push_recovery_descendants_terminal(
            evidence.recovery_capture
        ):
            return
        next_handle = self._owner._publish_push_recovery(
            evidence.recovery_capture,
            projection,
        )
        if self._uncertain_push is evidence:
            self._uncertain_push = replace(
                evidence,
                recovery_handle=next_handle,
                descendants_terminal=True,
                retained_child=None,
            )

    async def _run_push_recovery_cycle(
        self,
        evidence: _UncertainPushEvidence,
    ) -> PushRecoveryProjection:
        """Query only retained endpoint/ref and never reconstruct a push."""
        settings = evidence.context.command_settings()
        result: GitCommandResult | None = None
        retained_child: RetainedGitChildToken | None = None
        descendants_terminal = evidence.descendants_terminal
        try:
            result = await self._run_timed_push_command(
                evidence.context.build_query_argv(evidence.endpoint),
                settings=settings,
                timeout=self._push_query_timeout,
            )
            result = await self._settle_push_preflight_result(result)
            descendants_terminal = self._push_descendants_are_terminal(result)
        except GitRunCancelled as cancellation:
            try:
                if cancellation.retained_child is not None:
                    settlement = await self._drain_push_preflight_child(
                        cancellation.retained_child
                    )
                    result = self._push_result_from_settlement(settlement)
                    descendants_terminal = True
                elif cancellation.result is not None:
                    result = await self._settle_push_preflight_result(
                        cancellation.result
                    )
                    descendants_terminal = self._push_descendants_are_terminal(
                        result
                    )
            except _PushContainmentUnproved as unproved:
                retained_child = unproved.retained_child
                descendants_terminal = False
        except _PushContainmentUnproved as unproved:
            retained_child = unproved.retained_child
            descendants_terminal = False
        except (NetworkContextError, OSError, PushContractError, ValueError):
            result = None

        observation = RemoteRefObservation("missing")
        if (
            result is not None
            and self._proved_network_result(result)
            and result.returncode == 0
        ):
            observation = parse_ls_remote_refs(
                result.stdout,
                evidence.destination.destination_ref,
                evidence.parent_oid,
                evidence.candidate_oid,
            )
            if observation.state == "malformed":
                observation = RemoteRefObservation("missing")
        projection = push_recovery_copy(evidence.destination, observation)
        if observation.state == "candidate":
            self._owner.clear_push_candidate(evidence.candidate_capture)
            self._owner._clear_push_recovery(evidence.recovery_capture)
            self._release_uncertain_push_evidence(evidence)
            return projection

        if not descendants_terminal:
            self._owner._mark_push_recovery_descendants_unsettled(
                evidence.recovery_capture
            )
            next_handle = None
        else:
            next_handle = self._owner._publish_push_recovery(
                evidence.recovery_capture,
                projection,
            )
        current = self._uncertain_push
        parked_evidence: _UncertainPushEvidence | None = None
        if current is evidence or (
            current is not None
            and current.recovery_capture is evidence.recovery_capture
        ):
            parked_evidence = replace(
                evidence,
                recovery_handle=next_handle,
                descendants_terminal=descendants_terminal,
                retained_child=retained_child,
            )
            self._uncertain_push = parked_evidence
        if (
            parked_evidence is not None
            and retained_child is not None
            and not descendants_terminal
        ):
            self._start_push_recovery_settlement(
                parked_evidence,
                projection,
            )
        return projection

    def _release_uncertain_push_evidence(
        self,
        evidence: _UncertainPushEvidence,
    ) -> None:
        """Release exact certain evidence only after its descendants settle."""
        evidence.context_lease.release()
        self._close_or_retain_push_context(evidence.context)
        evidence.mutation_lease.release()
        current = self._uncertain_push
        if current is not None and (
            current is evidence
            or current.recovery_capture is evidence.recovery_capture
        ):
            self._uncertain_push = None

    def retained_push_operation(
        self,
        binding: SessionBinding,
    ) -> RetainedPushOperation | None:
        """Return the latest exact-binding push work without transferring it."""
        self._observe_push_rebinding()
        operation = self._retained_push_operation
        if operation is None or operation.binding != binding:
            return None
        return operation

    def cancel_push(
        self,
        binding: SessionBinding,
        operation: RetainedPushOperation,
    ) -> bool:
        """Drop one settled review while preserving its exact candidate."""
        self._observe_push_rebinding()
        if (
            operation is not self._retained_push_operation
            or operation.binding != binding
            or self._unsettled_push_preflight is not None
        ):
            return False
        local_cycle = self._push_local_proof_cycle
        if (
            operation.kind == "local_proof"
            and local_cycle is not None
            and not local_cycle.done()
        ):
            local_cycle.cancel()
            return True
        preflight_cycle = self._push_preflight_cycle
        if (
            operation.kind == "preflight"
            and preflight_cycle is not None
            and not preflight_cycle.done()
        ):
            preflight_cycle.cancel()
            return True
        if not operation.settled:
            push_cycle = self._push_cycle
            if (
                operation.kind == "push"
                and push_cycle is not None
                and not push_cycle.done()
                and not self._push_child_started
            ):
                push_cycle.cancel()
                return True
            return False
        if operation.kind == "local_proof":
            policy = self._push_destination_policy
            if policy is not None:
                self._owner._revoke_destination_policy(policy.owner_capture)
            self._push_destination_policy = None
            self._push_authorization = None
            self._retained_push_operation = None
            return True
        matching = tuple(
            (handle, snapshot)
            for handle, snapshot in self._push_review_snapshots.items()
            if snapshot.binding == binding
        )
        if len(matching) != 1:
            return False
        handle, snapshot = matching[0]
        self._push_review_snapshots.pop(handle, None)
        self._owner._revoke_destination_authorization(snapshot.authorization)
        snapshot.review_lease.release()
        self._close_or_retain_push_context(snapshot.context)
        self._retained_push_operation = None
        return True

    def _observe_push_rebinding(self) -> None:
        """Release reviews whose exact owner authority is no longer current."""
        self._retry_pending_push_context_cleanup()
        current_binding = self._owner.current_binding()
        for handle, snapshot in tuple(self._push_review_snapshots.items()):
            current = (
                snapshot.binding == current_binding
                and self._owner._destination_authorization_matches(
                    snapshot.policy.owner_capture,
                    snapshot.authorization,
                )
            )
            if current:
                continue
            if self._push_review_snapshots.get(handle) is not snapshot:
                continue
            self._push_review_snapshots.pop(handle)
            self._owner._revoke_destination_authorization(
                snapshot.authorization
            )
            snapshot.review_lease.release()
            self._close_or_retain_push_context(snapshot.context)
            if self._push_destination_policy is snapshot.policy:
                self._push_destination_policy = None
            if self._push_authorization is snapshot.authorization:
                self._push_authorization = None
            operation = self._retained_push_operation
            if (
                operation is not None
                and operation.kind == "preflight"
                and operation.binding == snapshot.binding
                and operation.settled
            ):
                self._retained_push_operation = None

    def _close_or_retain_push_context(
        self,
        context: NetworkGitExecutionContext,
    ) -> bool:
        """Close one exact context or retain its cleanup authority for retry."""
        try:
            cleaned = context.close()
        except NetworkContextError:
            cleaned = False
        if cleaned is True:
            self._pending_push_contexts.discard(context)
            return True
        self._pending_push_contexts.add(context)
        return False

    def _retry_pending_push_context_cleanup(self) -> None:
        """Retry only exact contexts whose prior close could not complete."""
        for context in tuple(self._pending_push_contexts):
            self._close_or_retain_push_context(context)

    async def _prove_push_destination(
        self,
        binding: SessionBinding,
        *,
        prior: _PushDestinationPolicySnapshot | None = None,
    ) -> tuple[
        _PushDestinationPolicySnapshot | None,
        Literal["blocked", "stale"],
    ]:
        current = self._owner.snapshot(binding)
        availability = current.push_candidate
        repository = current.trusted_repository
        if (
            self._sealed
            or binding != self._owner.current_binding()
            or availability is None
            or repository is None
            or self._git_executable is None
            or not self._repository_identity_matches(binding, repository)
        ):
            return None, "stale"

        candidate = availability.candidate
        proof: _PrivatePushProofDirectory | None = None
        try:
            proof = _create_private_push_proof_directory(repository)
            hooks_directory = proof.create_directory("hooks")
        except (OSError, RuntimeError):
            if proof is not None:
                proof.cleanup()
            return None, "blocked"
        with proof:
            prefix = self._local_push_proof_prefix(hooks_directory)

            exec_path_result = await self._run_local_push_proof_command(
                repository,
                (*prefix, "--exec-path"),
                proof=proof,
            )
            git_exec_path = (
                _canonical_directory_from_git(exec_path_result.stdout)
                if _command_succeeded(exec_path_result)
                else None
            )
            if git_exec_path is None:
                return None, "blocked"

            branch_result = await self._run_local_push_proof_command(
                repository,
                (*prefix, "symbolic-ref", "--quiet", "HEAD"),
                proof=proof,
            )
            head_result = await self._run_local_push_proof_command(
                repository,
                (
                    *prefix,
                    "rev-parse",
                    "--show-object-format=storage",
                    "--verify",
                    "--quiet",
                    "HEAD^{commit}",
                ),
                proof=proof,
            )
            branch = (
                _single_git_value(branch_result.stdout)
                if _command_succeeded(branch_result)
                else None
            )
            format_and_head = (
                _git_object_format_and_oid(head_result.stdout)
                if _command_succeeded(head_result)
                else None
            )
            if branch != candidate.local_branch_ref:
                return None, "stale"
            if format_and_head is None:
                return None, "blocked"
            object_format, head_oid = format_and_head
            if head_oid != candidate.candidate_oid:
                return None, "stale"
            expected_oid_width = 40 if object_format == "sha1" else 64
            if (
                len(candidate.parent_oid) != expected_oid_width
                or len(candidate.candidate_oid) != expected_oid_width
            ):
                return None, "blocked"

            object_result = await self._run_local_push_proof_command(
                repository,
                (*prefix, "cat-file", "commit", candidate.candidate_oid),
                proof=proof,
            )
            if not _command_succeeded(object_result):
                return None, "blocked"
            try:
                raw_commit = parse_raw_commit_object(object_result.stdout)
            except CommitContractError:
                return None, "blocked"
            if (
                raw_commit.parent_object_id != candidate.parent_oid
                or len(raw_commit.parent_object_id) != expected_oid_width
                or len(raw_commit.tree_object_id) != expected_oid_width
            ):
                return None, "blocked"
            fresh_head = HeadIdentity.attached(branch, head_oid)
            if prior is None:
                candidate_capture = (
                    self._owner._capture_push_candidate_after_fresh_proof(
                        binding,
                        candidate_generation=availability.generation,
                        repository=repository,
                        head=fresh_head,
                        sole_parent_oid=raw_commit.parent_object_id,
                    )
                )
                if candidate_capture is None:
                    return None, "stale"
            else:
                candidate_capture = prior.candidate_capture
                if not self._owner._revalidate_push_candidate_after_fresh_proof(
                    candidate_capture,
                    repository=repository,
                    head=fresh_head,
                    sole_parent_oid=raw_commit.parent_object_id,
                ):
                    return None, "stale"

            configuration_proof = await self._read_push_configuration(
                repository,
                prefix,
                candidate.local_branch_ref,
                proof=proof,
            )
            if configuration_proof is None:
                return None, "blocked"
            configuration, network_configuration = configuration_proof

            paths_result = await self._run_local_push_proof_command(
                repository,
                (
                    *prefix,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-z",
                    "-r",
                    "--no-renames",
                    candidate.parent_oid,
                    candidate.candidate_oid,
                    "--",
                ),
                proof=proof,
            )
            included_paths = (
                _parse_candidate_paths(paths_result.stdout)
                if _command_succeeded(paths_result)
                else None
            )
            if included_paths is None or not included_paths:
                return None, "blocked"

            attribute_context = self._prepare_candidate_attribute_proof(
                repository,
                proof,
                hooks_directory,
                object_format,
            )
            if attribute_context is None:
                return None, "blocked"
            (
                attribute_prefix,
                index_environment,
                source_objects,
            ) = attribute_context
            tree_result = await self._run_local_push_proof_command(
                repository,
                (
                    *attribute_prefix,
                    "read-tree",
                    raw_commit.tree_object_id,
                ),
                proof=proof,
                environment=index_environment,
                capture_index=True,
            )
            if not _command_succeeded(tree_result):
                return None, "blocked"
            attribute_result = await self._run_local_push_proof_command(
                repository,
                (
                    *attribute_prefix,
                    "check-attr",
                    "--cached",
                    "-z",
                    "--stdin",
                    "filter",
                ),
                proof=proof,
                environment=index_environment,
                stdin=b"\0".join(included_paths) + b"\0",
            )
            attribute_fingerprint = (
                _candidate_attribute_fingerprint(
                    included_paths,
                    attribute_result.stdout,
                )
                if _command_succeeded(attribute_result)
                else None
            )
            if attribute_fingerprint is None or not proof.validate():
                return None, "blocked"
            if not self._repository_identity_matches(binding, repository):
                return None, "stale"

            if prior is not None:
                unchanged = (
                    prior.candidate_tree_oid == raw_commit.tree_object_id
                    and prior.included_paths_fingerprint
                    == attribute_fingerprint
                    and prior.configuration.configuration_fingerprint
                    == configuration.configuration_fingerprint
                    and prior.configuration.transport.configured_identity
                    == configuration.transport.configured_identity
                    and prior.network_configuration.copy_fingerprint
                    == network_configuration.copy_fingerprint
                    and prior.source_objects.identity_fingerprint
                    == source_objects.identity_fingerprint
                    and prior.source_objects.object_format == object_format
                    and prior.git_exec_path == str(git_exec_path)
                )
                if not unchanged or not proof.cleanup():
                    return None, "blocked"
                return prior, "blocked"

            if not proof.cleanup():
                return None, "blocked"
            owner_capture = (
                self._owner._capture_destination_policy_after_fresh_proof(
                    candidate_capture,
                    configuration_fingerprint=(
                        configuration.configuration_fingerprint
                    ),
                    configured_remote_label=configuration.tracking_remote,
                    configured_destination_identity=(
                        configuration.transport.configured_identity
                    ),
                    destination=configuration.transport.destination,
                    candidate_tree_oid=raw_commit.tree_object_id,
                    included_paths_fingerprint=attribute_fingerprint,
                )
            )
            if owner_capture is None:
                return None, "stale"
            return (
                _PushDestinationPolicySnapshot(
                    owner_capture=owner_capture,
                    candidate_capture=candidate_capture,
                    configuration=configuration,
                    network_configuration=network_configuration,
                    source_objects=source_objects,
                    git_exec_path=str(git_exec_path),
                    candidate_tree_oid=raw_commit.tree_object_id,
                    included_paths_fingerprint=attribute_fingerprint,
                ),
                "blocked",
            )

    def _local_push_proof_prefix(
        self,
        hooks_directory: Path,
    ) -> tuple[str, ...]:
        return (
            self._git_executable_or_raise(),
            "--no-replace-objects",
            "-c",
            f"core.hooksPath={hooks_directory}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
        )

    def _prepare_candidate_attribute_proof(
        self,
        repository: RepositoryIdentity,
        proof: _PrivatePushProofDirectory,
        hooks_directory: Path,
        object_format: Literal["sha1", "sha256"],
    ) -> tuple[
        tuple[str, ...],
        dict[str, str],
        SourceObjectDirectoryAuthorization,
    ] | None:
        """Create an isolated exact-tree attribute reader with object-only access."""
        source_objects = Path(repository.git_common_dir) / "objects"
        try:
            canonical_objects = source_objects.resolve(strict=True)
            if canonical_objects != source_objects or not source_objects.is_dir():
                return None
            source_objects_identity = _filesystem_identity(source_objects)
            if not proof.track_external_directory(
                source_objects,
                source_objects_identity,
            ):
                return None
            source_objects_authorization = (
                _authorize_source_object_directory(
                    source_objects,
                    source_objects_identity,
                    object_format,
                )
            )
            git_dir = proof.create_directory("attribute.git")
            object_directory = proof.create_directory(
                "attribute.git/objects"
            )
            proof.create_directory("attribute.git/refs")
            worktree = proof.create_directory("attribute-worktree")
            home = proof.create_directory("attribute-home")
            config_home = proof.create_directory(
                "attribute-home/.config"
            )
            proof.create_file(
                "attribute.git/HEAD",
                b"ref: refs/heads/isolated\n",
            )
            proof.create_file(
                "attribute.git/config",
                _render_isolated_repository_config(object_format),
            )
            global_config = proof.create_file("global.gitconfig", b"")
            attributes_file = proof.create_file("global.attributes", b"")
            index_file = proof.reserve_index("candidate.index")
            if not proof.validate():
                return None
        except (NetworkContextError, OSError, RuntimeError):
            return None

        environment = build_local_push_proof_environment(
            self._environment,
            index_file=str(index_file),
        )
        environment.update(
            {
                "GIT_DIR": str(git_dir),
                "GIT_WORK_TREE": str(worktree),
                "GIT_OBJECT_DIRECTORY": str(object_directory),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(source_objects),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": str(global_config),
                "GIT_ATTR_NOSYSTEM": "1",
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(config_home),
            }
        )
        prefix = (
            *self._local_push_proof_prefix(hooks_directory),
            "-c",
            f"core.attributesFile={attributes_file}",
            f"--git-dir={git_dir}",
            f"--work-tree={worktree}",
        )
        return (
            prefix,
            environment,
            source_objects_authorization,
        )

    async def _read_push_configuration(
        self,
        repository: RepositoryIdentity,
        prefix: tuple[str, ...],
        branch_ref: str,
        *,
        proof: _PrivatePushProofDirectory,
    ) -> tuple[
        _ResolvedPushConfiguration,
        NetworkConfigAuthorization,
    ] | None:
        argv = (
            *prefix,
            "config",
            "--null",
            "--show-origin",
            "--show-scope",
            "--includes",
            "--get-regexp",
            _LOCAL_PUSH_CONFIG_PATTERN,
        )
        first = await self._run_local_push_proof_command(
            repository,
            argv,
            proof=proof,
        )
        if not _command_succeeded(first):
            return None
        first_facts = _parse_push_config_facts(
            first.stdout,
            repository,
            self._environment,
        )
        if first_facts is None:
            return None
        second = await self._run_local_push_proof_command(
            repository,
            argv,
            proof=proof,
        )
        if not _command_succeeded(second) or second.stdout != first.stdout:
            return None
        second_facts = _parse_push_config_facts(
            second.stdout,
            repository,
            self._environment,
        )
        if second_facts is None or second_facts != first_facts:
            return None
        try:
            resolved = _resolve_push_configuration(
                second_facts,
                branch_ref,
                self._transport_admission,
            )
            network_configuration = _authorize_network_config_snapshot(
                second_facts,
                configuration_fingerprint=(
                    resolved.configuration_fingerprint
                ),
                destination=resolved.transport.destination,
            )
        except (NetworkContextError, PushContractError):
            return None
        return resolved, network_configuration

    async def _run_local_push_proof_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
        *,
        proof: _PrivatePushProofDirectory,
        environment: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        capture_index: bool = False,
    ) -> GitCommandResult:
        if not proof.validate():
            return GitCommandResult(126, b"", b"")
        try:
            result = await self._runner.run(
                tuple(argv),
                cwd=repository.worktree_root,
                environment=(
                    build_local_push_proof_environment(self._environment)
                    if environment is None
                    else environment
                ),
                stdin=stdin,
                timeout=self._status_timeout,
                stdout_limit=_LOCAL_PUSH_STDOUT_LIMIT_BYTES,
                stderr_limit=DEFAULT_GIT_STDERR_LIMIT_BYTES,
            )
        except GitRunCancelled as cancellation:
            if cancellation.result is not None:
                result = await self._settle_commit_proof_result(
                    cancellation.result
                )
            else:
                retained_child = cancellation.retained_child
                assert retained_child is not None
                await self._drain_commit_proof_child(retained_child)
                raise
        except OSError:
            result = GitCommandResult(127, b"", b"")
        else:
            result = await self._settle_commit_proof_result(result)
        if capture_index and not proof.capture_index():
            return GitCommandResult(126, b"", b"")
        if not proof.validate():
            return GitCommandResult(126, b"", b"")
        return result

    def retained_status(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[SessionGitStatus] | None:
        """Return matching active status work without requesting a rerun."""
        self._observe_commit_rebinding()
        cycle = self._status_cycle
        waiter = self._status_waiter
        if (
            binding != self._owner.current_binding()
            or cycle is None
            or cycle.done()
            or waiter is None
            or waiter.done()
        ):
            return None
        pending = self._pending_status
        if self._status_cycle_binding == binding:
            return waiter
        if pending is not None and pending[0] == binding:
            return waiter
        return None

    def start_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
    ) -> asyncio.Task[SessionGitStatus]:
        """Synchronously admit one retained trusted status query."""
        if self._sealed:
            raise GitStatusAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        self._observe_commit_rebinding()
        snapshot = self._owner.snapshot(binding)
        repository = snapshot.trusted_repository
        if binding != self._owner.current_binding():
            raise GitStatusAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if repository is None:
            raise GitStatusAdmissionError(
                "untrusted",
                "Repository trust is required before Git status",
            )
        if self._status_cycle is not None and not self._status_cycle.done():
            admission = self._owner.admit_status(binding)
            if admission.reason == "mutation_active":
                raise GitStatusAdmissionError(
                    "mutation_active",
                    "A File Notes Git mutation is active",
                )
            if admission.reason != "status_active":
                if admission.lease is not None:
                    admission.lease.release()
                reason = admission.reason or "status_active"
                raise GitStatusAdmissionError(
                    reason,
                    "File Notes Git status cannot be coalesced",
                )
            if self._rerun_available:
                invalidation_generation = (
                    admission.invalidation_generation
                )
                if invalidation_generation is None:
                    raise RuntimeError(
                        "Active status admission omitted its generation"
                    )
                self._status_request_generation += 1
                self._pending_status = (
                    binding,
                    tuple(changes),
                    repository,
                    self._status_request_generation,
                    invalidation_generation,
                )
            else:
                self._status_request_generation += 1
                self._status_dirty = True
            waiter = self._status_waiter
            if waiter is None or waiter.done():
                waiter = self._create_task(
                    self._shield_status_cycle(self._status_cycle)
                )
                self._status_waiter = waiter
            return waiter

        admission = self._owner.admit_status(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "status_active"
            raise GitStatusAdmissionError(
                reason,
                "File Notes Git status admission was refused",
            )
        self._status_request_generation += 1
        request_generation = self._status_request_generation
        self._pending_status = None
        self._rerun_available = True
        self._status_dirty = False
        cycle: asyncio.Task[SessionGitStatus] | None = None
        try:
            cycle = self._create_task(
                self._run_status_cycle(
                    binding,
                    tuple(changes),
                    repository,
                    lease,
                    request_generation,
                )
            )
            waiter = self._create_task(self._shield_status_cycle(cycle))
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            self._pending_status = None
            self._rerun_available = False
            self._status_dirty = False
            raise
        self._status_cycle = cycle
        self._status_cycle_binding = binding
        self._status_waiter = waiter
        cycle.add_done_callback(self._status_cycle_completed)
        return waiter

    def start_stage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]:
        """Synchronously admit and retain one exact Stage operation."""
        requested = tuple(dict.fromkeys(group_ids))
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        orphaned_commit = self._observe_commit_rebinding()
        snapshot = self._owner.snapshot(binding)
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if orphaned_commit:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A retained guarded commit child is still settling",
            )
        repository = snapshot.trusted_repository
        if repository is None:
            raise GitMutationAdmissionError(
                "untrusted",
                "Repository trust is required before staging",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        status_cycle = self._status_cycle
        cycle: asyncio.Task[GitActionResult] | None = None
        try:
            cycle = self._create_action_task(
                self._run_stage_cycle(
                    binding,
                    requested,
                    repository,
                    lease,
                    status_cycle,
                )
            )
            waiter = self._create_action_task(
                self._shield_action_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._action_cycle = cycle
        self._action_waiter = waiter
        cycle.add_done_callback(self._action_cycle_completed)
        return waiter

    def start_unstage(
        self,
        binding: SessionBinding,
        group_ids: Collection[int],
    ) -> asyncio.Task[GitActionResult]:
        """Synchronously admit and retain one exact Unstage operation."""
        requested = tuple(dict.fromkeys(group_ids))
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        orphaned_commit = self._observe_commit_rebinding()
        snapshot = self._owner.snapshot(binding)
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if orphaned_commit:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A retained guarded commit child is still settling",
            )
        repository = snapshot.trusted_repository
        if repository is None:
            raise GitMutationAdmissionError(
                "untrusted",
                "Repository trust is required before unstaging",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        status_cycle = self._status_cycle
        cycle: asyncio.Task[GitActionResult] | None = None
        try:
            cycle = self._create_action_task(
                self._run_unstage_cycle(
                    binding,
                    requested,
                    repository,
                    lease,
                    status_cycle,
                )
            )
            waiter = self._create_action_task(
                self._shield_action_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._action_cycle = cycle
        self._action_waiter = waiter
        cycle.add_done_callback(self._action_cycle_completed)
        return waiter

    def start_commit_review(
        self,
        binding: SessionBinding,
        subject: str,
        body: str = "",
    ) -> asyncio.Task[CommitReviewResult]:
        """Admit and retain one read-only guarded commit review preflight."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        orphaned_commit = self._observe_commit_rebinding()
        snapshot = self._owner.snapshot(binding)
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if orphaned_commit:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A retained guarded commit child is still settling",
            )
        repository = snapshot.trusted_repository
        if repository is None:
            raise GitMutationAdmissionError(
                "untrusted",
                "Repository trust is required before commit review",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        self._discard_commit_review_snapshots()
        cycle: asyncio.Task[CommitReviewResult] | None = None
        try:
            cycle = self._create_task(
                self._run_commit_review_cycle(
                    binding,
                    repository,
                    snapshot.git_authority_generation,
                    subject,
                    body,
                    lease,
                )
            )
            waiter = self._create_task(self._shield_commit_review_cycle(cycle))
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            lease.release()
            raise
        self._commit_review_cycle = cycle
        self._commit_review_waiter = waiter
        self._retained_commit_operation = RetainedCommitOperation(
            binding,
            "review",
            cycle,
        )
        cycle.add_done_callback(self._commit_review_cycle_completed)
        return waiter

    async def _shield_commit_review_cycle(
        self,
        cycle: asyncio.Task[CommitReviewResult],
    ) -> CommitReviewResult:
        return await asyncio.shield(cycle)

    def _commit_review_cycle_completed(
        self,
        cycle: asyncio.Task[CommitReviewResult],
    ) -> None:
        if self._commit_review_cycle is cycle:
            self._commit_review_cycle = None
        if not cycle.cancelled():
            cycle.exception()

    def start_commit(
        self,
        binding: SessionBinding,
        handle: CommitReviewHandle,
        *,
        subject: str | None = None,
        body: str = "",
    ) -> asyncio.Task[CommitOutcome]:
        """Consume one review capability and retain one guarded commit cycle."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        orphaned_commit = self._observe_commit_rebinding()
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        if orphaned_commit:
            raise GitMutationAdmissionError(
                "mutation_active",
                "A retained guarded commit child is still settling",
            )
        if self._commit_cycle is not None and not self._commit_cycle.done():
            raise GitMutationAdmissionError(
                "mutation_active",
                "A guarded commit is already active",
            )
        if not isinstance(handle, CommitReviewHandle):
            raise GitMutationAdmissionError(
                "invalid_capability",
                "Commit review capability is invalid or already consumed",
            )
        admission = self._owner.admit_mutation(binding)
        lease = admission.lease
        if lease is None:
            reason = admission.reason or "mutation_active"
            raise GitMutationAdmissionError(
                reason,
                "File Notes Git mutation admission was refused",
            )
        snapshot = self._commit_review_snapshots.pop(handle._token, None)
        if snapshot is None:
            lease.release()
            raise GitMutationAdmissionError(
                "invalid_capability",
                "Commit review capability is invalid or already consumed",
            )
        self._discard_commit_review_snapshots()
        cycle: asyncio.Task[CommitOutcome] | None = None
        child_started_signal: asyncio.Future[bool] | None = None
        try:
            loop = asyncio.get_running_loop()
            child_started_signal = loop.create_future()
            cycle = loop.create_task(
                self._run_commit_cycle(
                    binding,
                    snapshot,
                    lease,
                    subject=subject,
                    body=body,
                )
            )
            waiter = loop.create_task(self._shield_commit_cycle(cycle))
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            if child_started_signal is not None and not child_started_signal.done():
                child_started_signal.set_result(False)
            self._owner._discard_commit_authority(snapshot.capture)
            lease.release()
            raise
        self._commit_child_started = False
        self._commit_child_started_signal = child_started_signal
        self._commit_cycle = cycle
        self._commit_waiter = waiter
        self._retained_commit_operation = RetainedCommitOperation(
            binding,
            "commit",
            cycle,
            child_started_signal,
        )
        cycle.add_done_callback(self._commit_cycle_completed)
        return waiter

    def _mark_commit_child_started(self) -> None:
        """Publish the exact successful-spawn boundary without yielding."""
        self._commit_child_started = True
        signal = self._commit_child_started_signal
        if signal is not None and not signal.done():
            signal.set_result(True)

    def retained_commit_operation(
        self,
        binding: SessionBinding,
    ) -> RetainedCommitOperation | None:
        """Return the latest exact-binding operation without transferring ownership."""
        self._observe_commit_rebinding()
        operation = self._retained_commit_operation
        if operation is None or operation.binding != binding:
            return None
        return operation

    def cancel_commit(
        self,
        binding: SessionBinding,
    ) -> bool:
        """Cancel exact review/confirmation work before a commit child starts."""
        operation = self._retained_commit_operation
        if operation is None:
            return False
        if operation.binding != binding:
            return False

        review_cycle = self._commit_review_cycle
        if (
            operation.kind == "review"
            and review_cycle is not None
            and not review_cycle.done()
        ):
            review_cycle.cancel()
            return True

        cycle = self._commit_cycle
        if (
            operation.kind == "commit"
            and cycle is not None
            and not cycle.done()
        ):
            if self._commit_child_started:
                return False
            cycle.cancel()
            return True

        if (
            operation.kind != "review"
            or not operation.settled
        ):
            return False
        matching_tokens = tuple(
            token
            for token, snapshot in self._commit_review_snapshots.items()
            if snapshot.capture.binding == binding
        )
        for token in matching_tokens:
            snapshot = self._commit_review_snapshots.pop(token, None)
            if snapshot is not None:
                self._owner._discard_commit_authority(snapshot.capture)
        self._retained_commit_operation = None
        return True

    async def _shield_commit_cycle(
        self,
        cycle: asyncio.Task[CommitOutcome],
    ) -> CommitOutcome:
        return await asyncio.shield(cycle)

    def _commit_cycle_completed(
        self,
        cycle: asyncio.Task[CommitOutcome],
    ) -> None:
        if self._commit_cycle is cycle:
            signal = self._commit_child_started_signal
            if signal is not None and not signal.done():
                signal.set_result(False)
            self._commit_cycle = None
            self._commit_waiter = None
            self._commit_child_started = False
            self._commit_child_started_signal = None
        if not cycle.cancelled():
            cycle.exception()

    def check_commit_again(
        self,
        binding: SessionBinding,
    ) -> asyncio.Task[CommitOutcome]:
        """Re-observe one exact attempt, proving it only when lifecycle-safe."""
        if self._sealed:
            raise GitMutationAdmissionError(
                "shutdown",
                "File Notes Git service is shut down",
            )
        self._observe_commit_rebinding()
        if binding != self._owner.current_binding():
            raise GitMutationAdmissionError(
                "stale_binding",
                "File Notes root binding is stale",
            )
        evidence = self._uncertain_commit
        if (
            evidence is None
            or evidence.proof.binding != binding
            or evidence.recovery_capability is None
        ):
            raise GitMutationAdmissionError(
                "invalid_capability",
                "No exact uncertain commit is available to check",
            )
        if (
            self._commit_recovery_cycle is not None
            and not self._commit_recovery_cycle.done()
        ):
            raise GitMutationAdmissionError(
                "mutation_active",
                "A guarded commit recovery is already active",
            )

        cycle: asyncio.Task[CommitOutcome] | None = None
        try:
            loop = asyncio.get_running_loop()
            cycle = loop.create_task(
                self._run_commit_recovery_cycle(binding, evidence)
            )
            waiter = loop.create_task(
                self._shield_commit_recovery_cycle(cycle)
            )
        except BaseException:
            if cycle is not None:
                cycle.cancel()
            raise
        self._commit_recovery_cycle = cycle
        self._commit_recovery_waiter = waiter
        self._retained_commit_operation = RetainedCommitOperation(
            binding,
            "recovery",
            cycle,
        )
        cycle.add_done_callback(self._commit_recovery_cycle_completed)
        return waiter

    async def _shield_commit_recovery_cycle(
        self,
        cycle: asyncio.Task[CommitOutcome],
    ) -> CommitOutcome:
        return await asyncio.shield(cycle)

    def _commit_recovery_cycle_completed(
        self,
        cycle: asyncio.Task[CommitOutcome],
    ) -> None:
        if self._commit_recovery_cycle is cycle:
            self._commit_recovery_cycle = None
            self._commit_recovery_waiter = None
        if not cycle.cancelled():
            cycle.exception()

    async def _run_commit_recovery_cycle(
        self,
        binding: SessionBinding,
        admitted_evidence: _UncertainCommitEvidence,
    ) -> CommitOutcome:
        """Prove one retained attempt without ever starting another commit."""
        evidence = self._uncertain_commit
        if evidence is not admitted_evidence:
            return _uncertain_commit_outcome()
        evidence, child_is_certain = self._settle_commit_recovery_child(evidence)
        if not child_is_certain:
            return _uncertain_commit_outcome()
        capability = evidence.recovery_capability
        if capability is None:
            return _uncertain_commit_outcome()
        admission = self._owner.admit_commit_recovery(binding, capability)
        lease = admission.lease
        capture = admission.capture
        if lease is None or capture is None:
            if admission.reason in {"stale_binding", "invalid_capability"}:
                self._observe_commit_rebinding()
            return _uncertain_commit_outcome()

        try:
            proof = evidence.proof
            repository = proof.repository
            if (
                not self._repository_identity_matches(binding, repository)
                or not await self._commit_local_state_is_supported(repository)
            ):
                return self._keep_commit_recovery_uncertain(
                    lease,
                    capture,
                    evidence,
                    can_check_again=False,
                )
            postflight = await self._read_commit_postflight(
                binding,
                repository,
            )
            if self._recovery_postflight_is_success(
                proof,
                capture,
                postflight,
            ):
                outcome = await self._publish_successful_commit(
                    binding,
                    None,
                    capture,
                    lease,
                    postflight,
                    recovery_evidence=evidence,
                )
                if outcome.state == "succeeded":
                    self._uncertain_commit = None
                return outcome
            if (
                evidence.known_normal_returncode is not None
                and evidence.known_normal_returncode > 0
                and self._recovery_postflight_is_unchanged(
                    proof,
                    postflight,
                )
            ):
                outcome = await self._publish_failed_commit(
                    binding,
                    None,
                    capture,
                    lease,
                    normal_returncode=evidence.known_normal_returncode,
                    recovery_evidence=evidence,
                )
                if outcome.state == "failed_unchanged":
                    self._uncertain_commit = None
                return outcome
            return self._keep_commit_recovery_uncertain(
                lease,
                capture,
                evidence,
                can_check_again=postflight.local_state_supported,
            )
        finally:
            lease.release()

    async def _run_commit_cycle(
        self,
        binding: SessionBinding,
        snapshot: _CommitReviewSnapshot,
        lease: GitMutationLease,
        *,
        subject: str | None,
        body: str,
    ) -> CommitOutcome:
        hooks_directory: Path | None = None
        confirmation: _CommitConfirmation | None = None
        retained_child: RetainedGitChildToken | None = None
        retained_claimed = False
        child_termination_known = False
        known_normal_returncode: int | None = None
        try:
            if subject is not None:
                try:
                    confirmed_message = normalize_commit_message(subject, body)
                except CommitContractError as error:
                    return CommitOutcome("blocked", str(error))
                if confirmed_message != snapshot.message:
                    return CommitOutcome(
                        "blocked",
                        "Commit message changed; review again.",
                    )
            confirmation = await self._revalidate_commit_confirmation(
                binding,
                snapshot,
                lease,
            )
            if confirmation is None:
                return CommitOutcome(
                    "blocked",
                    "Commit review is stale; review again.",
                )
            try:
                hooks_directory = _create_private_hooks_directory(
                    confirmation.capture.repository,
                    self._pending_hooks_cleanup,
                )
            except OSError:
                return CommitOutcome(
                    "blocked",
                    "A private commit hooks directory could not be prepared.",
                )
            try:
                child_result = await self._runner.run(
                    build_commit_argv(
                        self._git_executable_or_raise(),
                        str(hooks_directory),
                    ),
                    cwd=confirmation.capture.repository.worktree_root,
                    environment=build_commit_environment(
                        self._environment,
                        author=snapshot.author,
                        committer=snapshot.committer,
                    ),
                    stdin=snapshot.message,
                    timeout=self._status_timeout,
                    stdout_limit=DEFAULT_COMMIT_PROOF_STDERR_LIMIT_BYTES,
                    stderr_limit=DEFAULT_COMMIT_PROOF_STDERR_LIMIT_BYTES,
                    on_spawn=self._mark_commit_child_started,
                    cancel_before_spawn=True,
                )
            except OSError:
                if not self._commit_child_started:
                    self._remove_hooks_directory(hooks_directory)
                    hooks_directory = None
                    return CommitOutcome(
                        "blocked",
                        "Git could not start the commit child.",
                    )
                outcome = self._retain_uncertain_commit(
                    snapshot,
                    confirmation.capture,
                    lease,
                    hooks_directory=hooks_directory,
                )
                hooks_directory = None
                return outcome

            retained_child = child_result.retained_child
            if retained_child is not None:
                try:
                    retained_claimed = self._runner.claim_retained_child(retained_child)
                    settlement = self._runner.read_retained_child(retained_child)
                except (RuntimeError, ValueError):
                    settlement = RetainedGitChildSettlement("uncertain")
                if (
                    retained_claimed
                    and settlement.state == "natural"
                    and settlement.returncode is not None
                    and not settlement.stop_requested
                    and not settlement.force_stopped
                    and self._runner.release_retained_child(retained_child)
                ):
                    child_result = GitCommandResult(
                        settlement.returncode,
                        settlement.stdout,
                        settlement.stderr,
                        output_overflow=settlement.output_overflow,
                    )
                    retained_child = None
            child_termination_known = (
                retained_child is None
                and child_result.returncode is not None
            )
            child_is_natural = (
                child_termination_known
                and child_result.returncode >= 0
                and not child_result.termination_uncertain
                and not child_result.stop_requested
                and not child_result.force_stopped
            )
            if child_is_natural:
                known_normal_returncode = child_result.returncode
            if not child_is_natural:
                if retained_child is not None and not retained_claimed:
                    try:
                        self._runner.claim_retained_child(retained_child)
                    except (RuntimeError, ValueError):
                        retained_child = None
                if child_termination_known:
                    self._remove_hooks_directory(hooks_directory)
                    hooks_directory = None
                outcome = self._retain_uncertain_commit(
                    snapshot,
                    confirmation.capture,
                    lease,
                    retained_child=retained_child,
                    hooks_directory=hooks_directory,
                    termination_known=child_termination_known,
                    can_check_again=child_termination_known,
                )
                hooks_directory = None
                return outcome

            self._remove_hooks_directory(hooks_directory)
            hooks_directory = None
            postflight = await self._read_commit_postflight(
                binding,
                confirmation.capture.repository,
            )
            if child_result.returncode == 0 and self._postflight_is_success(
                snapshot,
                confirmation,
                postflight,
            ):
                return await self._publish_successful_commit(
                    binding,
                    snapshot,
                    confirmation.capture,
                    lease,
                    postflight,
                )
            if child_result.returncode != 0 and self._postflight_is_unchanged(
                snapshot,
                postflight,
            ):
                return await self._publish_failed_commit(
                    binding,
                    snapshot,
                    confirmation.capture,
                    lease,
                    normal_returncode=child_result.returncode,
                )
            return self._retain_uncertain_commit(
                snapshot,
                confirmation.capture,
                lease,
                termination_known=True,
                known_normal_returncode=child_result.returncode,
                can_check_again=(
                    postflight.repository_matches
                    and postflight.local_state_supported
                ),
            )
        except GitRunCancelled as cancellation:
            if not self._commit_child_started or confirmation is None:
                return CommitOutcome(
                    "cancelled",
                    "Commit confirmation was cancelled.",
                )
            cancelled_result = cancellation.result
            if cancelled_result is not None:
                retained_child = cancelled_result.retained_child
                child_termination_known = (
                    retained_child is None
                    and cancelled_result.returncode is not None
                )
                if (
                    child_termination_known
                    and cancelled_result.returncode >= 0
                    and not cancelled_result.termination_uncertain
                    and not cancelled_result.stop_requested
                    and not cancelled_result.force_stopped
                ):
                    known_normal_returncode = cancelled_result.returncode
            else:
                retained_child = cancellation.retained_child
            if retained_child is not None and not retained_claimed:
                try:
                    retained_claimed = self._runner.claim_retained_child(
                        retained_child
                    )
                except (RuntimeError, ValueError):
                    retained_child = None
            if (
                child_termination_known
                and hooks_directory is not None
                and self._remove_hooks_directory(hooks_directory)
            ):
                hooks_directory = None
            outcome = self._retain_uncertain_commit(
                snapshot,
                confirmation.capture,
                lease,
                retained_child=retained_child,
                hooks_directory=hooks_directory,
                termination_known=child_termination_known,
                known_normal_returncode=known_normal_returncode,
                can_check_again=child_termination_known,
            )
            hooks_directory = None
            return outcome
        except asyncio.CancelledError:
            if not self._commit_child_started or confirmation is None:
                return CommitOutcome(
                    "cancelled",
                    "Commit confirmation was cancelled.",
                )
            if (
                child_termination_known
                and hooks_directory is not None
                and self._remove_hooks_directory(hooks_directory)
            ):
                hooks_directory = None
            outcome = self._retain_uncertain_commit(
                snapshot,
                confirmation.capture,
                lease,
                retained_child=retained_child,
                hooks_directory=hooks_directory,
                termination_known=child_termination_known,
                known_normal_returncode=known_normal_returncode,
                can_check_again=child_termination_known,
            )
            hooks_directory = None
            return outcome
        except Exception:
            if not self._commit_child_started or confirmation is None:
                return CommitOutcome(
                    "blocked",
                    "Commit preparation failed before Git started.",
                )
            if retained_child is not None and not retained_claimed:
                try:
                    retained_claimed = self._runner.claim_retained_child(
                        retained_child
                    )
                except Exception:
                    retained_child = None
            if (
                child_termination_known
                and hooks_directory is not None
                and self._remove_hooks_directory(hooks_directory)
            ):
                hooks_directory = None
            outcome = self._retain_uncertain_commit(
                snapshot,
                confirmation.capture,
                lease,
                retained_child=retained_child,
                hooks_directory=hooks_directory,
                termination_known=child_termination_known,
                known_normal_returncode=known_normal_returncode,
                can_check_again=child_termination_known,
            )
            hooks_directory = None
            return outcome
        finally:
            if hooks_directory is not None and not self._commit_child_started:
                self._remove_hooks_directory(hooks_directory)
            evidence = self._uncertain_commit
            if evidence is None or evidence.mutation_lease is not lease:
                abandoned = (
                    snapshot.capture
                    if confirmation is None
                    else confirmation.capture
                )
                self._owner._discard_commit_authority(abandoned)
                lease.release()

    async def _revalidate_commit_confirmation(
        self,
        binding: SessionBinding,
        snapshot: _CommitReviewSnapshot,
        lease: GitMutationLease,
    ) -> _CommitConfirmation | None:
        reviewed = snapshot.capture
        repository = reviewed.repository
        root = self._safe_root(binding)
        if (
            root is None
            or self._git_executable is None
            or not await self._commit_repository_matches(binding, repository)
            or not await self._commit_local_state_is_supported(repository)
        ):
            return None
        head = await self._read_commit_head(repository)
        if head != reviewed.head:
            return None
        current = self._owner.snapshot(binding)
        if (
            current.git_authority_generation != reviewed.authority_generation
            or current.trusted_repository != repository
            or dict(current.staging_ownership) != dict(reviewed.ownership)
        ):
            return None
        groups_by_id = {
            group.group_id: group for group in coalesce_session_changes(current.changes)
        }
        if any(
            groups_by_id.get(group_id) is None
            or groups_by_id[group_id].sequence_ids != sequence_ids
            for group_id, sequence_ids in reviewed.group_sequence_ids.items()
        ):
            return None
        repository_ownership: dict[int, StagingOwnership] = {}
        for group_id, ownership in current.staging_ownership.items():
            group = groups_by_id.get(group_id)
            if group is None:
                return None
            repository_group, invalid = self._map_group(
                root,
                repository,
                group,
            )
            if invalid is not None or repository_group is None:
                return None
            mapped = _map_ownership_topology(
                ownership,
                group,
                repository_group,
            )
            if mapped is None:
                return None
            repository_ownership[group_id] = mapped
        proof = await self._complete_commit_proof(
            repository,
            reviewed.head,
            repository_ownership,
        )
        if proof != snapshot.proof:
            return None
        identities = await self._resolve_commit_identities(repository)
        if identities != (snapshot.author, snapshot.committer):
            return None
        capture = self._owner._recapture_commit_authority(
            lease,
            prior_capture=reviewed,
        )
        if capture is None:
            return None
        return _CommitConfirmation(capture)

    async def _read_commit_postflight(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> _CommitPostflight:
        repository_matches = await self._commit_repository_matches(
            binding,
            repository,
        )
        local_state_supported = (
            repository_matches
            and await self._commit_local_state_is_supported(repository)
        )
        if not local_state_supported:
            return _CommitPostflight(
                repository_matches,
                False,
                None,
                None,
                None,
                None,
                None,
            )
        head = await self._read_commit_head(repository)
        if head is None or head.object_id is None:
            return _CommitPostflight(True, True, head, None, None, None, None)
        index = await self._run_commit_proof_command(
            repository,
            build_commit_index_argv(self._git_executable_or_raise()),
        )
        delta = await self._run_commit_proof_command(
            repository,
            build_commit_delta_argv(
                self._git_executable_or_raise(),
                head.object_id,
            ),
        )
        tree = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "write-tree",
            ),
        )
        if not all(_command_succeeded(result) for result in (index, delta, tree)):
            return _CommitPostflight(True, True, head, None, None, None, None)
        if not _commit_index_semantics_are_supported(index.stdout):
            return _CommitPostflight(True, True, head, None, None, None, None)
        try:
            index_entries = parse_index_entries_z(index.stdout)
        except GitIndexParseError:
            return _CommitPostflight(True, True, head, None, None, None, None)
        if any(
            entry.stage != 0
            or entry.semantic_flags
            or entry.mode in {"040000", "160000"}
            for entry in index_entries
        ):
            return _CommitPostflight(True, True, head, None, None, None, None)
        tree_object_id = _ascii_object_id(tree.stdout)
        raw_commit: RawCommitObject | None = None
        raw = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "cat-file",
                "commit",
                head.object_id,
            ),
        )
        if _command_succeeded(raw):
            try:
                raw_commit = parse_raw_commit_object(raw.stdout)
            except CommitContractError:
                pass
        return _CommitPostflight(
            True,
            True,
            head,
            hashlib.sha256(index.stdout).hexdigest(),
            hashlib.sha256(delta.stdout).hexdigest(),
            tree_object_id,
            raw_commit,
        )

    @staticmethod
    def _postflight_is_success(
        snapshot: _CommitReviewSnapshot,
        confirmation: _CommitConfirmation,
        postflight: _CommitPostflight,
    ) -> bool:
        head = postflight.head
        raw = postflight.raw_commit
        return (
            postflight.repository_matches
            and postflight.local_state_supported
            and head is not None
            and head.kind == "attached"
            and head.branch == confirmation.capture.head.branch
            and head.object_id is not None
            and head.object_id != confirmation.capture.head.object_id
            and raw is not None
            and raw.parent_object_id == confirmation.capture.head.object_id
            and raw.tree_object_id == snapshot.proof.expected_tree
            and postflight.tree_object_id == raw.tree_object_id
            and raw.message == snapshot.message
            and raw.author == snapshot.author
            and raw.committer == snapshot.committer
            and not raw.has_signature
            and postflight.delta_signature == hashlib.sha256(b"").hexdigest()
        )

    @staticmethod
    def _postflight_is_unchanged(
        snapshot: _CommitReviewSnapshot,
        postflight: _CommitPostflight,
    ) -> bool:
        return (
            postflight.repository_matches
            and postflight.local_state_supported
            and postflight.head == snapshot.capture.head
            and postflight.index_signature == snapshot.proof.index_signature
            and postflight.delta_signature == snapshot.proof.delta_signature
            and postflight.tree_object_id == snapshot.proof.expected_tree
        )

    @staticmethod
    def _recovery_postflight_is_success(
        proof: _CommitRecoveryProof,
        capture: CommitAuthorityCapture,
        postflight: _CommitPostflight,
    ) -> bool:
        """Match exact success without retaining captured staging ownership."""
        head = postflight.head
        raw = postflight.raw_commit
        return (
            postflight.repository_matches
            and postflight.local_state_supported
            and head is not None
            and head.kind == "attached"
            and head.branch == capture.head.branch
            and head.object_id is not None
            and head.object_id != proof.old_head.object_id
            and raw is not None
            and raw.parent_object_id == proof.old_head.object_id
            and raw.tree_object_id == proof.complete_proof.expected_tree
            and postflight.tree_object_id == raw.tree_object_id
            and raw.message == proof.message
            and raw.author == proof.author
            and raw.committer == proof.committer
            and not raw.has_signature
            and postflight.delta_signature == hashlib.sha256(b"").hexdigest()
        )

    @staticmethod
    def _recovery_postflight_is_unchanged(
        proof: _CommitRecoveryProof,
        postflight: _CommitPostflight,
    ) -> bool:
        """Match the exact captured old branch and complete logical index."""
        return (
            postflight.repository_matches
            and postflight.local_state_supported
            and postflight.head == proof.old_head
            and (
                postflight.index_signature
                == proof.complete_proof.index_signature
            )
            and (
                postflight.delta_signature
                == proof.complete_proof.delta_signature
            )
            and (
                postflight.tree_object_id
                == proof.complete_proof.expected_tree
            )
        )

    async def _publish_successful_commit(
        self,
        binding: SessionBinding,
        snapshot: _CommitReviewSnapshot | None,
        capture: CommitAuthorityCapture,
        lease: GitMutationLease,
        postflight: _CommitPostflight,
        *,
        recovery_evidence: _UncertainCommitEvidence | None = None,
    ) -> CommitOutcome:
        head = postflight.head
        assert head is not None and head.object_id is not None
        current_changes = self._owner.snapshot(binding).changes
        status = await self._query_status(
            binding,
            current_changes,
            capture.repository,
            publish_ownership_changes=False,
        )
        if status.state != "ready":
            if recovery_evidence is not None:
                return self._keep_commit_recovery_uncertain(
                    lease,
                    capture,
                    recovery_evidence,
                    can_check_again=True,
                )
            assert snapshot is not None
            return self._retain_uncertain_commit(
                snapshot,
                capture,
                lease,
                termination_known=True,
                known_normal_returncode=0,
                can_check_again=True,
            )
        clean_groups = {row.group_id for row in status.rows if row.state == "clean"}
        retired: list[int] = []
        divergent: list[int] = []
        retired_groups: set[int] = set()
        for group_id, sequences in capture.group_sequence_ids.items():
            if group_id in clean_groups:
                retired_groups.add(group_id)
                target = retired
            else:
                target = divergent
            target.extend(sequences)
        refreshed_status = replace(
            status,
            rows=tuple(
                row
                for row in status.rows
                if row.group_id not in retired_groups
            ),
        )
        if recovery_evidence is not None:
            candidate_seed = recovery_evidence.proof.candidate_seed
        else:
            assert snapshot is not None
            candidate_seed = snapshot.candidate_seed
        publication = self._owner.publish_commit_outcome(
            lease,
            capture,
            CommitPublication(
                state="succeeded",
                new_head=head,
                retired_sequence_ids=tuple(retired),
                divergent_sequence_ids=tuple(divergent),
                refreshed_status=refreshed_status,
                candidate_seed=candidate_seed,
            ),
        )
        if not publication.published:
            if recovery_evidence is not None:
                self._uncertain_commit = recovery_evidence
                return _uncertain_commit_outcome()
            assert snapshot is not None
            return self._retain_uncertain_commit(
                snapshot,
                capture,
                lease,
                termination_known=True,
                known_normal_returncode=0,
                can_check_again=True,
            )
        if recovery_evidence is not None:
            included_group_ids = (
                recovery_evidence.proof.complete_proof.included_group_ids
            )
        else:
            assert snapshot is not None
            included_group_ids = snapshot.proof.included_group_ids
        count = len(included_group_ids)
        short_oid = head.object_id[:12]
        return CommitOutcome(
            "succeeded",
            f"Committed {count} session notes as {short_oid}; "
            "unrelated changes untouched.",
            qualification=(
                "No unrelated staged content was committed; "
                "Chatbook selected no unrelated worktree paths."
            ),
            commit_object_id=head.object_id,
            committed_note_count=count,
        )

    async def _publish_failed_commit(
        self,
        binding: SessionBinding,
        snapshot: _CommitReviewSnapshot | None,
        capture: CommitAuthorityCapture,
        lease: GitMutationLease,
        *,
        normal_returncode: int,
        recovery_evidence: _UncertainCommitEvidence | None = None,
    ) -> CommitOutcome:
        status = await self._query_status(
            binding,
            self._owner.snapshot(binding).changes,
            capture.repository,
            publish_ownership_changes=False,
        )
        if status.state != "ready":
            if recovery_evidence is not None:
                return self._keep_commit_recovery_uncertain(
                    lease,
                    capture,
                    recovery_evidence,
                    can_check_again=True,
                )
            assert snapshot is not None
            return self._retain_uncertain_commit(
                snapshot,
                capture,
                lease,
                termination_known=True,
                known_normal_returncode=normal_returncode,
                can_check_again=True,
            )
        publication = self._owner.publish_commit_outcome(
            lease,
            capture,
            CommitPublication(
                state="failed_unchanged",
                refreshed_status=status,
            ),
        )
        if not publication.published:
            if recovery_evidence is not None:
                self._uncertain_commit = recovery_evidence
                return _uncertain_commit_outcome()
            assert snapshot is not None
            return self._retain_uncertain_commit(
                snapshot,
                capture,
                lease,
                termination_known=True,
                known_normal_returncode=normal_returncode,
                can_check_again=True,
            )
        return CommitOutcome(
            "failed_unchanged",
            "Git did not create a commit; branch and staged state are unchanged.",
        )

    def _retain_uncertain_commit(
        self,
        snapshot: _CommitReviewSnapshot,
        capture: CommitAuthorityCapture,
        lease: GitMutationLease,
        *,
        retained_child: RetainedGitChildToken | None = None,
        hooks_directory: Path | None = None,
        termination_known: bool = False,
        known_normal_returncode: int | None = None,
        can_check_again: bool = False,
    ) -> CommitOutcome:
        recovery_capability = self._publish_uncertain_commit(
            lease,
            capture,
            can_check_again=can_check_again,
        )
        self._uncertain_commit = _UncertainCommitEvidence(
            proof=_CommitRecoveryProof(
                binding=capture.binding,
                repository=capture.repository,
                old_head=capture.head,
                complete_proof=snapshot.proof,
                message=snapshot.message,
                author=snapshot.author,
                committer=snapshot.committer,
                candidate_seed=snapshot.candidate_seed,
            ),
            recovery_capability=recovery_capability,
            retained_child=retained_child,
            hooks_directory=hooks_directory,
            mutation_lease=lease if recovery_capability is None else None,
            termination_known=termination_known,
            known_normal_returncode=known_normal_returncode,
        )
        return _uncertain_commit_outcome()

    def _publish_uncertain_commit(
        self,
        lease: GitMutationLease,
        capture: CommitAuthorityCapture,
        *,
        can_check_again: bool,
    ) -> CommitRecoveryCapability | None:
        publication = self._owner.publish_commit_outcome(
            lease,
            capture,
            CommitPublication(
                state="uncertain",
                recovery_projection=CommitRecoveryProjection(
                    _UNCERTAIN_COMMIT_MESSAGE,
                    can_check_again,
                ),
            ),
        )
        return publication.recovery_capability

    def _settle_commit_recovery_child(
        self,
        evidence: _UncertainCommitEvidence,
    ) -> tuple[_UncertainCommitEvidence, bool]:
        """Consume only certain termination from the exact retained child."""
        retained_child = evidence.retained_child
        if retained_child is None:
            return evidence, evidence.termination_known
        try:
            settlement = self._runner.read_retained_child(retained_child)
        except (RuntimeError, ValueError):
            return evidence, False
        if settlement.state in {"alive", "uncertain"}:
            return evidence, False
        try:
            released = self._runner.release_retained_child(retained_child)
        except (RuntimeError, ValueError):
            return evidence, False
        if not released:
            return evidence, False

        known_normal_returncode = (
            settlement.returncode
            if (
                settlement.state == "natural"
                and settlement.returncode is not None
                and settlement.returncode >= 0
                and not settlement.stop_requested
                and not settlement.force_stopped
            )
            else None
        )
        hooks_directory = evidence.hooks_directory
        if (
            hooks_directory is not None
            and self._remove_hooks_directory(hooks_directory)
        ):
            hooks_directory = None
        settled = replace(
            evidence,
            retained_child=None,
            hooks_directory=hooks_directory,
            termination_known=True,
            known_normal_returncode=known_normal_returncode,
        )
        self._uncertain_commit = settled
        return settled, True

    def _keep_commit_recovery_uncertain(
        self,
        lease: GitMutationLease,
        capture: CommitAuthorityCapture,
        evidence: _UncertainCommitEvidence,
        *,
        can_check_again: bool,
    ) -> CommitOutcome:
        """Keep the same quarantine without inventing fresh ownership."""
        projection = CommitRecoveryProjection(
            _UNCERTAIN_COMMIT_MESSAGE,
            can_check_again,
        )
        if self._owner.snapshot(capture.binding).commit_recovery == projection:
            self._uncertain_commit = replace(
                evidence,
                mutation_lease=None,
            )
            return _uncertain_commit_outcome()
        publication = self._owner.publish_commit_outcome(
            lease,
            capture,
            CommitPublication(
                state="uncertain",
                recovery_projection=projection,
            ),
        )
        capability = (
            publication.recovery_capability
            if publication.published
            else evidence.recovery_capability
        )
        self._uncertain_commit = replace(
            evidence,
            recovery_capability=capability,
            mutation_lease=None,
        )
        return _uncertain_commit_outcome()

    def _observe_commit_rebinding(self) -> bool:
        """Drop rebound proof and retain only exact child lifecycle resources."""
        evidence = self._uncertain_commit
        current_binding = self._owner.current_binding()
        operation = self._retained_commit_operation
        if operation is not None and operation.binding != current_binding:
            self._retained_commit_operation = None
        if evidence is not None and current_binding is not None:
            current_repository = self._owner.snapshot(
                current_binding
            ).trusted_repository
            if (
                evidence.proof.binding != current_binding
                or (
                    current_repository is not None
                    and current_repository != evidence.proof.repository
                )
            ):
                self._orphaned_commit = _OrphanedCommitLifecycle(
                    retained_child=evidence.retained_child,
                    hooks_directory=evidence.hooks_directory,
                    mutation_lease=evidence.mutation_lease,
                    termination_known=evidence.termination_known,
                )
                self._uncertain_commit = None
        self._settle_orphaned_commit()
        return self._orphaned_commit is not None

    def _discard_commit_review_snapshots(self) -> None:
        """Retire owner authority for every review capability being dropped."""
        snapshots = tuple(self._commit_review_snapshots.values())
        self._commit_review_snapshots.clear()
        for snapshot in snapshots:
            self._owner._discard_commit_authority(snapshot.capture)

    def _settle_orphaned_commit(self) -> None:
        """Settle one rebound child without recovering discarded commit proof."""
        lifecycle = self._orphaned_commit
        if lifecycle is None:
            return
        retained_child = lifecycle.retained_child
        if retained_child is not None:
            try:
                settlement = self._runner.read_retained_child(retained_child)
            except (RuntimeError, ValueError):
                return
            if settlement.state in {"alive", "uncertain"}:
                return
            try:
                released = self._runner.release_retained_child(retained_child)
            except (RuntimeError, ValueError):
                return
            if not released:
                return
            lifecycle = replace(
                lifecycle,
                retained_child=None,
                termination_known=True,
            )
            self._orphaned_commit = lifecycle
        if not lifecycle.termination_known:
            return

        hooks_directory = lifecycle.hooks_directory
        if (
            hooks_directory is not None
            and self._remove_hooks_directory(hooks_directory)
        ):
            lifecycle = replace(lifecycle, hooks_directory=None)
            self._orphaned_commit = lifecycle
        mutation_lease = lifecycle.mutation_lease
        if mutation_lease is not None:
            mutation_lease.release()
            lifecycle = replace(lifecycle, mutation_lease=None)
            self._orphaned_commit = lifecycle
        if lifecycle.hooks_directory is None:
            self._orphaned_commit = None

    async def _run_commit_review_cycle(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
        authority_generation: int,
        subject: str,
        body: str,
        lease: GitMutationLease,
    ) -> CommitReviewResult:
        try:
            return await self._prepare_commit_review(
                binding,
                repository,
                authority_generation,
                subject,
                body,
                lease,
            )
        except asyncio.CancelledError:
            return CommitReviewResult(
                "cancelled",
                message="Commit review was cancelled.",
            )
        finally:
            lease.release()

    async def _prepare_commit_review(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
        authority_generation: int,
        subject: str,
        body: str,
        lease: GitMutationLease,
    ) -> CommitReviewResult:
        """Run the ordered preflight and retain only a private proof snapshot."""
        try:
            message = normalize_commit_message(subject, body)
        except CommitContractError as error:
            return CommitReviewResult("blocked", message=str(error))
        root = self._safe_root(binding)
        if (
            root is None
            or self._git_executable is None
            or not await self._commit_repository_matches(
                binding,
                repository,
            )
        ):
            return _blocked_commit_review("Repository identity changed.")
        if not await self._commit_local_state_is_supported(repository):
            return _blocked_commit_review(
                "Repository state does not support guarded commit review."
            )

        head = await self._read_commit_head(repository)
        if head is None:
            return _blocked_commit_review(
                "Commit review requires an attached branch with an existing commit."
            )
        current = self._owner.snapshot(binding)
        if (
            current.git_authority_generation != authority_generation
            or current.trusted_repository != repository
            or not current.staging_ownership
        ):
            return _blocked_commit_review("Session staging authority changed.")
        ownership = dict(current.staging_ownership)
        groups_by_id = {
            group.group_id: group
            for group in coalesce_session_changes(current.changes)
        }
        group_sequence_ids = {
            group_id: groups_by_id[group_id].sequence_ids
            for group_id in ownership
            if group_id in groups_by_id
        }
        if len(group_sequence_ids) != len(ownership):
            return _blocked_commit_review("Session staging authority changed.")
        if any(
            item.repository != repository
            or item.head != head
            or groups_by_id[group_id].topology_signature
            != item.topology_signature
            for group_id, item in ownership.items()
        ):
            return _blocked_commit_review("Session staging authority changed.")
        repository_ownership: dict[int, StagingOwnership] = {}
        for group_id, item in ownership.items():
            session_group = groups_by_id[group_id]
            repository_group, invalid = self._map_group(
                root,
                repository,
                session_group,
            )
            if invalid is not None or repository_group is None:
                return _blocked_commit_review(
                    "Session staging authority changed."
                )
            mapped = _map_ownership_topology(
                item,
                session_group,
                repository_group,
            )
            if mapped is None:
                return _blocked_commit_review(
                    "Session staging authority changed."
                )
            repository_ownership[group_id] = mapped

        proof = await self._complete_commit_proof(
            repository,
            head,
            repository_ownership,
        )
        if proof is None:
            return _blocked_commit_review(
                "The complete staged state does not exactly match this session. "
                "If Git has unrelated staged changes, commit or unstage them "
                "outside Chatbook; then Refresh and review this session again."
            )
        identities = await self._resolve_commit_identities(repository)
        if identities is None:
            return _blocked_commit_review(
                "Configure Git user.name and user.email, then review again."
            )
        author, committer = identities
        included_notes: list[CommitIncludedNote] = []
        for group_id in proof.included_group_ids:
            change_type = _commit_review_change_type(
                repository_ownership[group_id]
            )
            if change_type is None:
                return _blocked_commit_review(
                    "The complete staged state cannot be classified safely."
                )
            included_notes.append(
                CommitIncludedNote(
                    group_id=group_id,
                    display_text=groups_by_id[group_id].display_text,
                    change_type=change_type,
                )
            )
        capture = self._owner._capture_commit_authority_after_review(
            lease,
            binding=binding,
            authority_generation=authority_generation,
            repository=repository,
            head=head,
            group_sequence_ids=group_sequence_ids,
            subject=message.partition(b"\n")[0].decode("utf-8"),
            included_notes=tuple(
                PushIncludedNote(note.group_id, note.display_text)
                for note in included_notes
            ),
            change_types=tuple(note.change_type for note in included_notes),
        )
        if capture is None:
            return _blocked_commit_review("Session staging authority changed.")

        token = object()
        projection = CommitReviewProjection(
            branch=head.branch or "",
            old_commit=head.object_id or "",
            message=message.decode("utf-8"),
            included_notes=tuple(included_notes),
            author=author,
            committer=committer,
        )
        self._commit_review_snapshots[token] = _CommitReviewSnapshot(
            capture=capture,
            proof=proof,
            message=message,
            author=author,
            committer=committer,
            candidate_seed=capture._candidate_seed,
        )
        return CommitReviewResult(
            "ready",
            handle=CommitReviewHandle(token),
            projection=projection,
        )

    async def _commit_repository_matches(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        """Re-read the complete repository mapping under proof isolation."""
        commands = (
            ("rev-parse", "--path-format=absolute", "--show-toplevel"),
            ("rev-parse", "--absolute-git-dir"),
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        )
        resolved: list[Path] = []
        for arguments in commands:
            result = await self._run_commit_proof_command(
                repository,
                (
                    self._git_executable_or_raise(),
                    "--no-replace-objects",
                    *arguments,
                ),
            )
            if not _command_succeeded(result):
                return False
            path = _canonical_directory_from_git(result.stdout)
            if path is None:
                return False
            resolved.append(path)
        return (
            resolved == [
                Path(repository.worktree_root),
                Path(repository.git_dir),
                Path(repository.git_common_dir),
            ]
            and self._repository_identity_matches(binding, repository)
        )

    async def _commit_local_state_is_supported(
        self,
        repository: RepositoryIdentity,
    ) -> bool:
        """Inspect local-only blockers before any object-resolving command."""
        git_dir = Path(repository.git_dir)
        common_dir = Path(repository.git_common_dir)
        marker_names = (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "REBASE_HEAD",
            "BISECT_START",
            "BISECT_LOG",
            "rebase-apply",
            "rebase-merge",
            "sequencer",
        )
        if any(
            _path_present(directory / marker)
            for directory in {git_dir, common_dir}
            for marker in marker_names
        ):
            return False
        fixed_blockers = (
            git_dir / "index.lock",
            git_dir / "HEAD.lock",
            common_dir / "packed-refs.lock",
            common_dir / "config.lock",
            common_dir / "info" / "grafts",
            git_dir / "info" / "sparse-checkout",
            common_dir / "info" / "sparse-checkout",
        )
        if any(_path_present(path) for path in fixed_blockers):
            return False
        try:
            if (common_dir / "refs" / "replace").is_dir() and any(
                (common_dir / "refs" / "replace").iterdir()
            ):
                return False
            heads = common_dir / "refs" / "heads"
            if heads.is_dir() and any(heads.rglob("*.lock")):
                return False
            pack_directory = common_dir / "objects" / "pack"
            if pack_directory.is_dir() and any(
                pack_directory.glob("*.promisor")
            ):
                return False
            packed_refs = common_dir / "packed-refs"
            if packed_refs.is_file():
                with packed_refs.open("rb") as stream:
                    packed_payload = stream.read(8 * 1024 * 1024 + 1)
                    if (
                        len(packed_payload) > 8 * 1024 * 1024
                        or b" refs/replace/" in packed_payload
                    ):
                        return False
        except OSError:
            return False

        local_config = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "config",
                "--includes",
                "--local",
                "--null",
                "--list",
            ),
        )
        if not _command_succeeded(local_config):
            return False
        if not _commit_local_config_is_supported(local_config.stdout):
            return False
        if not _commit_worktree_config_is_enabled(local_config.stdout):
            return True
        worktree_config = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "config",
                "--includes",
                "--worktree",
                "--null",
                "--list",
            ),
        )
        if not _command_succeeded(worktree_config):
            return False
        if not _commit_local_config_is_supported(worktree_config.stdout):
            return False
        return True

    async def _read_commit_head(
        self,
        repository: RepositoryIdentity,
    ) -> HeadIdentity | None:
        symbolic = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "symbolic-ref",
                "--quiet",
                "HEAD",
            ),
        )
        if not _command_succeeded(symbolic):
            return None
        branch = _single_git_value(symbolic.stdout)
        if branch is None or not branch.startswith("refs/heads/"):
            return None
        revision = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "rev-parse",
                "--verify",
                "--quiet",
                f"{branch}^{{commit}}",
            ),
        )
        if not _command_succeeded(revision):
            return None
        object_id = _ascii_object_id(revision.stdout)
        if object_id is None:
            return None
        return HeadIdentity.attached(branch, object_id)

    async def _complete_commit_proof(
        self,
        repository: RepositoryIdentity,
        head: HeadIdentity,
        ownership: Mapping[int, StagingOwnership],
    ) -> _CompleteCommitProof | None:
        """Prove the complete logical index while returning no raw paths."""
        if head.object_id is None:
            return None
        index_result = await self._run_commit_proof_command(
            repository,
            build_commit_index_argv(self._git_executable_or_raise()),
        )
        if not _command_succeeded(index_result):
            return None
        if not _commit_index_semantics_are_supported(index_result.stdout):
            return None
        try:
            index_entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError:
            return None
        if any(
            entry.stage != 0
            or bool(entry.semantic_flags)
            or entry.mode in {"040000", "160000"}
            or not any(character != "0" for character in entry.object_id)
            for entry in index_entries
        ):
            return None
        index_by_path = {entry.path: entry for entry in index_entries}
        for item in ownership.values():
            if any(
                index_by_path.get(path) != expected
                for path, expected in item.post_stage_entries.items()
            ):
                return None

        delta_result = await self._run_commit_proof_command(
            repository,
            build_commit_delta_argv(
                self._git_executable_or_raise(),
                head.object_id,
            ),
        )
        if not _command_succeeded(delta_result):
            return None
        try:
            delta = parse_raw_staged_delta(delta_result.stdout)
        except CommitContractError:
            return None
        ownership_entries = {
            group_id: (
                {
                    path: baseline.entry
                    for path, baseline in item.original_baselines.items()
                },
                item.post_stage_entries,
            )
            for group_id, item in ownership.items()
        }
        if not complete_commit_delta_matches_ownership(delta, ownership_entries):
            return None
        included_group_ids = tuple(
            group_id
            for group_id, pair in ownership_entries.items()
            if _expected_owned_delta({group_id: pair})
        )
        if not included_group_ids:
            return None

        tree_result = await self._run_commit_proof_command(
            repository,
            (
                self._git_executable_or_raise(),
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "write-tree",
            ),
        )
        if not _command_succeeded(tree_result):
            return None
        expected_tree = _ascii_object_id(tree_result.stdout)
        if expected_tree is None:
            return None

        freshness_paths = tuple(
            dict.fromkeys(
                os.fsencode(path)
                for group_id in included_group_ids
                for path in (
                    *ownership[group_id].approved_endpoint_topology,
                    *(
                        path
                        for edge in ownership[group_id].approved_move_edges
                        for path in edge
                    ),
                    *ownership[group_id].original_baselines.keys(),
                    *ownership[group_id].post_stage_entries.keys(),
                )
            )
        )
        freshness = await self._run_commit_proof_command(
            repository,
            build_commit_worktree_argv(
                self._git_executable_or_raise(),
                freshness_paths,
            ),
        )
        if not _command_succeeded(freshness):
            return None
        try:
            freshness_records = parse_porcelain_v2_z(
                freshness.stdout,
                allowed_paths=frozenset(
                    os.fsdecode(path) for path in freshness_paths
                ),
            )
        except PorcelainV2ParseError:
            return None
        if any(
            record.kind != "ordinary"
            or record.worktree_status != "."
            for record in freshness_records
        ):
            return None
        return _CompleteCommitProof(
            expected_tree=expected_tree,
            index_signature=hashlib.sha256(index_result.stdout).hexdigest(),
            delta_signature=hashlib.sha256(delta_result.stdout).hexdigest(),
            included_group_ids=included_group_ids,
        )

    async def _resolve_commit_identities(
        self,
        repository: RepositoryIdentity,
    ) -> tuple[GitIdentity, GitIdentity] | None:
        identities: list[GitIdentity] = []
        for variable in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            result = await self._run_commit_proof_command(
                repository,
                (
                    self._git_executable_or_raise(),
                    "--no-replace-objects",
                    "var",
                    variable,
                ),
            )
            if not _command_succeeded(result):
                return None
            try:
                identities.append(parse_git_identity(result.stdout))
            except CommitContractError:
                return None
        return identities[0], identities[1]

    async def _run_commit_proof_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
    ) -> GitCommandResult:
        try:
            result = await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_commit_environment(
                    self._environment,
                    read_only=True,
                ),
                timeout=self._status_timeout,
                stdout_limit=DEFAULT_COMMIT_PROOF_STDOUT_LIMIT_BYTES,
                stderr_limit=DEFAULT_COMMIT_PROOF_STDERR_LIMIT_BYTES,
            )
        except GitRunCancelled as cancellation:
            if cancellation.result is not None:
                return await self._settle_commit_proof_result(
                    cancellation.result
                )
            retained_child = cancellation.retained_child
            assert retained_child is not None
            await self._drain_commit_proof_child(retained_child)
            raise
        except OSError:
            return GitCommandResult(127, b"", b"")
        return await self._settle_commit_proof_result(result)

    async def _settle_commit_proof_result(
        self,
        result: GitCommandResult,
    ) -> GitCommandResult:
        retained_child = result.retained_child
        if retained_child is None:
            return result
        settlement = await self._drain_commit_proof_child(retained_child)
        return GitCommandResult(
            settlement.returncode,
            settlement.stdout,
            settlement.stderr,
            termination_uncertain=(
                settlement.state != "natural"
                or settlement.stop_requested
                or settlement.force_stopped
            ),
            stop_requested=settlement.stop_requested,
            force_stopped=settlement.force_stopped,
            output_overflow=settlement.output_overflow,
        )

    async def _drain_commit_proof_child(
        self,
        retained_child: RetainedGitChildToken,
    ) -> RetainedGitChildSettlement:
        """Retain through cancellation, then release one exact terminal token."""
        if not self._runner.claim_retained_child(retained_child):
            raise RuntimeError("Git proof child could not be retained")
        cancelled = False
        while True:
            try:
                settlement = await self._runner.settle_retained_child(
                    retained_child,
                    timeout=0.1,
                )
            except asyncio.CancelledError:
                cancelled = True
                continue
            if settlement.state not in {"alive", "uncertain"}:
                if not self._runner.release_retained_child(retained_child):
                    raise RuntimeError(
                        "Terminal Git proof child could not be released"
                    )
                if cancelled:
                    raise asyncio.CancelledError
                return settlement
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancelled = True

    async def _run_unstage_cycle(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        repository: RepositoryIdentity,
        lease: GitMutationLease,
        status_cycle: asyncio.Task[SessionGitStatus] | None,
    ) -> GitActionResult:
        try:
            if status_cycle is not None and not status_cycle.done():
                await asyncio.shield(status_cycle)
            if self._sealed:
                return GitActionResult(
                    "unstage",
                    "uncertain",
                    requested,
                    message="File Notes Git service shut down during Unstage",
                )
            inspection = await self._inspect_unstage(binding, repository)
            if isinstance(inspection, GitActionResult):
                return replace(
                    inspection,
                    action="unstage",
                    requested_group_ids=requested,
                    blocked_group_ids=(
                        requested
                        if inspection.state == "blocked"
                        else inspection.blocked_group_ids
                    ),
                )
            return await self._apply_unstage(binding, requested, inspection)
        finally:
            lease.release()

    async def _run_stage_cycle(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        repository: RepositoryIdentity,
        lease: GitMutationLease,
        status_cycle: asyncio.Task[SessionGitStatus] | None,
    ) -> GitActionResult:
        try:
            if status_cycle is not None and not status_cycle.done():
                await asyncio.shield(status_cycle)
            if self._sealed:
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    message="File Notes Git service shut down during Stage",
                )
            inspection = await self._inspect_stage(
                binding,
                repository,
            )
            if isinstance(inspection, GitActionResult):
                return replace(
                    inspection,
                    requested_group_ids=requested,
                    blocked_group_ids=(
                        requested
                        if inspection.state == "blocked"
                        else inspection.blocked_group_ids
                    ),
                )
            return await self._apply_stage(
                binding,
                requested,
                inspection,
            )
        finally:
            lease.release()

    async def _shield_action_cycle(
        self,
        cycle: asyncio.Task[GitActionResult],
    ) -> GitActionResult:
        return await asyncio.shield(cycle)

    def _action_cycle_completed(
        self,
        cycle: asyncio.Task[GitActionResult],
    ) -> None:
        if self._action_cycle is cycle:
            self._action_cycle = None
            self._action_waiter = None

    def _create_action_task(
        self,
        coroutine: object,
    ) -> asyncio.Task[GitActionResult]:
        try:
            return asyncio.get_running_loop().create_task(coroutine)  # type: ignore[arg-type]
        except BaseException:
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            raise

    def shutdown(self) -> Awaitable[None]:
        """Seal admission and return retained finite service settlement."""
        if self._shutdown_settlement is not None:
            settlement = self._shutdown_settlement
            if (
                isinstance(settlement, _ImmediateSettlement)
                or (
                    isinstance(settlement, _RetainedSettlement)
                    and settlement.task.done()
                )
            ):
                if self._shutdown_runner_confirmed is True:
                    self._settle_uncertain_commit_shutdown(True)
                    self._settle_uncertain_push_shutdown(True)
                    self._settle_push_shutdown(True)
                self._retry_pending_hooks_cleanup()
            return self._shutdown_settlement
        cycle = self._status_cycle
        waiter = self._status_waiter
        action_cycle = self._action_cycle
        action_waiter = self._action_waiter
        review_cycle = self._commit_review_cycle
        review_waiter = self._commit_review_waiter
        commit_cycle = self._commit_cycle
        commit_waiter = self._commit_waiter
        recovery_cycle = self._commit_recovery_cycle
        recovery_waiter = self._commit_recovery_waiter
        push_local_cycle = self._push_local_proof_cycle
        push_local_waiter = self._push_local_proof_waiter
        push_preflight_cycle = self._push_preflight_cycle
        push_preflight_waiter = self._push_preflight_waiter
        push_cycle = self._push_cycle
        push_waiter = self._push_waiter
        push_recovery_cycle = self._push_recovery_cycle
        push_recovery_waiter = self._push_recovery_waiter
        push_recovery_settlement_cycle = self._push_recovery_settlement_cycle
        active_task = next(
            (
                task
                for task in (
                    cycle,
                    waiter,
                    action_cycle,
                    action_waiter,
                    review_cycle,
                    review_waiter,
                    commit_cycle,
                    commit_waiter,
                    recovery_cycle,
                    recovery_waiter,
                    push_local_cycle,
                    push_local_waiter,
                    push_preflight_cycle,
                    push_preflight_waiter,
                    push_cycle,
                    push_waiter,
                    push_recovery_cycle,
                    push_recovery_waiter,
                    push_recovery_settlement_cycle,
                )
                if task is not None and not task.done()
            ),
            None,
        )
        if active_task is not None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError as error:
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                ) from error
            if running_loop is not active_task.get_loop():
                raise GitShutdownAffinityError(
                    "Active Git shutdown must be initiated from its event loop"
                )
        runner_settlement = self._runner.shutdown()
        self._sealed = True
        if push_local_cycle is not None and not push_local_cycle.done():
            push_local_cycle.cancel()
        if push_preflight_cycle is not None and not push_preflight_cycle.done():
            push_preflight_cycle.cancel()
        if (
            push_cycle is not None
            and not push_cycle.done()
            and not self._push_child_started
        ):
            push_cycle.cancel()
        binding = self._owner.current_binding()
        self._pending_status = None
        self._rerun_available = False
        self._status_dirty = False
        if (
            (cycle is None or cycle.done())
            and (waiter is None or waiter.done())
            and (action_cycle is None or action_cycle.done())
            and (action_waiter is None or action_waiter.done())
            and (review_cycle is None or review_cycle.done())
            and (review_waiter is None or review_waiter.done())
            and (commit_cycle is None or commit_cycle.done())
            and (commit_waiter is None or commit_waiter.done())
            and (recovery_cycle is None or recovery_cycle.done())
            and (recovery_waiter is None or recovery_waiter.done())
            and (push_local_cycle is None or push_local_cycle.done())
            and (push_local_waiter is None or push_local_waiter.done())
            and (push_preflight_cycle is None or push_preflight_cycle.done())
            and (push_preflight_waiter is None or push_preflight_waiter.done())
            and (push_cycle is None or push_cycle.done())
            and (push_waiter is None or push_waiter.done())
            and (
                push_recovery_cycle is None or push_recovery_cycle.done()
            )
            and (
                push_recovery_waiter is None or push_recovery_waiter.done()
            )
            and (
                push_recovery_settlement_cycle is None
                or push_recovery_settlement_cycle.done()
            )
            and (
                runner_settlement is None
                or isinstance(runner_settlement, _ImmediateSettlement)
            )
        ):
            runner_confirmed = (
                runner_settlement is None
                or bool(runner_settlement.value)
            )
            self._shutdown_runner_confirmed = runner_confirmed
            self._settle_uncertain_commit_shutdown(runner_confirmed)
            self._settle_uncertain_push_shutdown(runner_confirmed)
            self._settle_push_shutdown(runner_confirmed)
            self._status_cycle = None
            self._status_cycle_binding = None
            self._status_waiter = None
            self._action_cycle = None
            self._action_waiter = None
            self._commit_review_cycle = None
            self._commit_review_waiter = None
            self._commit_cycle = None
            self._commit_waiter = None
            self._commit_recovery_cycle = None
            self._commit_recovery_waiter = None
            self._commit_child_started = False
            self._commit_child_started_signal = None
            self._retained_commit_operation = None
            self._discard_commit_review_snapshots()
            if binding is not None:
                self._owner.clear_ownership(binding)
                self._owner.clear_status(binding)
            self._retry_pending_hooks_cleanup()
            self._shutdown_settlement = _ImmediateSettlement(None)
            return self._shutdown_settlement
        settlement = _RetainedSettlement(
            asyncio.get_running_loop().create_task(
                self._settle_shutdown(
                    binding,
                    cycle,
                    waiter,
                    action_cycle,
                    action_waiter,
                    review_cycle,
                    review_waiter,
                    commit_cycle,
                    commit_waiter,
                    recovery_cycle,
                    recovery_waiter,
                    push_local_cycle,
                    push_local_waiter,
                    push_preflight_cycle,
                    push_preflight_waiter,
                    push_cycle,
                    push_waiter,
                    push_recovery_cycle,
                    push_recovery_waiter,
                    push_recovery_settlement_cycle,
                    runner_settlement,
                )
            )
        )
        self._shutdown_settlement = settlement
        return settlement

    async def _settle_shutdown(
        self,
        binding: SessionBinding | None,
        cycle: asyncio.Task[SessionGitStatus] | None,
        waiter: asyncio.Task[SessionGitStatus] | None,
        action_cycle: asyncio.Task[GitActionResult] | None,
        action_waiter: asyncio.Task[GitActionResult] | None,
        review_cycle: asyncio.Task[CommitReviewResult] | None,
        review_waiter: asyncio.Task[CommitReviewResult] | None,
        commit_cycle: asyncio.Task[CommitOutcome] | None,
        commit_waiter: asyncio.Task[CommitOutcome] | None,
        recovery_cycle: asyncio.Task[CommitOutcome] | None,
        recovery_waiter: asyncio.Task[CommitOutcome] | None,
        push_local_cycle: asyncio.Task[PushDestinationPolicyResult] | None,
        push_local_waiter: asyncio.Task[PushDestinationPolicyResult] | None,
        push_preflight_cycle: asyncio.Task[PushPreflightResult] | None,
        push_preflight_waiter: asyncio.Task[PushPreflightResult] | None,
        push_cycle: asyncio.Task[PushExecutionResult] | None,
        push_waiter: asyncio.Task[PushExecutionResult] | None,
        push_recovery_cycle: asyncio.Task[PushRecoveryProjection] | None,
        push_recovery_waiter: asyncio.Task[PushRecoveryProjection] | None,
        push_recovery_settlement_cycle: asyncio.Task[None] | None,
        runner_settlement: Awaitable[bool] | None,
    ) -> None:
        """Join every retained task and preserve fail-closed shutdown state."""
        owned_tasks = tuple(
            task
            for task in (
                cycle,
                waiter,
                action_cycle,
                action_waiter,
                review_cycle,
                review_waiter,
                commit_cycle,
                commit_waiter,
                recovery_cycle,
                recovery_waiter,
            )
            if task is not None and not task.done()
        )
        if owned_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in owned_tasks),
                return_exceptions=True,
            )
        runner_confirmed = True
        if runner_settlement is not None:
            try:
                runner_confirmed = bool(
                    await asyncio.shield(runner_settlement)
                    if isinstance(runner_settlement, asyncio.Future)
                    else await runner_settlement
                )
            except BaseException:
                runner_confirmed = False
        push_tasks = tuple(
            task
            for task in (
                push_local_cycle,
                push_local_waiter,
                push_preflight_cycle,
                push_preflight_waiter,
                push_cycle,
                push_waiter,
                push_recovery_cycle,
                push_recovery_waiter,
                push_recovery_settlement_cycle,
            )
            if task is not None and not task.done()
        )
        self._shutdown_runner_confirmed = runner_confirmed
        if runner_confirmed and push_tasks:
            _done, pending = await asyncio.wait(
                push_tasks,
                timeout=_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS,
            )
            if (
                pending
                and push_preflight_cycle is not None
                and not push_preflight_cycle.done()
            ):
                self._push_quarantine_requested = True
                push_preflight_cycle.cancel()
        if push_preflight_cycle is not None and not push_preflight_cycle.done():
            quarantine_signal = self._push_quarantine_signal
            if quarantine_signal is not None and not quarantine_signal.done():
                await asyncio.wait(
                    (push_preflight_cycle, quarantine_signal),
                    timeout=_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )
        quarantine_cleared = False
        if runner_confirmed and self._unsettled_push_preflight is not None:
            quarantine_cleared = (
                await self._settle_unsettled_push_preflight_shutdown()
            )
            if quarantine_cleared:
                if (
                    push_preflight_cycle is not None
                    and not push_preflight_cycle.done()
                ):
                    push_preflight_cycle.cancel()
                if (
                    push_preflight_waiter is not None
                    and not push_preflight_waiter.done()
                ):
                    push_preflight_waiter.cancel()
                self._push_quarantine_requested = False
        if (
            runner_confirmed
            and self._unsettled_push_preflight is None
            and push_tasks
        ):
            await asyncio.wait(
                push_tasks,
                timeout=_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS,
            )
        if (
            self._unsettled_push_preflight is None
            and (
                push_preflight_cycle is None
                or push_preflight_cycle.done()
            )
        ):
            self._push_quarantine_requested = False
        self._settle_uncertain_commit_shutdown(runner_confirmed)
        self._settle_uncertain_push_shutdown(runner_confirmed)
        self._settle_push_shutdown(runner_confirmed)
        self._retry_pending_hooks_cleanup()
        self._status_cycle = None
        self._status_cycle_binding = None
        self._status_waiter = None
        self._action_cycle = None
        self._action_waiter = None
        self._commit_review_cycle = None
        self._commit_review_waiter = None
        self._commit_cycle = None
        self._commit_waiter = None
        self._commit_recovery_cycle = None
        self._commit_recovery_waiter = None
        self._commit_child_started = False
        self._commit_child_started_signal = None
        self._retained_commit_operation = None
        self._discard_commit_review_snapshots()
        self._pending_status = None
        self._rerun_available = False
        self._status_dirty = False
        if binding is not None:
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)

    async def _settle_unsettled_push_preflight_shutdown(self) -> bool:
        """Release one quarantined preflight only after exact terminal proof."""
        quarantine = self._unsettled_push_preflight
        if quarantine is None:
            return True
        token = quarantine.retained_child
        token_was_globally_released = False
        try:
            claimed = self._runner.claim_retained_child(token)
        except ValueError:
            token_was_globally_released = True
            claimed = False
        except Exception:
            return False
        if not token_was_globally_released:
            if claimed is not True:
                return False
            try:
                settlement = await self._runner.settle_retained_child(
                    token,
                    timeout=_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS,
                )
            except Exception:
                return False
            if (
                type(settlement) is not RetainedGitChildSettlement
                or settlement.state in {"alive", "uncertain"}
                or not settlement.owned_process_tree
                or not settlement.containment_proved
            ):
                return False
            try:
                released = self._runner.release_retained_child(token)
            except Exception:
                return False
            if released is not True:
                return False
        self._owner._revoke_destination_authorization(
            quarantine.authorization
        )
        if quarantine.active_lease.release() is not True:
            return False
        self._close_or_retain_push_context(quarantine.context)
        quarantine.mutation_lease.release()
        self._unsettled_push_preflight = None
        return True

    def _settle_push_shutdown(self, runner_confirmed: bool) -> None:
        """Release push review/context state only after runner settlement."""
        if not runner_confirmed:
            return
        for handle, snapshot in tuple(self._push_review_snapshots.items()):
            self._owner._revoke_destination_authorization(
                snapshot.authorization
            )
            snapshot.review_lease.release()
            self._push_review_snapshots.pop(handle, None)
            self._close_or_retain_push_context(snapshot.context)
        self._retry_pending_push_context_cleanup()
        push_cycles_active = any(
            task is not None and not task.done()
            for task in (
                self._push_local_proof_cycle,
                self._push_local_proof_waiter,
                self._push_preflight_cycle,
                self._push_preflight_waiter,
                self._push_cycle,
                self._push_waiter,
                self._push_recovery_cycle,
                self._push_recovery_waiter,
                self._push_recovery_settlement_cycle,
            )
        )
        if not push_cycles_active:
            self._push_local_proof_cycle = None
            self._push_local_proof_waiter = None
            self._push_preflight_cycle = None
            self._push_preflight_waiter = None
            self._push_cycle = None
            self._push_waiter = None
            self._push_recovery_cycle = None
            self._push_recovery_waiter = None
            self._push_recovery_settlement_cycle = None
        if (
            not self._push_review_snapshots
            and not self._pending_push_contexts
            and self._unsettled_push_preflight is None
            and self._uncertain_push is None
            and not push_cycles_active
        ):
            self._retained_push_operation = None

    def _settle_uncertain_push_shutdown(
        self,
        runner_confirmed: bool,
    ) -> None:
        """Discard process-only recovery after runner-owned trees terminate."""
        evidence = self._uncertain_push
        if evidence is None or not runner_confirmed:
            return
        if any(
            task is not None and not task.done()
            for task in (
                self._push_cycle,
                self._push_waiter,
                self._push_recovery_cycle,
                self._push_recovery_waiter,
                self._push_recovery_settlement_cycle,
            )
        ):
            return
        retained_child = evidence.retained_child
        if retained_child is not None:
            try:
                released = self._runner.release_retained_child(retained_child)
            except ValueError:
                released = True
            except Exception:
                return
            if released is not True:
                return
        self._owner._clear_push_recovery(evidence.recovery_capture)
        self._release_uncertain_push_evidence(evidence)

    def _settle_uncertain_commit_shutdown(
        self,
        runner_confirmed: bool,
    ) -> None:
        """Release only exact terminal evidence after bounded shutdown proof."""
        self._settle_orphaned_commit_shutdown(runner_confirmed)
        evidence = self._uncertain_commit
        if evidence is None or not runner_confirmed:
            return
        retained_child = evidence.retained_child
        if retained_child is not None:
            try:
                released = self._runner.release_retained_child(retained_child)
            except (RuntimeError, ValueError):
                return
            if not released:
                return
            evidence = replace(evidence, retained_child=None)
            self._uncertain_commit = evidence
        hooks_directory = evidence.hooks_directory
        if (
            hooks_directory is not None
            and self._remove_hooks_directory(hooks_directory)
        ):
            evidence = replace(evidence, hooks_directory=None)
            self._uncertain_commit = evidence
        mutation_lease = evidence.mutation_lease
        if mutation_lease is None:
            return
        mutation_lease.release()
        self._uncertain_commit = replace(
            evidence,
            mutation_lease=None,
        )

    def _settle_orphaned_commit_shutdown(
        self,
        runner_confirmed: bool,
    ) -> None:
        """Settle rebound lifecycle resources after the runner proves exit."""
        lifecycle = self._orphaned_commit
        if lifecycle is None or not runner_confirmed:
            return
        retained_child = lifecycle.retained_child
        if retained_child is not None:
            try:
                released = self._runner.release_retained_child(retained_child)
            except (RuntimeError, ValueError):
                return
            if not released:
                return
            lifecycle = replace(
                lifecycle,
                retained_child=None,
                termination_known=True,
            )
            self._orphaned_commit = lifecycle
        elif not lifecycle.termination_known:
            lifecycle = replace(lifecycle, termination_known=True)
            self._orphaned_commit = lifecycle
        hooks_directory = lifecycle.hooks_directory
        if (
            hooks_directory is not None
            and self._remove_hooks_directory(hooks_directory)
        ):
            lifecycle = replace(lifecycle, hooks_directory=None)
            self._orphaned_commit = lifecycle
        mutation_lease = lifecycle.mutation_lease
        if mutation_lease is not None:
            mutation_lease.release()
            lifecycle = replace(lifecycle, mutation_lease=None)
            self._orphaned_commit = lifecycle
        if lifecycle.hooks_directory is None:
            self._orphaned_commit = None

    def _remove_hooks_directory(self, directory: Path) -> bool:
        if _remove_private_hooks_directory(directory):
            self._pending_hooks_cleanup.discard(directory)
            return True
        self._pending_hooks_cleanup.add(directory)
        return False

    def _retry_pending_hooks_cleanup(self) -> None:
        evidence = self._uncertain_commit
        orphaned = self._orphaned_commit
        retained_hooks = {
            hooks
            for hooks in (
                None if evidence is None else evidence.hooks_directory,
                None if orphaned is None else orphaned.hooks_directory,
            )
            if hooks is not None
        }
        for directory in tuple(self._pending_hooks_cleanup):
            if directory in retained_hooks:
                continue
            self._remove_hooks_directory(directory)

    async def _shield_status_cycle(
        self,
        cycle: asyncio.Task[SessionGitStatus],
    ) -> SessionGitStatus:
        return await asyncio.shield(cycle)

    def _status_cycle_completed(
        self,
        cycle: asyncio.Task[SessionGitStatus],
    ) -> None:
        if self._status_cycle is cycle:
            self._status_cycle = None
            self._status_cycle_binding = None
            self._pending_status = None
            self._rerun_available = False
            self._status_dirty = False

    async def _run_status_cycle(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
        repository: RepositoryIdentity,
        lease: GitStatusLease,
        request_generation: int,
    ) -> SessionGitStatus:
        invalidation_generation = lease.invalidation_generation
        try:
            result = await self._query_status(
                binding,
                changes,
                repository,
            )
            lease.release()
            pending = self._pending_status
            self._pending_status = None
            if pending is None:
                return self._publish_cycle_result(
                    binding,
                    result,
                    request_generation,
                    invalidation_generation,
                )

            (
                pending_binding,
                pending_changes,
                pending_repository,
                pending_generation,
                pending_invalidation_generation,
            ) = pending
            self._status_cycle_binding = pending_binding
            admission = self._owner.admit_status(pending_binding)
            next_lease = admission.lease
            if next_lease is None:
                self._rerun_available = False
                message = (
                    "Git status rerun was suppressed because a mutation "
                    "was admitted"
                )
                stale = self._local_status(
                    pending_binding,
                    "stale",
                    rows=self._disabled_rows(result.rows, message),
                    repository=pending_repository,
                    head=result.head,
                    message=message,
                )
                return self._publish_cycle_result(
                    pending_binding,
                    stale,
                    pending_generation,
                    pending_invalidation_generation,
                )

            lease = next_lease
            invalidation_generation = pending_invalidation_generation
            self._rerun_available = False
            result = await self._query_status(
                pending_binding,
                pending_changes,
                pending_repository,
            )
            if self._status_dirty:
                self._status_dirty = False
                dirty_generation = self._status_request_generation
                message = (
                    "Newer File Notes changes are known; refresh Git status"
                )
                result = self._local_status(
                    pending_binding,
                    "stale",
                    rows=self._disabled_rows(result.rows, message),
                    repository=result.repository,
                    head=result.head,
                    message=message,
                )
                pending_generation = dirty_generation
            return self._publish_cycle_result(
                pending_binding,
                result,
                pending_generation,
                invalidation_generation,
            )
        finally:
            lease.release()

    def _create_task(
        self,
        coroutine: object,
    ) -> asyncio.Task[SessionGitStatus]:
        try:
            return asyncio.get_running_loop().create_task(coroutine)  # type: ignore[arg-type]
        except BaseException:
            close = getattr(coroutine, "close", None)
            if close is not None:
                close()
            raise

    def _publish_cycle_result(
        self,
        binding: SessionBinding,
        status: SessionGitStatus,
        request_generation: int,
        invalidation_generation: int,
    ) -> SessionGitStatus:
        if (
            not self._sealed
            and request_generation == self._status_request_generation
            and binding == self._owner.current_binding()
        ):
            self._owner.publish_status(
                binding,
                status,
                invalidation_generation=invalidation_generation,
            )
        return status

    async def _query_status(
        self,
        binding: SessionBinding,
        changes: tuple[SequencedSessionChange, ...],
        repository: RepositoryIdentity,
        *,
        publish_ownership_changes: bool = True,
    ) -> SessionGitStatus:
        if (
            self._owner.snapshot(binding).trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return self._local_status(
                binding,
                "stale",
                message="Repository identity changed; trust was cleared",
            )

        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust(binding)
            return self._local_status(
                binding,
                "stale",
                message="Selected File Notes root is no longer safe",
            )
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return self._failed_status(
                binding,
                "error",
                repository=repository,
                message="Unable to verify sparse-checkout state",
            )
        if sparse:
            return self._failed_status(
                binding,
                "unavailable",
                repository=repository,
                message="Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(changes)
        repository_groups: list[SessionChangeGroup] = []
        original_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped_group, invalid_row = self._map_group(
                root,
                repository,
                group,
            )
            if invalid_row is not None:
                invalid_rows[group.group_id] = invalid_row
                continue
            assert mapped_group is not None
            repository_groups.append(mapped_group)
            original_groups[group.group_id] = group

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            repository_groups,
            publish_ownership_changes=publish_ownership_changes,
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership and publish_ownership_changes:
                self._owner.clear_ownership(binding)
            return self._failed_status(
                binding,
                (
                    "stale"
                    if raw.state == "uncertain"
                    else raw.state
                ),
                repository=repository,
                head=raw.head,
                message=raw.message,
            )
        head = raw.head
        index_sequence = raw.index_entries
        index_entries = _stage_zero_index(index_sequence)
        conflicted_paths = {
            entry.path for entry in index_sequence if entry.stage != 0
        }
        status_records = raw.status_records

        ownership_by_id: dict[int, StagingOwnership] = {}
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = dict(current_ownership)
        for repository_group in repository_groups:
            owned = current_ownership.get(repository_group.group_id)
            if owned is None:
                continue
            original_group = original_groups[repository_group.group_id]
            mapped_ownership = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if (
                mapped_ownership is None
                or mapped_ownership.repository != repository
                or mapped_ownership.head != head
                or any(
                    path in conflicted_paths
                    or index_entries.get(path) != expected
                    for path, expected in mapped_ownership.post_stage_entries.items()
                )
            ):
                retained_ownership.pop(repository_group.group_id, None)
                continue
            ownership_by_id[repository_group.group_id] = mapped_ownership
        if publish_ownership_changes and len(retained_ownership) != len(
            current_ownership
        ):
            self._owner.publish_ownership(binding, retained_ownership)

        classified = classify_session_rows(
            repository_groups,
            status_records,
            index_sequence,
            ownership_by_id,
        )
        classified_by_id = {
            row.group_id: replace(
                row,
                group=original_groups[row.group_id],
            )
            for row in classified
        }
        rows = tuple(
            invalid_rows.get(group.group_id)
            or classified_by_id[group.group_id]
            for group in groups
        )
        return self._local_status(
            binding,
            "ready",
            rows=rows,
            repository=repository,
            head=head,
        )

    async def _read_raw_git_inspection(
        self,
        binding: SessionBinding,
        root: Path,
        repository: RepositoryIdentity,
        groups: Sequence[SessionChangeGroup],
        *,
        publish_ownership_changes: bool = True,
    ) -> _RawGitInspection | _RawGitInspectionFailure:
        """Read the shared fresh HEAD, complete index, and scoped status."""
        head_result = await self._read_head(root)
        if isinstance(head_result, _HeadReadFailure):
            return _RawGitInspectionFailure(
                (
                    "unavailable"
                    if head_result.kind == "unavailable"
                    else "error"
                ),
                head_result.message,
                revoke_ownership=True,
            )
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = {
            group_id: ownership
            for group_id, ownership in current_ownership.items()
            if (
                ownership.repository == repository
                and ownership.head == head_result
            )
        }
        if publish_ownership_changes and len(retained_ownership) != len(
            current_ownership
        ):
            self._owner.publish_ownership(binding, retained_ownership)

        index_result = await self._run_status_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
        )
        if not _command_succeeded(index_result):
            return _RawGitInspectionFailure(
                (
                    "uncertain"
                    if index_result.termination_uncertain
                    else "stale"
                ),
                _command_failure_message(index_result, "Git index read failed"),
                head=head_result,
                revoke_ownership=index_result.termination_uncertain,
            )
        try:
            index_entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return _RawGitInspectionFailure(
                "error",
                str(error),
                head=head_result,
                revoke_ownership=True,
            )

        index_by_path = _stage_zero_index(index_entries)
        conflicted_paths = {
            entry.path for entry in index_entries if entry.stage != 0
        }
        current_ownership = self._owner.snapshot(binding).staging_ownership
        retained_ownership = {
            group_id: ownership
            for group_id, ownership in current_ownership.items()
            if all(
                path not in conflicted_paths
                and index_by_path.get(path) == expected
                for path, expected in ownership.post_stage_entries.items()
            )
        }
        if publish_ownership_changes and len(retained_ownership) != len(
            current_ownership
        ):
            self._owner.publish_ownership(binding, retained_ownership)

        status_records: tuple[PorcelainRecord, ...] = ()
        repository_paths = tuple(
            os.fsencode(path)
            for group in groups
            for path in group.endpoints
        )
        if repository_paths:
            allowed_paths = frozenset(
                {
                    *(path for group in groups for path in group.endpoints),
                    *(
                        entry.path
                        for entry in index_entries
                        if any(
                            _paths_overlap(entry.path, endpoint)
                            for group in groups
                            for endpoint in group.endpoints
                        )
                    ),
                }
            )
            status_result = await self._run_status_command(
                repository,
                build_status_argv(
                    self._git_executable_or_raise(),
                    repository_paths,
                ),
            )
            if not _command_succeeded(status_result):
                return _RawGitInspectionFailure(
                    (
                        "uncertain"
                        if status_result.termination_uncertain
                        else "stale"
                    ),
                    _command_failure_message(status_result, "Git status failed"),
                    head=head_result,
                )
            try:
                status_records = parse_porcelain_v2_z(
                    status_result.stdout,
                    allowed_paths=allowed_paths,
                )
            except PorcelainV2ParseError as error:
                return _RawGitInspectionFailure(
                    "error",
                    str(error),
                    head=head_result,
                )
        return _RawGitInspection(
            head_result,
            index_entries,
            status_records,
        )

    async def _inspect_stage(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> _StageInspection | GitActionResult:
        """Read one fresh identity/HEAD/index/status Stage preflight."""
        failure = lambda state, message: GitActionResult(  # noqa: E731
            "stage",
            state,
            (),
            message=message,
        )
        snapshot = self._owner.snapshot(binding)
        if (
            snapshot.trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return failure("stale", "Repository identity changed; trust was cleared")
        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust_if_matches(binding, repository)
            return failure("stale", "Selected File Notes root is no longer safe")
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return failure("error", "Unable to verify sparse-checkout state")
        if sparse:
            return failure(
                "blocked",
                "Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(snapshot.changes)
        repository_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped, invalid = self._map_group(root, repository, group)
            if invalid is not None:
                invalid_rows[group.group_id] = invalid
            else:
                assert mapped is not None
                repository_groups[group.group_id] = mapped

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            tuple(repository_groups.values()),
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership:
                self._owner.clear_ownership(binding)
            return failure(
                (
                    "uncertain"
                    if raw.state in {"stale", "unavailable", "uncertain"}
                    else "error"
                ),
                raw.message,
            )
        head_result = raw.head
        index_sequence = raw.index_entries
        index_entries = _stage_zero_index(index_sequence)
        status_records = raw.status_records

        ownership_by_id: dict[int, StagingOwnership] = {}
        groups_by_id = {group.group_id: group for group in groups}
        for group_id, owned in snapshot.staging_ownership.items():
            original_group = groups_by_id.get(group_id)
            repository_group = repository_groups.get(group_id)
            if original_group is None or repository_group is None:
                continue
            mapped = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if mapped is not None:
                ownership_by_id[group_id] = mapped

        classified = classify_session_rows(
            tuple(repository_groups.values()),
            status_records,
            index_sequence,
            ownership_by_id,
        )
        rows = {row.group_id: row for row in classified}
        rows.update(invalid_rows)
        return _StageInspection(
            repository=repository,
            head=head_result,
            groups=groups,
            repository_groups=repository_groups,
            rows=rows,
            index_sequence=index_sequence,
            index_entries=index_entries,
            status_records=status_records,
        )

    async def _inspect_unstage(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> _StageInspection | GitActionResult:
        """Read exact Unstage facts without imposing Stage worktree types."""
        failure = lambda state, message: GitActionResult(  # noqa: E731
            "unstage",
            state,
            (),
            message=message,
        )
        snapshot = self._owner.snapshot(binding)
        if (
            snapshot.trusted_repository != repository
            or not await self.revalidate_repository(binding, repository)
        ):
            return failure("stale", "Repository identity changed; trust was cleared")
        root = self._safe_root(binding)
        if root is None:
            self._owner.clear_trust_if_matches(binding, repository)
            return failure("stale", "Selected File Notes root is no longer safe")
        sparse = await self._sparse_checkout_state(repository)
        if sparse is None:
            return failure("error", "Unable to verify sparse-checkout state")
        if sparse:
            return failure(
                "blocked",
                "Sparse checkout or sparse index is unsupported",
            )

        groups = coalesce_session_changes(snapshot.changes)
        repository_groups: dict[int, SessionChangeGroup] = {}
        invalid_rows: dict[int, SessionGitRow] = {}
        for group in groups:
            mapped, invalid = self._map_group_for_unstage(
                root,
                repository,
                group,
            )
            if invalid is not None:
                invalid_rows[group.group_id] = invalid
            else:
                assert mapped is not None
                repository_groups[group.group_id] = mapped

        raw = await self._read_raw_git_inspection(
            binding,
            root,
            repository,
            tuple(repository_groups.values()),
        )
        if isinstance(raw, _RawGitInspectionFailure):
            if raw.revoke_ownership:
                self._owner.clear_ownership(binding)
            return failure(
                (
                    "uncertain"
                    if raw.state in {"stale", "unavailable", "uncertain"}
                    else "error"
                ),
                raw.message,
            )
        index_entries = _stage_zero_index(raw.index_entries)
        ownership_by_id: dict[int, StagingOwnership] = {}
        groups_by_id = {group.group_id: group for group in groups}
        for group_id, owned in snapshot.staging_ownership.items():
            original_group = groups_by_id.get(group_id)
            repository_group = repository_groups.get(group_id)
            if original_group is None or repository_group is None:
                continue
            mapped = _map_ownership_topology(
                owned,
                original_group,
                repository_group,
            )
            if mapped is not None:
                ownership_by_id[group_id] = mapped
        rows = {
            row.group_id: row
            for row in classify_session_rows(
                tuple(repository_groups.values()),
                raw.status_records,
                raw.index_entries,
                ownership_by_id,
            )
        }
        rows.update(invalid_rows)
        return _StageInspection(
            repository=repository,
            head=raw.head,
            groups=groups,
            repository_groups=repository_groups,
            rows=rows,
            index_sequence=raw.index_entries,
            index_entries=index_entries,
            status_records=raw.status_records,
        )

    async def _apply_stage(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        inspection: _StageInspection,
    ) -> GitActionResult:
        """Apply one checked Stage command and publish exact ownership."""
        snapshot = self._owner.snapshot(binding)
        ownership = dict(snapshot.staging_ownership)
        staged: list[int] = []
        clean: list[int] = []
        blocked: list[int] = []
        pathspecs: list[bytes] = []
        baselines: dict[int, dict[str, IndexBaseline]] = {}
        affected_paths: dict[int, tuple[str, ...]] = {}
        groups_by_id = {group.group_id: group for group in inspection.groups}
        ownership_revoked = False

        for group_id in requested:
            group = groups_by_id.get(group_id)
            repository_group = inspection.repository_groups.get(group_id)
            row = inspection.rows.get(group_id)
            if group is None or repository_group is None or row is None:
                blocked.append(group_id)
                continue
            current_owned = ownership.get(group_id)
            owned_clean = False
            if current_owned is None:
                eligible = row.state == "unstaged" and row.stage_action == "stage"
            else:
                owned_matches = self._owned_stage_preflight_matches(
                    current_owned,
                    inspection,
                    repository_group,
                )
                eligible = (
                    owned_matches
                    and row.state
                    in {"owned_newer_edits", "owned_topology_changed"}
                    and row.stage_action == "stage_update"
                )
                owned_clean = (
                    owned_matches
                    and row.state == "owned"
                    and row.stage_action is None
                )
            effective = stage_pathspecs(
                repository_group,
                inspection.status_records,
                groups=tuple(inspection.repository_groups.values()),
            )
            if not eligible:
                if owned_clean:
                    clean.append(group_id)
                    continue
                if current_owned is not None:
                    ownership.pop(group_id, None)
                    ownership_revoked = True
                if row.state == "clean":
                    clean.append(group_id)
                else:
                    blocked.append(group_id)
                continue
            if not effective:
                clean.append(group_id)
                continue
            decoded = tuple(os.fsdecode(path) for path in effective)
            staged.append(group_id)
            pathspecs.extend(effective)
            affected_paths[group_id] = decoded
            original = (
                dict(current_owned.original_baselines)
                if current_owned is not None
                else {}
            )
            for path in decoded:
                original.setdefault(
                    path,
                    IndexBaseline(inspection.index_entries.get(path)),
                )
            baselines[group_id] = original

        if not staged:
            if ownership_revoked:
                self._owner.publish_stage_result(
                    binding,
                    inspection.repository,
                    ownership,
                )
            return GitActionResult(
                "stage",
                "blocked",
                requested,
                clean_group_ids=tuple(clean),
                blocked_group_ids=tuple(blocked),
                message="No requested session group is eligible for Stage",
            )
        if not await self.revalidate_repository(binding, inspection.repository):
            return GitActionResult(
                "stage",
                "stale",
                requested,
                blocked_group_ids=tuple(requested),
                message="Repository identity changed before Stage",
            )
        root = self._safe_root(binding)
        endpoint_safety_changed = root is None
        if root is not None:
            for group_id in staged:
                remapped, invalid = self._map_group(
                    root,
                    inspection.repository,
                    groups_by_id[group_id],
                )
                if (
                    invalid is not None
                    or remapped is None
                    or remapped != inspection.repository_groups[group_id]
                    or not stage_group_is_closed(
                        remapped,
                        inspection.index_entries,
                    )
                ):
                    endpoint_safety_changed = True
                    break
        if endpoint_safety_changed:
            return GitActionResult(
                "stage",
                "blocked",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="File Notes endpoint safety changed before Stage",
            )

        result = await self._run_stage_command(
            inspection.repository,
            build_stage_argv(
                self._git_executable_or_raise(),
                tuple(dict.fromkeys(pathspecs)),
            ),
        )
        if not _command_succeeded(result):
            for group_id in staged:
                ownership.pop(group_id, None)
            self._owner.publish_stage_result(
                binding,
                inspection.repository,
                ownership,
            )
            return GitActionResult(
                "stage",
                "uncertain" if result.termination_uncertain else "error",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message=_index_mutation_failure_message(
                    result,
                    "Git Stage failed",
                    inspection.repository,
                ),
            )

        postflight = await self._read_stage_postflight(
            binding,
            inspection.repository,
        )
        if isinstance(postflight, GitActionResult):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return replace(postflight, requested_group_ids=requested)
        post_head, post_index = postflight
        latest_groups = {
            group.group_id: group
            for group in coalesce_session_changes(
                self._owner.snapshot(binding).changes
            )
        }
        if post_head != inspection.head or any(
            latest_groups.get(group_id) is None
            or latest_groups[group_id].topology_signature
            != groups_by_id[group_id].topology_signature
            for group_id in staged
        ):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Git HEAD or session topology changed during Stage",
            )
        if any(
            not stage_group_is_closed(
                inspection.repository_groups[group_id],
                post_index,
            )
            for group_id in staged
        ):
            self._owner.clear_ownership(binding)
            self._owner.clear_status(binding)
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Git index topology changed during Stage",
            )

        for group_id in staged:
            previous = ownership.get(group_id)
            paths = set(affected_paths[group_id])
            if previous is not None:
                paths.update(previous.post_stage_entries)
            post_entries = {path: post_index.get(path) for path in paths}
            if any(
                entry is not None
                and (entry.stage != 0 or entry.semantic_flags)
                for entry in post_entries.values()
            ):
                self._owner.clear_ownership(binding)
                self._owner.clear_status(binding)
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    blocked_group_ids=tuple(staged + blocked),
                    clean_group_ids=tuple(clean),
                    message="Git index semantics changed during Stage",
                )
            if any(
                inspection.index_entries.get(path) == post_index.get(path)
                for path in affected_paths[group_id]
            ):
                self._owner.clear_ownership(binding)
                self._owner.clear_status(binding)
                return GitActionResult(
                    "stage",
                    "uncertain",
                    requested,
                    blocked_group_ids=tuple(staged + blocked),
                    clean_group_ids=tuple(clean),
                    message="Git Stage did not change every effective path",
                )
            ownership[group_id] = StagingOwnership(
                repository=inspection.repository,
                head=post_head,
                approved_endpoint_topology=groups_by_id[group_id].endpoints,
                approved_move_edges=groups_by_id[group_id].move_edges,
                approved_current_path=groups_by_id[group_id].current_path,
                original_baselines=baselines[group_id],
                post_stage_entries=post_entries,
            )

        if not self._owner.publish_stage_result(
            binding,
            inspection.repository,
            ownership,
        ):
            return GitActionResult(
                "stage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(staged + blocked),
                clean_group_ids=tuple(clean),
                message="Stage result no longer matches the selected session",
            )
        return GitActionResult(
            "stage",
            "success",
            requested,
            staged_group_ids=tuple(staged),
            clean_group_ids=tuple(clean),
            blocked_group_ids=tuple(blocked),
        )

    @staticmethod
    def _owned_stage_preflight_matches(
        ownership: StagingOwnership,
        inspection: _StageInspection,
        group: SessionChangeGroup,
    ) -> bool:
        if (
            ownership.repository != inspection.repository
            or ownership.head != inspection.head
            or any(
                inspection.index_entries.get(path) != expected
                for path, expected in ownership.post_stage_entries.items()
            )
            or not stage_group_is_closed(group, inspection.index_entries)
        ):
            return False
        return not any(
            path not in ownership.post_stage_entries
            for record in inspection.status_records
            for path in _staged_record_paths(record)
            if path in group.endpoints
        )

    async def _read_stage_postflight(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> tuple[HeadIdentity, Mapping[str, IndexEntry]] | GitActionResult:
        if not await self.revalidate_repository(binding, repository):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message="Repository identity changed after Stage",
            )
        root = self._safe_root(binding)
        if root is None:
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message="Selected File Notes root changed after Stage",
            )
        head = await self._read_head(root)
        if isinstance(head, _HeadReadFailure):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message=head.message,
            )
        index_result = await self._run_stage_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
        )
        if not _command_succeeded(index_result):
            return GitActionResult(
                "stage",
                "uncertain",
                (),
                message=_command_failure_message(
                    index_result,
                    "Git post-Stage index read failed",
                ),
            )
        try:
            entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return GitActionResult("stage", "uncertain", (), message=str(error))
        return head, _stage_zero_index(entries)

    async def _run_stage_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
    ) -> GitCommandResult:
        try:
            return await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(self._environment),
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )

    async def _apply_unstage(
        self,
        binding: SessionBinding,
        requested: tuple[int, ...],
        inspection: _StageInspection,
    ) -> GitActionResult:
        """Restore only exact saved baselines for valid selected ownership."""
        snapshot = self._owner.snapshot(binding)
        groups_by_id = {group.group_id: group for group in inspection.groups}
        valid: list[int] = []
        blocked: list[int] = []
        revoked: list[int] = []
        selected_ownership: dict[int, StagingOwnership] = {}
        mapped_ownership: dict[int, StagingOwnership] = {}

        for group_id in requested:
            owned = snapshot.staging_ownership.get(group_id)
            group = groups_by_id.get(group_id)
            repository_group = inspection.repository_groups.get(group_id)
            row = inspection.rows.get(group_id)
            if (
                owned is None
                or group is None
                or repository_group is None
                or row is None
            ):
                blocked.append(group_id)
                continue
            mapped = _map_ownership_topology(
                owned,
                group,
                repository_group,
            )
            if (
                mapped is None
                or owned.topology_signature != group.topology_signature
            ):
                blocked.append(group_id)
                continue
            if any(
                entry.stage != 0
                and entry.path in repository_group.endpoints
                for entry in inspection.index_sequence
            ):
                blocked.append(group_id)
                revoked.append(group_id)
                continue
            if not ownership_signature_matches(
                mapped,
                repository=inspection.repository,
                head=inspection.head,
                topology=repository_group.topology_signature,
                current_index_entries=inspection.index_entries,
            ):
                blocked.append(group_id)
                revoked.append(group_id)
                continue
            if (
                not row.unstage_eligible
                or not unstage_group_is_closed(
                    repository_group,
                    mapped.original_baselines,
                    inspection.index_entries,
                    mapped,
                )
            ):
                blocked.append(group_id)
                continue
            valid.append(group_id)
            selected_ownership[group_id] = owned
            mapped_ownership[group_id] = mapped

        if revoked and not self._owner.publish_unstage_result(
            binding,
            inspection.repository,
            {
                group_id: snapshot.staging_ownership[group_id]
                for group_id in revoked
            },
            revoked,
        ):
            return GitActionResult(
                "unstage",
                "uncertain",
                requested,
                blocked_group_ids=requested,
                message="Unstage ownership changed during preflight",
            )
        if not valid:
            return GitActionResult(
                "unstage",
                "blocked",
                requested,
                blocked_group_ids=tuple(blocked),
                message="No requested session group is eligible for Unstage",
            )
        if not await self.revalidate_repository(binding, inspection.repository):
            self._owner.clear_ownership(binding)
            return GitActionResult(
                "unstage",
                "stale",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message="Repository identity changed before Unstage",
            )

        try:
            payload = _build_combined_update_index_payload(
                tuple(mapped_ownership[group_id] for group_id in valid),
                inspection.index_entries,
            )
        except ValueError as error:
            self._owner.publish_unstage_result(
                binding,
                inspection.repository,
                selected_ownership,
                valid,
            )
            return GitActionResult(
                "unstage",
                "blocked",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message=str(error),
            )
        result = await self._run_unstage_command(
            inspection.repository,
            build_unstage_argv(self._git_executable_or_raise()),
            payload,
        )
        postflight = await self._read_unstage_postflight(
            binding,
            inspection.repository,
        )
        if isinstance(postflight, GitActionResult):
            if self._owner.snapshot(binding).trusted_repository == inspection.repository:
                self._owner.publish_unstage_result(
                    binding,
                    inspection.repository,
                    selected_ownership,
                    valid,
                )
            self._owner.clear_status(binding)
            return replace(
                postflight,
                requested_group_ids=requested,
                blocked_group_ids=tuple(valid + blocked),
            )
        post_head, post_index = postflight
        latest_groups = {
            group.group_id: group
            for group in coalesce_session_changes(
                self._owner.snapshot(binding).changes
            )
        }
        verified = (
            _command_succeeded(result)
            and post_head == inspection.head
            and all(
                latest_groups.get(group_id) is not None
                and latest_groups[group_id].topology_signature
                == groups_by_id[group_id].topology_signature
                and _baseline_matches_index(
                    mapped_ownership[group_id],
                    post_index,
                )
                for group_id in valid
            )
        )
        if not verified:
            if not self._owner.publish_unstage_result(
                binding,
                inspection.repository,
                selected_ownership,
                valid,
            ):
                self._owner.clear_ownership(binding)
            return GitActionResult(
                "unstage",
                (
                    "uncertain"
                    if result.termination_uncertain or _command_succeeded(result)
                    else "error"
                ),
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message=(
                    "Unstage postflight did not verify the saved baseline"
                    if _command_succeeded(result)
                    else _index_mutation_failure_message(
                        result,
                        "Git Unstage failed",
                        inspection.repository,
                    )
                ),
            )
        if not self._owner.publish_unstage_result(
            binding,
            inspection.repository,
            selected_ownership,
            valid,
        ):
            return GitActionResult(
                "unstage",
                "uncertain",
                requested,
                blocked_group_ids=tuple(valid + blocked),
                message="Unstage result no longer matches the selected session",
            )
        return GitActionResult(
            "unstage",
            "success",
            requested,
            unstaged_group_ids=tuple(valid),
            blocked_group_ids=tuple(blocked),
        )

    async def _read_unstage_postflight(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> tuple[HeadIdentity, Mapping[str, IndexEntry]] | GitActionResult:
        if not await self.revalidate_repository(binding, repository):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message="Repository identity changed after Unstage",
            )
        root = self._safe_root(binding)
        if root is None:
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message="Selected File Notes root changed after Unstage",
            )
        head = await self._read_head(root)
        if isinstance(head, _HeadReadFailure):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message=head.message,
            )
        index_result = await self._run_unstage_command(
            repository,
            build_index_argv(self._git_executable_or_raise()),
            None,
        )
        if not _command_succeeded(index_result):
            return GitActionResult(
                "unstage",
                "uncertain",
                (),
                message=_command_failure_message(
                    index_result,
                    "Git post-Unstage index read failed",
                ),
            )
        try:
            entries = parse_index_entries_z(index_result.stdout)
        except GitIndexParseError as error:
            return GitActionResult("unstage", "uncertain", (), message=str(error))
        return head, _stage_zero_index(entries)

    async def _run_unstage_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
        stdin: bytes | None,
    ) -> GitCommandResult:
        try:
            return await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(self._environment),
                stdin=stdin,
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )

    async def _sparse_checkout_state(
        self,
        repository: RepositoryIdentity,
    ) -> bool | None:
        for key in ("core.sparseCheckout", "index.sparse"):
            result = await self._run_status_command(
                repository,
                (
                    self._git_executable_or_raise(),
                    "config",
                    "--bool",
                    "--get",
                    key,
                ),
            )
            if result.timed_out or result.termination_uncertain:
                return None
            if result.returncode == 1:
                continue
            if result.returncode != 0:
                return None
            if result.stdout == b"true\n":
                return True
            if result.stdout != b"false\n":
                return None
        return False

    async def _run_status_command(
        self,
        repository: RepositoryIdentity,
        argv: Sequence[GitArg],
    ) -> GitCommandResult:
        try:
            result = await self._runner.run(
                argv,
                cwd=repository.worktree_root,
                environment=build_git_environment(
                    self._environment,
                    for_status=True,
                ),
                timeout=self._status_timeout,
            )
        except OSError as error:
            return GitCommandResult(
                127,
                b"",
                str(error).encode("utf-8", "replace"),
            )
        if result.termination_uncertain:
            binding = self._owner.current_binding()
            if (
                binding is not None
                and self._owner.snapshot(binding).trusted_repository
                == repository
            ):
                self._owner.clear_ownership(binding)
        return result

    def _map_group(
        self,
        notes_root: Path,
        repository: RepositoryIdentity,
        group: SessionChangeGroup,
    ) -> tuple[SessionChangeGroup | None, SessionGitRow | None]:
        mapping: dict[str, str] = {}
        for endpoint in group.endpoints:
            mapped, problem = _map_session_endpoint(
                notes_root,
                Path(repository.worktree_root),
                endpoint,
            )
            if problem == "nested_repository":
                return None, SessionGitRow(
                    group,
                    "nested_repository",
                    disabled_reason="Nested repository unsupported",
                )
            if problem is not None:
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason=problem,
                )
            assert mapped is not None
            mapping[endpoint] = mapped
        return (
            replace(
                group,
                endpoints=tuple(mapping[path] for path in group.endpoints),
                source_path=mapping[group.source_path],
                destination_path=(
                    None
                    if group.destination_path is None
                    else mapping[group.destination_path]
                ),
                current_path=mapping[group.current_path],
                move_edges=tuple(
                    (mapping[source], mapping[destination])
                    for source, destination in group.move_edges
                ),
            ),
            None,
        )

    def _map_group_for_unstage(
        self,
        notes_root: Path,
        repository: RepositoryIdentity,
        group: SessionChangeGroup,
    ) -> tuple[SessionChangeGroup | None, SessionGitRow | None]:
        """Map safe endpoint text without rejecting owned replacements."""
        worktree_root = Path(repository.worktree_root)
        try:
            root_prefix = notes_root.relative_to(worktree_root)
        except ValueError:
            return None, SessionGitRow(
                group,
                "unsupported",
                disabled_reason="Unsafe File Notes path",
            )
        mapping: dict[str, str] = {}
        for endpoint in group.endpoints:
            components = endpoint.split("/")
            if (
                not endpoint
                or endpoint.startswith("/")
                or "\0" in endpoint
                or ".git" in components
                or any(component in {"", ".", ".."} for component in components)
            ):
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason="Unsafe File Notes path",
                )
            for depth in range(1, len(components) + 1):
                boundary = notes_root.joinpath(*components[:depth])
                try:
                    boundary_stat = boundary.stat(follow_symlinks=False)
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except OSError:
                    return None, SessionGitRow(
                        group,
                        "unsupported",
                        disabled_reason="Unsafe File Notes path",
                    )
                if stat.S_ISLNK(boundary_stat.st_mode):
                    return None, SessionGitRow(
                        group,
                        "unsupported",
                        disabled_reason="Unsafe File Notes path",
                    )
                if stat.S_ISDIR(boundary_stat.st_mode):
                    try:
                        (boundary / ".git").stat(follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        return None, SessionGitRow(
                            group,
                            "unsupported",
                            disabled_reason="Unsafe nested repository boundary",
                        )
                    else:
                        return None, SessionGitRow(
                            group,
                            "nested_repository",
                            disabled_reason="Nested repository unsupported",
                        )
            mapped = root_prefix.joinpath(*components).as_posix()
            if not mapped or mapped == ".":
                return None, SessionGitRow(
                    group,
                    "unsupported",
                    disabled_reason="Unsafe File Notes path",
                )
            mapping[endpoint] = mapped
        return (
            replace(
                group,
                endpoints=tuple(mapping[path] for path in group.endpoints),
                source_path=mapping[group.source_path],
                destination_path=(
                    None
                    if group.destination_path is None
                    else mapping[group.destination_path]
                ),
                current_path=mapping[group.current_path],
                move_edges=tuple(
                    (mapping[source], mapping[destination])
                    for source, destination in group.move_edges
                ),
            ),
            None,
        )

    def _failed_status(
        self,
        binding: SessionBinding,
        state: Literal["stale", "unavailable", "error"],
        *,
        repository: RepositoryIdentity,
        head: HeadIdentity | None = None,
        message: str | None = None,
    ) -> SessionGitStatus:
        rows: tuple[SessionGitRow, ...] = ()
        snapshot = self._owner.snapshot(binding)
        previous = snapshot.git_status
        if (
            binding == self._owner.current_binding()
            and snapshot.trusted_repository == repository
            and previous is not None
            and previous.binding_generation == binding.generation
            and previous.repository == repository
        ):
            rows = self._disabled_rows(previous.rows, message)
        return self._local_status(
            binding,
            state,
            rows=rows,
            repository=repository,
            head=head,
            message=message,
        )

    @staticmethod
    def _disabled_rows(
        rows: tuple[SessionGitRow, ...],
        message: str | None,
    ) -> tuple[SessionGitRow, ...]:
        reason = message or "Git status refresh is required"
        return tuple(
            replace(
                row,
                stage_action=None,
                unstage_eligible=False,
                disabled_reason=row.disabled_reason or reason,
            )
            for row in rows
        )

    def _local_status(
        self,
        binding: SessionBinding,
        state: Literal["ready", "stale", "unavailable", "error"],
        *,
        rows: tuple[SessionGitRow, ...] = (),
        repository: RepositoryIdentity | None = None,
        head: HeadIdentity | None = None,
        message: str | None = None,
    ) -> SessionGitStatus:
        generation = self._owner.next_status_generation(binding)
        return SessionGitStatus(
            binding_generation=binding.generation,
            status_generation=0 if generation is None else generation,
            state=state,
            rows=rows,
            repository=repository,
            head=head,
            message=message,
        )

    def _git_executable_or_raise(self) -> str:
        if self._git_executable is None:
            raise RuntimeError("Git is not installed")
        return self._git_executable

    def _safe_root(self, binding: SessionBinding) -> Path | None:
        root = Path(binding.root_key)
        try:
            root_stat = root.stat(follow_symlinks=False)
            canonical = root.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or canonical != root
        ):
            return None
        return canonical

    async def _run_discovery(
        self,
        root: Path,
        arguments: Sequence[str],
    ) -> GitCommandResult | None:
        assert self._git_executable is not None
        try:
            result = await self._runner.run(
                (self._git_executable, *arguments),
                cwd=str(root),
                environment=build_git_environment(
                    self._environment,
                    stable_locale=True,
                ),
                timeout=self._discovery_timeout,
            )
        except OSError:
            return None
        if result.timed_out or result.termination_uncertain:
            return None
        return result

    async def _read_repository_paths(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path] | None:
        """Read the canonical worktree, Git-dir, and common-dir mapping."""
        path_commands = (
            ("rev-parse", "--path-format=absolute", "--show-toplevel"),
            ("rev-parse", "--absolute-git-dir"),
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        )
        resolved_paths: list[Path] = []
        for arguments in path_commands:
            result = await self._run_discovery(root, arguments)
            if result is None or result.returncode != 0:
                return None
            resolved = _canonical_directory_from_git(result.stdout)
            if resolved is None:
                return None
            resolved_paths.append(resolved)
        return resolved_paths[0], resolved_paths[1], resolved_paths[2]

    async def _read_head(
        self,
        root: Path,
    ) -> HeadIdentity | _HeadReadFailure:
        symbolic = await self._run_head_probe(
            root,
            ("symbolic-ref", "--quiet", "HEAD"),
        )
        if isinstance(symbolic, _HeadReadFailure):
            return symbolic
        if symbolic.timed_out or symbolic.termination_uncertain:
            return self._head_command_failure(
                symbolic,
                "Git symbolic HEAD read failed",
            )

        branch: str | None
        if symbolic.returncode == 0:
            branch = _single_git_value(symbolic.stdout)
            if branch is None or not branch.startswith("refs/"):
                return _HeadReadFailure(
                    "error",
                    "Git symbolic HEAD output is malformed",
                )
        elif (
            symbolic.returncode == 1
            and symbolic.stdout == b""
            and symbolic.stderr == b""
        ):
            branch = None
        else:
            return self._head_command_failure(
                symbolic,
                "Git symbolic HEAD read failed",
            )

        revision = await self._run_head_probe(
            root,
            ("rev-parse", "--verify", "--quiet", "HEAD^{commit}"),
        )
        if isinstance(revision, _HeadReadFailure):
            return revision
        if revision.timed_out or revision.termination_uncertain:
            return self._head_command_failure(
                revision,
                "Git HEAD commit read failed",
            )
        if revision.returncode == 0:
            object_id = _ascii_object_id(revision.stdout)
            if object_id is None:
                return _HeadReadFailure(
                    "error",
                    "Git HEAD commit output is malformed",
                )
            if branch is None:
                return HeadIdentity.detached(object_id)
            return HeadIdentity.attached(branch, object_id)

        if branch is None:
            return self._head_command_failure(
                revision,
                "Detached Git HEAD does not resolve to a commit",
            )
        if not (
            revision.returncode == 1
            and revision.stdout == b""
            and revision.stderr == b""
        ):
            return self._head_command_failure(
                revision,
                "Git HEAD commit read failed",
            )

        reference = await self._run_head_probe(
            root,
            ("show-ref", "--exists", branch),
        )
        if isinstance(reference, _HeadReadFailure):
            return reference
        if reference.timed_out or reference.termination_uncertain:
            return self._head_command_failure(
                reference,
                "Git HEAD reference lookup failed",
            )
        if (
            reference.returncode == 2
            and reference.stdout == b""
            and reference.stderr == b""
        ):
            return HeadIdentity.unborn(branch)
        if (
            reference.returncode == 0
            and reference.stdout == b""
            and reference.stderr == b""
        ):
            return _HeadReadFailure(
                "error",
                "Git HEAD reference exists but does not resolve to a commit",
            )
        if reference.returncode != 129:
            return self._head_command_failure(
                reference,
                "Git HEAD reference lookup failed",
            )

        fallback = await self._run_head_probe(
            root,
            ("show-ref", "--verify", "--quiet", branch),
        )
        if isinstance(fallback, _HeadReadFailure):
            return fallback
        if fallback.timed_out or fallback.termination_uncertain:
            return self._head_command_failure(
                fallback,
                "Git HEAD reference lookup failed",
            )
        if (
            fallback.returncode == 1
            and fallback.stdout == b""
            and fallback.stderr == b""
        ):
            return HeadIdentity.unborn(branch)
        if (
            fallback.returncode == 0
            and fallback.stdout == b""
            and fallback.stderr == b""
        ):
            return _HeadReadFailure(
                "error",
                "Git HEAD reference exists but does not resolve to a commit",
            )
        return self._head_command_failure(
            fallback,
            "Git HEAD reference lookup failed",
        )

    async def _run_head_probe(
        self,
        root: Path,
        arguments: Sequence[str],
    ) -> GitCommandResult | _HeadReadFailure:
        assert self._git_executable is not None
        try:
            return await self._runner.run(
                (self._git_executable, *arguments),
                cwd=str(root),
                environment=build_git_environment(
                    self._environment,
                    stable_locale=True,
                ),
                timeout=self._discovery_timeout,
            )
        except OSError as error:
            diagnostic = sanitize_git_stderr(
                str(error).encode("utf-8", "replace")
            )
            message = "Git HEAD process could not start"
            if diagnostic:
                message = f"{message}: {diagnostic}"
            return _HeadReadFailure("unavailable", message)

    @staticmethod
    def _head_command_failure(
        result: GitCommandResult,
        fallback: str,
    ) -> _HeadReadFailure:
        kind: HeadReadFailureKind = (
            "unavailable"
            if result.timed_out or result.termination_uncertain
            else "error"
        )
        return _HeadReadFailure(
            kind,
            _command_failure_message(result, fallback),
        )

    def _repository_identity_matches(
        self,
        binding: SessionBinding,
        repository: RepositoryIdentity,
    ) -> bool:
        root = self._safe_root(binding)
        if root is None:
            return False
        expected = (
            (
                repository.worktree_root,
                repository.worktree_identity,
            ),
            (repository.git_dir, repository.git_dir_identity),
            (
                repository.git_common_dir,
                repository.git_common_dir_identity,
            ),
        )
        resolved: list[Path] = []
        for path_text, identity in expected:
            path = Path(path_text)
            try:
                canonical = path.resolve(strict=True)
            except (OSError, RuntimeError):
                return False
            if (
                canonical != path
                or not canonical.is_dir()
                or _filesystem_identity(canonical) != identity
            ):
                return False
            resolved.append(canonical)
        worktree_root = resolved[0]
        try:
            root.relative_to(worktree_root)
        except ValueError:
            return False
        return True


def _parse_candidate_paths(payload: bytes) -> tuple[bytes, ...] | None:
    """Parse one bounded NUL path batch without decoding repository paths."""
    if (
        not payload
        or not payload.endswith(b"\0")
        or len(payload) > _LOCAL_PUSH_STDOUT_LIMIT_BYTES
    ):
        return None
    paths = tuple(payload[:-1].split(b"\0"))
    if len(paths) > 100_000 or len(paths) != len(set(paths)):
        return None
    for path in paths:
        components = path.split(b"/")
        if (
            not path
            or path.startswith(b"/")
            or any(
                component in {b"", b".", b"..", b".git"}
                for component in components
            )
        ):
            return None
    return paths


def _candidate_attribute_fingerprint(
    paths: tuple[bytes, ...],
    payload: bytes,
) -> str | None:
    """Prove one batched exact-tree filter result and reject LFS/custom filters."""
    if not payload.endswith(b"\0"):
        return None
    fields = payload[:-1].split(b"\0")
    if len(fields) != len(paths) * 3:
        return None
    digest = hashlib.sha256()
    digest.update(b"file-notes-candidate-attributes-v1\0")
    for index, expected_path in enumerate(paths):
        path, attribute, value = fields[index * 3 : index * 3 + 3]
        if (
            path != expected_path
            or attribute != b"filter"
            or value not in {b"unspecified", b"unset"}
        ):
            return None
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(b"\0")
        digest.update(value)
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_push_config_facts(
    payload: bytes,
    repository: RepositoryIdentity,
    environment: Mapping[str, str],
) -> tuple[_GitConfigFact, ...] | None:
    """Parse relevant scoped config and fingerprint each canonical source."""
    if not payload or not payload.endswith(b"\0"):
        return None
    fields = payload[:-1].split(b"\0")
    if len(fields) % 3:
        return None
    facts: list[_GitConfigFact] = []
    for offset in range(0, len(fields), 3):
        scope_bytes, origin_bytes, item = fields[offset : offset + 3]
        if not origin_bytes.startswith(b"file:") or b"\n" not in item:
            return None
        key_bytes, value_bytes = item.split(b"\n", 1)
        try:
            scope = scope_bytes.decode("ascii")
            key = key_bytes.decode("utf-8")
            value = value_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
        source = _config_source_identity(
            origin_bytes[5:],
            repository.worktree_root,
        )
        if source is None:
            return None
        origin_identity, canonical_origin = source
        if scope == "unknown":
            scope = _classify_unknown_config_scope(
                canonical_origin,
                repository,
                environment,
            )
            if scope is None:
                return None
        try:
            facts.append(
                _GitConfigFact(
                    scope,
                    origin_identity,
                    key,
                    value,
                )
            )
        except PushContractError:
            return None
    return tuple(facts)


def _config_source_identity(
    raw_path: bytes,
    worktree_root: str,
) -> tuple[str, Path] | None:
    """Hash canonical config source identity and change metadata, never values."""
    if not raw_path or b"\0" in raw_path:
        return None
    path = Path(os.fsdecode(raw_path))
    if not path.is_absolute():
        path = Path(worktree_root) / path
    try:
        canonical = path.resolve(strict=True)
        metadata = canonical.stat(follow_symlinks=False)
    except (OSError, RuntimeError):
        return None
    if canonical != path or not stat.S_ISREG(metadata.st_mode):
        return None
    digest = hashlib.sha256()
    digest.update(b"file-notes-config-source-v1\0")
    for value in (
        os.fsencode(str(canonical)),
        str(metadata.st_dev).encode("ascii"),
        str(metadata.st_ino).encode("ascii"),
        str(metadata.st_mode).encode("ascii"),
        str(metadata.st_uid).encode("ascii"),
        str(metadata.st_gid).encode("ascii"),
        str(metadata.st_size).encode("ascii"),
        str(metadata.st_mtime_ns).encode("ascii"),
        str(metadata.st_ctime_ns).encode("ascii"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest(), canonical


def _classify_unknown_config_scope(
    canonical_origin: Path,
    repository: RepositoryIdentity,
    environment: Mapping[str, str],
) -> Literal["system", "global", "local", "worktree"] | None:
    """Classify old-Git unknown scope only for exact trusted repo config."""
    candidates = (
        ("local", Path(repository.git_common_dir) / "config"),
        ("local", Path(repository.git_dir) / "config"),
        ("worktree", Path(repository.git_dir) / "config.worktree"),
    )
    for scope, candidate in candidates:
        try:
            canonical_candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if (
            canonical_candidate == candidate
            and canonical_origin == canonical_candidate
        ):
            return scope
    for candidate in _known_global_config_paths(environment):
        try:
            canonical_candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if (
            canonical_candidate == candidate
            and canonical_origin == canonical_candidate
        ):
            return "global"
    if (
        canonical_origin in _known_system_config_paths()
        and _is_root_protected_system_config(canonical_origin)
    ):
        return "system"
    return None


def _known_global_config_paths(
    environment: Mapping[str, str],
) -> tuple[Path, ...]:
    """Return only global config paths selected by the sanitized environment."""
    candidates: list[Path] = []
    home_text = environment.get("HOME")
    if home_text:
        home = Path(home_text)
        candidates.append(home / ".gitconfig")
    config_home_text = environment.get("XDG_CONFIG_HOME")
    if config_home_text:
        candidates.append(Path(config_home_text) / "git" / "config")
    elif home_text:
        candidates.append(Path(home_text) / ".config" / "git" / "config")
    return tuple(candidates)


def _known_system_config_paths() -> frozenset[Path]:
    """Return exact conventional system Git config paths accepted fail-closed."""
    return frozenset(
        {
            Path("/etc/gitconfig"),
            Path("/usr/local/etc/gitconfig"),
            Path("/opt/homebrew/etc/gitconfig"),
            Path("/usr/share/git-core/gitconfig"),
            Path("/usr/lib/git-core/gitconfig"),
            Path(
                "/Library/Developer/CommandLineTools/usr/share/"
                "git-core/gitconfig"
            ),
            Path(
                "/Applications/Xcode.app/Contents/Developer/usr/share/"
                "git-core/gitconfig"
            ),
        }
    )


def _is_root_protected_system_config(path: Path) -> bool:
    """Prove a known POSIX system config and every parent are root-protected."""
    if os.name != "posix":
        return False
    try:
        for component in (path, *path.parents):
            metadata = component.stat(follow_symlinks=False)
            if metadata.st_uid != 0 or metadata.st_mode & (
                stat.S_IWGRP | stat.S_IWOTH
            ):
                return False
    except OSError:
        return False
    return True


def build_file_notes_session_owner() -> FileNotesSessionOwner:
    """Build one process owner with exactly one attached Git service."""
    owner = FileNotesSessionOwner()
    owner.attach_git_service(FileNotesGitService(owner))
    return owner


def _blocked_commit_review(message: str) -> CommitReviewResult:
    """Build one path-free blocked review settlement."""
    return CommitReviewResult("blocked", message=message)


def _uncertain_commit_outcome() -> CommitOutcome:
    """Return the exact bounded uncertainty copy."""
    return CommitOutcome("uncertain", _UNCERTAIN_COMMIT_MESSAGE)


def _iter_safe_private_directory_parents(
    repository: RepositoryIdentity,
) -> Generator[tuple[Path, Path, int], None, None]:
    """Yield canonical safe parents in preferred fallback order."""
    worktree = Path(repository.worktree_root).resolve(strict=True)
    repository_device = worktree.stat().st_dev
    checked_parents: set[Path] = set()
    for candidate in (Path(tempfile.gettempdir()), worktree.parent):
        try:
            parent = candidate.resolve(strict=True)
            if parent in checked_parents:
                continue
            checked_parents.add(parent)
            if not _hooks_parent_is_safe(parent, repository_device):
                continue
        except (OSError, RuntimeError):
            continue
        yield worktree, parent, repository_device


def _create_private_push_proof_directory(
    repository: RepositoryIdentity,
) -> _PrivatePushProofDirectory:
    """Create one identity-pinned proof root under an already-safe parent."""
    if not _private_push_proof_apis_available():
        raise OSError("Private push proof safety requires POSIX descriptor APIs")
    for worktree, parent, repository_device in (
        _iter_safe_private_directory_parents(repository)
    ):
        proof: _PrivatePushProofDirectory | None = None
        root: Path | None = None
        try:
            parent_identity = _filesystem_identity(parent)
            root = Path(
                tempfile.mkdtemp(
                    prefix=".chatbook-push-proof-",
                    dir=str(parent),
                )
            )
            if (
                root.parent != parent
                or root.is_relative_to(worktree)
            ):
                raise OSError("Private proof root location is unsafe")
            root.chmod(0o700)
            proof = _PrivatePushProofDirectory(
                root=root,
                parent_identity=parent_identity,
                repository_device=repository_device,
            )
            if proof.validate():
                return proof
        except (OSError, RuntimeError):
            pass
        if proof is not None:
            proof.cleanup()
        elif root is not None:
            _remove_private_hooks_directory(root)
    raise OSError("Unable to create a private push proof directory")


def _private_push_proof_apis_available() -> bool:
    """Return whether descriptor-relative proof safety can run fail-closed."""
    required_attributes = (
        "O_CLOEXEC",
        "O_NONBLOCK",
        "O_NOFOLLOW",
        "fchmod",
        "pread",
    )
    return (
        _private_hooks_posix_ownership_apis_available()
        and all(hasattr(os, attribute) for attribute in required_attributes)
    )


def _identity_from_stat_result(
    metadata: os.stat_result,
) -> FileSystemIdentity:
    """Project one stat result to the repository identity primitive."""
    return FileSystemIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _bounded_file_descriptor_digest(
    file_descriptor: int,
    expected_size: int,
) -> bytes | None:
    """Hash exactly one bounded pinned file without changing its offset."""
    if (
        expected_size < 0
        or expected_size > _LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES
    ):
        return None
    digest = hashlib.sha256()
    offset = 0
    try:
        while offset < expected_size:
            chunk = os.pread(
                file_descriptor,
                min(_GIT_STREAM_CHUNK_BYTES, expected_size - offset),
                offset,
            )
            if not chunk:
                return None
            digest.update(chunk)
            offset += len(chunk)
        if os.pread(file_descriptor, 1, expected_size):
            return None
    except OSError:
        return None
    return digest.digest()


def _create_private_hooks_directory(
    repository: RepositoryIdentity,
    pending_cleanup: set[Path],
) -> Path:
    """Create and verify one empty owner-only hooks directory outside the repo."""
    if not _private_hooks_posix_ownership_apis_available():
        raise OSError("Private hooks safety requires POSIX ownership APIs")
    for worktree, parent, repository_device in (
        _iter_safe_private_directory_parents(repository)
    ):
        directory: Path | None = None
        try:
            directory = Path(
                tempfile.mkdtemp(
                    prefix=".chatbook-hooks-",
                    dir=str(parent),
                )
            )
            pending_cleanup.add(directory)
            directory.chmod(0o700)
            metadata = directory.stat()
            verified = (
                directory.parent == parent
                and not directory.is_relative_to(worktree)
                and directory.is_dir()
                and not directory.is_symlink()
                and metadata.st_dev == repository_device
                and metadata.st_uid == os.geteuid()
                and stat.S_IMODE(metadata.st_mode) == 0o700
                and not any(directory.iterdir())
                and _hooks_parent_is_safe(parent, repository_device)
            )
            if verified:
                return directory
        except (OSError, RuntimeError):
            pass
        if (
            directory is not None
            and _remove_private_hooks_directory(directory)
        ):
            pending_cleanup.discard(directory)
    raise OSError("Unable to create a private hooks directory")


def _private_hooks_posix_ownership_apis_available() -> bool:
    """Return whether private-hooks ownership checks can run safely."""
    return os.name == "posix" and all(
        hasattr(os, attribute)
        for attribute in ("geteuid", "getegid", "getgroups")
    )


def _hooks_parent_is_safe(
    parent: Path,
    repository_device: int,
) -> bool:
    """Validate one canonical hooks parent against cross-principal substitution."""
    if not _private_hooks_posix_ownership_apis_available():
        return False
    try:
        if parent != parent.resolve(strict=True):
            return False
        chain = tuple(reversed((parent, *parent.parents)))
        metadata = tuple(component.lstat() for component in chain)
    except (OSError, RuntimeError):
        return False
    effective_uid = os.geteuid()
    trusted_uids = {0, effective_uid}
    if any(
        not stat.S_ISDIR(item.st_mode) or item.st_uid not in trusted_uids
        for item in metadata
    ):
        return False
    for index, item in enumerate(metadata):
        mode = stat.S_IMODE(item.st_mode)
        shared_writable = bool(mode & (stat.S_IWGRP | stat.S_IWOTH))
        if shared_writable and not mode & stat.S_ISVTX:
            return False
        if index + 1 >= len(metadata) or not shared_writable:
            continue
        child = metadata[index + 1]
        if child.st_uid not in {effective_uid, item.st_uid}:
            return False
    immediate = metadata[-1]
    return (
        immediate.st_dev == repository_device
        and _directory_is_writable_by_current_process(immediate)
    )


def _directory_is_writable_by_current_process(metadata: os.stat_result) -> bool:
    """Project Unix directory write/search permission for the effective IDs."""
    if not _private_hooks_posix_ownership_apis_available():
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    effective_uid = os.geteuid()
    if effective_uid == 0:
        return True
    if effective_uid == metadata.st_uid:
        required = stat.S_IWUSR | stat.S_IXUSR
    else:
        try:
            effective_groups = {os.getegid(), *os.getgroups()}
        except OSError:
            effective_groups = {os.getegid()}
        if metadata.st_gid in effective_groups:
            required = stat.S_IWGRP | stat.S_IXGRP
        else:
            required = stat.S_IWOTH | stat.S_IXOTH
    return (mode & required) == required


def _remove_private_hooks_directory(directory: Path) -> bool:
    """Remove one verified-empty directory without recursive deletion."""
    try:
        if directory.is_dir() and not directory.is_symlink():
            directory.rmdir()
            return True
    except OSError:
        return False
    return False


def _path_present(path: Path) -> bool:
    """Treat every filesystem entry, including a symlink, as present."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _commit_local_config_is_supported(payload: bytes) -> bool:
    """Reject sparse/partial/promisor config without retaining its values."""
    if not payload:
        return True
    if not payload.endswith(b"\0"):
        return False
    for record in payload[:-1].split(b"\0"):
        try:
            raw_key, raw_value = record.split(b"\n", 1)
            key = raw_key.decode("ascii").lower()
        except (ValueError, UnicodeDecodeError):
            return False
        value = raw_value.strip().lower()
        if key == "extensions.partialclone":
            return False
        if (
            key in {"core.sparsecheckout", "index.sparse"}
            and value not in _DISABLED_GIT_BOOLEAN_VALUES
        ):
            return False
        if (
            key.startswith("remote.")
            and key.endswith(".promisor")
            and value not in _DISABLED_GIT_BOOLEAN_VALUES
        ):
            return False
    return True


def _commit_worktree_config_is_enabled(payload: bytes) -> bool:
    """Return the effective repository opt-in for worktree config."""
    if not payload:
        return False
    enabled = False
    for record in payload[:-1].split(b"\0"):
        raw_key, raw_value = record.split(b"\n", 1)
        if raw_key.decode("ascii").lower() == "extensions.worktreeconfig":
            enabled = (
                raw_value.strip().lower()
                not in _DISABLED_GIT_BOOLEAN_VALUES
            )
    return enabled


def _commit_index_semantics_are_supported(payload: bytes) -> bool:
    """Accept only ordinary cached records before the shared byte parser."""
    if not payload:
        return True
    if not payload.endswith(b"\0"):
        return False
    return all(
        len(record) >= 2 and record[:2] == b"H "
        for record in payload[:-1].split(b"\0")
    )


def _expected_owned_delta(
    ownership_entries: Mapping[
        int,
        tuple[Mapping[str, IndexEntry | None], Mapping[str, IndexEntry | None]],
    ],
) -> dict[bytes, tuple[str, str, str, str, str]]:
    """Build the exact raw no-renames delta authorized by ownership."""
    expected: dict[bytes, tuple[str, str, str, str, str]] = {}
    object_id_widths = {
        len(entry.object_id)
        for before, after in ownership_entries.values()
        for entry in (*before.values(), *after.values())
        if entry is not None
    }
    if len(object_id_widths) != 1 or not object_id_widths.issubset({40, 64}):
        return {}
    zero_oid = "0" * object_id_widths.pop()
    owned_paths: set[str] = set()
    for before, after in ownership_entries.values():
        group_paths = set(before).union(after)
        if owned_paths.intersection(group_paths):
            return {}
        owned_paths.update(group_paths)
        for path in group_paths:
            old_entry = before.get(path)
            new_entry = after.get(path)
            if old_entry == new_entry:
                continue
            if (
                (old_entry is not None and not _proof_index_entry_is_supported(
                    path,
                    old_entry,
                ))
                or (
                    new_entry is not None
                    and not _proof_index_entry_is_supported(path, new_entry)
                )
            ):
                return {}
            raw_path = os.fsencode(path)
            if raw_path in expected:
                return {}
            old_mode = "000000" if old_entry is None else old_entry.mode
            new_mode = "000000" if new_entry is None else new_entry.mode
            old_oid = zero_oid if old_entry is None else old_entry.object_id.lower()
            new_oid = zero_oid if new_entry is None else new_entry.object_id.lower()
            status = "A" if old_entry is None else "D" if new_entry is None else "M"
            expected[raw_path] = (
                old_mode,
                new_mode,
                old_oid,
                new_oid,
                status,
            )
    return expected


def _commit_review_change_type(
    ownership: StagingOwnership,
) -> CommitReviewChangeType | None:
    """Classify one included note from its exact proven Git delta."""
    expected = _expected_owned_delta(
        {
            0: (
                {
                    path: baseline.entry
                    for path, baseline in ownership.original_baselines.items()
                },
                ownership.post_stage_entries,
            )
        }
    )
    statuses = frozenset(item[-1] for item in expected.values())
    return {
        frozenset({"A"}): "New",
        frozenset({"M"}): "Modified",
        frozenset({"D"}): "Deleted",
        frozenset({"A", "D"}): "Moved",
    }.get(statuses)


def _proof_index_entry_is_supported(path: str, entry: IndexEntry) -> bool:
    return (
        entry.path == path
        and entry.stage == 0
        and not entry.semantic_flags
        and entry.mode != "160000"
        and len(entry.mode) == 6
        and all(character in "01234567" for character in entry.mode)
        and len(entry.object_id) in {40, 64}
        and all(
            character in "0123456789abcdefABCDEF"
            for character in entry.object_id
        )
        and any(character != "0" for character in entry.object_id)
    )


def _canonical_directory_from_git(payload: bytes) -> Path | None:
    value = _single_git_value(payload)
    if value is None or "\0" in value:
        return None
    try:
        path = Path(value).resolve(strict=True)
        path_stat = path.stat(follow_symlinks=False)
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISDIR(path_stat.st_mode):
        return None
    return path


def _single_git_value(payload: bytes) -> str | None:
    if not payload.endswith(b"\n") or b"\0" in payload:
        return None
    value = payload[:-1]
    if not value:
        return None
    return os.fsdecode(value)


def _ascii_object_id(payload: bytes) -> str | None:
    if not payload.endswith(b"\n"):
        return None
    object_id = payload[:-1]
    if len(object_id) not in {40, 64}:
        return None
    try:
        value = object_id.decode("ascii")
    except UnicodeDecodeError:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        return None
    return value.lower()


def _git_object_format_and_oid(
    payload: bytes,
) -> tuple[Literal["sha1", "sha256"], str] | None:
    """Parse exact storage-format and complete object-ID proof output."""
    parts = payload.split(b"\n")
    if len(parts) != 3 or parts[-1] != b"":
        return None
    object_format_bytes, object_id_bytes, _terminator = parts
    if object_format_bytes == b"sha1":
        object_format: Literal["sha1", "sha256"] = "sha1"
        expected_width = 40
    elif object_format_bytes == b"sha256":
        object_format = "sha256"
        expected_width = 64
    else:
        return None
    object_id = _ascii_object_id(object_id_bytes + b"\n")
    if object_id is None or len(object_id) != expected_width:
        return None
    return object_format, object_id


def _render_isolated_repository_config(
    object_format: Literal["sha1", "sha256"],
) -> bytes:
    """Render the minimum exact config for a format-matched private Git dir."""
    if object_format == "sha1":
        return b"[core]\n\trepositoryFormatVersion = 0\n"
    if object_format == "sha256":
        return (
            b"[core]\n"
            b"\trepositoryFormatVersion = 1\n"
            b"[extensions]\n"
            b"\tobjectFormat = sha256\n"
        )
    raise ValueError("unsupported object format")


def _filesystem_identity(path: Path) -> FileSystemIdentity:
    path_stat = path.stat(follow_symlinks=False)
    return FileSystemIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _map_session_endpoint(
    notes_root: Path,
    worktree_root: Path,
    relative_path: str,
) -> tuple[str | None, str | None]:
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\0" in relative_path
    ):
        return None, "Unsafe File Notes path"
    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return None, "Unsafe File Notes path"
    if ".git" in components:
        return None, "nested_repository"

    candidate = notes_root.joinpath(*components)
    for depth in range(1, len(components)):
        parent = notes_root.joinpath(*components[:depth])
        try:
            parent_stat = parent.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError:
            return None, "Unsafe File Notes path"
        if (
            stat.S_ISLNK(parent_stat.st_mode)
            or not stat.S_ISDIR(parent_stat.st_mode)
        ):
            return None, "Unsafe File Notes path"
        git_boundary = parent / ".git"
        try:
            git_boundary.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            return None, "Unsafe nested repository boundary"
        else:
            return None, "nested_repository"

    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(notes_root)
        candidate.relative_to(worktree_root)
    except (OSError, RuntimeError, ValueError):
        return None, "File Notes path leaves the trusted worktree"

    try:
        endpoint_stat = candidate.stat(follow_symlinks=False)
    except FileNotFoundError:
        endpoint_stat = None
    except OSError:
        return None, "Unable to inspect File Notes path"
    if endpoint_stat is not None:
        if stat.S_ISLNK(endpoint_stat.st_mode):
            return None, "Symlink endpoints are unsupported"
        if stat.S_ISDIR(endpoint_stat.st_mode):
            return None, "Directory endpoints are unsupported"
        if not stat.S_ISREG(endpoint_stat.st_mode):
            return None, "Non-regular endpoints are unsupported"

    return candidate.relative_to(worktree_root).as_posix(), None


def _command_succeeded(result: GitCommandResult) -> bool:
    return (
        result.returncode == 0
        and not result.timed_out
        and not result.termination_uncertain
        and not result.output_overflow
    )


def _stage_zero_index(
    entries: Sequence[IndexEntry],
) -> dict[str, IndexEntry]:
    """Return an exact path map only when each entry is unconflicted stage 0."""
    mapped: dict[str, IndexEntry] = {}
    for entry in entries:
        if entry.stage == 0:
            mapped[entry.path] = entry
    return mapped


def _command_failure_message(
    result: GitCommandResult,
    fallback: str,
) -> str:
    if result.termination_uncertain:
        return f"{fallback}: child termination is uncertain"
    if result.timed_out:
        return f"{fallback}: timed out"
    diagnostic = sanitize_git_stderr(result.stderr)
    if not diagnostic:
        return fallback
    return f"{fallback}: {diagnostic}"


def _index_mutation_failure_message(
    result: GitCommandResult,
    fallback: str,
    repository: RepositoryIdentity,
) -> str:
    """Report an existing Git index lock without deleting or retrying it."""
    if (
        result.returncode != 0
        and not result.timed_out
        and not result.termination_uncertain
    ):
        try:
            (Path(repository.git_dir) / "index.lock").lstat()
        except OSError:
            pass
        else:
            return "Git index busy; retry"
    return _command_failure_message(result, fallback)


class PorcelainV2ParseError(ValueError):
    """Raised when porcelain-v2 bytes are incomplete or malformed."""


class GitIndexParseError(ValueError):
    """Raised when a complete NUL-safe index read is ambiguous or malformed."""


class PorcelainPathOutsideSessionError(PorcelainV2ParseError):
    """Raised when Git reports a path outside the complete session whitelist."""


@dataclass(frozen=True, slots=True)
class PorcelainRecord:
    """One byte-safely decoded porcelain-v2 status record."""

    kind: PorcelainKind
    path: str | None
    index_status: str = "."
    worktree_status: str = "."
    submodule: str | None = None
    modes: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    original_path: str | None = None
    score: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "object_ids", tuple(self.object_ids))


def parse_index_entries_z(payload: bytes) -> tuple[IndexEntry, ...]:
    """Parse `ls-files --stage -v -z` without losing filename bytes."""
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise GitIndexParseError("Git index payload is not NUL terminated")
    entries: list[IndexEntry] = []
    identities: set[tuple[str, int]] = set()
    for raw_entry in payload[:-1].split(b"\0"):
        if not raw_entry:
            raise GitIndexParseError("Git index payload contains an empty entry")
        if len(raw_entry) < 3 or raw_entry[1:2] != b" ":
            raise GitIndexParseError("Git index entry lacks a semantic tag")
        raw_tag = raw_entry[:1]
        if not (
            raw_tag.isalpha()
            or raw_tag == b"?"
        ):
            raise GitIndexParseError("Git index semantic tag is unsupported")
        try:
            metadata, raw_path = raw_entry[2:].split(b"\t", 1)
        except ValueError as error:
            raise GitIndexParseError(
                "Git index entry lacks a path boundary"
            ) from error
        fields = metadata.split(b" ")
        if len(fields) != 3:
            raise GitIndexParseError("Git index entry metadata is malformed")
        raw_mode, raw_object_id, raw_stage = fields
        if (
            len(raw_mode) != 6
            or any(byte not in b"01234567" for byte in raw_mode)
        ):
            raise GitIndexParseError("Git index mode is malformed")
        if (
            len(raw_object_id) not in {40, 64}
            or any(
                byte not in b"0123456789abcdefABCDEF"
                for byte in raw_object_id
            )
        ):
            raise GitIndexParseError("Git index object ID is malformed")
        if raw_stage not in {b"0", b"1", b"2", b"3"}:
            raise GitIndexParseError("Git index stage is malformed")
        if not raw_path:
            raise GitIndexParseError("Git index path is empty")
        path = os.fsdecode(raw_path)
        if (
            path.startswith("/")
            or "\0" in path
            or any(component in {"", ".", ".."} for component in path.split("/"))
        ):
            raise GitIndexParseError("Git index path is unsafe")
        stage = int(raw_stage)
        identity = path, stage
        if identity in identities:
            raise GitIndexParseError(
                "Git index contains a duplicate path and stage"
            )
        identities.add(identity)
        tag = raw_tag.decode("ascii")
        flags: list[str] = []
        if tag.upper() == "S":
            flags.append("skip-worktree")
        if tag.islower():
            flags.append("assume-unchanged")
        entries.append(
            IndexEntry(
                path=path,
                mode=raw_mode.decode("ascii"),
                object_id=raw_object_id.decode("ascii").lower(),
                stage=stage,
                semantic_flags=tuple(flags),
            )
        )
    return tuple(entries)


def parse_porcelain_v2_z(
    payload: bytes,
    *,
    allowed_paths: frozenset[str],
) -> tuple[PorcelainRecord, ...]:
    """Parse NUL-delimited porcelain-v2 bytes and fail closed on every path."""
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise PorcelainV2ParseError("Porcelain-v2 payload is not NUL terminated")

    fields = payload[:-1].split(b"\0")
    records: list[PorcelainRecord] = []
    position = 0
    while position < len(fields):
        raw_record = fields[position]
        position += 1
        if not raw_record:
            raise PorcelainV2ParseError("Porcelain-v2 payload contains an empty record")
        if raw_record.startswith(b"# "):
            continue
        marker = raw_record[:1]
        if marker == b"1":
            records.append(_parse_ordinary(raw_record, allowed_paths))
            continue
        if marker == b"2":
            if position >= len(fields):
                raise PorcelainV2ParseError(
                    "Porcelain-v2 rename record lacks its original path"
                )
            original_path = _decode_allowed_path(
                fields[position],
                allowed_paths,
            )
            position += 1
            records.append(
                _parse_rename(
                    raw_record,
                    original_path,
                    allowed_paths,
                )
            )
            continue
        if marker == b"u":
            records.append(_parse_unmerged(raw_record, allowed_paths))
            continue
        if marker == b"?":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="untracked",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        if marker == b"!":
            records.append(
                _parse_simple_record(
                    raw_record,
                    kind="ignored",
                    allowed_paths=allowed_paths,
                )
            )
            continue
        raise PorcelainV2ParseError("Unsupported porcelain-v2 record type")
    return tuple(records)


def compute_stage_closure(
    endpoints: Collection[str],
    index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return literal endpoints plus tracked index ancestors/descendants."""
    closure = set(endpoints)
    for endpoint in endpoints:
        closure.update(
            path
            for path in index_entries
            if _paths_overlap(endpoint, path)
        )
    return frozenset(closure)


def compute_unstage_closure(
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]:
    """Return paths an exact baseline restoration may replace in the index."""
    closure = set(baselines)
    for path, baseline in baselines.items():
        if baseline.entry is None:
            continue
        closure.update(
            current_path
            for current_path in current_index_entries
            if _paths_overlap(path, current_path)
        )
    return frozenset(closure)


def stage_group_is_closed(
    group: SessionChangeGroup,
    index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Return whether the Stage closure remains inside one session lineage."""
    return compute_stage_closure(
        group.endpoints,
        index_entries,
    ).issubset(group.endpoints)


def unstage_group_is_closed(
    group: SessionChangeGroup,
    baselines: Mapping[str, IndexBaseline],
    current_index_entries: Mapping[str, IndexEntry],
    ownership: StagingOwnership,
) -> bool:
    """Return whether replacement conflicts are exact same-group ownership."""
    closure = compute_unstage_closure(
        baselines,
        current_index_entries,
    )
    if (
        not closure.issubset(group.endpoints)
        or ownership.topology_signature != group.topology_signature
    ):
        return False
    return all(
        ownership.post_stage_entries.get(path) == entry
        and path in ownership.post_stage_entries
        for path, entry in current_index_entries.items()
        if path in closure
    )


def stage_pathspecs(
    group: SessionChangeGroup,
    status_records: Sequence[PorcelainRecord],
    *,
    groups: Sequence[SessionChangeGroup],
) -> tuple[bytes, ...]:
    """Encode only effective endpoints, omitting absent transient lineage."""
    if group.group_id in _ambiguous_group_ids(groups, status_records):
        return ()
    changed_paths = _effective_mutation_paths(group, status_records)
    return tuple(
        os.fsencode(path)
        for path in group.endpoints
        if path in changed_paths
    )


def ownership_signature_matches(
    ownership: StagingOwnership,
    *,
    repository: RepositoryIdentity,
    head: HeadIdentity,
    topology: SessionChangeTopology,
    current_index_entries: Mapping[str, IndexEntry],
) -> bool:
    """Compare exact repository, HEAD, topology, entry, and flag evidence."""
    if (
        ownership.repository != repository
        or ownership.head != head
        or ownership.topology_signature != topology
    ):
        return False
    return all(
        current_index_entries.get(path) == expected
        for path, expected in ownership.post_stage_entries.items()
    )


def _map_ownership_topology(
    ownership: StagingOwnership,
    original_group: SessionChangeGroup,
    repository_group: SessionChangeGroup,
) -> StagingOwnership | None:
    """Project owner topology into repository coordinates for row policy."""
    mapping = dict(zip(original_group.endpoints, repository_group.endpoints))
    try:
        endpoints = tuple(
            mapping[path] for path in ownership.approved_endpoint_topology
        )
        move_edges = tuple(
            (mapping[source], mapping[destination])
            for source, destination in ownership.approved_move_edges
        )
        current_path = mapping[ownership.approved_current_path]
    except KeyError:
        return None
    return replace(
        ownership,
        approved_endpoint_topology=endpoints,
        approved_move_edges=move_edges,
        approved_current_path=current_path,
    )


def classify_session_rows(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
    index_entries: Mapping[str, IndexEntry] | Sequence[IndexEntry],
    ownership: Mapping[int, StagingOwnership],
) -> tuple[SessionGitRow, ...]:
    """Apply the frozen row/action policy to every coalesced session group."""
    flattened_entries = (
        tuple(index_entries.values())
        if isinstance(index_entries, Mapping)
        else tuple(index_entries)
    )
    entries_by_path: dict[str, IndexEntry] = {}
    for entry in flattened_entries:
        if entry.path not in entries_by_path or entry.stage == 0:
            entries_by_path[entry.path] = entry
    global_records = tuple(
        record for record in status_records if record.path is None
    )
    ambiguous_groups = _ambiguous_group_ids(groups, status_records)
    rows: list[SessionGitRow] = []
    for group in groups:
        records = global_records + tuple(
            record
            for record in status_records
            if _record_touches_group(record, group)
        )
        entries = tuple(
            entry
            for entry in flattened_entries
            if entry.path in group.endpoints
        )
        rows.append(
            _classify_group(
                group,
                records,
                entries,
                entries_by_path,
                ownership.get(group.group_id),
                group.group_id in ambiguous_groups,
            )
        )
    return tuple(rows)


def _parse_ordinary(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 8)
    if len(fields) != 9 or fields[0] != b"1":
        raise PorcelainV2ParseError("Malformed ordinary porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="ordinary",
        path=_decode_allowed_path(fields[8], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
    )


def _parse_rename(
    raw_record: bytes,
    original_path: str,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 9)
    if len(fields) != 10 or fields[0] != b"2":
        raise PorcelainV2ParseError("Malformed rename porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="rename",
        path=_decode_allowed_path(fields[9], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:6]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[6:8]
        ),
        original_path=original_path,
        score=_decode_ascii(fields[8], "rename score"),
    )


def _parse_unmerged(
    raw_record: bytes,
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    fields = raw_record.split(b" ", 10)
    if len(fields) != 11 or fields[0] != b"u":
        raise PorcelainV2ParseError("Malformed unmerged porcelain-v2 record")
    index_status, worktree_status = _decode_xy(fields[1])
    return PorcelainRecord(
        kind="unmerged",
        path=_decode_allowed_path(fields[10], allowed_paths),
        index_status=index_status,
        worktree_status=worktree_status,
        submodule=_decode_ascii(fields[2], "submodule state"),
        modes=tuple(
            _decode_ascii(field, "file mode")
            for field in fields[3:7]
        ),
        object_ids=tuple(
            _decode_ascii(field, "object ID")
            for field in fields[7:10]
        ),
    )


def _parse_simple_record(
    raw_record: bytes,
    *,
    kind: Literal["untracked", "ignored"],
    allowed_paths: frozenset[str],
) -> PorcelainRecord:
    if len(raw_record) < 3 or raw_record[1:2] != b" ":
        raise PorcelainV2ParseError(f"Malformed {kind} porcelain-v2 record")
    return PorcelainRecord(
        kind=kind,
        path=_decode_allowed_path(raw_record[2:], allowed_paths),
    )


def _decode_allowed_path(
    raw_path: bytes,
    allowed_paths: frozenset[str],
) -> str:
    if not raw_path:
        raise PorcelainV2ParseError("Porcelain-v2 record contains an empty path")
    path = os.fsdecode(raw_path)
    if path not in allowed_paths:
        raise PorcelainPathOutsideSessionError(
            f"Git reported a path outside the session whitelist: {path!r}"
        )
    return path


def _decode_xy(raw_xy: bytes) -> tuple[str, str]:
    if len(raw_xy) != 2:
        raise PorcelainV2ParseError("Porcelain-v2 XY status must contain two bytes")
    xy = _decode_ascii(raw_xy, "XY status")
    return xy[0], xy[1]


def _decode_ascii(value: bytes, label: str) -> str:
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise PorcelainV2ParseError(
            f"Porcelain-v2 {label} is not ASCII"
        ) from error


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _record_touches_group(
    record: PorcelainRecord,
    group: SessionChangeGroup,
) -> bool:
    return (
        record.path in group.endpoints
        or record.original_path in group.endpoints
    )


def _classify_group(
    group: SessionChangeGroup,
    records: Sequence[PorcelainRecord],
    entries: Sequence[IndexEntry],
    all_index_entries: Mapping[str, IndexEntry],
    owned: StagingOwnership | None,
    ambiguous_lineage: bool,
) -> SessionGitRow:
    if ambiguous_lineage:
        return SessionGitRow(
            group,
            "ambiguous_lineage",
            disabled_reason=(
                "Ambiguous session lineage: effective path belongs to multiple groups"
            ),
        )
    error = next((record for record in records if record.kind == "error"), None)
    if error is not None:
        return SessionGitRow(
            group,
            "error",
            disabled_reason=error.message or "Git status failed",
        )
    unavailable = next(
        (record for record in records if record.kind == "unavailable"),
        None,
    )
    if unavailable is not None:
        return SessionGitRow(
            group,
            "unavailable",
            disabled_reason=unavailable.message or "Git is unavailable",
        )
    if any(record.kind == "nested_repository" for record in records):
        return SessionGitRow(
            group,
            "nested_repository",
            disabled_reason="Nested repository unsupported",
        )
    if (
        any(record.kind == "unmerged" for record in records)
        or any(entry.stage != 0 for entry in entries)
    ):
        return SessionGitRow(
            group,
            "conflict",
            disabled_reason="Git conflict",
        )
    if any(record.kind == "ignored" for record in records):
        return SessionGitRow(
            group,
            "ignored",
            disabled_reason="Ignored by Git",
        )
    unsupported_reason = _unsupported_semantic_reason(entries)
    if unsupported_reason is not None:
        return SessionGitRow(
            group,
            "unsupported",
            disabled_reason=unsupported_reason,
        )
    if not stage_group_is_closed(group, all_index_entries):
        return SessionGitRow(
            group,
            "unsafe_closure",
            disabled_reason="Git mutation closure leaves this session lineage",
        )

    unstaged = any(
        record.kind == "untracked" or record.worktree_status != "."
        for record in records
    )
    staged = any(
        record.kind in {"ordinary", "rename"}
        and record.index_status != "."
        for record in records
    )

    if owned is not None:
        owned_entries_match = all(
            all_index_entries.get(path) == expected
            for path, expected in owned.post_stage_entries.items()
        )
        unowned_staged = any(
            path not in owned.post_stage_entries
            for record in records
            for path in _staged_record_paths(record)
        )
        if (
            owned_entries_match
            and not unowned_staged
            and owned.topology_signature != group.topology_signature
        ):
            return SessionGitRow(
                group,
                "owned_topology_changed",
                stage_action="stage_update",
                disabled_reason="Path lineage changed; Stage update required",
            )
        if (
            owned_entries_match
            and not unowned_staged
            and owned.topology_signature == group.topology_signature
        ):
            if unstaged:
                return SessionGitRow(
                    group,
                    "owned_newer_edits",
                    stage_action="stage_update",
                    unstage_eligible=True,
                )
            return SessionGitRow(
                group,
                "owned",
                unstage_eligible=True,
            )

    if staged and unstaged:
        return SessionGitRow(
            group,
            "external_partial",
            disabled_reason="External index state",
        )
    if staged:
        return SessionGitRow(
            group,
            "external_staged",
            disabled_reason="External index state",
        )
    if unstaged:
        return SessionGitRow(group, "unstaged", stage_action="stage")
    return SessionGitRow(group, "clean")


def _staged_record_paths(record: PorcelainRecord) -> tuple[str, ...]:
    if (
        record.kind not in {"ordinary", "rename"}
        or record.index_status == "."
    ):
        return ()
    return tuple(
        path
        for path in (record.path, record.original_path)
        if path is not None
    )


def _effective_mutation_paths(
    group: SessionChangeGroup,
    status_records: Sequence[PorcelainRecord],
) -> frozenset[str]:
    paths: set[str] = set()
    for record in status_records:
        if (
            not _record_touches_group(record, group)
            or (
                record.kind != "untracked"
                and record.worktree_status == "."
            )
        ):
            continue
        for path in (record.path, record.original_path):
            if path in group.endpoints:
                paths.add(path)
    return frozenset(paths)


def _ambiguous_group_ids(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
) -> frozenset[int]:
    path_groups: dict[str, set[int]] = {}
    for group in groups:
        for path in _effective_mutation_paths(group, status_records):
            path_groups.setdefault(path, set()).add(group.group_id)
    return frozenset(
        group_id
        for group_ids in path_groups.values()
        if len(group_ids) > 1
        for group_id in group_ids
    )


def _unsupported_semantic_reason(
    entries: Sequence[IndexEntry],
) -> str | None:
    reasons = {
        flag
        for entry in entries
        for flag in entry.semantic_flags
    }
    if any(
        entry.object_id and set(entry.object_id) == {"0"}
        for entry in entries
    ):
        reasons.add("intent-to-add")
    if not reasons:
        return None
    return f"Unsupported Git index state: {', '.join(sorted(reasons))}"
