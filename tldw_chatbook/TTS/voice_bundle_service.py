"""Application-owned lifecycle for explicit clone-voice bundle portability."""

from __future__ import annotations

import asyncio
import errno
import os
import re
import threading
import stat
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from threading import Event
from time import monotonic
from typing import Any, Final, Literal, NoReturn, Protocol, TypeVar, cast
from uuid import UUID, uuid4

try:  # pragma: no cover - exercised through the platform support seam.
    import fcntl
except ImportError:  # pragma: no cover - Windows is intentionally unsupported.
    fcntl = None  # type: ignore[assignment]

from tldw_chatbook.TTS.TTS_Generation import AudioCppGuidedDependencySnapshot
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactConsumerRequirement,
)
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_migration_namespace import rename_noreplace_at
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReference,
)
from tldw_chatbook.TTS.profile_repository import (
    TTSBundleImportCommand,
    TTSBundleImportResult,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
)
from tldw_chatbook.TTS.voice_bundle_codec import (
    MAX_BUNDLE_ARCHIVE_BYTES,
    TTSCloneVoiceBundle,
    TTSVoiceBundleError,
    TTSVoiceBundleSinks,
    encode_clone_voice_bundle,
    inspect_clone_voice_bundle,
)
from tldw_chatbook.Utils.private_paths import secure_private_directory

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | _DIRECTORY | _CLOEXEC | _NOFOLLOW
_SOURCE_FLAGS = os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _NONBLOCK
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_SESSION_LIMIT = 4
_SESSION_TTL_SECONDS = 600.0
_COPY_CHUNK_BYTES = 128 * 1024
_MEMBER_LEAVES: Final[tuple[str, ...]] = (
    "manifest.json",
    "profile.json",
    "reference.wav",
    "reference.txt",
)
_HANDLE_KEY = object()
_T = TypeVar("_T")

TTSVoiceBundleDependencyState = Literal["exact", "missing", "mismatch", "pending"]
TTSVoiceBundleImportResultStatus = Literal["created", "reused", "stale_inspection"]


class _Repository(Protocol):
    @property
    def generation(self) -> int: ...

    async def get_profile_collisions(
        self, profile_id: UUID, draft: TTSProfileDraft
    ) -> ProfileStoreResult[TTSProfileCollisionSnapshot]: ...

    async def get_profile(
        self, profile_id: UUID
    ) -> ProfileStoreResult[TTSGenerationProfile]: ...

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSCloneReference]: ...

    async def commit_bundle_import(
        self, command: TTSBundleImportCommand
    ) -> ProfileStoreResult[TTSBundleImportResult]: ...


class _DependencyService(Protocol):
    async def audio_cpp_guided_dependency_snapshot(
        self, requirement: TTSCloneRecipeRequirement
    ) -> AudioCppGuidedDependencySnapshot: ...


class _ArtifactLeaseCoordinator(Protocol):
    def lease_consumers(
        self,
        consumers: tuple[AudioCppArtifactConsumerRequirement, ...],
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class TTSVoiceBundleImportChoice:
    """One exact reviewed import action and orthogonal inactive consent."""

    choice: Literal["create", "reuse", "copy"]
    inactive_consent: bool

    def __post_init__(self) -> None:
        if self.choice not in {"create", "reuse", "copy"}:
            raise TTSVoiceBundleError("operation_failed")
        if type(self.inactive_consent) is not bool:
            raise TTSVoiceBundleError("operation_failed")


class TTSVoiceBundleHandle:
    """Unforgeable, redacted, single-service authority for one review."""

    __slots__ = ("__service_identity", "__token")

    def __init__(self, key: object, service_identity: object) -> None:
        if key is not _HANDLE_KEY:
            raise TypeError("Voice-bundle handles cannot be constructed directly")
        self.__service_identity = service_identity
        self.__token = object()

    @property
    def is_redacted(self) -> bool:
        return True

    def _belongs_to(self, service_identity: object) -> bool:
        return self.__service_identity is service_identity

    def __copy__(self) -> TTSVoiceBundleHandle:
        raise TypeError("Voice-bundle handles cannot be copied")

    def __deepcopy__(self, memo: object) -> TTSVoiceBundleHandle:
        del memo
        raise TypeError("Voice-bundle handles cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("Voice-bundle handles cannot be serialized")

    def __repr__(self) -> str:
        return "TTSVoiceBundleHandle(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class TTSVoiceBundleReview:
    """Safe canonical facts presented for one explicit import decision."""

    handle: TTSVoiceBundleHandle
    profile_id: UUID
    profile_name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    recipe_id: str
    recipe_revision: int
    dependency_state: TTSVoiceBundleDependencyState
    allowed_choices: tuple[Literal["create", "reuse", "copy"], ...]
    copy_profile_id: UUID | None
    copy_profile_name: str | None
    exact_private_duplicate: bool
    uuid_conflict: bool = False
    name_conflict: bool = False

    def __post_init__(self) -> None:
        if type(self.uuid_conflict) is not bool or type(self.name_conflict) is not bool:
            raise TTSVoiceBundleError("operation_failed")

    def __repr__(self) -> str:
        return "TTSVoiceBundleReview(<safe facts>)"


@dataclass(frozen=True, slots=True)
class TTSVoiceBundleImportResult:
    """One terminal import result or a replacement review after staleness."""

    status: TTSVoiceBundleImportResultStatus
    profile: TTSGenerationProfile | None = None
    review: TTSVoiceBundleReview | None = None

    def __post_init__(self) -> None:
        stale = self.status == "stale_inspection"
        if stale != (self.review is not None) or stale == (self.profile is not None):
            raise TTSVoiceBundleError("operation_failed")


@dataclass(frozen=True, slots=True, repr=False)
class _SourceFingerprint:
    parent_identity: tuple[int, int, int, int]
    identity: tuple[int, int, int, int, int, int, int, int, int]
    digest: str


@dataclass(frozen=True, slots=True, repr=False)
class _CopyResult:
    source_fingerprint: _SourceFingerprint
    bundle: TTSCloneVoiceBundle


@dataclass(frozen=True, slots=True, repr=False)
class _WorkerOutcome:
    code: str | None
    result: object | None


@dataclass(frozen=True, slots=True, repr=False)
class _PublicOutcome:
    code: str | None = None
    control: Literal["cancelled", "keyboard_interrupt", "system_exit"] | None = None
    result: object | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _ReviewEvidence:
    review: TTSVoiceBundleReview
    source_collisions: TTSProfileCollisionSnapshot
    portable: PortableTTSProfile
    requirement: TTSCloneRecipeRequirement
    repository_generation: int
    dependency_revision: int

    def visible_key(self) -> tuple[object, ...]:
        review = self.review
        return (
            review.profile_id,
            review.profile_name,
            review.provider_id,
            review.model_id,
            review.voice_id,
            review.recipe_id,
            review.recipe_revision,
            review.dependency_state,
            review.allowed_choices,
            review.copy_profile_id,
            review.copy_profile_name,
            review.exact_private_duplicate,
            review.uuid_conflict,
            review.name_conflict,
            self.repository_generation,
            self.dependency_revision,
            self.source_collisions,
        )


@dataclass(slots=True, repr=False)
class _Session:
    handle: TTSVoiceBundleHandle
    expires_at: float
    source: Path
    source_fingerprint: _SourceFingerprint
    evidence: _ReviewEvidence


@dataclass(slots=True, repr=False)
class _Operation:
    root_fd: int
    operation_fd: int
    operation_leaf: str
    operation_identity: tuple[int, int]
    files: dict[str, tuple[int, int]]


def _test_boundary(boundary: str) -> None:
    """Deterministic test seam for source-authority boundary mutations."""

    del boundary


def _posix_supported() -> bool:
    return (
        os.name == "posix"
        and fcntl is not None
        and bool(_NOFOLLOW)
        and bool(_DIRECTORY)
        and hasattr(os, "geteuid")
        and hasattr(os, "fchmod")
        and hasattr(os, "fsync")
        and os.stat in os.supports_follow_symlinks
    )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _source_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _parent_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_uid,
    )


def _private_directory(info: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == _PRIVATE_DIRECTORY_MODE
    )


def _private_file(info: os.stat_result, *, links: tuple[int, ...] = (1,)) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == _PRIVATE_FILE_MODE
        and info.st_nlink in links
    )


def _owned_source(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_nlink == 1
        and 0 < info.st_size <= MAX_BUNDLE_ARCHIVE_BYTES
    )


def _source_path(value: Path | str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or "\x00" in raw:
        raise TTSVoiceBundleError("operation_failed")
    return Path(os.path.abspath(os.path.normpath(os.path.expanduser(raw))))


def _classify_and_sever_public_failure(error: BaseException) -> _PublicOutcome:
    """Reduce one private failure graph to a bounded, type-only outcome."""

    if type(error) is asyncio.CancelledError:
        outcome = _PublicOutcome(control="cancelled")
    elif type(error) is KeyboardInterrupt:
        outcome = _PublicOutcome(control="keyboard_interrupt")
    elif type(error) is SystemExit:
        outcome = _PublicOutcome(control="system_exit")
    elif type(error) is TTSVoiceBundleError:
        outcome = _PublicOutcome(code=cast(TTSVoiceBundleError, error).code)
    else:
        outcome = _PublicOutcome(code="operation_failed")
    BaseException.__setattr__(error, "__traceback__", None)
    BaseException.__setattr__(error, "__cause__", None)
    BaseException.__setattr__(error, "__context__", None)
    BaseException.__setattr__(error, "args", ())
    return outcome


def _reconstruct_public_failure(outcome: _PublicOutcome) -> BaseException | None:
    if outcome.control == "cancelled":
        return asyncio.CancelledError()
    if outcome.control == "keyboard_interrupt":
        return KeyboardInterrupt()
    if outcome.control == "system_exit":
        return SystemExit()
    if outcome.code is not None:
        return TTSVoiceBundleError(cast(Any, outcome.code))
    return None


_ORIGINAL_OPEN = os.open
_ORIGINAL_FDOPEN = os.fdopen


@dataclass(eq=False)
class _BundleOpenOutcome:
    descriptor: int | None = None
    rejected: bool = False
    identity: tuple[int, int] | None = None


def _open_bundle_descriptor(*args, _outcome, **kwargs):
    primitive = os.open
    try:
        fd = primitive(*args, **kwargs)
    except OSError:
        if primitive is _ORIGINAL_OPEN:
            _outcome.rejected = True
        raise
    _outcome.descriptor = fd
    _outcome.identity = _identity(os.fstat(fd))
    return fd


@dataclass(eq=False)
class _BundleStreamOutcome:
    stream: object | None = None
    rejected: bool = False


def _open_bundle_stream(fd, *, _outcome):
    primitive = os.fdopen
    try:
        stream = primitive(fd, "w+b", closefd=False)
    except OSError:
        if primitive is _ORIGINAL_FDOPEN:
            _outcome.rejected = True
        raise
    _outcome.stream = stream
    return stream


class _BundleNativeOperation:
    """Actual bundle worker outcomes, never descendant or capture authority."""

    def __init__(self, service, selected, *, publication=False):
        self.service = service
        self.publication = publication
        self.published = None
        self.root = service._root
        self.selected = tuple(selected)
        self.paths = {self.root, *selected, *(p.parent for p in selected)}
        self.parents = {}
        self.descriptors = {}
        self.identities = {}
        self.streams = {}
        self.pending_stream_fds = set()
        self.pending = set()
        self.failed_closes = set()
        self.leases = []
        self.errors = []
        self.uncertain = False
        self.residue = False
        self.operation = None
        self.pid = os.getpid()
        self.thread = None

    def check(self):
        if (
            self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or self not in self.service._native_operations
        ):
            raise TTSVoiceBundleError("operation_failed")

    def admit(self, selected=None):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.check()
        if self.uncertain or self.pending:
            raise TTSVoiceBundleError("cleanup_failed")
        self.service._check_source()
        if self.service._root != self.root:
            raise TTSVoiceBundleError("operation_failed")
        lease = storage.acquire_storage(self.root if selected is None else selected)
        lease.native_owner = self
        self.leases.append(lease)

    def select(self, parent_fd, leaf):
        self.check()
        if (
            sys._getframe(1).f_code is not _bundle_select.__code__
            or sys._getframe(2).f_code not in _BUNDLE_SELECT_CODES
        ):
            raise TTSVoiceBundleError("operation_failed")
        parent = self.parents.get(parent_fd)
        operation = self.operation
        valid = (
            parent == self.root
            and re.fullmatch(r"operation-[0-9a-f]{32}", leaf)
            or operation is not None
            and parent_fd == operation.operation_fd
            and leaf in ("source.bundle", *_MEMBER_LEAVES)
            or any(
                parent == p.parent
                and re.fullmatch(
                    re.escape("." + p.name + ".") + r"[0-9a-f]{32}\.tmp", leaf
                )
                for p in self.selected
            )
        )
        if not valid:
            raise TTSVoiceBundleError("operation_failed")
        self.paths.add(parent / leaf)

    def open(self, *args, **kwargs):
        self.check()
        if "dir_fd" in kwargs:
            parent = self.parents.get(kwargs["dir_fd"])
            if parent is None:
                raise TTSVoiceBundleError("operation_failed")
            selected = parent / args[0]
        else:
            selected = Path(args[0])
        if selected not in self.paths:
            raise TTSVoiceBundleError("operation_failed")
        self.admit(selected)
        outcome = _BundleOpenOutcome()
        self.pending.add(outcome)
        try:
            return _open_bundle_descriptor(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self.descriptors[outcome.descriptor] = None
                self.identities[outcome.descriptor] = outcome.identity
                if (
                    self.operation is not None
                    and kwargs.get("dir_fd") == self.operation.operation_fd
                    and selected.name in ("source.bundle", *_MEMBER_LEAVES)
                    and args[1] & os.O_CREAT
                    and outcome.identity is not None
                ):
                    self.operation.files[selected.name] = outcome.identity
                if args[1] & _DIRECTORY:
                    self.parents[outcome.descriptor] = selected
                self.pending.discard(outcome)
            elif outcome.rejected:
                self.pending.discard(outcome)

    def open_directory(self, *args, **kwargs):
        from tldw_chatbook.Utils import private_paths

        self.admit()
        outcome = private_paths._NativeOpenOutcome()
        attempt = object()
        self.pending.add(attempt)
        try:
            return private_paths._native_open(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self.descriptors[outcome.descriptor] = private_paths._native_close
                self.pending.discard(attempt)
            elif outcome.rejected:
                self.pending.discard(attempt)

    def close(self, fd):
        self.check()
        if fd in self.failed_closes:
            raise TTSVoiceBundleError("cleanup_failed")
        if fd not in self.descriptors:
            raise TTSVoiceBundleError("cleanup_failed")
        try:
            (self.descriptors[fd] or os.close)(fd)
        except BaseException as error:
            self.failed_closes.add(fd)
            self.errors.append(error)
            self.uncertain = True
            raise
        del self.descriptors[fd]
        self.parents.pop(fd, None)
        self.identities.pop(fd, None)

    def stream(self, fd):
        self.check()
        if fd not in self.descriptors or fd in self.failed_closes:
            raise TTSVoiceBundleError("cleanup_failed")
        outcome = _BundleStreamOutcome()
        self.pending.add(outcome)
        self.pending_stream_fds.add(fd)
        # Buffered objects never own the descriptor; there is one native closer.
        try:
            return _open_bundle_stream(fd, _outcome=outcome)
        finally:
            if outcome.stream is not None:
                self.streams[outcome.stream] = fd
                self.pending_stream_fds.remove(fd)
                self.pending.remove(outcome)
            elif outcome.rejected:
                self.pending_stream_fds.remove(fd)
                self.pending.remove(outcome)

    def close_stream(self, stream):
        self.check()
        if stream not in self.streams or stream in self.failed_closes:
            raise TTSVoiceBundleError("cleanup_failed")
        try:
            stream.close()
        except BaseException as error:
            self.failed_closes.add(stream)
            self.errors.append(error)
            self.uncertain = True
            raise
        del self.streams[stream]

    def finish(self):
        self.check()
        for stream in tuple(self.streams):
            if stream not in self.failed_closes:
                try:
                    self.close_stream(stream)
                except BaseException:
                    pass  # Exact error and native outcome remain on this operation.
        for fd in tuple(self.descriptors):
            if (
                fd not in self.failed_closes
                and fd not in self.streams.values()
                and fd not in self.pending_stream_fds
            ):
                try:
                    self.close(fd)
                except BaseException:
                    pass  # Do not retry this descriptor after an uncertain close.
        if (
            self.pending
            or self.streams
            or self.descriptors
            or self.uncertain
            or self.residue
        ):
            return False
        while self.leases:
            try:
                self.leases[-1].close()
            except BaseException as error:
                self.errors.append(error)
                self.uncertain = True
                return False
            self.leases.pop()
        self.service._native_operations.discard(self)
        return True


def _bundle_native(native):
    if type(native) is not _BundleNativeOperation:
        raise TTSVoiceBundleError("operation_failed")
    native.check()
    return native


def _bundle_open(native, *args, **kwargs):
    return (
        os.open(*args, **kwargs)
        if native is None
        else _bundle_native(native).open(*args, **kwargs)
    )


def _bundle_close(native, fd):
    return os.close(fd) if native is None else _bundle_native(native).close(fd)


def _bundle_select(native, parent_fd, leaf):
    if native is not None:
        _bundle_native(native).select(parent_fd, leaf)


def _prepare_root_sync(root: Path, *, _native=None) -> tuple[Path, tuple[int, int]]:
    if _native is not None and (
        root != _bundle_native(_native).root or _native.selected
    ):
        raise TTSVoiceBundleError("operation_failed")
    if not _posix_supported():
        raise TTSVoiceBundleError("unsupported_platform")
    selected = secure_private_directory(
        root,
        create=True,
        application_owned=True,
        **(
            {
                "_open": _bundle_native(_native).open_directory,
                "_close": _bundle_native(_native).close,
            }
            if _native is not None
            else {}
        ),
    ).lexical_path
    descriptor = _bundle_open(_native, selected, _DIRECTORY_FLAGS)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(selected, follow_symlinks=False)
        if not _private_directory(opened) or _identity(opened) != _identity(named):
            raise OSError
        assert fcntl is not None
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.listdir(descriptor):
            raise TTSVoiceBundleError("cleanup_failed")
        return selected, _identity(opened)
    finally:
        _bundle_close(_native, descriptor)


def _create_operation(
    root: Path, root_identity: tuple[int, int], *, _native=None
) -> _Operation:
    if _native is not None and (
        root != _bundle_native(_native).root
        or _native.operation is not None
        or _native.publication
    ):
        raise TTSVoiceBundleError("operation_failed")
    root_fd = _bundle_open(_native, root, _DIRECTORY_FLAGS)
    operation_fd = -1
    operation_leaf = ""
    try:
        root_info = os.fstat(root_fd)
        named_root = os.stat(root, follow_symlinks=False)
        if (
            _identity(root_info) != root_identity
            or _identity(named_root) != root_identity
            or not _private_directory(root_info)
        ):
            raise OSError
        assert fcntl is not None
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        if os.listdir(root_fd):
            raise OSError
        for _ in range(16):
            operation_leaf = f"operation-{token_hex(16)}"
            try:
                if _native is not None:
                    _native.admit()
                    _native.residue = True
                os.mkdir(operation_leaf, _PRIVATE_DIRECTORY_MODE, dir_fd=root_fd)
                break
            except FileExistsError:
                continue
        else:
            raise OSError
        _bundle_select(_native, root_fd, operation_leaf)
        operation_fd = _bundle_open(
            _native, operation_leaf, _DIRECTORY_FLAGS, dir_fd=root_fd
        )
        os.fchmod(operation_fd, _PRIVATE_DIRECTORY_MODE)
        opened = os.fstat(operation_fd)
        named = os.stat(operation_leaf, dir_fd=root_fd, follow_symlinks=False)
        if not _private_directory(opened) or _identity(opened) != _identity(named):
            raise OSError
        operation = _Operation(
            root_fd=root_fd,
            operation_fd=operation_fd,
            operation_leaf=operation_leaf,
            operation_identity=_identity(opened),
            files={},
        )
        if _native is not None:
            _native.operation = operation
            _native.residue = True
        return operation
    except BaseException:
        # Only an observed original opened inode can authorize failed-create removal.
        known_fd = operation_fd
        expected = None
        if _native is not None:
            for fd, selected in _native.parents.items():
                if selected == root / operation_leaf:
                    known_fd = fd
                    expected = _native.identities.get(fd)
                    break
        elif known_fd >= 0:
            expected = _identity(os.fstat(known_fd))
        try:
            if operation_leaf and known_fd >= 0 and expected is not None:
                opened = os.fstat(known_fd)
                named = os.stat(operation_leaf, dir_fd=root_fd, follow_symlinks=False)
                if (
                    _identity(opened) == expected == _identity(named)
                    and _identity(os.fstat(root_fd)) == root_identity
                    and _identity(os.stat(root, follow_symlinks=False)) == root_identity
                    and _private_directory(opened)
                    and not os.listdir(known_fd)
                ):
                    os.rmdir(operation_leaf, dir_fd=root_fd)
                    os.fsync(root_fd)
                    if _native is not None:
                        _native.residue = False
        except BaseException as error:
            if _native is not None:
                _native.errors.append(error)
        if _native is None:
            for fd in (known_fd, root_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        # The original worker retires all independently known FDs, preserving body.
        raise


def _create_operation_file(operation: _Operation, leaf: str, *, _native=None) -> int:
    if _native is not None and _bundle_native(_native).operation is not operation:
        raise TTSVoiceBundleError("operation_failed")
    _bundle_select(_native, operation.operation_fd, leaf)
    descriptor = _bundle_open(
        _native,
        leaf,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW,
        _PRIVATE_FILE_MODE,
        dir_fd=operation.operation_fd,
    )
    os.fchmod(descriptor, _PRIVATE_FILE_MODE)
    info = os.fstat(descriptor)
    named = os.stat(leaf, dir_fd=operation.operation_fd, follow_symlinks=False)
    if not _private_file(info) or _identity(info) != _identity(named):
        _bundle_close(_native, descriptor)
        raise OSError
    operation.files[leaf] = _identity(info)
    return descriptor


def _cleanup_operation(operation: _Operation, *, _native=None) -> bool:
    if _native is not None:
        native = _bundle_native(_native)
        if (
            native.operation is not operation
            or operation.root_fd not in native.descriptors
            or operation.operation_fd not in native.descriptors
        ):
            raise TTSVoiceBundleError("cleanup_failed")
    failed = False
    for leaf, expected in tuple(operation.files.items()):
        try:
            named = os.stat(leaf, dir_fd=operation.operation_fd, follow_symlinks=False)
            if _identity(named) != expected or not _private_file(named):
                failed = True
                continue
            os.unlink(leaf, dir_fd=operation.operation_fd)
        except FileNotFoundError:
            continue
        except OSError:
            failed = True
    try:
        os.fsync(operation.operation_fd)
        opened = os.fstat(operation.operation_fd)
        named = os.stat(
            operation.operation_leaf,
            dir_fd=operation.root_fd,
            follow_symlinks=False,
        )
        if (
            _identity(opened) != operation.operation_identity
            or _identity(named) != operation.operation_identity
            or not _private_directory(opened)
            or os.listdir(operation.operation_fd)
        ):
            failed = True
        else:
            os.rmdir(operation.operation_leaf, dir_fd=operation.root_fd)
            os.fsync(operation.root_fd)
    except OSError:
        failed = True
    _bundle_close(_native, operation.operation_fd)
    try:
        assert fcntl is not None
        fcntl.flock(operation.root_fd, fcntl.LOCK_UN)
    finally:
        _bundle_close(_native, operation.root_fd)
    if _native is not None:
        _native.residue = failed
    return not failed


def _copy_and_inspect(
    root: Path,
    root_identity: tuple[int, int],
    source: Path,
    expected: _SourceFingerprint | None,
    *,
    _native=None,
) -> _CopyResult:
    operation = _create_operation(root, root_identity, _native=_native)
    source_fd = -1
    copy_fd = -1
    primary_code: str | None = None
    result: _CopyResult | None = None
    streams: list[Any] = []
    try:
        parent_before = os.lstat(source.parent)
        if not stat.S_ISDIR(parent_before.st_mode):
            raise TTSVoiceBundleError("source_changed")
        parent_identity = _parent_identity(parent_before)
        parent_fd = _bundle_open(_native, source.parent, _DIRECTORY_FLAGS)
        try:
            parent_opened = os.fstat(parent_fd)
            parent_named = os.lstat(source.parent)
            if (
                _parent_identity(parent_opened) != parent_identity
                or _parent_identity(parent_named) != parent_identity
            ):
                raise TTSVoiceBundleError("source_changed")
            initial = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _owned_source(initial):
                raise TTSVoiceBundleError("source_changed")
            source_fd = _bundle_open(
                _native, source.name, _SOURCE_FLAGS, dir_fd=parent_fd
            )
            opened = os.fstat(source_fd)
            if _identity(opened) != _identity(initial):
                raise TTSVoiceBundleError("source_changed")
            if stat.S_IMODE(opened.st_mode) != _PRIVATE_FILE_MODE:
                os.fchmod(source_fd, _PRIVATE_FILE_MODE)
            opened = os.fstat(source_fd)
            named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            identity = _source_identity(opened)
            if (
                _source_identity(named) != identity
                or not _private_file(opened)
                or (expected is not None and expected.identity != identity)
            ):
                raise TTSVoiceBundleError("source_changed")
            _test_boundary("source_initial_open")

            copy_fd = _create_operation_file(
                operation, "source.bundle", _native=_native
            )
            digest = sha256()
            copied = 0
            while True:
                chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_BUNDLE_ARCHIVE_BYTES:
                    raise TTSVoiceBundleError("bundle_limit_exceeded")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(copy_fd, view)
                    if written <= 0:
                        raise OSError
                    view = view[written:]
                _test_boundary("source_copy_progress")
            os.fsync(copy_fd)
            _test_boundary("source_copy_complete")
            final_opened = os.fstat(source_fd)
            final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            current = _SourceFingerprint(parent_identity, identity, digest.hexdigest())
            if (
                copied != identity[6]
                or _source_identity(final_opened) != identity
                or _source_identity(final_named) != identity
                or (expected is not None and expected != current)
            ):
                raise TTSVoiceBundleError("source_changed")
            _test_boundary("source_post_copy")
        finally:
            _bundle_close(_native, parent_fd)
        _bundle_close(_native, source_fd)
        source_fd = -1
        os.lseek(copy_fd, 0, os.SEEK_SET)
        parts: list[bytes] = []
        while True:
            chunk = os.read(copy_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            parts.append(chunk)
        payload = b"".join(parts)
        del parts

        sink_files = []
        for leaf in _MEMBER_LEAVES:
            descriptor = _create_operation_file(operation, leaf, _native=_native)
            stream = (
                os.fdopen(descriptor, "w+b")
                if _native is None
                else _native.stream(descriptor)
            )
            streams.append(stream)
            sink_files.append(stream)
        bundle = inspect_clone_voice_bundle(
            payload,
            sinks=TTSVoiceBundleSinks(*sink_files),
        )
        del payload
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
        _test_boundary("source_post_inspection")
        parent_fd = _bundle_open(_native, source.parent, _DIRECTORY_FLAGS)
        try:
            parent_opened = os.fstat(parent_fd)
            parent_named = os.lstat(source.parent)
            final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                _parent_identity(parent_opened) != parent_identity
                or _parent_identity(parent_named) != parent_identity
                or _source_identity(final_named) != identity
            ):
                raise TTSVoiceBundleError("source_changed")
        finally:
            _bundle_close(_native, parent_fd)
        _test_boundary("source_pre_fingerprint")
        parent_fd = _bundle_open(_native, source.parent, _DIRECTORY_FLAGS)
        try:
            parent_opened = os.fstat(parent_fd)
            parent_named = os.lstat(source.parent)
            final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                _parent_identity(parent_opened) != parent_identity
                or _parent_identity(parent_named) != parent_identity
                or _source_identity(final_named) != identity
            ):
                raise TTSVoiceBundleError("source_changed")
        finally:
            _bundle_close(_native, parent_fd)
        if expected is not None and current != expected:
            raise TTSVoiceBundleError("source_changed")
        result = _CopyResult(current, bundle)
    except TTSVoiceBundleError as error:
        primary_code = error.code
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        primary_code = "operation_failed"
    finally:
        if source_fd >= 0:
            _bundle_close(_native, source_fd)
        if copy_fd >= 0:
            _bundle_close(_native, copy_fd)
        for stream in streams:
            try:
                (stream.close() if _native is None else _native.close_stream(stream))
            except OSError:
                primary_code = primary_code or "cleanup_failed"
        if not _cleanup_operation(operation, _native=_native):
            primary_code = "cleanup_failed"
    if primary_code is not None or result is None:
        raise TTSVoiceBundleError(
            cast(Any, primary_code or "operation_failed")
        ) from None
    return result


def _copy_and_inspect_sync(
    root: Path,
    root_identity: tuple[int, int],
    source: Path,
    expected: _SourceFingerprint | None,
    *,
    _native=None,
) -> _WorkerOutcome:
    try:
        return _WorkerOutcome(
            None,
            _copy_and_inspect(root, root_identity, source, expected, _native=_native),
        )
    except TTSVoiceBundleError as error:
        return _WorkerOutcome(error.code, None)
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        return _WorkerOutcome("operation_failed", None)


def _fingerprint_source_sync(
    source: Path, expected: _SourceFingerprint, *, _native=None
) -> _WorkerOutcome:
    """Revalidate exact parent, inode metadata, and all source bytes."""

    parent_fd = -1
    source_fd = -1
    try:
        parent_named = os.lstat(source.parent)
        if _parent_identity(parent_named) != expected.parent_identity:
            raise TTSVoiceBundleError("source_changed")
        parent_fd = _bundle_open(_native, source.parent, _DIRECTORY_FLAGS)
        parent_opened = os.fstat(parent_fd)
        if _parent_identity(parent_opened) != expected.parent_identity:
            raise TTSVoiceBundleError("source_changed")
        named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if _source_identity(named) != expected.identity or not _private_file(named):
            raise TTSVoiceBundleError("source_changed")
        source_fd = _bundle_open(_native, source.name, _SOURCE_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(source_fd)
        if _source_identity(opened) != expected.identity:
            raise TTSVoiceBundleError("source_changed")
        digest = sha256()
        size = 0
        while True:
            chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BUNDLE_ARCHIVE_BYTES:
                raise TTSVoiceBundleError("source_changed")
            digest.update(chunk)
        final_opened = os.fstat(source_fd)
        final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        final_parent = os.fstat(parent_fd)
        lexical_parent = os.lstat(source.parent)
        if (
            size != expected.identity[6]
            or digest.hexdigest() != expected.digest
            or _source_identity(final_opened) != expected.identity
            or _source_identity(final_named) != expected.identity
            or _parent_identity(final_parent) != expected.parent_identity
            or _parent_identity(lexical_parent) != expected.parent_identity
        ):
            raise TTSVoiceBundleError("source_changed")
        return _WorkerOutcome(None, True)
    except TTSVoiceBundleError as error:
        return _WorkerOutcome(error.code, None)
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        return _WorkerOutcome("operation_failed", None)
    finally:
        if source_fd >= 0:
            _bundle_close(_native, source_fd)
        if parent_fd >= 0:
            _bundle_close(_native, parent_fd)


def _published_file_matches(
    parent_fd: int,
    destination: Path,
    parent_identity: tuple[int, int, int, int],
    published_identity: tuple[int, int],
    payload: bytes,
    *,
    _native=None,
) -> bool:
    os.fsync(parent_fd)
    if (
        _parent_identity(os.fstat(parent_fd)) != parent_identity
        or _parent_identity(os.lstat(destination.parent)) != parent_identity
    ):
        return False
    named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(named) != published_identity or not _private_file(named):
        return False
    descriptor = _bundle_open(
        _native, destination.name, _SOURCE_FLAGS, dir_fd=parent_fd
    )
    try:
        opened = os.fstat(descriptor)
        digest = sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        final_opened = os.fstat(descriptor)
    finally:
        _bundle_close(_native, descriptor)
    final_named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        _identity(opened) != published_identity
        or _identity(final_opened) != published_identity
        or _identity(final_named) != published_identity
        or _source_identity(opened) != _source_identity(final_opened)
        or _source_identity(final_opened) != _source_identity(final_named)
        or not _private_file(final_opened)
        or size != len(payload)
        or digest.digest() != sha256(payload).digest()
        or _parent_identity(os.fstat(parent_fd)) != parent_identity
        or _parent_identity(os.lstat(destination.parent)) != parent_identity
    ):
        return False
    return (
        _identity(os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False))
        == published_identity
        and _parent_identity(os.fstat(parent_fd)) == parent_identity
        and _parent_identity(os.lstat(destination.parent)) == parent_identity
    )


def _converge_published_file(
    parent_fd: int,
    destination: Path,
    parent_identity: tuple[int, int, int, int],
    published_identity: tuple[int, int],
    payload: bytes,
    *,
    _native=None,
) -> bool:
    for _ in range(3):
        try:
            return _published_file_matches(
                parent_fd,
                destination,
                parent_identity,
                published_identity,
                payload,
                _native=_native,
            )
        except OSError:
            continue
    return False


def _publish_sync(
    destination: Path,
    payload: bytes,
    cancellation: Event | None = None,
    *,
    _native=None,
) -> _WorkerOutcome:
    parent_fd = -1
    temporary_fd = -1
    temporary_leaf = f".{destination.name}.{token_hex(16)}.tmp"
    temporary_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    parent_identity: tuple[int, int, int, int] | None = None
    code: str | None = None
    try:
        if not _posix_supported():
            raise TTSVoiceBundleError("unsupported_platform")
        before = os.lstat(destination.parent)
        if not stat.S_ISDIR(before.st_mode):
            raise TTSVoiceBundleError("destination_changed")
        parent_fd = _bundle_open(_native, destination.parent, _DIRECTORY_FLAGS)
        opened_parent = os.fstat(parent_fd)
        parent_identity = _parent_identity(opened_parent)
        if (
            _parent_identity(before) != parent_identity
            or _parent_identity(os.lstat(destination.parent)) != parent_identity
        ):
            raise TTSVoiceBundleError("destination_changed")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise TTSVoiceBundleError("destination_changed")
        _bundle_select(_native, parent_fd, temporary_leaf)
        temporary_fd = _bundle_open(
            _native,
            temporary_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW,
            _PRIVATE_FILE_MODE,
            dir_fd=parent_fd,
        )
        os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
        temporary_info = os.fstat(temporary_fd)
        temporary_named = os.stat(
            temporary_leaf, dir_fd=parent_fd, follow_symlinks=False
        )
        if not _private_file(temporary_info) or _identity(temporary_info) != _identity(
            temporary_named
        ):
            raise TTSVoiceBundleError("destination_changed")
        temporary_identity = _identity(temporary_info)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(temporary_fd)
        _test_boundary("destination_pre_publish")
        if cancellation is not None and cancellation.is_set():
            raise asyncio.CancelledError
        if _parent_identity(os.lstat(destination.parent)) != parent_identity:
            raise TTSVoiceBundleError("destination_changed")
        temporary_named = os.stat(
            temporary_leaf, dir_fd=parent_fd, follow_symlinks=False
        )
        if _identity(temporary_named) != temporary_identity or not _private_file(
            temporary_named
        ):
            raise TTSVoiceBundleError("destination_changed")
        _test_boundary("destination_pre_rename")
        if _parent_identity(os.lstat(destination.parent)) != parent_identity:
            raise TTSVoiceBundleError("destination_changed")
        temporary_named = os.stat(
            temporary_leaf, dir_fd=parent_fd, follow_symlinks=False
        )
        if _identity(temporary_named) != temporary_identity or not _private_file(
            temporary_named
        ):
            raise TTSVoiceBundleError("destination_changed")
        if cancellation is not None and cancellation.is_set():
            raise asyncio.CancelledError
        try:
            rename_noreplace_at(
                parent_fd,
                temporary_leaf,
                destination.name,
            )
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise TTSVoiceBundleError("destination_changed") from None
            if error.errno in {
                errno.ENOSYS,
                errno.ENOTSUP,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            }:
                raise TTSVoiceBundleError("unsupported_platform") from None
            raise

        # The no-replace move is the publication PONR and consumes the temp.
        published_identity = temporary_identity
        temporary_identity = None
        try:
            _test_boundary("destination_post_link")
        except Exception:
            pass

        if not _converge_published_file(
            parent_fd,
            destination,
            parent_identity,
            published_identity,
            payload,
            _native=_native,
        ):
            return _WorkerOutcome("cleanup_failed", None)
        try:
            _test_boundary("destination_post_fsync")
        except Exception:
            pass
        if not _converge_published_file(
            parent_fd,
            destination,
            parent_identity,
            published_identity,
            payload,
            _native=_native,
        ):
            return _WorkerOutcome("cleanup_failed", None)
        if _native is not None:
            _bundle_native(_native).published = (destination, published_identity)
        return _WorkerOutcome(None, True)
    except TTSVoiceBundleError as error:
        code = error.code
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            code = "cancelled"
        elif not isinstance(error, Exception):
            raise
        else:
            code = "operation_failed"
    finally:
        if temporary_fd >= 0:
            try:
                _bundle_close(_native, temporary_fd)
            except OSError:
                code = "cleanup_failed"
        if parent_fd >= 0:
            try:
                _bundle_close(_native, parent_fd)
            except OSError:
                code = "cleanup_failed"
    if published_identity is not None:
        return _WorkerOutcome(code or "cleanup_failed", None)
    return _WorkerOutcome(code or "operation_failed", None)


_BUNDLE_SELECT_CODES = frozenset(
    (
        _create_operation.__code__,
        _create_operation_file.__code__,
        _publish_sync.__code__,
    )
)


async def _await_retained(
    task: asyncio.Task[_T],
    cancellation_callback: Callable[[], None] | None = None,
) -> tuple[asyncio.CancelledError | None, _T]:
    cancellation: asyncio.CancelledError | None = None
    caller = asyncio.current_task()
    requests = caller.cancelling() if caller is not None else 0
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            current = caller.cancelling() if caller is not None else 0
            if current > requests:
                cancellation = cancellation or error
                requests = current
                if cancellation_callback is not None:
                    cancellation_callback()
        except BaseException:
            if not task.done():
                raise
    result = task.result()
    return cancellation, result


def _canonical_reference(reference: TTSCloneReference) -> CanonicalTTSCloneReference:
    return CanonicalTTSCloneReference(
        wav_bytes=reference.wav_bytes,
        reference_text=reference.reference_text,
        sha256=reference.sha256,
        byte_length=reference.summary.byte_length,
        duration_ms=reference.summary.duration_ms,
        sample_rate_hz=reference.summary.sample_rate_hz,
        channels=reference.summary.channels,
        sample_encoding=reference.summary.sample_encoding,
    )


def _valid_dependency_snapshot(
    value: object, requirement: TTSCloneRecipeRequirement
) -> AudioCppGuidedDependencySnapshot | None:
    if type(value) is not AudioCppGuidedDependencySnapshot:
        return None
    snapshot = cast(AudioCppGuidedDependencySnapshot, value)
    if snapshot.state not in {"exact", "missing", "mismatch", "pending"}:
        return None
    if any(
        type(item) is not int or item < 0
        for item in (
            snapshot.provider_configuration_revision,
            snapshot.saved_generation,
            snapshot.applied_generation,
        )
    ):
        return None
    if type(snapshot.pending_configuration) is not bool:
        return None
    for item in (snapshot.saved_requirement, snapshot.applied_requirement):
        if item is not None and type(item) is not TTSCloneRecipeRequirement:
            return None
        if item is not None:
            try:
                rebuilt = TTSCloneRecipeRequirement(
                    recipe_id=item.recipe_id,
                    recipe_revision=item.recipe_revision,
                    model_id=item.model_id,
                )
            except (TypeError, ValueError, TTSVoiceBundleError):
                return None
            if rebuilt != item:
                return None
    if snapshot.pending_configuration != (
        snapshot.saved_generation != snapshot.applied_generation
    ):
        return None
    if snapshot.state == "exact" and snapshot.applied_requirement != requirement:
        return None
    if snapshot.state == "missing" and (
        snapshot.saved_requirement is not None
        or snapshot.applied_requirement is not None
    ):
        return None
    if snapshot.state == "mismatch" and (
        snapshot.saved_requirement == requirement
        and snapshot.applied_requirement == requirement
    ):
        return None
    if snapshot.state == "pending" and not (
        snapshot.pending_configuration and snapshot.saved_requirement == requirement
    ):
        return None
    return snapshot


def _profile_matches(
    profile: TTSGenerationProfile, portable: PortableTTSProfile
) -> bool:
    draft = portable.draft
    return (
        profile.profile_id == portable.profile_id
        and profile.display_name == draft.display_name
        and (
            profile.provider_id,
            profile.model_id,
            profile.voice_id,
            profile.response_format,
            profile.speed,
            profile.options,
        )
        == (
            draft.provider_id,
            draft.model_id,
            draft.voice_id,
            draft.response_format,
            draft.speed,
            draft.options,
        )
    )


class TTSVoiceBundlePortabilityService:
    """Own retained import inspection sessions and atomic bundle publication."""

    def __init__(
        self,
        operation_root: Path,
        repository: _Repository,
        dependency_service: _DependencyService,
        *,
        profile_mutation_fence: Callable[[], object],
        artifact_lease_coordinator: _ArtifactLeaseCoordinator | None = None,
        clock: Callable[[], float] = monotonic,
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._root = Path(operation_root)
        self._configured_source = None
        self._native_operations = set()
        self._maintenance_admission_closed = False
        self._owner_loop = None
        self._repository = repository
        self._dependency_service = dependency_service
        if not callable(profile_mutation_fence):
            raise TTSVoiceBundleError("operation_failed")
        self._profile_mutation_fence = profile_mutation_fence
        self._artifact_lease_coordinator = artifact_lease_coordinator
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._identity = object()
        self._root_identity: tuple[int, int] | None = None
        self._root_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._sessions: dict[TTSVoiceBundleHandle, _Session] = {}
        self._workers: set[asyncio.Task[object]] = set()
        self._owned_calls: set[asyncio.Task[object]] = set()
        self._calls: set[asyncio.Task[object]] = set()
        self._inspection_reservations = 0
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def _lease_import_dependency(
        self,
        command: TTSBundleImportCommand,
    ) -> AsyncIterator[None]:
        coordinator = self._artifact_lease_coordinator
        if coordinator is None:
            yield
            return
        async with cast(
            Any,
            coordinator.lease_consumers(
                (
                    AudioCppArtifactConsumerRequirement(
                        provider_id=command.source_draft.provider_id,
                        model_id=command.source_draft.model_id,
                        recipe_requirement=command.recipe_requirement,
                    ),
                )
            ),
        ):
            yield

    @asynccontextmanager
    async def _lease_profile_mutation(self) -> AsyncIterator[None]:
        """Join the app-owned profile service mutation fence."""

        async with cast(Any, self._profile_mutation_fence()):
            yield

    def _check_source(self):
        from .profile_source import check_bundle_source

        check_bundle_source(self)

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise TTSVoiceBundleError("operation_failed")

    def _maintenance_close_admission(self):
        self._check_loop()
        self._maintenance_admission_closed = True

    async def _maintenance_drain(self, deadline: float) -> bool:
        self._maintenance_close_admission()
        while self._calls or self._workers or self._owned_calls:
            if monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.01, max(0, deadline - monotonic())))
        return not self._native_operations

    async def _maintenance_resume(self):
        self._check_loop()
        self._check_source()
        if self._closed:
            raise TTSVoiceBundleError("operation_failed")
        self._maintenance_admission_closed = False

    def _run_native_worker(self, native, function, args):
        native.thread = threading.current_thread()
        failure = None
        result = None
        try:
            native.admit()
            native.admit()
            for selected in native.selected:
                native.admit(selected)
                native.admit(selected.parent)
            result = function(*args, _native=native)
            return result
        except BaseException as error:
            failure = error
            raise
        finally:
            prior = len(native.errors)
            retired = native.finish()
            # Preserve an already evaluated durable publication acknowledgement.
            published = (
                native.publication
                and native.published is not None
                and type(result) is _WorkerOutcome
                and result.result is True
                and result.code is None
            )
            body_failed = type(result) is _WorkerOutcome and result.code is not None
            if failure is None and not published and not body_failed:
                for error in native.errors[prior:]:
                    if not isinstance(error, Exception):
                        raise error
                if not retired:
                    raise TTSVoiceBundleError("cleanup_failed")

    async def _ensure_root(self) -> None:
        async with self._root_lock:
            if self._root_identity is not None:
                return
            if self._closed or self._maintenance_admission_closed:
                raise TTSVoiceBundleError("operation_failed")
            try:
                outcome = await self._run_worker(_prepare_root_sync, self._root)
            except asyncio.CancelledError:
                raise
            except TTSVoiceBundleError as error:
                raise TTSVoiceBundleError(error.code) from None
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                raise TTSVoiceBundleError("operation_failed") from None
            self._root, self._root_identity = cast(
                tuple[Path, tuple[int, int]], outcome
            )

    async def _admit(self, *, inspection: bool = False) -> asyncio.Task[object]:
        self._check_loop()
        self._check_source()
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is None:
            raise TTSVoiceBundleError("operation_failed")
        async with self._session_lock:
            if self._closed or self._maintenance_admission_closed:
                raise TTSVoiceBundleError("operation_failed")
            self._expire_sessions()
            if (
                inspection
                and len(self._sessions) + self._inspection_reservations
                >= _SESSION_LIMIT
            ):
                raise TTSVoiceBundleError("operation_failed")
            self._calls.add(call)
            if inspection:
                self._inspection_reservations += 1
        return call

    async def _release(
        self, call: asyncio.Task[object], *, inspection: bool = False
    ) -> None:
        async with self._session_lock:
            self._calls.discard(call)
            if inspection:
                self._inspection_reservations -= 1

    async def _run_worker(
        self, function: Callable[..., object], *args: object, _selected=()
    ) -> object:
        native = _BundleNativeOperation(self, _selected)
        self._native_operations.add(native)
        worker: asyncio.Task[object] = asyncio.create_task(
            asyncio.to_thread(self._run_native_worker, native, function, args)
        )
        self._workers.add(worker)
        try:
            cancellation, result = await _await_retained(worker)
        finally:
            self._workers.discard(worker)
        if cancellation is not None:
            raise cancellation
        return result

    async def _run_worker_settled(
        self, function: Callable[..., object], *args: object, _selected=()
    ) -> tuple[Literal["cancelled"] | None, object]:
        cancellation_event = Event()
        native = _BundleNativeOperation(self, _selected, publication=True)
        self._native_operations.add(native)
        worker: asyncio.Task[object] = asyncio.create_task(
            asyncio.to_thread(
                self._run_native_worker, native, function, (*args, cancellation_event)
            )
        )
        self._workers.add(worker)
        try:
            cancellation, result = await _await_retained(worker, cancellation_event.set)
        finally:
            self._workers.discard(worker)
        return ("cancelled" if cancellation is not None else None), result

    async def _run_owned_call(
        self, awaitable: object
    ) -> tuple[Literal["cancelled"] | None, object]:
        task = asyncio.create_task(cast(Any, awaitable))
        self._owned_calls.add(task)
        try:
            cancellation, result = await _await_retained(task)
        finally:
            self._owned_calls.discard(task)
        return ("cancelled" if cancellation is not None else None), result

    async def _copy_source(
        self, source: Path, expected: _SourceFingerprint | None
    ) -> _CopyResult:
        await self._ensure_root()
        assert self._root_identity is not None
        outcome = cast(
            _WorkerOutcome,
            await self._run_worker(
                _copy_and_inspect_sync,
                self._root,
                self._root_identity,
                source,
                expected,
                _selected=(source,),
            ),
        )
        if outcome.code is not None or type(outcome.result) is not _CopyResult:
            raise TTSVoiceBundleError(
                cast(Any, outcome.code or "operation_failed")
            ) from None
        return cast(_CopyResult, outcome.result)

    async def _inspect_impl(
        self, source: Path | str | os.PathLike[str]
    ) -> TTSVoiceBundleReview:
        call = await self._admit(inspection=True)
        try:
            selected = _source_path(source)
            copied = await self._copy_source(selected, None)
            evidence = await self._review_evidence(copied.bundle)
            return await self._issue_session(
                selected, copied.source_fingerprint, evidence
            )
        finally:
            await self._release(call, inspection=True)

    async def _inspect_outcome(self, source: object) -> _PublicOutcome:
        try:
            result = await self._inspect_impl(cast(Any, source))
            return _PublicOutcome(result=result)
        except BaseException as error:
            return _classify_and_sever_public_failure(error)

    async def inspect(
        self, source: Path | str | os.PathLike[str]
    ) -> TTSVoiceBundleReview:
        """Copy and validate one unchanged hostile source into a safe review.

        Args:
            source: User-selected bundle path to copy and inspect.

        Returns:
            Sanitized review facts plus opaque commit authority.

        Raises:
            TTSVoiceBundleError: Inspection cannot complete safely.
            asyncio.CancelledError: The inspection caller is cancelled.
        """

        outcome = await self._inspect_outcome(source)
        del source
        failure = _reconstruct_public_failure(outcome)
        if failure is not None:
            del outcome
            raise failure from None
        if type(outcome.result) is not TTSVoiceBundleReview:
            del outcome
            raise TTSVoiceBundleError("operation_failed") from None
        return cast(TTSVoiceBundleReview, outcome.result)

    def _expire_sessions(self) -> None:
        now = self._clock()
        for handle, session in tuple(self._sessions.items()):
            if session.expires_at <= now:
                self._sessions.pop(handle, None)

    async def _issue_session(
        self,
        source: Path,
        fingerprint: _SourceFingerprint,
        evidence: _ReviewEvidence,
    ) -> TTSVoiceBundleReview:
        self._check_source()
        async with self._session_lock:
            self._expire_sessions()
            if self._closed or len(self._sessions) >= _SESSION_LIMIT:
                raise TTSVoiceBundleError("operation_failed")
            handle = TTSVoiceBundleHandle(_HANDLE_KEY, self._identity)
            review = TTSVoiceBundleReview(
                handle=handle,
                profile_id=evidence.review.profile_id,
                profile_name=evidence.review.profile_name,
                provider_id=evidence.review.provider_id,
                model_id=evidence.review.model_id,
                voice_id=evidence.review.voice_id,
                recipe_id=evidence.review.recipe_id,
                recipe_revision=evidence.review.recipe_revision,
                dependency_state=evidence.review.dependency_state,
                allowed_choices=evidence.review.allowed_choices,
                copy_profile_id=evidence.review.copy_profile_id,
                copy_profile_name=evidence.review.copy_profile_name,
                exact_private_duplicate=evidence.review.exact_private_duplicate,
                uuid_conflict=evidence.review.uuid_conflict,
                name_conflict=evidence.review.name_conflict,
            )
            final_evidence = _ReviewEvidence(
                review=review,
                source_collisions=evidence.source_collisions,
                portable=evidence.portable,
                requirement=evidence.requirement,
                repository_generation=evidence.repository_generation,
                dependency_revision=evidence.dependency_revision,
            )
            self._sessions[handle] = _Session(
                handle=handle,
                expires_at=self._clock() + _SESSION_TTL_SECONDS,
                source=source,
                source_fingerprint=fingerprint,
                evidence=final_evidence,
            )
            return review

    async def _review_evidence(self, bundle: TTSCloneVoiceBundle) -> _ReviewEvidence:
        generation = self._repository.generation
        collision_result = await self._repository.get_profile_collisions(
            bundle.profile.profile_id,
            bundle.profile.draft,
        )
        if (
            type(collision_result) is not ProfileStoreResult
            or collision_result.generation != generation
            or type(collision_result.value) is not TTSProfileCollisionSnapshot
        ):
            raise TTSVoiceBundleError("operation_failed")
        collisions = collision_result.value
        exact_duplicate = await self._exact_duplicate(bundle, collisions, generation)
        dependency_value = (
            await self._dependency_service.audio_cpp_guided_dependency_snapshot(
                bundle.recipe_requirement
            )
        )
        dependency = _valid_dependency_snapshot(
            dependency_value,
            bundle.recipe_requirement,
        )
        if dependency is None:
            raise TTSVoiceBundleError("operation_failed")
        if self._repository.generation != generation:
            raise TTSVoiceBundleError("stale_inspection")
        has_collision = (
            collisions.profile_id_match is not None
            or collisions.normalized_name_match is not None
        )
        copy_profile_id: UUID | None = None
        copy_name: str | None = None
        if not has_collision:
            choices: tuple[Literal["create", "reuse", "copy"], ...] = ("create",)
        else:
            copy_profile_id, copy_name = await self._copy_destination(
                bundle.profile, generation
            )
            choices = ("reuse", "copy") if exact_duplicate else ("copy",)
        placeholder = TTSVoiceBundleHandle(_HANDLE_KEY, self._identity)
        review = TTSVoiceBundleReview(
            handle=placeholder,
            profile_id=bundle.profile.profile_id,
            profile_name=bundle.profile.draft.display_name,
            provider_id=bundle.profile.draft.provider_id,
            model_id=bundle.profile.draft.model_id,
            voice_id=bundle.profile.draft.voice_id,
            recipe_id=bundle.recipe_requirement.recipe_id,
            recipe_revision=bundle.recipe_requirement.recipe_revision,
            dependency_state=cast(TTSVoiceBundleDependencyState, dependency.state),
            allowed_choices=choices,
            copy_profile_id=copy_profile_id,
            copy_profile_name=copy_name,
            exact_private_duplicate=exact_duplicate,
            uuid_conflict=collisions.profile_id_match is not None,
            name_conflict=collisions.normalized_name_match is not None,
        )
        return _ReviewEvidence(
            review=review,
            source_collisions=collisions,
            portable=bundle.profile,
            requirement=bundle.recipe_requirement,
            repository_generation=generation,
            dependency_revision=dependency.provider_configuration_revision,
        )

    async def _exact_duplicate(
        self,
        bundle: TTSCloneVoiceBundle,
        collisions: TTSProfileCollisionSnapshot,
        generation: int,
    ) -> bool:
        candidate = collisions.profile_id_match
        if (
            candidate is None
            or candidate != collisions.normalized_name_match
            or not _profile_matches(candidate, bundle.profile)
            or candidate.reference is None
            or candidate.reference.recipe_requirement != bundle.recipe_requirement
        ):
            return False
        result = await self._repository.get_reference(
            candidate.profile_id,
            expected_revision=candidate.revision,
            expected_generation=generation,
        )
        if (
            type(result) is not ProfileStoreResult
            or result.generation != generation
            or type(result.value) is not TTSCloneReference
        ):
            raise TTSVoiceBundleError("operation_failed")
        return (
            result.value.recipe_requirement == bundle.recipe_requirement
            and _canonical_reference(result.value) == bundle.reference
        )

    async def _copy_destination(
        self, portable: PortableTTSProfile, generation: int
    ) -> tuple[UUID, str]:
        base = portable.draft.display_name
        for index in range(1, 33):
            candidate_id = self._uuid_factory()
            if type(candidate_id) is not UUID:
                raise TTSVoiceBundleError("operation_failed")
            suffix = " copy" if index == 1 else f" copy {index}"
            candidate_name = f"{base[: max(1, 128 - len(suffix))]}{suffix}"
            draft = TTSProfileDraft(
                display_name=candidate_name,
                provider_id=portable.draft.provider_id,
                model_id=portable.draft.model_id,
                voice_id=portable.draft.voice_id,
                response_format=portable.draft.response_format,
                speed=portable.draft.speed,
                options=portable.draft.options,
            )
            result = await self._repository.get_profile_collisions(candidate_id, draft)
            if result.generation != generation:
                raise TTSVoiceBundleError("stale_inspection")
            if (
                result.value.profile_id_match is None
                and result.value.normalized_name_match is None
            ):
                return candidate_id, candidate_name
        raise TTSVoiceBundleError("operation_failed")

    async def _consume_session(
        self, handle: object
    ) -> tuple[asyncio.Task[object], _Session]:
        self._check_loop()
        self._check_source()
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is None:
            raise TTSVoiceBundleError("operation_failed")
        async with self._session_lock:
            if self._closed or self._maintenance_admission_closed:
                raise TTSVoiceBundleError("operation_failed")
            self._expire_sessions()
            if type(handle) is not TTSVoiceBundleHandle or not handle._belongs_to(
                self._identity
            ):
                raise TTSVoiceBundleError("stale_inspection")
            session = self._sessions.pop(handle, None)
            if session is None:
                raise TTSVoiceBundleError("stale_inspection")
            self._calls.add(call)
        return call, session

    async def _verify_source(self, session: _Session) -> None:
        outcome = cast(
            _WorkerOutcome,
            await self._run_worker(
                _fingerprint_source_sync,
                session.source,
                session.source_fingerprint,
                _selected=(session.source,),
            ),
        )
        if outcome.code is not None or outcome.result is not True:
            raise TTSVoiceBundleError(
                cast(Any, outcome.code or "operation_failed")
            ) from None

    async def _commit_impl(
        self,
        handle: TTSVoiceBundleHandle,
        choice: TTSVoiceBundleImportChoice,
    ) -> TTSVoiceBundleImportResult:
        if type(choice) is not TTSVoiceBundleImportChoice:
            raise TTSVoiceBundleError("operation_failed")
        call, session = await self._consume_session(handle)
        try:
            copied = await self._copy_source(session.source, session.source_fingerprint)
            refreshed = await self._review_evidence(copied.bundle)
            await self._verify_source(session)
            if refreshed.visible_key() != session.evidence.visible_key():
                review = await self._issue_session(
                    session.source, copied.source_fingerprint, refreshed
                )
                return TTSVoiceBundleImportResult("stale_inspection", review=review)
            if choice.choice not in session.evidence.review.allowed_choices:
                raise TTSVoiceBundleError("stale_inspection")
            dependency_state = (
                "exact" if refreshed.review.dependency_state == "exact" else "missing"
            )
            consent_required = dependency_state == "missing" and choice.choice in {
                "create",
                "copy",
            }
            if choice.inactive_consent is not consent_required:
                raise TTSVoiceBundleError("operation_failed")
            command = TTSBundleImportCommand(
                choice=choice.choice,
                source_profile_id=copied.bundle.profile.profile_id,
                source_draft=copied.bundle.profile.draft,
                recipe_requirement=copied.bundle.recipe_requirement,
                canonical_reference=copied.bundle.reference,
                expected_generation=refreshed.repository_generation,
                reviewed_source_collisions=refreshed.source_collisions,
                copy_profile_id=(
                    refreshed.review.copy_profile_id
                    if choice.choice == "copy"
                    else None
                ),
                copy_display_name=(
                    refreshed.review.copy_profile_name
                    if choice.choice == "copy"
                    else None
                ),
                dependency_state=cast(Any, dependency_state),
                inactive_consent=choice.inactive_consent,
            )
            _test_boundary("commit_pre_repository")
            await self._verify_source(session)
            async with self._lease_import_dependency(command):
                async with self._lease_profile_mutation():
                    control, result = await self._run_owned_call(
                        self._repository.commit_bundle_import(command)
                    )
            if control is not None:
                raise asyncio.CancelledError from None
            if type(result) is not ProfileStoreResult:
                raise TTSVoiceBundleError("operation_failed")
            repository_result = cast(ProfileStoreResult[TTSBundleImportResult], result)
            if (
                repository_result.generation != refreshed.repository_generation
                or type(repository_result.value) is not TTSBundleImportResult
            ):
                raise TTSVoiceBundleError("operation_failed")
            if repository_result.value.kind == "stale_inspection":
                successor = await self._review_evidence(copied.bundle)
                review = await self._issue_session(
                    session.source, copied.source_fingerprint, successor
                )
                return TTSVoiceBundleImportResult("stale_inspection", review=review)
            assert repository_result.value.profile is not None
            return TTSVoiceBundleImportResult(
                cast(TTSVoiceBundleImportResultStatus, repository_result.value.kind),
                profile=repository_result.value.profile,
            )
        finally:
            await self._release(call)

    async def _commit_outcome(self, handle: object, choice: object) -> _PublicOutcome:
        try:
            result = await self._commit_impl(cast(Any, handle), cast(Any, choice))
            return _PublicOutcome(result=result)
        except BaseException as error:
            return _classify_and_sever_public_failure(error)

    async def commit(
        self,
        handle: TTSVoiceBundleHandle,
        choice: TTSVoiceBundleImportChoice,
    ) -> TTSVoiceBundleImportResult:
        """Consume, revalidate, and commit one reviewed import exactly once."""

        outcome = await self._commit_outcome(handle, choice)
        del handle, choice
        failure = _reconstruct_public_failure(outcome)
        if failure is not None:
            del outcome
            raise failure from None
        if type(outcome.result) is not TTSVoiceBundleImportResult:
            del outcome
            raise TTSVoiceBundleError("operation_failed") from None
        return cast(TTSVoiceBundleImportResult, outcome.result)

    async def _export_impl(
        self,
        profile_id: UUID,
        destination: Path | str | os.PathLike[str],
        *,
        expected_generation: int,
        expected_revision: int,
        acknowledged: bool,
    ) -> None:
        call = await self._admit()
        try:
            if acknowledged is not True:
                raise TTSVoiceBundleError("acknowledgement_required")
            if type(profile_id) is not UUID:
                raise TTSVoiceBundleError("operation_failed")
            result = await self._repository.get_profile(profile_id)
            if (
                type(result) is not ProfileStoreResult
                or result.generation != expected_generation
                or type(result.value) is not TTSGenerationProfile
                or result.value.revision != expected_revision
                or result.value.reference is None
            ):
                raise TTSVoiceBundleError("stale_inspection")
            profile = result.value
            reference_result = await self._repository.get_reference(
                profile_id,
                expected_revision=expected_revision,
                expected_generation=expected_generation,
            )
            if (
                type(reference_result) is not ProfileStoreResult
                or reference_result.generation != expected_generation
                or type(reference_result.value) is not TTSCloneReference
                or reference_result.value.recipe_requirement is None
            ):
                raise TTSVoiceBundleError("operation_failed")
            reference = reference_result.value
            requirement = reference.recipe_requirement
            assert requirement is not None
            bundle = TTSCloneVoiceBundle(
                profile=PortableTTSProfile(
                    profile_id=profile.profile_id,
                    draft=TTSProfileDraft(
                        display_name=profile.display_name,
                        provider_id=profile.provider_id,
                        model_id=profile.model_id,
                        voice_id=profile.voice_id,
                        response_format=profile.response_format,
                        speed=profile.speed,
                        options=profile.options,
                    ),
                ),
                reference=_canonical_reference(reference),
                recipe_requirement=requirement,
            )
            payload = encode_clone_voice_bundle(bundle)
            selected = _source_path(destination)
            control, raw_outcome = await self._run_worker_settled(
                _publish_sync, selected, payload, _selected=(selected,)
            )
            outcome = cast(_WorkerOutcome, raw_outcome)
            del payload, bundle, reference
            if outcome.result is True and outcome.code is None:
                return
            if control is not None or outcome.code == "cancelled":
                raise asyncio.CancelledError from None
            if outcome.code is not None:
                raise TTSVoiceBundleError(cast(Any, outcome.code)) from None
        finally:
            await self._release(call)

    async def _export_outcome(
        self,
        profile_id: object,
        destination: object,
        expected_generation: object,
        expected_revision: object,
        acknowledged: object,
    ) -> _PublicOutcome:
        try:
            await self._export_impl(
                cast(Any, profile_id),
                cast(Any, destination),
                expected_generation=cast(Any, expected_generation),
                expected_revision=cast(Any, expected_revision),
                acknowledged=cast(Any, acknowledged),
            )
            return _PublicOutcome(result=True)
        except BaseException as error:
            return _classify_and_sever_public_failure(error)

    async def export(
        self,
        profile_id: UUID,
        destination: Path | str | os.PathLike[str],
        *,
        expected_generation: int,
        expected_revision: int,
        acknowledged: bool,
    ) -> None:
        """Publish one deterministic, acknowledged bundle without overwrite."""

        outcome = await self._export_outcome(
            profile_id,
            destination,
            expected_generation,
            expected_revision,
            acknowledged,
        )
        del (
            profile_id,
            destination,
            expected_generation,
            expected_revision,
            acknowledged,
        )
        failure = _reconstruct_public_failure(outcome)
        if failure is not None:
            del outcome
            raise failure from None
        if outcome.result is not True:
            del outcome
            raise TTSVoiceBundleError("operation_failed") from None

    def seal(self) -> None:
        self._closed = True

    async def _invalidate_impl(self, handle: object) -> None:
        call = await self._admit()
        try:
            async with self._session_lock:
                if type(handle) is not TTSVoiceBundleHandle or not handle._belongs_to(
                    self._identity
                ):
                    raise TTSVoiceBundleError("stale_inspection")
                self._sessions.pop(handle, None)
        finally:
            await self._release(call)

    async def _invalidate_outcome(self, handle: object) -> _PublicOutcome:
        try:
            await self._invalidate_impl(handle)
            return _PublicOutcome(result=True)
        except BaseException as error:
            return _classify_and_sever_public_failure(error)

    async def invalidate(self, handle: TTSVoiceBundleHandle) -> None:
        """Idempotently discard one retained review authority."""

        outcome = await self._invalidate_outcome(handle)
        del handle
        failure = _reconstruct_public_failure(outcome)
        if failure is not None:
            del outcome
            raise failure from None

    async def _close_impl(self) -> None:
        async with self._session_lock:
            if self._close_task is None:
                self._closed = True
                self._sessions.clear()
                self._close_task = asyncio.create_task(self._complete_close())
            close_task = self._close_task
        cancellation, _ = await _await_retained(close_task)
        if cancellation is not None:
            raise cancellation

    async def _close_outcome(self) -> _PublicOutcome:
        try:
            await self._close_impl()
            return _PublicOutcome(result=True)
        except BaseException as error:
            return _classify_and_sever_public_failure(error)

    async def close(self) -> None:
        """Seal admission and retain settlement of every service-owned call."""

        outcome = await self._close_outcome()
        failure = _reconstruct_public_failure(outcome)
        if failure is not None:
            del outcome
            raise failure from None
        if outcome.result is not True:
            del outcome
            raise TTSVoiceBundleError("operation_failed") from None

    async def wait_closed(self) -> None:
        """Join definitive close; ``close`` remains idempotent."""

        await self.close()

    async def _complete_close(self) -> None:
        current = asyncio.current_task()
        while True:
            async with self._session_lock:
                retained = tuple(
                    task
                    for task in self._calls | self._workers | self._owned_calls
                    if task is not current and not task.done()
                )
                if not retained:
                    self._sessions.clear()
                    if self._native_operations:
                        raise TTSVoiceBundleError("cleanup_failed")
                    return
            await asyncio.gather(
                *(asyncio.shield(task) for task in retained), return_exceptions=True
            )


__all__ = [
    "TTSVoiceBundleHandle",
    "TTSVoiceBundleImportChoice",
    "TTSVoiceBundleImportResult",
    "TTSVoiceBundlePortabilityService",
    "TTSVoiceBundleReview",
]
