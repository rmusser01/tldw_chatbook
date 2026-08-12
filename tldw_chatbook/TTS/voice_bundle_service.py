"""Application-owned lifecycle for explicit clone-voice bundle portability."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from time import monotonic
from typing import Any, Final, Literal, NoReturn, Protocol, TypeVar, cast
from uuid import UUID, uuid4

try:  # pragma: no cover - exercised through the platform support seam.
    import fcntl
except ImportError:  # pragma: no cover - Windows is intentionally unsupported.
    fcntl = None  # type: ignore[assignment]

from tldw_chatbook.TTS.TTS_Generation import AudioCppGuidedDependencySnapshot
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
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
    identity: tuple[int, int, int, int, int, int, int, int]
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
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
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


def _prepare_root_sync(root: Path) -> tuple[Path, tuple[int, int]]:
    if not _posix_supported():
        raise TTSVoiceBundleError("unsupported_platform")
    selected = secure_private_directory(
        root, create=True, application_owned=True
    ).lexical_path
    descriptor = os.open(selected, _DIRECTORY_FLAGS)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(selected, follow_symlinks=False)
        if not _private_directory(opened) or _identity(opened) != _identity(named):
            raise OSError
        assert fcntl is not None
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.listdir(descriptor):
            raise OSError
        return selected, _identity(opened)
    finally:
        os.close(descriptor)


def _create_operation(root: Path, root_identity: tuple[int, int]) -> _Operation:
    root_fd = os.open(root, _DIRECTORY_FLAGS)
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
                os.mkdir(operation_leaf, _PRIVATE_DIRECTORY_MODE, dir_fd=root_fd)
                break
            except FileExistsError:
                continue
        else:
            raise OSError
        operation_fd = os.open(operation_leaf, _DIRECTORY_FLAGS, dir_fd=root_fd)
        os.fchmod(operation_fd, _PRIVATE_DIRECTORY_MODE)
        opened = os.fstat(operation_fd)
        named = os.stat(operation_leaf, dir_fd=root_fd, follow_symlinks=False)
        if not _private_directory(opened) or _identity(opened) != _identity(named):
            raise OSError
        return _Operation(
            root_fd=root_fd,
            operation_fd=operation_fd,
            operation_leaf=operation_leaf,
            operation_identity=_identity(opened),
            files={},
        )
    except BaseException:
        if operation_fd >= 0:
            os.close(operation_fd)
        if operation_leaf:
            try:
                os.rmdir(operation_leaf, dir_fd=root_fd)
            except OSError:
                pass
        os.close(root_fd)
        raise


def _create_operation_file(operation: _Operation, leaf: str) -> int:
    descriptor = os.open(
        leaf,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW,
        _PRIVATE_FILE_MODE,
        dir_fd=operation.operation_fd,
    )
    os.fchmod(descriptor, _PRIVATE_FILE_MODE)
    info = os.fstat(descriptor)
    named = os.stat(leaf, dir_fd=operation.operation_fd, follow_symlinks=False)
    if not _private_file(info) or _identity(info) != _identity(named):
        os.close(descriptor)
        raise OSError
    operation.files[leaf] = _identity(info)
    return descriptor


def _cleanup_operation(operation: _Operation) -> bool:
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
    os.close(operation.operation_fd)
    try:
        assert fcntl is not None
        fcntl.flock(operation.root_fd, fcntl.LOCK_UN)
    finally:
        os.close(operation.root_fd)
    return not failed


def _copy_and_inspect(
    root: Path,
    root_identity: tuple[int, int],
    source: Path,
    expected: _SourceFingerprint | None,
) -> _CopyResult:
    operation = _create_operation(root, root_identity)
    source_fd = -1
    copy_fd = -1
    primary_code: str | None = None
    result: _CopyResult | None = None
    streams: list[Any] = []
    try:
        parent_before = os.lstat(source.parent)
        if not stat.S_ISDIR(parent_before.st_mode):
            raise TTSVoiceBundleError("source_changed")
        parent_fd = os.open(source.parent, _DIRECTORY_FLAGS)
        try:
            parent_opened = os.fstat(parent_fd)
            parent_named = os.lstat(source.parent)
            if _identity(parent_before) != _identity(parent_opened) or _identity(
                parent_opened
            ) != _identity(parent_named):
                raise TTSVoiceBundleError("source_changed")
            initial = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _owned_source(initial):
                raise TTSVoiceBundleError("source_changed")
            source_fd = os.open(source.name, _SOURCE_FLAGS, dir_fd=parent_fd)
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

            copy_fd = _create_operation_file(operation, "source.bundle")
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
            current = _SourceFingerprint(identity, digest.hexdigest())
            if (
                copied != identity[6]
                or _source_identity(final_opened) != identity
                or _source_identity(final_named) != identity
                or (expected is not None and expected != current)
            ):
                raise TTSVoiceBundleError("source_changed")
            _test_boundary("source_post_copy")
        finally:
            os.close(parent_fd)
        os.close(source_fd)
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
            descriptor = _create_operation_file(operation, leaf)
            stream = os.fdopen(descriptor, "w+b")
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
        parent_fd = os.open(source.parent, _DIRECTORY_FLAGS)
        try:
            final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if _source_identity(final_named) != identity:
                raise TTSVoiceBundleError("source_changed")
        finally:
            os.close(parent_fd)
        _test_boundary("source_pre_fingerprint")
        parent_fd = os.open(source.parent, _DIRECTORY_FLAGS)
        try:
            final_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if _source_identity(final_named) != identity:
                raise TTSVoiceBundleError("source_changed")
        finally:
            os.close(parent_fd)
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
            os.close(source_fd)
        if copy_fd >= 0:
            os.close(copy_fd)
        for stream in streams:
            try:
                stream.close()
            except OSError:
                primary_code = primary_code or "cleanup_failed"
        if not _cleanup_operation(operation):
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
) -> _WorkerOutcome:
    try:
        return _WorkerOutcome(
            None, _copy_and_inspect(root, root_identity, source, expected)
        )
    except TTSVoiceBundleError as error:
        return _WorkerOutcome(error.code, None)
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        return _WorkerOutcome("operation_failed", None)


def _publish_sync(destination: Path, payload: bytes) -> _WorkerOutcome:
    parent_fd = -1
    temporary_fd = -1
    temporary_leaf = f".{destination.name}.{token_hex(16)}.tmp"
    temporary_identity: tuple[int, int] | None = None
    published = False
    code: str | None = None
    try:
        if not _posix_supported():
            raise TTSVoiceBundleError("unsupported_platform")
        before = os.lstat(destination.parent)
        if not stat.S_ISDIR(before.st_mode):
            raise TTSVoiceBundleError("destination_changed")
        parent_fd = os.open(destination.parent, _DIRECTORY_FLAGS)
        opened_parent = os.fstat(parent_fd)
        named_parent = os.lstat(destination.parent)
        if _identity(before) != _identity(opened_parent) or _identity(
            opened_parent
        ) != _identity(named_parent):
            raise TTSVoiceBundleError("destination_changed")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise TTSVoiceBundleError("destination_changed")
        temporary_fd = os.open(
            temporary_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW,
            _PRIVATE_FILE_MODE,
            dir_fd=parent_fd,
        )
        os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
        info = os.fstat(temporary_fd)
        named = os.stat(temporary_leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not _private_file(info) or _identity(info) != _identity(named):
            raise TTSVoiceBundleError("destination_changed")
        temporary_identity = _identity(info)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(temporary_fd)
        _test_boundary("destination_pre_publish")
        if _identity(os.lstat(destination.parent)) != _identity(opened_parent):
            raise TTSVoiceBundleError("destination_changed")
        temporary_named = os.stat(
            temporary_leaf, dir_fd=parent_fd, follow_symlinks=False
        )
        if _identity(temporary_named) != temporary_identity or not _private_file(
            temporary_named
        ):
            raise TTSVoiceBundleError("destination_changed")
        try:
            os.link(
                temporary_leaf,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise TTSVoiceBundleError("destination_changed") from None
        published = True
        destination_info = os.stat(
            destination.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _identity(destination_info) != temporary_identity or not _private_file(
            destination_info, links=(2,)
        ):
            raise TTSVoiceBundleError("destination_changed")
        os.unlink(temporary_leaf, dir_fd=parent_fd)
        temporary_identity = None
        os.fsync(parent_fd)
        settled = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if not _private_file(settled) or _identity(settled) != _identity(
            destination_info
        ):
            raise TTSVoiceBundleError("destination_changed")
    except TTSVoiceBundleError as error:
        code = error.code
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        code = "operation_failed"
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if parent_fd >= 0 and temporary_identity is not None:
            try:
                named = os.stat(temporary_leaf, dir_fd=parent_fd, follow_symlinks=False)
                if _identity(named) == temporary_identity and _private_file(named):
                    os.unlink(temporary_leaf, dir_fd=parent_fd)
                else:
                    code = "cleanup_failed"
            except FileNotFoundError:
                pass
            except OSError:
                code = "cleanup_failed"
        if code is not None and published and parent_fd >= 0:
            # Once linked, publication is complete; never remove an ambiguous final.
            code = "destination_changed"
        if parent_fd >= 0:
            os.close(parent_fd)
    return _WorkerOutcome(code, None if code else True)


async def _await_retained(
    task: asyncio.Task[_T],
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
        clock: Callable[[], float] = monotonic,
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._root = Path(operation_root)
        self._repository = repository
        self._dependency_service = dependency_service
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._identity = object()
        self._root_identity: tuple[int, int] | None = None
        self._root_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._sessions: dict[TTSVoiceBundleHandle, _Session] = {}
        self._workers: set[asyncio.Task[object]] = set()
        self._calls: set[asyncio.Task[object]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def _ensure_root(self) -> None:
        async with self._root_lock:
            if self._root_identity is not None:
                return
            if self._closed:
                raise TTSVoiceBundleError("operation_failed")
            try:
                outcome = await self._run_worker(_prepare_root_sync, self._root)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                raise TTSVoiceBundleError("operation_failed") from None
            self._root, self._root_identity = cast(
                tuple[Path, tuple[int, int]], outcome
            )

    def _track_call(self) -> asyncio.Task[object] | None:
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._calls.add(call)
        return call

    async def _run_worker(
        self, function: Callable[..., object], *args: object
    ) -> object:
        worker: asyncio.Task[object] = asyncio.create_task(
            asyncio.to_thread(function, *args)
        )
        self._workers.add(worker)
        try:
            cancellation, result = await _await_retained(worker)
        finally:
            self._workers.discard(worker)
        if cancellation is not None:
            raise cancellation
        return result

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
            ),
        )
        if outcome.code is not None or type(outcome.result) is not _CopyResult:
            raise TTSVoiceBundleError(
                cast(Any, outcome.code or "operation_failed")
            ) from None
        return cast(_CopyResult, outcome.result)

    async def inspect(
        self, source: Path | str | os.PathLike[str]
    ) -> TTSVoiceBundleReview:
        """Copy and validate one unchanged hostile source into a safe review."""

        call = self._track_call()
        try:
            if self._closed:
                raise TTSVoiceBundleError("operation_failed")
            selected = _source_path(source)
            async with self._session_lock:
                self._expire_sessions()
                if len(self._sessions) >= _SESSION_LIMIT:
                    raise TTSVoiceBundleError("operation_failed")
            copied = await self._copy_source(selected, None)
            evidence = await self._review_evidence(copied.bundle)
            return await self._issue_session(
                selected, copied.source_fingerprint, evidence
            )
        finally:
            if call is not None:
                self._calls.discard(call)

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
        dependency = (
            await self._dependency_service.audio_cpp_guided_dependency_snapshot(
                bundle.recipe_requirement
            )
        )
        if type(dependency) is not AudioCppGuidedDependencySnapshot:
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
        try:
            result = await self._repository.get_reference(
                candidate.profile_id,
                expected_revision=candidate.revision,
                expected_generation=generation,
            )
        except Exception:
            return False
        return (
            type(result) is ProfileStoreResult
            and result.generation == generation
            and type(result.value) is TTSCloneReference
            and result.value.recipe_requirement == bundle.recipe_requirement
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

    async def commit(
        self,
        handle: TTSVoiceBundleHandle,
        choice: TTSVoiceBundleImportChoice,
    ) -> TTSVoiceBundleImportResult:
        """Consume, revalidate, and commit one reviewed import exactly once."""

        call = self._track_call()
        try:
            if type(choice) is not TTSVoiceBundleImportChoice:
                raise TTSVoiceBundleError("operation_failed")
            async with self._session_lock:
                self._expire_sessions()
                if type(handle) is not TTSVoiceBundleHandle or not handle._belongs_to(
                    self._identity
                ):
                    raise TTSVoiceBundleError("stale_inspection")
                session = self._sessions.pop(handle, None)
            if session is None:
                raise TTSVoiceBundleError("stale_inspection")
            copied = await self._copy_source(session.source, session.source_fingerprint)
            refreshed = await self._review_evidence(copied.bundle)
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
            result = await self._repository.commit_bundle_import(command)
            if (
                type(result) is not ProfileStoreResult
                or result.generation != refreshed.repository_generation
                or type(result.value) is not TTSBundleImportResult
            ):
                raise TTSVoiceBundleError("operation_failed")
            if result.value.kind == "stale_inspection":
                successor = await self._review_evidence(copied.bundle)
                review = await self._issue_session(
                    session.source, copied.source_fingerprint, successor
                )
                return TTSVoiceBundleImportResult("stale_inspection", review=review)
            assert result.value.profile is not None
            return TTSVoiceBundleImportResult(
                cast(TTSVoiceBundleImportResultStatus, result.value.kind),
                profile=result.value.profile,
            )
        finally:
            if call is not None:
                self._calls.discard(call)

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

        call = self._track_call()
        try:
            if acknowledged is not True:
                raise TTSVoiceBundleError("acknowledgement_required")
            if self._closed or type(profile_id) is not UUID:
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
                recipe_requirement=reference.recipe_requirement,
            )
            payload = encode_clone_voice_bundle(bundle)
            selected = _source_path(destination)
            outcome = cast(
                _WorkerOutcome, await self._run_worker(_publish_sync, selected, payload)
            )
            del payload, bundle, reference
            if outcome.code is not None:
                raise TTSVoiceBundleError(cast(Any, outcome.code)) from None
        finally:
            if call is not None:
                self._calls.discard(call)

    def seal(self) -> None:
        self._closed = True

    async def close(self) -> None:
        """Seal admission and retain settlement of every service-owned call."""

        if self._close_task is None:
            self.seal()
            self._close_task = asyncio.create_task(self._complete_close())
        cancellation, _ = await _await_retained(self._close_task)
        if cancellation is not None:
            raise cancellation

    async def wait_closed(self) -> None:
        """Join definitive close; ``close`` remains idempotent."""

        await self.close()

    async def _complete_close(self) -> None:
        current = asyncio.current_task()
        calls = [call for call in self._calls if call is not current]
        if calls:
            await asyncio.gather(
                *(asyncio.shield(call) for call in calls), return_exceptions=True
            )
        workers = tuple(self._workers)
        if workers:
            await asyncio.gather(
                *(asyncio.shield(worker) for worker in workers), return_exceptions=True
            )
        async with self._session_lock:
            self._sessions.clear()


__all__ = [
    "TTSVoiceBundleHandle",
    "TTSVoiceBundleImportChoice",
    "TTSVoiceBundleImportResult",
    "TTSVoiceBundlePortabilityService",
    "TTSVoiceBundleReview",
]
