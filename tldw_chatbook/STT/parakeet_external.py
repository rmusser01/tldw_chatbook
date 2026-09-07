"""Descriptor-backed verification for user-owned Parakeet ONNX roots."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, Literal, Mapping

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
)
from tldw_chatbook.Utils.path_validation import validate_path, validate_path_simple

from .executor import (
    LocalSourceChangedError,
    LocalSourceSnapshot,
    snapshot_local_source,
)


_HASH_CHUNK_BYTES = 64 * 1024
_SUPPORTED_MODELS = {
    "nemo-parakeet-tdt-0.6b-v2": "parakeet-v2",
    "nemo-parakeet-tdt-0.6b-v3": "parakeet-v3",
}
_SUPPORTED_PRECISIONS = frozenset({"int8", "f32"})
_CHANGED_DIAGNOSTIC_CODES = frozenset(
    {
        "ancestor_identity",
        "file_path_identity",
        "open_file_identity",
        "post_read_file_identity",
        "file_read",
        "snapshot_identity",
    }
)


class ExternalParakeetErrorCode(str, Enum):
    """Stable external-root verification failures."""

    UNSUPPORTED = "unsupported_descriptor"
    MISSING = "missing_file"
    IRREGULAR = "irregular_file"
    CHANGED = "changed_file"
    CORRUPT = "corrupt_file"
    CANCELLED = "cancelled"


_EXTERNAL_PARAKEET_RECOVERY = {
    ExternalParakeetErrorCode.MISSING: (
        "Required model files are missing. Choose a complete model directory.",
        True,
    ),
    ExternalParakeetErrorCode.IRREGULAR: (
        "Model files must be regular files without links. Choose a safe model directory.",
        True,
    ),
    ExternalParakeetErrorCode.CHANGED: (
        "Model files changed during verification. Wait for file changes to finish, then retry.",
        True,
    ),
    ExternalParakeetErrorCode.CORRUPT: (
        "Model files do not match the curated model. Choose an unmodified model directory.",
        True,
    ),
    ExternalParakeetErrorCode.UNSUPPORTED: (
        "This curated model does not support an external directory.",
        True,
    ),
    ExternalParakeetErrorCode.CANCELLED: (
        "Verification cancelled. The prior source is unchanged.",
        False,
    ),
}


def format_external_parakeet_recovery(
    code: ExternalParakeetErrorCode,
) -> tuple[str, bool]:
    """Return stable path-free recovery copy for one verification failure.

    Args:
        code: Stable verification failure code.

    Returns:
        User-facing recovery message and whether it is an error.
    """

    return _EXTERNAL_PARAKEET_RECOVERY[code]


class ExternalParakeetVerificationError(RuntimeError):
    """Path-safe failure raised while verifying an external Parakeet root."""

    code: ExternalParakeetErrorCode

    def __init__(
        self,
        code: ExternalParakeetErrorCode,
        *,
        diagnostic_code: str | None = None,
    ) -> None:
        """Create one stable, path-free verification failure.

        Args:
            code: Machine-readable failure category.
            diagnostic_code: Optional path-free internal failure phase.
        """

        if (
            diagnostic_code is not None
            and diagnostic_code not in _CHANGED_DIAGNOSTIC_CODES
        ):
            raise ValueError("unsupported external verification diagnostic")
        self.code = code
        self.diagnostic_code = diagnostic_code
        super().__init__(f"External Parakeet verification failed: {code.value}")


@dataclass(frozen=True, repr=False)
class VerifiedExternalParakeet:
    """Path-private verified identity for a user-owned Parakeet root."""

    reference: ArtifactRef
    directory: Path = field(repr=False)
    snapshot: LocalSourceSnapshot = field(repr=False)

    def __repr__(self) -> str:
        return f"VerifiedExternalParakeet(reference={self.reference!r})"


_FileIdentity = tuple[int, int, int, int, int, int]
_DirectoryIdentity = tuple[int, int, int]
_AncestorIdentities = tuple[tuple[Path, _DirectoryIdentity], ...]
_VerifiedFile = tuple[Path, _FileIdentity, _AncestorIdentities]
_Owner = tuple[Literal["configured", "scope"], str]
_CacheKey = tuple[ArtifactRef, Path, str]


def _file_identity(metadata: os.stat_result) -> _FileIdentity:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_identity(metadata: os.stat_result) -> _DirectoryIdentity:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
    )


def _fail(
    code: ExternalParakeetErrorCode,
    *,
    diagnostic_code: str | None = None,
) -> None:
    raise ExternalParakeetVerificationError(
        code,
        diagnostic_code=diagnostic_code,
    ) from None


def _fail_changed(diagnostic_code: str) -> None:
    _fail(
        ExternalParakeetErrorCode.CHANGED,
        diagnostic_code=diagnostic_code,
    )


def _emit_progress(
    progress: Callable[[int, int], None] | None,
    bytes_done: int,
    bytes_total: int,
) -> None:
    if progress is None:
        return
    try:
        progress(bytes_done, bytes_total)
    except Exception:
        return


def _is_supported(descriptor: object) -> bool:
    if type(descriptor) is not ArtifactDescriptor:
        return False
    expected_artifact_id = _SUPPORTED_MODELS.get(descriptor.model_id)
    return (
        descriptor.role is ArtifactRole.ROOT
        and descriptor.format is ArtifactFormat.ONNX
        and descriptor.consumer == "stt"
        and descriptor.model_family == "parakeet"
        and descriptor.precision in _SUPPORTED_PRECISIONS
        and descriptor.reference.artifact_id == expected_artifact_id
    )


def _validated_root(directory: Path) -> Path:
    try:
        selected = Path(directory)
        validated = validate_path_simple(selected, require_exists=True)
        absolute = validated.absolute()
        resolved = validated.resolve(strict=True)
        metadata = absolute.lstat()
    except (OSError, TypeError, ValueError):
        _fail(ExternalParakeetErrorCode.IRREGULAR)
    if absolute != resolved or not stat.S_ISDIR(metadata.st_mode):
        _fail(ExternalParakeetErrorCode.IRREGULAR)
    return absolute


def _declared_path(root: Path, declared: ArtifactFile) -> Path:
    candidate = root.joinpath(*PurePosixPath(declared.path).parts)
    try:
        candidate.relative_to(root)
        resolved_parent = validate_path(
            candidate.parent,
            root,
            redact_paths=True,
            allow_hidden=True,
        )
    except (OSError, TypeError, ValueError):
        _fail(ExternalParakeetErrorCode.IRREGULAR)
    if resolved_parent != candidate.parent:
        _fail(ExternalParakeetErrorCode.IRREGULAR)
    return candidate


def _ancestor_identities(
    root: Path,
    parent: Path,
) -> tuple[tuple[Path, _DirectoryIdentity], ...]:
    try:
        relative = parent.relative_to(root)
    except ValueError:
        _fail(ExternalParakeetErrorCode.IRREGULAR)

    paths = [root]
    current = root
    for component in relative.parts:
        current /= component
        paths.append(current)

    identities: list[tuple[Path, _DirectoryIdentity]] = []
    for path in paths:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            _fail(ExternalParakeetErrorCode.MISSING)
        except OSError:
            _fail(ExternalParakeetErrorCode.IRREGULAR)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(ExternalParakeetErrorCode.IRREGULAR)
        identities.append((path, _directory_identity(metadata)))
    return tuple(identities)


def _require_unchanged_ancestors(
    identities: _AncestorIdentities,
) -> None:
    for path, before in identities:
        try:
            metadata = path.lstat()
        except OSError:
            _fail_changed("ancestor_identity")
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != before
        ):
            _fail_changed("ancestor_identity")


def _require_unchanged_file(
    root: Path,
    verified: _VerifiedFile,
) -> None:
    path, before, ancestors = verified
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail_changed("file_path_identity")
    if not stat.S_ISREG(metadata.st_mode) or resolved != path:
        _fail_changed("file_path_identity")
    if _file_identity(metadata) != before:
        _fail_changed("post_read_file_identity")
    _require_unchanged_ancestors(ancestors)


class ExternalParakeetVerifier:
    """Verify catalog-declared Parakeet files without parsing or owning them."""

    class _Entry:
        def __init__(self) -> None:
            self.stop = threading.Event()
            self.future: Future[VerifiedExternalParakeet] | None = None
            self.waiters: dict[object, Callable[[int, int], None] | None] = {}
            self.owners: set[_Owner] = set()
            self.result: VerifiedExternalParakeet | None = None

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="parakeet-verify",
        )
        self._lock = threading.Lock()
        self._entries: dict[_CacheKey, ExternalParakeetVerifier._Entry] = {}
        self._owner_tokens: dict[_Owner, object] = {}
        self._fallback_stops: set[threading.Event] = set()
        self._closed = False

    def verify(
        self,
        descriptor: ArtifactDescriptor,
        directory: Path,
        *,
        owner: _Owner | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] | None = None,
    ) -> VerifiedExternalParakeet:
        """Verify all and only the files declared by a trusted descriptor.

        Args:
            descriptor: Trusted catalog descriptor for the requested bundle.
            directory: User-owned directory that supplies the declared bytes.
            owner: Optional configured-selection or live-scope cache owner.
            cancelled: Callback polled on this calling waiter thread.
            progress: Optional determinate ``(bytes_done, bytes_total)`` callback.

        Returns:
            The exact descriptor reference and a path-private local snapshot.

        Raises:
            ExternalParakeetVerificationError: The descriptor or root fails a
                stable verification guard, or verification is cancelled.
        """

        if not _is_supported(descriptor):
            _fail(ExternalParakeetErrorCode.UNSUPPORTED)
        self._validate_owner(owner)
        if cancelled():
            _fail(ExternalParakeetErrorCode.CANCELLED)
        root = _validated_root(directory)
        cache_key = self._cache_key(descriptor, root)
        if cache_key is None:
            return self._verify_without_cache(
                descriptor,
                root,
                cancelled=cancelled,
                progress=progress,
            )

        waiter = object()
        with self._lock:
            if self._closed:
                _fail(ExternalParakeetErrorCode.CANCELLED)
            owner_token = (
                self._owner_tokens.setdefault(owner, object())
                if owner is not None
                else None
            )
            entry = self._entries.get(cache_key)
            if entry is not None and entry.result is not None:
                self._retain_owner(cache_key, entry, owner, owner_token)
                return entry.result
            if entry is None:
                entry = self._Entry()
                self._entries[cache_key] = entry
                entry.waiters[waiter] = progress
                entry.future = self._executor.submit(
                    self._verify_uncached,
                    descriptor,
                    root,
                    entry.stop.is_set,
                    lambda done, total: self._fanout(entry, done, total),
                )
            else:
                entry.waiters[waiter] = progress
            future = entry.future
        assert future is not None

        try:
            while True:
                if cancelled():
                    self._drop_waiter(cache_key, entry, waiter)
                    _fail(ExternalParakeetErrorCode.CANCELLED)
                try:
                    verified = future.result(timeout=0.01)
                    if cancelled():
                        self._drop_waiter(cache_key, entry, waiter)
                        _fail(ExternalParakeetErrorCode.CANCELLED)
                    break
                except TimeoutError:
                    continue
                except CancelledError:
                    _fail(ExternalParakeetErrorCode.CANCELLED)
        except BaseException:
            self._drop_waiter(cache_key, entry, waiter)
            raise

        with self._lock:
            if self._entries.get(cache_key) is entry:
                entry.waiters.pop(waiter, None)
                entry.result = verified
                self._retain_owner(cache_key, entry, owner, owner_token)
                if not entry.waiters and not entry.owners:
                    self._entries.pop(cache_key, None)
        return verified

    @staticmethod
    def _validate_owner(owner: _Owner | None) -> None:
        if owner is None:
            return
        if (
            type(owner) is not tuple
            or len(owner) != 2
            or owner[0] not in ("configured", "scope")
            or type(owner[1]) is not str
            or not owner[1]
        ):
            raise ValueError("owner must be a configured or scope identity")

    @staticmethod
    def _cache_key(
        descriptor: ArtifactDescriptor,
        root: Path,
    ) -> _CacheKey | None:
        paths = tuple(_declared_path(root, declared) for declared in descriptor.files)
        try:
            identities = tuple(
                (
                    int(metadata.st_dev),
                    int(metadata.st_ino),
                    int(metadata.st_size),
                    int(metadata.st_mtime_ns),
                )
                for path in paths
                for metadata in (path.lstat(),)
                if stat.S_ISREG(metadata.st_mode)
            )
        except OSError:
            return None
        if len(identities) != len(paths):
            return None
        digest = hashlib.sha256()
        for index, identity in enumerate(identities):
            digest.update(f"{index}:{identity!r};".encode("ascii"))
        return descriptor.reference, root, digest.hexdigest()

    def _verify_without_cache(
        self,
        descriptor: ArtifactDescriptor,
        root: Path,
        *,
        cancelled: Callable[[], bool],
        progress: Callable[[int, int], None] | None,
    ) -> VerifiedExternalParakeet:
        stop = threading.Event()
        with self._lock:
            if self._closed:
                _fail(ExternalParakeetErrorCode.CANCELLED)
            self._fallback_stops.add(stop)
            try:
                future = self._executor.submit(
                    self._verify_uncached,
                    descriptor,
                    root,
                    stop.is_set,
                    progress,
                )
            except BaseException:
                self._fallback_stops.discard(stop)
                raise
        try:
            while True:
                if cancelled():
                    stop.set()
                    _fail(ExternalParakeetErrorCode.CANCELLED)
                try:
                    verified = future.result(timeout=0.01)
                    if cancelled():
                        stop.set()
                        _fail(ExternalParakeetErrorCode.CANCELLED)
                    return verified
                except TimeoutError:
                    continue
                except CancelledError:
                    _fail(ExternalParakeetErrorCode.CANCELLED)
        finally:
            with self._lock:
                self._fallback_stops.discard(stop)

    def _fanout(self, entry: _Entry, done: int, total: int) -> None:
        with self._lock:
            callbacks = tuple(entry.waiters.values())
        for callback in callbacks:
            _emit_progress(callback, done, total)

    def _drop_waiter(
        self,
        cache_key: _CacheKey,
        entry: _Entry,
        waiter: object,
    ) -> None:
        with self._lock:
            if self._entries.get(cache_key) is not entry:
                return
            entry.waiters.pop(waiter, None)
            if not entry.waiters and entry.result is None:
                entry.stop.set()
                self._entries.pop(cache_key, None)

    def _retain_owner(
        self,
        cache_key: _CacheKey,
        entry: _Entry,
        owner: _Owner | None,
        owner_token: object | None,
    ) -> None:
        if owner is None or self._owner_tokens.get(owner) is not owner_token:
            return
        entry.owners.add(owner)
        if owner[0] == "configured":
            for other_key, other in self._entries.items():
                if other_key != cache_key:
                    other.owners.discard(owner)
            self._prune_unowned()

    def set_configured_owners(
        self,
        owners: Mapping[str, tuple[ArtifactRef, Path]],
    ) -> None:
        """Retain cached results matching current persistent selections."""

        desired = {
            name: (reference, Path(directory).absolute())
            for name, (reference, directory) in owners.items()
            if type(name) is str and name
        }
        with self._lock:
            for owner in tuple(self._owner_tokens):
                if owner[0] == "configured":
                    self._owner_tokens.pop(owner, None)
            for entry in self._entries.values():
                entry.owners = {
                    owner for owner in entry.owners if owner[0] != "configured"
                }
            for entry in self._entries.values():
                result = entry.result
                if result is None:
                    continue
                for name, (reference, directory) in desired.items():
                    if result.reference == reference and result.directory == directory:
                        owner: _Owner = ("configured", name)
                        self._owner_tokens.setdefault(owner, object())
                        entry.owners.add(owner)
            self._prune_unowned()

    def release_scope(self, scope_id: str) -> None:
        """Release all verification retention owned by one live job scope."""

        if type(scope_id) is not str or not scope_id:
            raise ValueError("scope_id must be a non-empty string")
        with self._lock:
            owner: _Owner = ("scope", scope_id)
            self._owner_tokens.pop(owner, None)
            for entry in self._entries.values():
                entry.owners.discard(owner)
            self._prune_unowned()

    def _prune_unowned(self) -> None:
        for key, entry in tuple(self._entries.items()):
            if entry.result is not None and not entry.owners and not entry.waiters:
                self._entries.pop(key, None)

    def close(self) -> None:
        """Cancel owned work and release process-lifetime cache state."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._owner_tokens.clear()
            for entry in entries:
                entry.stop.set()
            for stop in self._fallback_stops:
                stop.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _verify_uncached(
        descriptor: ArtifactDescriptor,
        root: Path,
        cancelled: Callable[[], bool],
        progress: Callable[[int, int], None] | None,
    ) -> VerifiedExternalParakeet:
        bytes_total = sum(item.size_bytes for item in descriptor.files)
        bytes_done = 0
        verified_files: list[_VerifiedFile] = []
        _emit_progress(progress, bytes_done, bytes_total)

        for declared in descriptor.files:
            if cancelled():
                _fail(ExternalParakeetErrorCode.CANCELLED)
            path = _declared_path(root, declared)
            ancestors = _ancestor_identities(root, path.parent)
            try:
                before_metadata = path.lstat()
            except FileNotFoundError:
                _fail(ExternalParakeetErrorCode.MISSING)
            except OSError:
                _fail(ExternalParakeetErrorCode.IRREGULAR)
            if not stat.S_ISREG(before_metadata.st_mode):
                _fail(ExternalParakeetErrorCode.IRREGULAR)
            before = _file_identity(before_metadata)
            if before_metadata.st_size != declared.size_bytes:
                _fail(ExternalParakeetErrorCode.CORRUPT)
            digest = hashlib.sha256()
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                descriptor_fd = os.open(path, flags)
                with os.fdopen(descriptor_fd, "rb") as source:
                    opened = _file_identity(os.fstat(source.fileno()))
                    # Windows path stat reports birth time as ctime while fstat
                    # reports change time; the other fields still identify the file.
                    if opened[:-1] != before[:-1]:
                        _fail_changed("open_file_identity")
                    while True:
                        if cancelled():
                            _fail(ExternalParakeetErrorCode.CANCELLED)
                        chunk = source.read(_HASH_CHUNK_BYTES)
                        if not chunk:
                            break
                        digest.update(chunk)
                        bytes_done += len(chunk)
                        _emit_progress(progress, bytes_done, bytes_total)
            except OSError:
                _fail_changed("file_read")

            verified = (path, before, ancestors)
            _require_unchanged_file(root, verified)
            if digest.hexdigest() != declared.sha256:
                _fail(ExternalParakeetErrorCode.CORRUPT)
            verified_files.append(verified)

        verified_paths = tuple(path for path, _identity, _ancestors in verified_files)
        try:
            snapshot = snapshot_local_source(verified_paths)
        except (LocalSourceChangedError, OSError):
            _fail_changed("snapshot_identity")
        if snapshot.paths != verified_paths or snapshot.identities != tuple(
            (identity[0], identity[1], identity[3], identity[4])
            for _path, identity, _ancestors in verified_files
        ):
            _fail_changed("snapshot_identity")
        for verified in verified_files:
            _require_unchanged_file(root, verified)
        return VerifiedExternalParakeet(
            reference=descriptor.reference,
            directory=root,
            snapshot=snapshot,
        )
