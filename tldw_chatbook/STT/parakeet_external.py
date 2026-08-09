"""Descriptor-backed verification for user-owned Parakeet ONNX roots."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable

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


class ExternalParakeetErrorCode(str, Enum):
    """Stable external-root verification failures."""

    UNSUPPORTED = "unsupported_descriptor"
    MISSING = "missing_file"
    IRREGULAR = "irregular_file"
    CHANGED = "changed_file"
    CORRUPT = "corrupt_file"
    CANCELLED = "cancelled"


class ExternalParakeetVerificationError(RuntimeError):
    """Path-safe failure raised while verifying an external Parakeet root."""

    code: ExternalParakeetErrorCode

    def __init__(self, code: ExternalParakeetErrorCode) -> None:
        """Create one stable, path-free verification failure.

        Args:
            code: Machine-readable failure category.
        """

        self.code = code
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


def _fail(code: ExternalParakeetErrorCode) -> None:
    raise ExternalParakeetVerificationError(code) from None


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
            _fail(ExternalParakeetErrorCode.CHANGED)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != before
        ):
            _fail(ExternalParakeetErrorCode.CHANGED)


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
        _fail(ExternalParakeetErrorCode.CHANGED)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or resolved != path
        or _file_identity(metadata) != before
    ):
        _fail(ExternalParakeetErrorCode.CHANGED)
    _require_unchanged_ancestors(ancestors)


class ExternalParakeetVerifier:
    """Verify catalog-declared Parakeet files without parsing or owning them."""

    def verify(
        self,
        descriptor: ArtifactDescriptor,
        directory: Path,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] | None = None,
    ) -> VerifiedExternalParakeet:
        """Verify all and only the files declared by a trusted descriptor.

        Args:
            descriptor: Trusted catalog descriptor for the requested bundle.
            directory: User-owned directory that supplies the declared bytes.
            cancelled: Callback polled between fixed-size hash chunks.
            progress: Optional determinate ``(bytes_done, bytes_total)`` callback.

        Returns:
            The exact descriptor reference and a path-private local snapshot.

        Raises:
            ExternalParakeetVerificationError: The descriptor or root fails a
                stable verification guard, or verification is cancelled.
        """

        if not _is_supported(descriptor):
            _fail(ExternalParakeetErrorCode.UNSUPPORTED)
        root = _validated_root(directory)
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
                    if _file_identity(os.fstat(source.fileno())) != before:
                        _fail(ExternalParakeetErrorCode.CHANGED)
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
                _fail(ExternalParakeetErrorCode.CHANGED)

            verified = (path, before, ancestors)
            _require_unchanged_file(root, verified)
            if digest.hexdigest() != declared.sha256:
                _fail(ExternalParakeetErrorCode.CORRUPT)
            verified_files.append(verified)

        verified_paths = tuple(path for path, _identity, _ancestors in verified_files)
        try:
            snapshot = snapshot_local_source(verified_paths)
        except (LocalSourceChangedError, OSError):
            _fail(ExternalParakeetErrorCode.CHANGED)
        if snapshot.paths != verified_paths or snapshot.identities != tuple(
            (identity[0], identity[1], identity[3], identity[4])
            for _path, identity, _ancestors in verified_files
        ):
            _fail(ExternalParakeetErrorCode.CHANGED)
        for verified in verified_files:
            _require_unchanged_file(root, verified)
        return VerifiedExternalParakeet(
            reference=descriptor.reference,
            directory=root,
            snapshot=snapshot,
        )
