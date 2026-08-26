"""Bounded read-only local package discovery for guided audio.cpp setup."""

from __future__ import annotations

import asyncio
import math
import os
import stat
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import monotonic as _monotonic
from typing import Any

from tldw_chatbook.Utils.path_validation import validate_path_simple

from .audio_cpp_guided_config import AudioCppManagedArtifactIdentity
from .audio_cpp_recipes import (
    AUDIO_CPP_RECIPE_REGISTRY,
    AudioCppFileKind,
    AudioCppFileSignal,
    AudioCppMatchResult,
    AudioCppMatchState,
    AudioCppPackageDescription,
    AudioCppPackageFileEvidence,
    AudioCppRecipeRegistry,
    _managed_artifact_matches_recipe,
)
from .windows_artifact_fs import (
    OS_WINDOWS_ARTIFACT_FILESYSTEM,
    WindowsArtifactError,
    WindowsArtifactFilesystem,
    WindowsFileIdentity,
    windows_audio_cpp_platform_supported,
)


_scandir = os.scandir
_windows_artifact_filesystem: WindowsArtifactFilesystem | None = (
    OS_WINDOWS_ARTIFACT_FILESYSTEM if windows_audio_cpp_platform_supported() else None
)
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SCAN_CLEANUP_OWNER = "_audio_cpp_scan_cleanup_owner"


class AudioCppPackageScanError(ValueError):
    """Stable path-independent error for an unusable selected root."""

    __slots__ = ("_cleanup_owner",)

    def __init__(self, message: str, *, cleanup_owner: object | None = None) -> None:
        self._cleanup_owner = cleanup_owner
        super().__init__(message)

    def take_cleanup_owner(self) -> object | None:
        """Transfer one exact retained Windows handle owner once."""

        owner = self._cleanup_owner
        self._cleanup_owner = None
        return owner


def take_audio_cpp_scan_cleanup_owner(error: BaseException) -> object | None:
    """Transfer one retained Windows scan handle from an error or cancellation."""

    if isinstance(error, AudioCppPackageScanError):
        return AudioCppPackageScanError.take_cleanup_owner(error)
    owner = getattr(error, _SCAN_CLEANUP_OWNER, None)
    if owner is not None:
        setattr(error, _SCAN_CLEANUP_OWNER, None)
    return owner


class AudioCppScanOutcome(StrEnum):
    """Overall scan completeness independent of individual recipe matches."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    PERMISSION_LIMITED = "permission_limited"
    CANCELLED = "cancelled"


class AudioCppScanLimit(StrEnum):
    """Finite scanner budget that stopped or downgraded discovery."""

    DEPTH = "depth"
    ENTRIES = "entries"
    CANDIDATE_ROOTS = "candidate_roots"
    RESULTS = "results"
    METADATA_PER_FILE = "metadata_per_file"
    METADATA_TOTAL = "metadata_total"
    ENTRY_TIME = "entry_time"
    TOTAL_TIME = "total_time"


class AudioCppScanIssueCode(StrEnum):
    """Sanitized per-entry issue code."""

    PERMISSION_DENIED = "permission_denied"
    UNREADABLE = "unreadable"
    SOURCE_CHANGED = "source_changed"
    SYMLINK_SKIPPED = "symlink_skipped"
    REPARSE_SKIPPED = "reparse_skipped"
    SPECIAL_FILE_SKIPPED = "special_file_skipped"
    NO_FOLLOW_UNAVAILABLE = "no_follow_unavailable"


def _positive_integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"audio.cpp scanner {label} is invalid")
    return value


def _positive_seconds(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"audio.cpp scanner {label} is invalid")
    return float(value)


@dataclass(frozen=True, slots=True)
class AudioCppScanLimits:
    """All finite work and retained-detail budgets for one explicit root."""

    max_depth: int = 8
    max_entries: int = 4096
    max_candidate_roots: int = 128
    max_results: int = 64
    max_metadata_bytes_per_file: int = 64 * 1024
    max_metadata_bytes_total: int = 1024 * 1024
    max_entry_seconds: float = 0.25
    max_total_seconds: float = 5.0
    max_issues: int = 16
    max_unknown_names: int = 16

    def __post_init__(self) -> None:
        _positive_integer(self.max_depth, "max_depth", allow_zero=True)
        for name in (
            "max_entries",
            "max_candidate_roots",
            "max_results",
            "max_metadata_bytes_per_file",
            "max_metadata_bytes_total",
            "max_issues",
            "max_unknown_names",
        ):
            _positive_integer(getattr(self, name), name)
        _positive_seconds(self.max_entry_seconds, "max_entry_seconds")
        _positive_seconds(self.max_total_seconds, "max_total_seconds")
        if self.max_metadata_bytes_per_file > self.max_metadata_bytes_total:
            raise ValueError(
                "audio.cpp scanner per-file metadata limit exceeds total limit"
            )


@dataclass(frozen=True, slots=True)
class AudioCppScanIssue:
    """One path-sanitized retained scanner issue."""

    code: AudioCppScanIssueCode
    safe_name: str


@dataclass(frozen=True, slots=True)
class AudioCppPackageDiscovery:
    """One deduplicated candidate root and its pure recipe match."""

    description: AudioCppPackageDescription
    match: AudioCppMatchResult


@dataclass(frozen=True, slots=True)
class AudioCppPackageScanResult:
    """Bounded immutable result for one explicit scan request."""

    outcome: AudioCppScanOutcome
    request_revision: int
    selected_root_name: str
    canonical_root: str = field(repr=False)
    canonical_root_identity: str
    root_was_symlink: bool
    discoveries: tuple[AudioCppPackageDiscovery, ...]
    limits_reached: tuple[AudioCppScanLimit, ...]
    issues: tuple[AudioCppScanIssue, ...]
    issues_truncated: bool
    unknown_names: tuple[str, ...]
    unknown_names_truncated: bool
    visited_entries: int
    metadata_bytes_read: int


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character
        for character in value
        if not unicodedata.category(character).startswith("C")
    ).strip()
    return (cleaned or "Selected package")[:128]


def _filesystem_identity(info: os.stat_result) -> str:
    encoded = ":".join(
        str(value)
        for value in (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    )
    return sha256(encoded.encode("ascii")).hexdigest()


def _windows_filesystem_identity(identity: WindowsFileIdentity) -> str:
    encoded = b"\0".join(
        (
            identity.volume_serial_number.to_bytes(8, "little", signed=False),
            identity.file_id,
            identity.kind.encode("ascii"),
            identity.reparse_tag.to_bytes(4, "little", signed=False),
        )
    )
    return sha256(encoded).hexdigest()


def _is_reparse_or_symlink(info: Any) -> bool:
    """Return whether a no-follow stat identifies a link/reparse object."""
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    )


def _same_source(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_mode,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_mode,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _read_only_no_follow_flags() -> int | None:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        return None
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | no_follow
    )


def _required_metadata_bytes(kind: AudioCppFileKind) -> int:
    if kind is AudioCppFileKind.GGUF:
        return 8
    if kind is AudioCppFileKind.SAFETENSORS:
        return 9
    if kind is AudioCppFileKind.JSON:
        return 1
    return 0


def _metadata_is_valid(kind: AudioCppFileKind, data: bytes, size_bytes: int) -> bool:
    if kind is AudioCppFileKind.GGUF:
        return (
            len(data) == 8
            and data[:4] == b"GGUF"
            and int.from_bytes(data[4:8], "little") == 3
        )
    if kind is AudioCppFileKind.SAFETENSORS:
        if len(data) != 9:
            return False
        header_size = int.from_bytes(data[:8], "little")
        return 2 <= header_size <= size_bytes - 8 and data[8:9] == b"{"
    if kind is AudioCppFileKind.JSON:
        return data in {b"{", b"["}
    return True


@dataclass(slots=True)
class _ScanState:
    limits: AudioCppScanLimits
    started_at: float
    cancellation_event: threading.Event
    visited_entries: int = 0
    metadata_bytes_read: int = 0
    limits_reached: set[AudioCppScanLimit] = field(default_factory=set)
    issues: list[AudioCppScanIssue] = field(default_factory=list)
    issues_truncated: bool = False
    unknown_names: list[str] = field(default_factory=list)
    unknown_names_truncated: bool = False
    permission_limited: bool = False
    permission_paths: set[tuple[str, ...]] = field(default_factory=set)
    incomplete_paths: set[tuple[str, ...]] = field(default_factory=set)
    stopped: bool = False

    def add_limit(self, limit: AudioCppScanLimit) -> None:
        self.limits_reached.add(limit)

    def add_issue(self, code: AudioCppScanIssueCode, name: str) -> None:
        if len(self.issues) >= self.limits.max_issues:
            self.issues_truncated = True
            return
        self.issues.append(AudioCppScanIssue(code, _safe_name(name)))

    def add_permission(self, path: tuple[str, ...], name: str) -> None:
        self.permission_limited = True
        self.permission_paths.add(path)
        self.add_issue(AudioCppScanIssueCode.PERMISSION_DENIED, name)

    def add_incomplete(
        self,
        code: AudioCppScanIssueCode,
        path: tuple[str, ...],
        name: str,
    ) -> None:
        self.incomplete_paths.add(path)
        self.add_issue(code, name)

    def add_unknown(self, name: str) -> None:
        if len(self.unknown_names) >= self.limits.max_unknown_names:
            self.unknown_names_truncated = True
            return
        self.unknown_names.append(_safe_name(name))

    def cancellation_or_total_limit(self) -> bool:
        if self.cancellation_event.is_set():
            self.stopped = True
            return True
        if _monotonic() - self.started_at > self.limits.max_total_seconds:
            self.add_limit(AudioCppScanLimit.TOTAL_TIME)
            self.stopped = True
            return True
        return False


def _close_windows_owner(
    owner: Any,
    *,
    state: _ScanState | None,
    relative_parts: tuple[str, ...],
    safe_name: str,
) -> None:
    close = getattr(owner, "close", None)
    if not callable(close):
        raise AudioCppPackageScanError(
            "Windows audio.cpp package handle cleanup did not complete.",
            cleanup_owner=owner,
        )
    retained = owner
    failed_once = False
    for _attempt in range(2):
        try:
            close = getattr(retained, "close")
            close()
            if failed_once and state is not None:
                state.add_incomplete(
                    AudioCppScanIssueCode.UNREADABLE,
                    relative_parts,
                    safe_name,
                )
            return
        except WindowsArtifactError as error:
            failed_once = True
            retained = error.take_cleanup_owner() or retained
    raise AudioCppPackageScanError(
        "Windows audio.cpp package handle cleanup did not complete.",
        cleanup_owner=retained,
    ) from None


def _pin_windows_directory_identity(
    path: Path,
    *,
    state: _ScanState | None = None,
    relative_parts: tuple[str, ...] = (),
) -> str:
    filesystem = _windows_artifact_filesystem
    if filesystem is None:
        raise AudioCppPackageScanError(
            "Windows audio.cpp package handles are unavailable."
        )
    try:
        owner = filesystem.pin_directory_no_reparse(path)
    except WindowsArtifactError as error:
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is unavailable.",
            cleanup_owner=error.take_cleanup_owner(),
        ) from None
    except OSError:
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is unavailable."
        ) from None
    try:
        identity = owner.identity
        if identity.kind != "directory" or identity.reparse_tag:
            raise AudioCppPackageScanError(
                "Selected audio.cpp package root is unavailable."
            )
        return _windows_filesystem_identity(identity)
    finally:
        _close_windows_owner(
            owner,
            state=state,
            relative_parts=relative_parts,
            safe_name=path.name,
        )


def _inspect_file_windows(
    path: Path,
    relative_parts: tuple[str, ...],
    info: os.stat_result,
    signal: AudioCppFileSignal,
    state: _ScanState,
) -> AudioCppPackageFileEvidence:
    filesystem = _windows_artifact_filesystem
    if filesystem is None:
        raise AssertionError("Windows artifact filesystem is not selected")
    required_bytes = _required_metadata_bytes(signal.kind)
    readable = True
    valid = info.st_size >= signal.minimum_size_bytes
    identity = _filesystem_identity(info)
    if required_bytes > state.limits.max_metadata_bytes_per_file:
        state.add_limit(AudioCppScanLimit.METADATA_PER_FILE)
        valid = False
        required_bytes = 0
    elif (
        state.metadata_bytes_read + required_bytes
        > state.limits.max_metadata_bytes_total
    ):
        state.add_limit(AudioCppScanLimit.METADATA_TOTAL)
        valid = False
        required_bytes = 0

    owner: Any = None
    try:
        owner = filesystem.open_file_no_reparse(path)
        opened_identity = owner.identity
        opened_info = os.stat(path, follow_symlinks=False)
        if (
            opened_identity.kind != "file"
            or opened_identity.reparse_tag
            or not stat.S_ISREG(opened_info.st_mode)
            or not _same_source(info, opened_info)
        ):
            readable = False
            valid = False
            state.add_incomplete(
                AudioCppScanIssueCode.SOURCE_CHANGED,
                relative_parts,
                path.name,
            )
        else:
            if required_bytes:
                data = owner.read(required_bytes)
                state.metadata_bytes_read += len(data)
                valid = valid and _metadata_is_valid(
                    signal.kind,
                    data,
                    opened_info.st_size,
                )
            identity = _windows_filesystem_identity(opened_identity)
            info = opened_info
    except PermissionError:
        readable = False
        valid = False
        state.add_permission(relative_parts, path.name)
    except WindowsArtifactError as error:
        cleanup = error.take_cleanup_owner()
        if cleanup is not None:
            raise AudioCppPackageScanError(
                "Windows audio.cpp package handle cleanup did not complete.",
                cleanup_owner=cleanup,
            ) from None
        readable = False
        valid = False
        state.add_incomplete(
            AudioCppScanIssueCode.UNREADABLE,
            relative_parts,
            path.name,
        )
    except OSError:
        readable = False
        valid = False
        state.add_incomplete(
            AudioCppScanIssueCode.UNREADABLE,
            relative_parts,
            path.name,
        )
    finally:
        if owner is not None:
            _close_windows_owner(
                owner,
                state=state,
                relative_parts=relative_parts,
                safe_name=path.name,
            )

    return AudioCppPackageFileEvidence(
        relative_path=signal.relative_path,
        size_bytes=info.st_size,
        identity=identity,
        readable=readable,
        metadata_valid=valid,
    )


def _inspect_file(
    path: Path,
    relative_parts: tuple[str, ...],
    info: os.stat_result,
    signal: AudioCppFileSignal,
    state: _ScanState,
) -> AudioCppPackageFileEvidence:
    if _windows_artifact_filesystem is not None:
        return _inspect_file_windows(path, relative_parts, info, signal, state)
    required_bytes = _required_metadata_bytes(signal.kind)
    readable = True
    valid = info.st_size >= signal.minimum_size_bytes
    identity = _filesystem_identity(info)
    open_flags = _read_only_no_follow_flags()
    if open_flags is None:
        state.add_incomplete(
            AudioCppScanIssueCode.NO_FOLLOW_UNAVAILABLE,
            relative_parts,
            path.name,
        )
        return AudioCppPackageFileEvidence(
            relative_path=signal.relative_path,
            size_bytes=info.st_size,
            identity=identity,
            readable=False,
            metadata_valid=False,
        )
    if required_bytes > state.limits.max_metadata_bytes_per_file:
        state.add_limit(AudioCppScanLimit.METADATA_PER_FILE)
        valid = False
        required_bytes = 0
    elif (
        state.metadata_bytes_read + required_bytes
        > state.limits.max_metadata_bytes_total
    ):
        state.add_limit(AudioCppScanLimit.METADATA_TOTAL)
        valid = False
        required_bytes = 0

    descriptor: int | None = None
    try:
        descriptor = os.open(path, open_flags)
        opened_info = os.fstat(descriptor)
        if not stat.S_ISREG(opened_info.st_mode) or not _same_source(info, opened_info):
            readable = False
            valid = False
            state.add_incomplete(
                AudioCppScanIssueCode.SOURCE_CHANGED,
                relative_parts,
                path.name,
            )
        else:
            if required_bytes:
                data = os.read(descriptor, required_bytes)
                state.metadata_bytes_read += len(data)
                valid = valid and _metadata_is_valid(
                    signal.kind,
                    data,
                    opened_info.st_size,
                )
            identity = _filesystem_identity(opened_info)
    except PermissionError:
        readable = False
        valid = False
        state.add_permission(relative_parts, path.name)
    except OSError:
        readable = False
        valid = False
        state.add_incomplete(
            AudioCppScanIssueCode.UNREADABLE,
            relative_parts,
            path.name,
        )
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                readable = False
                valid = False
                state.add_incomplete(
                    AudioCppScanIssueCode.UNREADABLE,
                    relative_parts,
                    path.name,
                )

    return AudioCppPackageFileEvidence(
        relative_path=signal.relative_path,
        size_bytes=info.st_size,
        identity=identity,
        readable=readable,
        metadata_valid=valid,
    )


def _signal_index(
    registry: AudioCppRecipeRegistry,
) -> tuple[tuple[tuple[str, ...], AudioCppFileSignal], ...]:
    unique: dict[str, AudioCppFileSignal] = {}
    for recipe in registry.recipes:
        for signal in recipe.required_files:
            unique.setdefault(signal.relative_path, signal)
    return tuple(
        sorted(
            (
                (PurePosixPath(signal.relative_path).parts, signal)
                for signal in unique.values()
            ),
            key=lambda item: (item[1].relative_path, item[1].kind.value),
        )
    )


def _path_failure_affects_candidate(
    affected_paths: set[tuple[str, ...]],
    candidate_root: tuple[str, ...],
    signals: tuple[tuple[tuple[str, ...], AudioCppFileSignal], ...],
) -> bool:
    for affected_path in affected_paths:
        if affected_path[: len(candidate_root)] != candidate_root:
            continue
        relative_failure = affected_path[len(candidate_root) :]
        if not relative_failure or any(
            signal_parts[: len(relative_failure)] == relative_failure
            for signal_parts, _signal in signals
        ):
            return True
    return False


def _close_directory_iterator(
    iterator: object,
    state: _ScanState,
    directory_name: str,
) -> None:
    close = getattr(iterator, "close", None)
    if close is None:
        return
    try:
        close()
    except OSError:
        state.add_issue(AudioCppScanIssueCode.UNREADABLE, directory_name)


def _resolve_selected_root(
    root: str | os.PathLike[str],
    *,
    allow_root_symlink: bool,
) -> tuple[Path, os.stat_result, bool]:
    try:
        selected = validate_path_simple(
            root,
            require_exists=False,
            probe_existing=False,
        )
        selected_info = os.lstat(selected)
    except (OSError, TypeError, ValueError):
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is unavailable."
        ) from None
    root_is_link = _is_reparse_or_symlink(selected_info)
    if root_is_link and not allow_root_symlink:
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is a symlink; review its target first."
        )
    try:
        canonical = selected.resolve(strict=True)
        canonical_info = os.stat(canonical, follow_symlinks=False)
    except OSError:
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is unavailable."
        ) from None
    root_is_link = root_is_link or (
        os.path.normcase(os.path.abspath(os.fspath(selected)))
        != os.path.normcase(os.path.abspath(os.fspath(canonical)))
    )
    if root_is_link and not allow_root_symlink:
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root is a symlink; review its target first."
        )
    if _is_reparse_or_symlink(canonical_info) or not stat.S_ISDIR(
        canonical_info.st_mode
    ):
        raise AudioCppPackageScanError(
            "Selected audio.cpp package root must be a readable directory."
        )
    return canonical, canonical_info, root_is_link


def _cancelled_result(
    *,
    request_revision: int,
    canonical_root: Path,
    root_info: os.stat_result,
    root_was_symlink: bool,
    root_identity: str | None = None,
) -> AudioCppPackageScanResult:
    return AudioCppPackageScanResult(
        outcome=AudioCppScanOutcome.CANCELLED,
        request_revision=request_revision,
        selected_root_name=_safe_name(canonical_root.name),
        canonical_root=str(canonical_root),
        canonical_root_identity=root_identity or _filesystem_identity(root_info),
        root_was_symlink=root_was_symlink,
        discoveries=(),
        limits_reached=(),
        issues=(),
        issues_truncated=False,
        unknown_names=(),
        unknown_names_truncated=False,
        visited_entries=0,
        metadata_bytes_read=0,
    )


def _managed_expectation(
    expected_managed_artifact: AudioCppManagedArtifactIdentity | None,
    expected_canonical_root: str | os.PathLike[str] | None,
) -> tuple[AudioCppManagedArtifactIdentity, str] | None:
    values = (
        expected_managed_artifact,
        expected_canonical_root,
    )
    if not any(value is not None for value in values):
        return None
    if expected_managed_artifact is None or expected_canonical_root is None:
        raise ValueError("audio.cpp managed scan expectation must be complete")
    if type(expected_managed_artifact) is not AudioCppManagedArtifactIdentity:
        raise TypeError("audio.cpp managed artifact identity is required")
    invalid_root = False
    try:
        root = os.fspath(expected_canonical_root)
    except (OSError, TypeError, ValueError):
        invalid_root = True
        root = ""
    if invalid_root:
        raise TypeError("audio.cpp managed canonical root is required") from None
    if type(root) is not str:
        raise TypeError("audio.cpp managed canonical root is required")
    return expected_managed_artifact, root


def _require_managed_exact_result(
    result: AudioCppPackageScanResult,
    *,
    root: str | os.PathLike[str],
    allow_root_symlink: bool,
    expectation: tuple[AudioCppManagedArtifactIdentity, str, str] | None,
) -> AudioCppPackageScanResult:
    if expectation is None or result.outcome is AudioCppScanOutcome.CANCELLED:
        return result
    managed_artifact, expected_root, expected_root_identity = expectation
    try:
        final_root, final_info, final_was_symlink = _resolve_selected_root(
            root,
            allow_root_symlink=allow_root_symlink,
        )
    except AudioCppPackageScanError:
        raise AudioCppPackageScanError(
            "Managed audio.cpp package no longer matches its installed identity."
        ) from None
    candidates = tuple(
        candidate
        for discovery in result.discoveries
        for candidate in discovery.match.candidates
    )
    exact_discovery = (
        len(result.discoveries) == 1
        and result.discoveries[0].match.state is AudioCppMatchState.EXACT
    )
    candidate_matches = False
    if len(candidates) == 1:
        candidate = candidates[0]
        if _managed_artifact_matches_recipe(candidate.recipe, managed_artifact):
            candidate_matches = bool(
                candidate.canonical_root == expected_root
                and candidate.canonical_root_identity == expected_root_identity
            )
    if not (
        result.outcome is AudioCppScanOutcome.COMPLETE
        and exact_discovery
        and candidate_matches
        and not result.root_was_symlink
        and not final_was_symlink
        and result.canonical_root == expected_root == str(final_root)
        and result.canonical_root_identity
        == expected_root_identity
        == (
            _pin_windows_directory_identity(final_root)
            if _windows_artifact_filesystem is not None
            else _filesystem_identity(final_info)
        )
    ):
        raise AudioCppPackageScanError(
            "Managed audio.cpp package no longer matches its installed identity."
        )
    return result


def scan_audio_cpp_package_root(
    root: str | os.PathLike[str],
    *,
    registry: AudioCppRecipeRegistry = AUDIO_CPP_RECIPE_REGISTRY,
    limits: AudioCppScanLimits | None = None,
    cancellation_event: threading.Event | None = None,
    allow_root_symlink: bool = False,
    request_revision: int = 0,
    expected_managed_artifact: AudioCppManagedArtifactIdentity | None = None,
    expected_canonical_root: str | os.PathLike[str] | None = None,
) -> AudioCppPackageScanResult:
    """Inspect exactly one user-selected root with finite no-follow budgets.

    Args:
        root: User-selected package directory. The scanner never searches a
            parent or sibling automatically.
        registry: Sealed recipe registry used for exact matching.
        limits: Optional finite scanner budgets; defaults are used when absent.
        cancellation_event: Optional cross-thread cancellation signal.
        allow_root_symlink: Whether to resolve a disclosed top-level symlink.
            Nested links and reparse points are always skipped.
        request_revision: Non-negative caller revision copied into the result.
        expected_managed_artifact: Optional exact managed-store identity.
        expected_canonical_root: Canonical managed root, present with identity.

    Returns:
        One immutable, bounded scan result with sanitized retained evidence.

    Raises:
        AudioCppPackageScanError: If the selected root is invalid or unusable.
        TypeError: If the registry, limits, or cancellation signal is invalid.
        ValueError: If the request revision is invalid.
    """
    if not isinstance(registry, AudioCppRecipeRegistry):
        raise TypeError("audio.cpp recipe registry is required")
    managed_contract = _managed_expectation(
        expected_managed_artifact,
        expected_canonical_root,
    )
    if type(request_revision) is not int or request_revision < 0:
        raise ValueError("audio.cpp scan request revision is invalid")
    active_limits = AudioCppScanLimits() if limits is None else limits
    if not isinstance(active_limits, AudioCppScanLimits):
        raise TypeError("audio.cpp scan limits are required")
    cancellation = cancellation_event or threading.Event()
    if not isinstance(cancellation, threading.Event):
        raise TypeError("audio.cpp scan cancellation event is invalid")
    canonical_root, root_info, root_was_symlink = _resolve_selected_root(
        root,
        allow_root_symlink=allow_root_symlink,
    )
    selected_root_identity = (
        _pin_windows_directory_identity(canonical_root)
        if _windows_artifact_filesystem is not None
        else _filesystem_identity(root_info)
    )
    if managed_contract is not None and (
        str(canonical_root) != managed_contract[1] or root_was_symlink
    ):
        raise AudioCppPackageScanError(
            "Managed audio.cpp package no longer matches its installed identity."
        )
    managed_expectation = (
        None
        if managed_contract is None
        else (*managed_contract, selected_root_identity)
    )
    if cancellation.is_set():
        return _cancelled_result(
            request_revision=request_revision,
            canonical_root=canonical_root,
            root_info=root_info,
            root_was_symlink=root_was_symlink,
            root_identity=selected_root_identity,
        )

    state = _ScanState(active_limits, _monotonic(), cancellation)
    signals = _signal_index(registry)
    queue: deque[tuple[Path, tuple[str, ...], int]] = deque(((canonical_root, (), 0),))
    directory_identities: dict[tuple[str, ...], str] = {(): selected_root_identity}
    candidate_evidence: dict[
        tuple[str, ...],
        dict[str, AudioCppPackageFileEvidence],
    ] = {}
    inspection_cache: dict[
        tuple[str, AudioCppFileKind, int],
        AudioCppPackageFileEvidence,
    ] = {}

    while queue and not state.stopped:
        directory, relative_directory, depth = queue.popleft()
        if state.cancellation_or_total_limit():
            break
        iterator: object | None = None
        windows_directory_owner: Any = None
        try:
            if _windows_artifact_filesystem is not None:
                windows_directory_owner = (
                    _windows_artifact_filesystem.pin_directory_no_reparse(directory)
                )
                opened_identity = windows_directory_owner.identity
                if (
                    opened_identity.kind != "directory"
                    or opened_identity.reparse_tag
                    or _windows_filesystem_identity(opened_identity)
                    != directory_identities[relative_directory]
                ):
                    state.add_incomplete(
                        AudioCppScanIssueCode.SOURCE_CHANGED,
                        relative_directory,
                        directory.name,
                    )
                    _close_windows_owner(
                        windows_directory_owner,
                        state=state,
                        relative_parts=relative_directory,
                        safe_name=directory.name,
                    )
                    windows_directory_owner = None
                    continue
            else:
                current_info = os.stat(directory, follow_symlinks=False)
                if (
                    _is_reparse_or_symlink(current_info)
                    or not stat.S_ISDIR(current_info.st_mode)
                    or _filesystem_identity(current_info)
                    != directory_identities[relative_directory]
                ):
                    state.add_incomplete(
                        AudioCppScanIssueCode.SOURCE_CHANGED,
                        relative_directory,
                        directory.name,
                    )
                    continue
            iterator = _scandir(directory)
            if _windows_artifact_filesystem is None:
                opened_info = os.stat(directory, follow_symlinks=False)
                if (
                    _is_reparse_or_symlink(opened_info)
                    or not stat.S_ISDIR(opened_info.st_mode)
                    or _filesystem_identity(opened_info)
                    != directory_identities[relative_directory]
                ):
                    state.add_incomplete(
                        AudioCppScanIssueCode.SOURCE_CHANGED,
                        relative_directory,
                        directory.name,
                    )
                    _close_directory_iterator(iterator, state, directory.name)
                    continue
        except PermissionError:
            if iterator is not None:
                _close_directory_iterator(iterator, state, directory.name)
            if windows_directory_owner is not None:
                _close_windows_owner(
                    windows_directory_owner,
                    state=state,
                    relative_parts=relative_directory,
                    safe_name=directory.name,
                )
            state.add_permission(relative_directory, directory.name)
            continue
        except WindowsArtifactError as error:
            cleanup = error.take_cleanup_owner()
            if cleanup is not None:
                raise AudioCppPackageScanError(
                    "Windows audio.cpp package handle cleanup did not complete.",
                    cleanup_owner=cleanup,
                ) from None
            if iterator is not None:
                _close_directory_iterator(iterator, state, directory.name)
            if windows_directory_owner is not None:
                _close_windows_owner(
                    windows_directory_owner,
                    state=state,
                    relative_parts=relative_directory,
                    safe_name=directory.name,
                )
            state.add_incomplete(
                AudioCppScanIssueCode.UNREADABLE,
                relative_directory,
                directory.name,
            )
            continue
        except OSError:
            if iterator is not None:
                _close_directory_iterator(iterator, state, directory.name)
            if windows_directory_owner is not None:
                _close_windows_owner(
                    windows_directory_owner,
                    state=state,
                    relative_parts=relative_directory,
                    safe_name=directory.name,
                )
            state.add_incomplete(
                AudioCppScanIssueCode.UNREADABLE,
                relative_directory,
                directory.name,
            )
            continue
        try:
            windows_names: set[str] = set()
            for entry in iterator:
                if state.cancellation_or_total_limit():
                    break
                if state.visited_entries >= active_limits.max_entries:
                    state.add_limit(AudioCppScanLimit.ENTRIES)
                    state.stopped = True
                    break
                state.visited_entries += 1
                entry_started = _monotonic()
                relative_parts = (*relative_directory, entry.name)
                if _windows_artifact_filesystem is not None:
                    folded_name = entry.name.casefold()
                    if folded_name in windows_names:
                        state.add_incomplete(
                            AudioCppScanIssueCode.SOURCE_CHANGED,
                            relative_directory,
                            entry.name,
                        )
                        continue
                    windows_names.add(folded_name)
                try:
                    info = entry.stat(follow_symlinks=False)
                except PermissionError:
                    state.add_permission(relative_parts, entry.name)
                    continue
                except OSError:
                    state.add_incomplete(
                        AudioCppScanIssueCode.UNREADABLE,
                        relative_parts,
                        entry.name,
                    )
                    continue

                entry_path = directory / entry.name
                is_reparse = bool(
                    getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
                )
                if _is_reparse_or_symlink(info):
                    state.add_issue(
                        AudioCppScanIssueCode.REPARSE_SKIPPED
                        if is_reparse and not stat.S_ISLNK(info.st_mode)
                        else AudioCppScanIssueCode.SYMLINK_SKIPPED,
                        entry.name,
                    )
                elif stat.S_ISDIR(info.st_mode):
                    if depth >= active_limits.max_depth:
                        state.add_limit(AudioCppScanLimit.DEPTH)
                    else:
                        directory_identities[relative_parts] = (
                            _pin_windows_directory_identity(
                                entry_path,
                                state=state,
                                relative_parts=relative_parts,
                            )
                            if _windows_artifact_filesystem is not None
                            else _filesystem_identity(info)
                        )
                        queue.append((entry_path, relative_parts, depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    matched = False
                    for signal_parts, signal in signals:
                        if (
                            len(relative_parts) < len(signal_parts)
                            or relative_parts[-len(signal_parts) :] != signal_parts
                        ):
                            continue
                        matched = True
                        candidate_root = relative_parts[: -len(signal_parts)]
                        if candidate_root not in candidate_evidence:
                            if (
                                len(candidate_evidence)
                                >= active_limits.max_candidate_roots
                            ):
                                state.add_limit(AudioCppScanLimit.CANDIDATE_ROOTS)
                                continue
                            candidate_evidence[candidate_root] = {}
                        cache_key = (
                            str(entry_path),
                            signal.kind,
                            signal.minimum_size_bytes,
                        )
                        cached = inspection_cache.get(cache_key)
                        if cached is None:
                            cached = _inspect_file(
                                entry_path,
                                relative_parts,
                                info,
                                signal,
                                state,
                            )
                            inspection_cache[cache_key] = cached
                        evidence = AudioCppPackageFileEvidence(
                            relative_path=signal.relative_path,
                            size_bytes=cached.size_bytes,
                            identity=cached.identity,
                            readable=cached.readable,
                            metadata_valid=cached.metadata_valid,
                        )
                        candidate_evidence[candidate_root][signal.relative_path] = (
                            evidence
                        )
                    if not matched and entry.name.casefold().endswith(
                        (".gguf", ".safetensors")
                    ):
                        state.add_unknown(entry.name)
                else:
                    state.add_issue(
                        AudioCppScanIssueCode.SPECIAL_FILE_SKIPPED, entry.name
                    )

                if _monotonic() - entry_started > active_limits.max_entry_seconds:
                    state.add_limit(AudioCppScanLimit.ENTRY_TIME)
        except PermissionError:
            state.add_permission(relative_directory, directory.name)
        except OSError:
            state.add_incomplete(
                AudioCppScanIssueCode.UNREADABLE,
                relative_directory,
                directory.name,
            )
        finally:
            _close_directory_iterator(iterator, state, directory.name)
            if windows_directory_owner is not None:
                _close_windows_owner(
                    windows_directory_owner,
                    state=state,
                    relative_parts=relative_directory,
                    safe_name=directory.name,
                )

    state.cancellation_or_total_limit()
    if cancellation.is_set():
        return _cancelled_result(
            request_revision=request_revision,
            canonical_root=canonical_root,
            root_info=root_info,
            root_was_symlink=root_was_symlink,
            root_identity=selected_root_identity,
        )

    global_partial = bool(state.limits_reached)
    discoveries: list[AudioCppPackageDiscovery] = []
    seen_discoveries: set[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]] = (
        set()
    )
    candidate_items = sorted(candidate_evidence.items(), key=lambda item: item[0])
    for relative_root, evidence_by_path in candidate_items:
        candidate_path = canonical_root.joinpath(*relative_root)
        root_identity = directory_identities.get(relative_root)
        if root_identity is None:
            try:
                candidate_info = os.stat(candidate_path, follow_symlinks=False)
            except OSError:
                state.add_issue(
                    AudioCppScanIssueCode.SOURCE_CHANGED, candidate_path.name
                )
                continue
            if _is_reparse_or_symlink(candidate_info) or not stat.S_ISDIR(
                candidate_info.st_mode
            ):
                state.add_issue(
                    AudioCppScanIssueCode.SOURCE_CHANGED, candidate_path.name
                )
                continue
            root_identity = _filesystem_identity(candidate_info)
        description = AudioCppPackageDescription(
            canonical_root=str(candidate_path),
            canonical_root_identity=root_identity,
            safe_name=_safe_name(candidate_path.name),
            files=tuple(
                sorted(
                    evidence_by_path.values(),
                    key=lambda item: item.relative_path,
                )
            ),
            partial=global_partial
            or _path_failure_affects_candidate(
                state.incomplete_paths,
                relative_root,
                signals,
            ),
            permission_limited=_path_failure_affects_candidate(
                state.permission_paths,
                relative_root,
                signals,
            ),
        )
        match = registry.match(description)
        if match.state is AudioCppMatchState.UNKNOWN:
            continue
        identity = (
            str(candidate_path),
            match.recipe_ids,
            tuple(
                sorted(
                    (item.relative_path, item.identity) for item in description.files
                )
            ),
        )
        if identity in seen_discoveries:
            continue
        if len(discoveries) >= active_limits.max_results:
            state.add_limit(AudioCppScanLimit.RESULTS)
            break
        seen_discoveries.add(identity)
        discoveries.append(AudioCppPackageDiscovery(description, match))

    limits_reached = tuple(sorted(state.limits_reached, key=lambda item: item.value))
    if state.permission_limited:
        outcome = AudioCppScanOutcome.PERMISSION_LIMITED
    elif limits_reached or state.incomplete_paths:
        outcome = AudioCppScanOutcome.PARTIAL
    else:
        outcome = AudioCppScanOutcome.COMPLETE
    result = AudioCppPackageScanResult(
        outcome=outcome,
        request_revision=request_revision,
        selected_root_name=_safe_name(canonical_root.name),
        canonical_root=str(canonical_root),
        canonical_root_identity=selected_root_identity,
        root_was_symlink=root_was_symlink,
        discoveries=tuple(discoveries),
        limits_reached=limits_reached,
        issues=tuple(state.issues),
        issues_truncated=state.issues_truncated,
        unknown_names=tuple(state.unknown_names),
        unknown_names_truncated=state.unknown_names_truncated,
        visited_entries=state.visited_entries,
        metadata_bytes_read=state.metadata_bytes_read,
    )
    return _require_managed_exact_result(
        result,
        root=root,
        allow_root_symlink=allow_root_symlink,
        expectation=managed_expectation,
    )


async def scan_audio_cpp_package_root_async(
    root: str | os.PathLike[str],
    *,
    registry: AudioCppRecipeRegistry = AUDIO_CPP_RECIPE_REGISTRY,
    limits: AudioCppScanLimits | None = None,
    cancellation_event: threading.Event | None = None,
    allow_root_symlink: bool = False,
    request_revision: int = 0,
    expected_managed_artifact: AudioCppManagedArtifactIdentity | None = None,
    expected_canonical_root: str | os.PathLike[str] | None = None,
) -> AudioCppPackageScanResult:
    """Run one package scan off-loop and propagate caller cancellation.

    Args:
        root: User-selected package directory.
        registry: Sealed recipe registry used for exact matching.
        limits: Optional finite scanner budgets; defaults are used when absent.
        cancellation_event: Optional cross-thread cancellation signal.
        allow_root_symlink: Whether to resolve a disclosed top-level symlink.
        request_revision: Non-negative caller revision copied into the result.
        expected_managed_artifact: Optional exact managed-store identity.
        expected_canonical_root: Canonical managed root, present with identity.

    Returns:
        The immutable result produced by :func:`scan_audio_cpp_package_root`.

    Raises:
        asyncio.CancelledError: If the awaiting caller cancels the scan.
        AudioCppPackageScanError: If the selected root is invalid or unusable.
        TypeError: If the registry, limits, or cancellation signal is invalid.
        ValueError: If the request revision is invalid.
    """
    cancellation = cancellation_event or threading.Event()
    work = asyncio.create_task(
        asyncio.to_thread(
            scan_audio_cpp_package_root,
            root,
            registry=registry,
            limits=limits,
            cancellation_event=cancellation,
            allow_root_symlink=allow_root_symlink,
            request_revision=request_revision,
            expected_managed_artifact=expected_managed_artifact,
            expected_canonical_root=expected_canonical_root,
        )
    )
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError as cancelled:
        cancellation.set()
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if work.done() and not work.cancelled():
            error = work.exception()
            if isinstance(error, AudioCppPackageScanError):
                cleanup = error.take_cleanup_owner()
                if cleanup is not None:
                    setattr(cancelled, _SCAN_CLEANUP_OWNER, cleanup)
        raise cancelled


__all__ = (
    "AudioCppPackageDiscovery",
    "AudioCppPackageScanError",
    "AudioCppPackageScanResult",
    "AudioCppScanIssue",
    "AudioCppScanIssueCode",
    "AudioCppScanLimit",
    "AudioCppScanLimits",
    "AudioCppScanOutcome",
    "scan_audio_cpp_package_root",
    "scan_audio_cpp_package_root_async",
    "take_audio_cpp_scan_cleanup_owner",
)
