"""Read-only Windows filesystem adapter for one-time note imports.

This path-based adapter is intentionally separate from the stronger POSIX
descriptor-relative walker. It compensates for Windows' missing ``dir_fd`` and
``O_NOFOLLOW`` support by rejecting reparse components, recording complete
lexical identities, and verifying the opened regular-file handle before reads.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tldw_chatbook.Notes.note_folder_models import (
    FolderValidationError,
    normalize_folder_name,
)
from tldw_chatbook.Notes.note_import_discovery import (
    DiscoveredImportSource,
    ImportDiscovery,
    ImportDiscoveryFailure,
    ImportSelectionError,
    SourceIdentity,
    VerifiedSourceReadError,
    _bounded_message,
    _copy_bounded_selection,
    _disambiguate_failure_paths,
    _display_collision_key,
    _display_path,
    _display_sort_key,
    _is_safe_display_path,
    _is_safe_display_segment,
    _reject,
    _selection_error,
    _unsafe_failure_display_path,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportBounds,
    ImportSource,
    ImportSourceKind,
)

_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class WindowsReadOnlyFilesystem(Protocol):
    """Minimal injectable filesystem capabilities used by the adapter."""

    def absolute(self, path: Path) -> Path: ...

    def lstat(self, path: Path) -> Any: ...

    def scandir(self, path: Path) -> Any: ...

    def open(self, path: Path, flags: int) -> int: ...

    def set_inheritable(self, descriptor: int, inheritable: bool) -> None: ...

    def fstat(self, descriptor: int) -> Any: ...

    def read(self, descriptor: int, count: int) -> bytes: ...

    def close(self, descriptor: int) -> None: ...


class OSWindowsReadOnlyFilesystem:
    """Windows-capable implementation using only Python's portable ``os`` API."""

    def absolute(self, path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def lstat(self, path: Path) -> os.stat_result:
        return os.lstat(path)

    def scandir(self, path: Path) -> os.ScandirIterator[str]:
        return os.scandir(path)

    def open(self, path: Path, flags: int) -> int:
        return os.open(path, flags)

    def set_inheritable(self, descriptor: int, inheritable: bool) -> None:
        os.set_inheritable(descriptor, inheritable)

    def fstat(self, descriptor: int) -> os.stat_result:
        return os.fstat(descriptor)

    def read(self, descriptor: int, count: int) -> bytes:
        return os.read(descriptor, count)

    def close(self, descriptor: int) -> None:
        os.close(descriptor)


OS_WINDOWS_FILESYSTEM = OSWindowsReadOnlyFilesystem()


@dataclass(frozen=True, slots=True)
class _SelectedPath:
    path: Path = field(repr=False)
    metadata: Any = field(repr=False)
    parent_identities: tuple[SourceIdentity, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ScannedEntry:
    name: str
    metadata: Any | None = field(repr=False)


@dataclass(slots=True)
class _DiscoveryState:
    bounds: ImportBounds
    candidates: list[DiscoveredImportSource] = field(default_factory=list)
    failures: list[ImportDiscoveryFailure] = field(default_factory=list)
    total_bytes: int = 0
    entry_count: int = 0


class _DirectoryChanged(RuntimeError):
    """A path-based directory scan no longer matches its recorded identity."""


def discover_import_sources(
    paths: Iterable[Path],
    bounds: ImportBounds,
    *,
    filesystem: WindowsReadOnlyFilesystem = OS_WINDOWS_FILESYSTEM,
) -> ImportDiscovery:
    """Discover sources through bounded, non-following Windows path checks."""
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")

    selected_paths = _copy_bounded_selection(paths, bounds)
    selected = [
        _inspect_selected_path(path, bounds, filesystem) for path in selected_paths
    ]
    kinds = [_mode_kind(item.metadata.st_mode) for item in selected]
    if any(kind == "symlink" for kind in kinds):
        _reject(bounds, "selected_symlink")
    if any(kind == "other" for kind in kinds):
        _reject(bounds, "selection_not_regular")

    directory_count = kinds.count("directory")
    if directory_count and len(selected) != 1:
        reason_code = (
            "multiple_directories"
            if directory_count == len(selected)
            else "mixed_selection"
        )
        _reject(bounds, reason_code)

    state = _DiscoveryState(bounds=bounds)
    if directory_count:
        selected_root = selected[0]
        root_label = selected_root.path.name
        if not _is_safe_display_segment(root_label):
            _reject(bounds, "unsafe_display_path")
        try:
            normalize_folder_name(root_label)
        except FolderValidationError:
            _reject(bounds, "unsafe_display_path")
        root_identity = _identity_from_stat(selected_root.metadata)
        try:
            _validate_directory_binding(
                selected_root.path,
                selected_root.parent_identities,
                root_identity,
                filesystem,
            )
            _scan_directory(
                directory_path=selected_root.path,
                relative_parts=(),
                parent_depth=0,
                root_label=root_label,
                directory_identities=(
                    *selected_root.parent_identities,
                    root_identity,
                ),
                state=state,
                filesystem=filesystem,
            )
            _validate_directory_binding(
                selected_root.path,
                selected_root.parent_identities,
                root_identity,
                filesystem,
            )
        except _DirectoryChanged:
            _reject(bounds, "selection_changed")
        except ImportSelectionError:
            raise
        except (OSError, TypeError, ValueError):
            raise _selection_error(bounds, "selection_unreadable") from None
    else:
        root_label = None
        _admit_selected_files(selected, state)

    candidates = tuple(
        sorted(
            state.candidates,
            key=lambda item: _display_sort_key(item.source.display_path),
        )
    )
    ordered_failures = tuple(
        sorted(state.failures, key=lambda item: _display_sort_key(item.display_path))
    )
    return ImportDiscovery(
        candidates=candidates,
        failures=_disambiguate_failure_paths(candidates, ordered_failures),
        root_label=root_label,
        total_bytes=state.total_bytes,
        entry_count=state.entry_count,
    )


def read_discovered_source(
    candidate: DiscoveredImportSource,
    bounds: ImportBounds,
    *,
    filesystem: WindowsReadOnlyFilesystem = OS_WINDOWS_FILESYSTEM,
) -> bytes:
    """Read only after the opened Windows handle matches discovery exactly."""
    if not isinstance(candidate, DiscoveredImportSource):
        raise TypeError("candidate must be a DiscoveredImportSource.")
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")
    if (
        candidate.size_bytes != candidate.identity.size
        or candidate.size_bytes < 0
        or candidate.size_bytes > bounds.max_file_bytes
    ):
        raise VerifiedSourceReadError("source_changed")

    descriptor: int | None = None
    content = bytearray()
    primary_error: VerifiedSourceReadError | None = None
    close_failed = False
    try:
        _validate_candidate_binding(candidate, filesystem)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = filesystem.open(candidate.source.source_path, flags)
        filesystem.set_inheritable(descriptor, False)
        opened_metadata = filesystem.fstat(descriptor)
        if not _file_identity_matches(candidate.identity, opened_metadata):
            raise VerifiedSourceReadError("source_changed")

        while True:
            chunk = filesystem.read(
                descriptor,
                min(64 * 1024, bounds.max_file_bytes + 1),
            )
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > bounds.max_file_bytes:
                raise VerifiedSourceReadError("max_file_bytes_exceeded")

        after_read = filesystem.fstat(descriptor)
        if len(content) != candidate.size_bytes or not _file_identity_matches(
            candidate.identity,
            after_read,
        ):
            raise VerifiedSourceReadError("source_changed")
    except VerifiedSourceReadError as error:
        primary_error = error
    except (FileNotFoundError, NotADirectoryError):
        primary_error = VerifiedSourceReadError("source_changed")
    except (OSError, TypeError, ValueError):
        primary_error = VerifiedSourceReadError("source_unavailable")
    finally:
        if descriptor is not None:
            try:
                filesystem.close(descriptor)
            except BaseException:  # noqa: BLE001 - normalize cleanup failures.
                close_failed = True

    if primary_error is not None:
        raise primary_error
    if close_failed:
        raise VerifiedSourceReadError("source_unavailable")
    try:
        _validate_candidate_binding(candidate, filesystem)
    except VerifiedSourceReadError:
        raise
    except (FileNotFoundError, NotADirectoryError):
        raise VerifiedSourceReadError("source_changed") from None
    except (OSError, TypeError, ValueError):
        raise VerifiedSourceReadError("source_unavailable") from None
    return bytes(content)


def _inspect_selected_path(
    path: Path,
    bounds: ImportBounds,
    filesystem: WindowsReadOnlyFilesystem,
) -> _SelectedPath:
    try:
        absolute_path = filesystem.absolute(path)
        anchor, components = _path_components(absolute_path)
        current_path = anchor
        root_metadata = filesystem.lstat(current_path)
        if _is_reparse(root_metadata):
            _reject(bounds, "selected_symlink")
        if not stat.S_ISDIR(root_metadata.st_mode):
            _reject(bounds, "selection_not_regular")
        parent_identities: list[SourceIdentity] = []
        for component in components[:-1]:
            parent_identities.append(_identity_from_stat(root_metadata))
            current_path /= component
            root_metadata = filesystem.lstat(current_path)
            if _is_reparse(root_metadata):
                _reject(bounds, "selected_symlink")
            if not stat.S_ISDIR(root_metadata.st_mode):
                _reject(bounds, "selection_not_regular")
        if not components:
            metadata = root_metadata
        else:
            parent_identities.append(_identity_from_stat(root_metadata))
            metadata = filesystem.lstat(current_path / components[-1])
        if _is_reparse(metadata):
            _reject(bounds, "selected_symlink")
        return _SelectedPath(
            path=absolute_path,
            metadata=metadata,
            parent_identities=tuple(parent_identities),
        )
    except ImportSelectionError:
        raise
    except FileNotFoundError:
        raise _selection_error(bounds, "selection_missing") from None
    except ValueError:
        raise _selection_error(bounds, "invalid_selection") from None
    except (OSError, TypeError):
        raise _selection_error(bounds, "selection_unreadable") from None


def _path_components(path: Path) -> tuple[Path, tuple[str, ...]]:
    if not path.anchor:
        raise ValueError("absolute path required")
    return Path(path.anchor), tuple(path.parts[1:])


def _mode_kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _admit_selected_files(
    selected: list[_SelectedPath],
    state: _DiscoveryState,
) -> None:
    names: set[str] = set()
    for item in sorted(selected, key=lambda value: _display_sort_key(value.path.name)):
        display_path = item.path.name
        collision_key = _display_collision_key(display_path)
        if not _is_safe_display_segment(display_path):
            _reject(state.bounds, "unsafe_display_path")
        if collision_key in names:
            _reject(state.bounds, "ambiguous_display_path")
        names.add(collision_key)
        state.entry_count += 1
        _admit_file(
            item.path,
            display_path,
            item.metadata,
            ImportSourceKind.SELECTED_FILE,
            item.parent_identities,
            state,
        )


def _scan_directory(
    *,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    directory_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
    filesystem: WindowsReadOnlyFilesystem,
) -> None:
    _require_directory_chain(directory_path, directory_identities, filesystem)
    scanned_entries: list[_ScannedEntry] = []
    with filesystem.scandir(directory_path) as iterator:
        entries = []
        for entry in iterator:
            state.entry_count += 1
            if state.entry_count > state.bounds.max_entries:
                _reject(state.bounds, "max_entries_exceeded")
            entries.append(entry)
        entries.sort(key=lambda entry: _display_sort_key(entry.name))
        for entry in entries:
            metadata: Any | None = None
            if _is_safe_display_segment(entry.name):
                try:
                    metadata = filesystem.lstat(directory_path / entry.name)
                except (OSError, TypeError, ValueError):
                    pass
            scanned_entries.append(_ScannedEntry(entry.name, metadata))
    _require_directory_chain(directory_path, directory_identities, filesystem)
    _validate_sibling_namespace(scanned_entries, state.bounds)

    for scanned_entry in scanned_entries:
        _scan_entry(
            scanned_entry,
            directory_path=directory_path,
            relative_parts=relative_parts,
            parent_depth=parent_depth,
            root_label=root_label,
            directory_identities=directory_identities,
            state=state,
            filesystem=filesystem,
        )
    _require_directory_chain(directory_path, directory_identities, filesystem)


def _validate_sibling_namespace(
    entries: Iterable[_ScannedEntry],
    bounds: ImportBounds,
) -> None:
    folder_keys: set[str] = set()
    for entry in entries:
        metadata = entry.metadata
        if (
            metadata is None
            or _is_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            continue
        try:
            normalized = normalize_folder_name(entry.name)
        except FolderValidationError:
            continue
        if normalized.key in folder_keys:
            _reject(bounds, "ambiguous_folder_path")
        folder_keys.add(normalized.key)


def _scan_entry(
    scanned_entry: _ScannedEntry,
    *,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    directory_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
    filesystem: WindowsReadOnlyFilesystem,
) -> None:
    entry_path = directory_path / scanned_entry.name
    entry_parts = (*relative_parts, scanned_entry.name)
    display_path = _display_path(root_label, entry_parts)
    if not _is_safe_display_path(display_path):
        _add_failure(
            state,
            entry_path,
            _unsafe_failure_display_path(root_label, entry_parts),
            "nested_unsafe_name",
        )
        return
    metadata = scanned_entry.metadata
    if metadata is None:
        _add_failure(state, entry_path, display_path, "nested_unavailable")
        return
    if _is_reparse(metadata):
        _add_failure(state, entry_path, display_path, "nested_symlink")
        return
    kind = _mode_kind(metadata.st_mode)
    if kind == "other":
        _add_failure(state, entry_path, display_path, "nested_not_regular")
        return
    if kind == "file":
        _admit_file(
            entry_path,
            display_path,
            metadata,
            ImportSourceKind.DIRECTORY_MEMBER,
            directory_identities,
            state,
        )
        return

    try:
        normalize_folder_name(scanned_entry.name)
    except FolderValidationError:
        _add_failure(
            state,
            entry_path,
            _unsafe_failure_display_path(root_label, entry_parts),
            "nested_unsafe_name",
        )
        return
    child_depth = parent_depth + 1
    if child_depth > state.bounds.max_depth:
        _reject(state.bounds, "max_depth_exceeded")
    child_identity = _identity_from_stat(metadata)
    candidate_mark = len(state.candidates)
    failure_mark = len(state.failures)
    total_bytes_mark = state.total_bytes
    try:
        _scan_directory(
            directory_path=entry_path,
            relative_parts=entry_parts,
            parent_depth=child_depth,
            root_label=root_label,
            directory_identities=(*directory_identities, child_identity),
            state=state,
            filesystem=filesystem,
        )
    except ImportSelectionError:
        raise
    except (_DirectoryChanged, OSError, TypeError, ValueError):
        del state.candidates[candidate_mark:]
        del state.failures[failure_mark:]
        state.total_bytes = total_bytes_mark
        _add_failure(state, entry_path, display_path, "nested_unavailable")


def _admit_file(
    path: Path,
    display_path: str,
    metadata: Any,
    kind: ImportSourceKind,
    parent_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
) -> None:
    size = metadata.st_size
    if size < 0 or size > state.bounds.max_file_bytes:
        _reject(state.bounds, "max_file_bytes_exceeded")
    if len(state.candidates) >= state.bounds.max_files:
        _reject(state.bounds, "max_files_exceeded")
    if state.total_bytes + size > state.bounds.max_total_bytes:
        _reject(state.bounds, "max_total_bytes_exceeded")
    state.total_bytes += size
    state.candidates.append(
        DiscoveredImportSource(
            source=ImportSource(
                kind=kind,
                display_path=display_path,
                source_path=path,
            ),
            size_bytes=size,
            identity=_identity_from_stat(metadata),
            parent_identities=parent_identities,
        )
    )


def _add_failure(
    state: _DiscoveryState,
    source_path: Path,
    display_path: str,
    reason_code: str,
) -> None:
    state.failures.append(
        ImportDiscoveryFailure(
            display_path=display_path,
            reason_code=reason_code,
            user_message=_bounded_message(state.bounds, reason_code),
            source_path=source_path,
        )
    )


def _validate_directory_binding(
    path: Path,
    parent_identities: tuple[SourceIdentity, ...],
    directory_identity: SourceIdentity,
    filesystem: WindowsReadOnlyFilesystem,
) -> None:
    anchor, components = _path_components(path)
    if len(parent_identities) != len(components):
        raise _DirectoryChanged
    current_path = anchor
    for index, expected_identity in enumerate(parent_identities):
        metadata = filesystem.lstat(current_path)
        if not _directory_identity_matches(expected_identity, metadata):
            raise _DirectoryChanged
        if index < len(components) - 1:
            current_path /= components[index]
    leaf_metadata = filesystem.lstat(path)
    if not _directory_identity_matches(directory_identity, leaf_metadata):
        raise _DirectoryChanged


def _validate_candidate_binding(
    candidate: DiscoveredImportSource,
    filesystem: WindowsReadOnlyFilesystem,
) -> None:
    path = candidate.source.source_path
    anchor, components = _path_components(path)
    if not components or len(candidate.parent_identities) != len(components):
        raise VerifiedSourceReadError("source_changed")
    current_path = anchor
    for index, expected_identity in enumerate(candidate.parent_identities):
        metadata = filesystem.lstat(current_path)
        if not _directory_identity_matches(expected_identity, metadata):
            raise VerifiedSourceReadError("source_changed")
        if index < len(components) - 1:
            current_path /= components[index]
    leaf_metadata = filesystem.lstat(path)
    if not _file_identity_matches(candidate.identity, leaf_metadata):
        raise VerifiedSourceReadError("source_changed")


def _require_directory_chain(
    path: Path,
    expected_identities: tuple[SourceIdentity, ...],
    filesystem: WindowsReadOnlyFilesystem,
) -> None:
    anchor, components = _path_components(path)
    if len(expected_identities) != len(components) + 1:
        raise _DirectoryChanged
    current_path = anchor
    for index, expected_identity in enumerate(expected_identities):
        if not _directory_identity_matches(
            expected_identity,
            filesystem.lstat(current_path),
        ):
            raise _DirectoryChanged
        if index < len(components):
            current_path /= components[index]


def _identity_from_stat(metadata: Any) -> SourceIdentity:
    return SourceIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _directory_identity_matches(identity: SourceIdentity, metadata: Any) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not _is_reparse(metadata)
        and identity == _identity_from_stat(metadata)
    )


def _file_identity_matches(identity: SourceIdentity, metadata: Any) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not _is_reparse(metadata)
        and identity == _identity_from_stat(metadata)
    )


def _is_reparse(metadata: Any) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _REPARSE_ATTRIBUTE)
