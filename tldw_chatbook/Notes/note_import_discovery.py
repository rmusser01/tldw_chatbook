"""Read-only, bounded source discovery for one-time Database Notes imports."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from unicodedata import normalize
from urllib.parse import quote_from_bytes

from tldw_chatbook.Notes.note_folder_models import (
    FolderValidationError,
    normalize_folder_name,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportBounds,
    ImportSource,
    ImportSourceKind,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple


class ImportSelectionError(ValueError):
    """A stable, path-redacted failure that rejects the whole selection."""

    def __init__(self, reason_code: str, user_message: str) -> None:
        self.reason_code = reason_code
        self.user_message = user_message
        super().__init__(user_message)


class _SecureDiscoveryUnavailable(RuntimeError):
    """Secure descriptor-relative filesystem operations are unavailable."""


class VerifiedSourceReadError(RuntimeError):
    """Path-free parser-facing failure from a descriptor-verified source read."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Non-following metadata for a parser's later pre-open recheck."""

    device: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    size: int = field(repr=False)
    modified_ns: int = field(repr=False)
    changed_ns: int = field(repr=False)

    def __post_init__(self) -> None:
        integer_fields = (
            "device",
            "inode",
            "mode",
            "size",
            "modified_ns",
            "changed_ns",
        )
        for field_name in integer_fields:
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer.")
        for field_name in ("device", "inode", "mode", "size"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class DiscoveredImportSource:
    """One regular file admitted by bounded discovery."""

    source: ImportSource
    size_bytes: int
    identity: SourceIdentity = field(repr=False)
    parent_identities: tuple[SourceIdentity, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ImportSource):
            raise TypeError("source must be an ImportSource.")
        if type(self.size_bytes) is not int:
            raise TypeError("size_bytes must be an integer.")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative.")
        if not isinstance(self.identity, SourceIdentity):
            raise TypeError("identity must be a SourceIdentity.")
        if isinstance(self.parent_identities, (str, bytes)):
            raise TypeError("parent_identities must be a collection.")
        try:
            parent_identities = tuple(self.parent_identities)
        except TypeError as error:
            raise TypeError("parent_identities must be a collection.") from error
        if not parent_identities:
            raise ValueError("parent_identities cannot be empty.")
        if not all(
            isinstance(identity, SourceIdentity) for identity in parent_identities
        ):
            raise ValueError("parent_identities must contain SourceIdentity values.")
        if self.size_bytes != self.identity.size:
            raise ValueError("size_bytes must match identity.size.")
        object.__setattr__(self, "parent_identities", parent_identities)


@dataclass(frozen=True, slots=True)
class ImportDiscoveryFailure:
    """One nested unsafe entry retained for a later visible Skip item."""

    display_path: str
    reason_code: str
    user_message: str
    source_path: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ImportDiscovery:
    """Complete read-only result from one approved source selection."""

    candidates: tuple[DiscoveredImportSource, ...]
    failures: tuple[ImportDiscoveryFailure, ...]
    root_label: str | None
    total_bytes: int
    entry_count: int

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        failures = tuple(self.failures)
        if not all(
            isinstance(candidate, DiscoveredImportSource) for candidate in candidates
        ):
            raise ValueError("candidates must contain discovered sources.")
        if not all(isinstance(failure, ImportDiscoveryFailure) for failure in failures):
            raise ValueError("failures must contain discovery failures.")
        if self.root_label is not None and not _is_safe_display_segment(
            self.root_label
        ):
            raise ValueError("root_label must be a safe folder segment.")
        for field_name in ("total_bytes", "entry_count"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "failures", failures)


@dataclass(slots=True)
class _DiscoveryState:
    bounds: ImportBounds
    candidates: list[DiscoveredImportSource] = field(default_factory=list)
    failures: list[ImportDiscoveryFailure] = field(default_factory=list)
    total_bytes: int = 0
    entry_count: int = 0


@dataclass(frozen=True, slots=True)
class _SelectedPath:
    path: Path = field(repr=False)
    metadata: os.stat_result = field(repr=False)
    parent_identities: tuple[SourceIdentity, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ScannedDirectoryEntry:
    entry: os.DirEntry[str] = field(repr=False)
    metadata: os.stat_result | None = field(repr=False)


_MESSAGES = {
    "empty_selection": "Choose at least one file or one folder.",
    "invalid_selection": "The selected source is not valid.",
    "selection_missing": "A selected source is no longer available.",
    "selection_unreadable": "A selected source cannot be inspected safely.",
    "secure_discovery_unavailable": "Secure source discovery is unavailable.",
    "selected_symlink": "Linked files and folders cannot be selected.",
    "selection_not_regular": "Select regular files or one folder.",
    "mixed_selection": "Choose files or one folder, not both.",
    "multiple_directories": "Choose only one folder at a time.",
    "ambiguous_display_path": "Selected files must have distinct names.",
    "ambiguous_folder_path": "The selected folder has ambiguous subfolder names.",
    "unsafe_display_path": "A selected source has an unsafe display name.",
    "selection_changed": "A selected source changed during inspection.",
    "max_depth_exceeded": "The selected folder is nested too deeply.",
    "max_files_exceeded": "The selection contains too many files.",
    "max_file_bytes_exceeded": "A selected file is too large.",
    "max_total_bytes_exceeded": "The selected files are too large in total.",
    "max_entries_exceeded": "The selected folder contains too many entries.",
    "nested_symlink": "Linked entries are skipped for safety.",
    "nested_unsafe_name": "This entry has an unsafe name and will be skipped.",
    "nested_not_regular": "This entry is not a regular file and will be skipped.",
    "nested_unavailable": "This entry cannot be inspected safely and will be skipped.",
}


def _platform_uses_windows_adapter() -> bool:
    """Return whether the dedicated Windows path adapter is required."""
    return os.name == "nt"


def _windows_filesystem() -> object:
    from tldw_chatbook.Notes.note_import_windows_fs import OS_WINDOWS_FILESYSTEM

    return OS_WINDOWS_FILESYSTEM


def discover_import_sources(
    paths: Iterable[Path],
    bounds: ImportBounds,
) -> ImportDiscovery:
    """Dispatch source discovery to the platform's read-only strategy.

    Args:
        paths: User-selected files or one directory to discover.
        bounds: Resource and diagnostic limits for the discovery pass.

    Returns:
        An immutable description of admitted sources and safe failures.

    Raises:
        ImportSelectionError: The selection is invalid or cannot be inspected safely.
        TypeError: ``bounds`` or a selected path has an invalid type.
    """
    if _platform_uses_windows_adapter():
        from tldw_chatbook.Notes.note_import_windows_fs import (
            discover_import_sources as discover_windows_sources,
        )

        return discover_windows_sources(
            paths,
            bounds,
            filesystem=_windows_filesystem(),  # type: ignore[arg-type]
        )
    return _discover_import_sources_posix(paths, bounds)


def _discover_import_sources_posix(
    paths: Iterable[Path],
    bounds: ImportBounds,
) -> ImportDiscovery:
    """Validate a file-only or single-directory selection without reading content."""
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")

    selected_paths = _copy_bounded_selection(paths, bounds)
    selected = [_inspect_selected_path(path, bounds) for path in selected_paths]
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
    if directory_count == 1:
        selected_root = selected[0]
        root_label = selected_root.path.name
        if not _is_safe_display_segment(root_label):
            _reject(bounds, "unsafe_display_path")
        try:
            normalize_folder_name(root_label)
        except FolderValidationError:
            _reject(bounds, "unsafe_display_path")
        _scan_selected_directory(selected_root, root_label, state)
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
    failures = _disambiguate_failure_paths(candidates, ordered_failures)
    return ImportDiscovery(
        candidates=candidates,
        failures=failures,
        root_label=root_label,
        total_bytes=state.total_bytes,
        entry_count=state.entry_count,
    )


def read_discovered_source(
    candidate: DiscoveredImportSource,
    bounds: ImportBounds,
) -> bytes:
    """Dispatch a verified source read to the platform-specific strategy.

    Args:
        candidate: Source identity and path admitted by discovery.
        bounds: Resource limits that must still admit the source.

    Returns:
        The source bytes read through verified handles or descriptors.

    Raises:
        TypeError: ``candidate`` or ``bounds`` has an invalid type.
        VerifiedSourceReadError: The source is unavailable, changed, or unsafe.
    """
    if _platform_uses_windows_adapter():
        from tldw_chatbook.Notes.note_import_windows_fs import (
            read_discovered_source as read_windows_source,
        )

        return read_windows_source(
            candidate,
            bounds,
            filesystem=_windows_filesystem(),  # type: ignore[arg-type]
        )
    return _read_discovered_source_posix(candidate, bounds)


def _read_discovered_source_posix(
    candidate: DiscoveredImportSource,
    bounds: ImportBounds,
) -> bytes:
    """Read one admitted leaf through its recorded, non-following identities."""
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

    descriptors: list[int] = []
    content = bytearray()
    primary_error: VerifiedSourceReadError | None = None
    close_failed = False
    try:
        path = candidate.source.source_path
        components = path.parts[1:]
        if not path.anchor or not components:
            raise VerifiedSourceReadError("source_unavailable")
        if len(candidate.parent_identities) != len(components):
            raise VerifiedSourceReadError("source_changed")

        current_fd = os.open(path.anchor, _directory_open_flags())
        descriptors.append(current_fd)
        if not _identity_matches(
            candidate.parent_identities[0],
            os.fstat(current_fd),
        ):
            raise VerifiedSourceReadError("source_changed")

        for index, component in enumerate(components[:-1], start=1):
            metadata = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise VerifiedSourceReadError("source_changed")
            next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            descriptors.append(next_fd)
            if not _identity_matches(
                candidate.parent_identities[index],
                os.fstat(next_fd),
            ):
                raise VerifiedSourceReadError("source_changed")
            current_fd = next_fd

        leaf_name = components[-1]
        before_open = os.stat(
            leaf_name,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        if not _file_identity_matches(candidate.identity, before_open):
            raise VerifiedSourceReadError("source_changed")
        leaf_fd = os.open(leaf_name, _file_open_flags(), dir_fd=current_fd)
        descriptors.append(leaf_fd)
        after_open = os.fstat(leaf_fd)
        if not _file_identity_matches(candidate.identity, after_open):
            raise VerifiedSourceReadError("source_changed")

        while True:
            chunk = os.read(leaf_fd, min(64 * 1024, bounds.max_file_bytes + 1))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > bounds.max_file_bytes:
                raise VerifiedSourceReadError("max_file_bytes_exceeded")

        after_read = os.fstat(leaf_fd)
        if len(content) != candidate.size_bytes or not _file_identity_matches(
            candidate.identity, after_read
        ):
            raise VerifiedSourceReadError("source_changed")
        _verify_lexical_source_binding(candidate)
    except VerifiedSourceReadError as error:
        primary_error = error
    except (FileNotFoundError, NotADirectoryError):
        primary_error = VerifiedSourceReadError("source_changed")
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = VerifiedSourceReadError("secure_read_unavailable")
    except (OSError, TypeError, ValueError):
        primary_error = VerifiedSourceReadError("source_unavailable")
    finally:
        close_failed = _close_descriptors(descriptors)

    if primary_error is not None:
        raise primary_error
    if close_failed:
        raise VerifiedSourceReadError("source_unavailable")
    return bytes(content)


def _verify_lexical_source_binding(candidate: DiscoveredImportSource) -> None:
    """Recheck that the recorded leaf remains bound at its lexical path."""
    descriptors: list[int] = []
    primary_error: VerifiedSourceReadError | None = None
    close_failed = False
    try:
        path = candidate.source.source_path
        components = path.parts[1:]
        if (
            not path.anchor
            or not components
            or len(candidate.parent_identities) != len(components)
        ):
            raise VerifiedSourceReadError("source_changed")

        current_fd = os.open(path.anchor, _directory_open_flags())
        descriptors.append(current_fd)
        if not _identity_matches(
            candidate.parent_identities[0],
            os.fstat(current_fd),
        ):
            raise VerifiedSourceReadError("source_changed")

        for index, component in enumerate(components[:-1], start=1):
            metadata = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise VerifiedSourceReadError("source_changed")
            next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            descriptors.append(next_fd)
            if not _identity_matches(
                candidate.parent_identities[index],
                os.fstat(next_fd),
            ):
                raise VerifiedSourceReadError("source_changed")
            current_fd = next_fd

        leaf_metadata = os.stat(
            components[-1],
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        if not _file_identity_matches(candidate.identity, leaf_metadata):
            raise VerifiedSourceReadError("source_changed")
    except VerifiedSourceReadError as error:
        primary_error = error
    except (FileNotFoundError, NotADirectoryError):
        primary_error = VerifiedSourceReadError("source_changed")
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = VerifiedSourceReadError("secure_read_unavailable")
    except (OSError, TypeError, ValueError):
        primary_error = VerifiedSourceReadError("source_unavailable")
    finally:
        close_failed = _close_descriptors(descriptors)

    if primary_error is not None:
        raise primary_error
    if close_failed:
        raise VerifiedSourceReadError("source_unavailable")


def _copy_bounded_selection(
    paths: Iterable[Path], bounds: ImportBounds
) -> tuple[Path, ...]:
    if isinstance(paths, (str, bytes, Path)):
        _reject(bounds, "invalid_selection")
    selected: list[Path] = []
    for path in paths:
        if not isinstance(path, Path):
            _reject(bounds, "invalid_selection")
        try:
            validated_path = validate_path_simple(path, probe_existing=False)
        except ValueError:
            _reject(bounds, "invalid_selection")
        selected.append(validated_path)
        if len(selected) > bounds.max_entries:
            _reject(bounds, "max_entries_exceeded")
        if len(selected) > bounds.max_files:
            _reject(bounds, "max_files_exceeded")
    if not selected:
        _reject(bounds, "empty_selection")
    return tuple(selected)


def _absolute_path(path: Path) -> Path:
    """Make a lexical absolute path without resolving or following links."""
    return Path(os.path.abspath(os.fspath(path)))


def _inspect_selected_path(path: Path, bounds: ImportBounds) -> _SelectedPath:
    """Inspect every absolute path component from a pinned root descriptor."""
    descriptors: list[int] = []
    result: _SelectedPath | None = None
    primary_error: ImportSelectionError | None = None
    close_failed = False
    try:
        absolute_path = _absolute_path(path)
        flags = _directory_open_flags()
        anchor = absolute_path.anchor
        if not anchor:
            _reject(bounds, "invalid_selection")
        current_fd = os.open(anchor, flags)
        descriptors.append(current_fd)
        root_metadata = os.fstat(current_fd)
        components = absolute_path.parts[1:]
        if not components:
            result = _SelectedPath(
                path=absolute_path,
                metadata=root_metadata,
                parent_identities=(),
            )
        else:
            parent_identities = [_identity_from_stat(root_metadata)]
            for component in components[:-1]:
                metadata = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                if _is_link_or_reparse(metadata):
                    _reject(bounds, "selected_symlink")
                if not stat.S_ISDIR(metadata.st_mode):
                    _reject(bounds, "selection_not_regular")
                next_fd = os.open(component, flags, dir_fd=current_fd)
                descriptors.append(next_fd)
                opened_metadata = os.fstat(next_fd)
                if not _same_object(metadata, opened_metadata):
                    _reject(bounds, "selection_changed")
                current_fd = next_fd
                parent_identities.append(_identity_from_stat(opened_metadata))
            leaf_metadata = os.stat(
                components[-1],
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(leaf_metadata):
                _reject(bounds, "selected_symlink")
            result = _SelectedPath(
                path=absolute_path,
                metadata=leaf_metadata,
                parent_identities=tuple(parent_identities),
            )
    except ImportSelectionError as error:
        primary_error = error
    except FileNotFoundError:
        primary_error = _selection_error(bounds, "selection_missing")
    except ValueError:
        primary_error = _selection_error(bounds, "invalid_selection")
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = _selection_error(bounds, "secure_discovery_unavailable")
    except (OSError, TypeError):
        primary_error = _selection_error(bounds, "selection_unreadable")
    finally:
        close_failed = _close_descriptors(descriptors)

    if primary_error is not None:
        raise primary_error
    if close_failed:
        _reject(bounds, "selection_unreadable")
    if result is None:
        _reject(bounds, "selection_unreadable")
    return result


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
    ordered = sorted(
        selected,
        key=lambda item: _display_sort_key(item.path.name),
    )
    for item in ordered:
        display_path = item.path.name
        normalized_name = _display_collision_key(display_path)
        if not _is_safe_display_segment(display_path):
            _reject(state.bounds, "unsafe_display_path")
        if normalized_name in names:
            _reject(state.bounds, "ambiguous_display_path")
        names.add(normalized_name)
        state.entry_count += 1
        _admit_file(
            item.path,
            display_path,
            item.metadata,
            ImportSourceKind.SELECTED_FILE,
            item.parent_identities,
            state,
        )


def _scan_selected_directory(
    selected_root: _SelectedPath,
    root_label: str,
    state: _DiscoveryState,
) -> None:
    root_fd: int | None = None
    primary_error: ImportSelectionError | None = None
    close_failed = False
    try:
        root_fd = _open_verified_directory(selected_root, state.bounds)
        scan_succeeded = _scan_directory_fd(
            directory_fd=root_fd,
            directory_path=selected_root.path,
            relative_parts=(),
            parent_depth=0,
            root_label=root_label,
            directory_identities=(
                *selected_root.parent_identities,
                _identity_from_stat(selected_root.metadata),
            ),
            state=state,
        )
        if not scan_succeeded:
            primary_error = _selection_error(state.bounds, "selection_unreadable")
    except ImportSelectionError as error:
        primary_error = error
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = _selection_error(
            state.bounds,
            "secure_discovery_unavailable",
        )
    except (OSError, TypeError, ValueError):
        primary_error = _selection_error(state.bounds, "selection_unreadable")
    finally:
        close_failed = _close_descriptors([root_fd]) if root_fd is not None else False

    if primary_error is not None:
        raise primary_error
    if close_failed:
        _reject(state.bounds, "selection_unreadable")


def _scan_directory_fd(
    *,
    directory_fd: int,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    directory_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
) -> bool:
    try:
        with os.scandir(directory_fd) as iterator:
            entries: list[os.DirEntry[str]] = []
            for entry in iterator:
                state.entry_count += 1
                if state.entry_count > state.bounds.max_entries:
                    _reject(state.bounds, "max_entries_exceeded")
                entries.append(entry)
            entries.sort(key=lambda entry: _display_sort_key(entry.name))
            scanned_entries = _snapshot_directory_entries(entries)
            _validate_sibling_folder_namespace(scanned_entries, state.bounds)
            for scanned_entry in scanned_entries:
                _scan_entry(
                    scanned_entry=scanned_entry,
                    directory_fd=directory_fd,
                    directory_path=directory_path,
                    relative_parts=relative_parts,
                    parent_depth=parent_depth,
                    root_label=root_label,
                    directory_identities=directory_identities,
                    state=state,
                )
    except ImportSelectionError:
        raise
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        raise
    except (OSError, TypeError, ValueError):
        return False
    return True


def _snapshot_directory_entries(
    entries: Iterable[os.DirEntry[str]],
) -> tuple[_ScannedDirectoryEntry, ...]:
    """Capture one non-following metadata observation for every safe entry name."""
    scanned_entries: list[_ScannedDirectoryEntry] = []
    for entry in entries:
        metadata: os.stat_result | None = None
        if _is_safe_display_segment(entry.name):
            try:
                metadata = entry.stat(follow_symlinks=False)
            except NotImplementedError as error:
                raise _SecureDiscoveryUnavailable from error
            except (OSError, TypeError, ValueError):
                pass
        scanned_entries.append(_ScannedDirectoryEntry(entry=entry, metadata=metadata))
    return tuple(scanned_entries)


def _validate_sibling_folder_namespace(
    scanned_entries: Iterable[_ScannedDirectoryEntry],
    bounds: ImportBounds,
) -> None:
    """Reject canonically ambiguous directory siblings before traversal."""
    folder_keys: set[str] = set()
    for scanned_entry in scanned_entries:
        entry = scanned_entry.entry
        metadata = scanned_entry.metadata
        if metadata is None:
            continue
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            continue
        try:
            normalized = normalize_folder_name(entry.name)
        except FolderValidationError:
            continue
        if normalized.key in folder_keys:
            _reject(bounds, "ambiguous_folder_path")
        folder_keys.add(normalized.key)


def _scan_entry(
    *,
    scanned_entry: _ScannedDirectoryEntry,
    directory_fd: int,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    directory_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
) -> None:
    entry = scanned_entry.entry
    entry_path = directory_path / entry.name
    entry_parts = (*relative_parts, entry.name)
    display_path = _display_path(root_label, entry_parts)
    if not _is_safe_display_path(display_path):
        safe_display_path = _unsafe_failure_display_path(root_label, entry_parts)
        _add_failure(state, entry_path, safe_display_path, "nested_unsafe_name")
        return
    metadata = scanned_entry.metadata
    if metadata is None:
        _add_failure(state, entry_path, display_path, "nested_unavailable")
        return

    kind = _mode_kind(metadata.st_mode)
    if _is_link_or_reparse(metadata):
        _add_failure(state, entry_path, display_path, "nested_symlink")
    elif kind == "other":
        _add_failure(state, entry_path, display_path, "nested_not_regular")
    elif kind == "file":
        _admit_file(
            entry_path,
            display_path,
            metadata,
            ImportSourceKind.DIRECTORY_MEMBER,
            directory_identities,
            state,
        )
    else:
        try:
            normalize_folder_name(entry.name)
        except FolderValidationError:
            safe_display_path = _unsafe_failure_display_path(root_label, entry_parts)
            _add_failure(
                state,
                entry_path,
                safe_display_path,
                "nested_unsafe_name",
            )
            return
        child_depth = parent_depth + 1
        if child_depth > state.bounds.max_depth:
            _reject(state.bounds, "max_depth_exceeded")
        _scan_child_directory(
            entry=entry,
            parent_fd=directory_fd,
            entry_path=entry_path,
            entry_parts=entry_parts,
            metadata=metadata,
            child_depth=child_depth,
            root_label=root_label,
            display_path=display_path,
            parent_identities=directory_identities,
            state=state,
        )


def _scan_child_directory(
    *,
    entry: os.DirEntry[str],
    parent_fd: int,
    entry_path: Path,
    entry_parts: tuple[str, ...],
    metadata: os.stat_result,
    child_depth: int,
    root_label: str,
    display_path: str,
    parent_identities: tuple[SourceIdentity, ...],
    state: _DiscoveryState,
) -> None:
    candidate_mark = len(state.candidates)
    failure_mark = len(state.failures)
    total_bytes_mark = state.total_bytes
    child_fd: int | None = None
    primary_error: ImportSelectionError | None = None
    scan_succeeded = False
    close_failed = False
    try:
        child_fd = os.open(
            entry.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        opened_metadata = os.fstat(child_fd)
        if _same_object(metadata, opened_metadata):
            scan_succeeded = _scan_directory_fd(
                directory_fd=child_fd,
                directory_path=entry_path,
                relative_parts=entry_parts,
                parent_depth=child_depth,
                root_label=root_label,
                directory_identities=(
                    *parent_identities,
                    _identity_from_stat(opened_metadata),
                ),
                state=state,
            )
    except ImportSelectionError as error:
        primary_error = error
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = _selection_error(
            state.bounds,
            "secure_discovery_unavailable",
        )
    except (OSError, TypeError, ValueError):
        scan_succeeded = False
    finally:
        close_failed = _close_descriptors([child_fd]) if child_fd is not None else False
    if primary_error is not None:
        raise primary_error
    if close_failed:
        scan_succeeded = False
    if not scan_succeeded:
        del state.candidates[candidate_mark:]
        del state.failures[failure_mark:]
        state.total_bytes = total_bytes_mark
        _add_failure(state, entry_path, display_path, "nested_unavailable")


def _admit_file(
    path: Path,
    display_path: str,
    metadata: os.stat_result,
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


def _identity_from_stat(metadata: os.stat_result) -> SourceIdentity:
    return SourceIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _same_object(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(second.st_mode)
        and first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
    )


def _open_verified_directory(
    selected: _SelectedPath,
    bounds: ImportBounds,
) -> int:
    """Reopen a selected directory through its verified component identities."""
    descriptors: list[int] = []
    returned_leaf_fd: int | None = None
    primary_error: ImportSelectionError | None = None
    close_failed = False
    try:
        flags = _directory_open_flags()
        current_fd = os.open(selected.path.anchor, flags)
        descriptors.append(current_fd)
        root_metadata = os.fstat(current_fd)
        if not _identity_matches(selected.parent_identities[0], root_metadata):
            _reject(bounds, "selection_changed")

        components = selected.path.parts[1:]
        for index, component in enumerate(components):
            metadata = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata):
                _reject(bounds, "selected_symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                _reject(bounds, "selection_changed")
            next_fd = os.open(component, flags, dir_fd=current_fd)
            descriptors.append(next_fd)
            opened_metadata = os.fstat(next_fd)
            expected_identity = (
                _identity_from_stat(selected.metadata)
                if index == len(components) - 1
                else selected.parent_identities[index + 1]
            )
            if not _identity_matches(expected_identity, opened_metadata):
                _reject(bounds, "selection_changed")
            current_fd = next_fd
        returned_leaf_fd = descriptors.pop()
    except ImportSelectionError as error:
        primary_error = error
    except FileNotFoundError:
        primary_error = _selection_error(bounds, "selection_missing")
    except (_SecureDiscoveryUnavailable, NotImplementedError):
        primary_error = _selection_error(bounds, "secure_discovery_unavailable")
    except (OSError, TypeError, ValueError):
        primary_error = _selection_error(bounds, "selection_unreadable")
    finally:
        close_failed = _close_descriptors(descriptors)

    if primary_error is not None:
        raise primary_error
    if returned_leaf_fd is None:
        _reject(bounds, "selection_unreadable")
    if close_failed:
        _close_descriptors([returned_leaf_fd])
        _reject(bounds, "selection_unreadable")
    return returned_leaf_fd


def _directory_open_flags() -> int:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow_flag = getattr(os, "O_NOFOLLOW", None)
    if (
        not isinstance(directory_flag, int)
        or directory_flag <= 0
        or not isinstance(no_follow_flag, int)
        or no_follow_flag <= 0
    ):
        raise _SecureDiscoveryUnavailable
    flags = os.O_RDONLY | directory_flag | no_follow_flag
    close_on_exec_flag = getattr(os, "O_CLOEXEC", 0)
    if isinstance(close_on_exec_flag, int) and close_on_exec_flag > 0:
        flags |= close_on_exec_flag
    return flags


def _file_open_flags() -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", None)
    nonblocking_flag = getattr(os, "O_NONBLOCK", None)
    if (
        not isinstance(no_follow_flag, int)
        or no_follow_flag <= 0
        or not isinstance(nonblocking_flag, int)
        or nonblocking_flag <= 0
    ):
        raise _SecureDiscoveryUnavailable
    flags = os.O_RDONLY | no_follow_flag | nonblocking_flag
    close_on_exec_flag = getattr(os, "O_CLOEXEC", 0)
    if isinstance(close_on_exec_flag, int) and close_on_exec_flag > 0:
        flags |= close_on_exec_flag
    return flags


def _close_descriptors(descriptors: Iterable[int]) -> bool:
    """Close descriptors completely and report failure without leaking OS text."""
    failed = False
    interruption: BaseException | None = None
    for descriptor in reversed(tuple(descriptors)):
        try:
            os.close(descriptor)
        except OSError:  # Descriptor errors become a path-free cleanup failure.
            failed = True
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
            if interruption is None:
                interruption = error
    if interruption is not None:
        raise interruption
    return failed


def _identity_matches(identity: SourceIdentity, metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and identity.device == metadata.st_dev
        and identity.inode == metadata.st_ino
    )


def _file_identity_matches(
    identity: SourceIdentity,
    metadata: os.stat_result,
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not _is_link_or_reparse(metadata)
        and identity.device == metadata.st_dev
        and identity.inode == metadata.st_ino
        and identity.mode == metadata.st_mode
        and identity.size == metadata.st_size
        and identity.modified_ns == metadata.st_mtime_ns
        and identity.changed_ns == metadata.st_ctime_ns
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


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


def _display_path(root_label: str, relative_parts: tuple[str, ...]) -> str:
    return PurePosixPath(root_label, *relative_parts).as_posix()


def _unsafe_failure_display_path(
    root_label: str,
    relative_parts: tuple[str, ...],
) -> str:
    encoded_parts = tuple(_quote_filename_segment(part) for part in relative_parts)
    return _display_path(root_label, (".unsafe-entry", *encoded_parts))


def _quote_filename_segment(segment: str) -> str:
    try:
        encoded = os.fsencode(segment)
    except UnicodeEncodeError:
        encoded = segment.encode("utf-8", errors="surrogatepass")
    return quote_from_bytes(encoded, safe="-._~")


def _disambiguate_failure_paths(
    candidates: tuple[DiscoveredImportSource, ...],
    failures: tuple[ImportDiscoveryFailure, ...],
) -> tuple[ImportDiscoveryFailure, ...]:
    used = {
        _display_collision_key(candidate.source.display_path)
        for candidate in candidates
    }
    disambiguated: list[ImportDiscoveryFailure] = []
    for failure in failures:
        display_path = failure.display_path
        suffix = 1
        while _display_collision_key(display_path) in used:
            display_path = f"{failure.display_path}~failure-{suffix}"
            suffix += 1
        used.add(_display_collision_key(display_path))
        disambiguated.append(
            failure
            if display_path == failure.display_path
            else replace(failure, display_path=display_path)
        )
    return tuple(disambiguated)


def _is_safe_display_segment(segment: str) -> bool:
    return bool(
        segment
        and segment == segment.strip()
        and segment not in {".", ".."}
        and "/" not in segment
        and "\\" not in segment
        and "\x00" not in segment
        and all(character.isprintable() for character in segment)
    )


def _is_safe_display_path(display_path: str) -> bool:
    path = PurePosixPath(display_path)
    return (
        not path.is_absolute()
        and all(_is_safe_display_segment(part) for part in path.parts)
        and path != PurePosixPath(".")
    )


def _display_sort_key(display_path: str) -> tuple[str, str]:
    return _display_collision_key(display_path), display_path


def _display_collision_key(display_path: str) -> str:
    return normalize("NFKC", display_path).casefold()


def _bounded_message(bounds: ImportBounds, reason_code: str) -> str:
    return _MESSAGES[reason_code][: bounds.max_reason_length]


def _reject(bounds: ImportBounds, reason_code: str) -> None:
    raise ImportSelectionError(reason_code, _bounded_message(bounds, reason_code))


def _selection_error(
    bounds: ImportBounds,
    reason_code: str,
) -> ImportSelectionError:
    return ImportSelectionError(reason_code, _bounded_message(bounds, reason_code))
