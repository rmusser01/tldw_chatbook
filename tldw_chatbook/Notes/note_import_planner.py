"""Read-only, bounded source discovery for one-time Database Notes imports."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from tldw_chatbook.Notes.note_import_plan_models import (
    ImportBounds,
    ImportSource,
    ImportSourceKind,
)


class ImportSelectionError(ValueError):
    """A stable, path-redacted failure that rejects the whole selection."""

    def __init__(self, reason_code: str, user_message: str) -> None:
        self.reason_code = reason_code
        self.user_message = user_message
        super().__init__(user_message)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Non-following metadata for a parser's later pre-open recheck."""

    device: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    size: int = field(repr=False)
    modified_ns: int = field(repr=False)
    changed_ns: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class DiscoveredImportSource:
    """One regular file admitted by bounded discovery."""

    source: ImportSource
    size_bytes: int
    identity: SourceIdentity = field(repr=False)


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


_MESSAGES = {
    "empty_selection": "Choose at least one file or one folder.",
    "invalid_selection": "The selected source is not valid.",
    "selection_missing": "A selected source is no longer available.",
    "selection_unreadable": "A selected source cannot be inspected safely.",
    "selected_symlink": "Linked files and folders cannot be selected.",
    "selection_not_regular": "Select regular files or one folder.",
    "mixed_selection": "Choose files or one folder, not both.",
    "multiple_directories": "Choose only one folder at a time.",
    "ambiguous_display_path": "Selected files must have distinct names.",
    "unsafe_display_path": "A selected source has an unsafe display name.",
    "selection_changed": "A selected source changed during inspection.",
    "max_depth_exceeded": "The selected folder is nested too deeply.",
    "max_files_exceeded": "The selection contains too many files.",
    "max_file_bytes_exceeded": "A selected file is too large.",
    "max_total_bytes_exceeded": "The selected files are too large in total.",
    "max_entries_exceeded": "The selected folder contains too many entries.",
    "nested_symlink": "Linked entries are skipped for safety.",
    "nested_not_regular": "This entry is not a regular file and will be skipped.",
    "nested_unavailable": "This entry cannot be inspected safely and will be skipped.",
}


def discover_import_sources(
    paths: Iterable[Path],
    bounds: ImportBounds,
) -> ImportDiscovery:
    """Validate a file-only or single-directory selection without reading content."""
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")

    selected_paths = _copy_bounded_selection(paths, bounds)
    selected = [
        (_absolute_path(path), _selected_lstat(path, bounds)) for path in selected_paths
    ]
    kinds = [_mode_kind(metadata.st_mode) for _, metadata in selected]

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
        root_path, root_metadata = selected[0]
        root_label = root_path.name
        if not _is_safe_display_segment(root_label):
            _reject(bounds, "unsafe_display_path")
        _scan_selected_directory(root_path, root_metadata, root_label, state)
    else:
        root_label = None
        _admit_selected_files(selected, state)

    candidates = tuple(
        sorted(
            state.candidates,
            key=lambda item: _display_sort_key(item.source.display_path),
        )
    )
    failures = tuple(
        sorted(state.failures, key=lambda item: _display_sort_key(item.display_path))
    )
    return ImportDiscovery(
        candidates=candidates,
        failures=failures,
        root_label=root_label,
        total_bytes=state.total_bytes,
        entry_count=state.entry_count,
    )


def _copy_bounded_selection(
    paths: Iterable[Path], bounds: ImportBounds
) -> tuple[Path, ...]:
    if isinstance(paths, (str, bytes, Path)):
        _reject(bounds, "invalid_selection")
    selected: list[Path] = []
    for path in paths:
        if not isinstance(path, Path):
            _reject(bounds, "invalid_selection")
        selected.append(path)
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


def _selected_lstat(path: Path, bounds: ImportBounds) -> os.stat_result:
    absolute_path = _absolute_path(path)
    try:
        return absolute_path.lstat()
    except FileNotFoundError:
        _reject(bounds, "selection_missing")
    except OSError:
        _reject(bounds, "selection_unreadable")


def _mode_kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _admit_selected_files(
    selected: list[tuple[Path, os.stat_result]],
    state: _DiscoveryState,
) -> None:
    names: set[str] = set()
    ordered = sorted(selected, key=lambda item: _display_sort_key(item[0].name))
    for path, metadata in ordered:
        display_path = path.name
        normalized_name = display_path.casefold()
        if not _is_safe_display_segment(display_path):
            _reject(state.bounds, "unsafe_display_path")
        if normalized_name in names:
            _reject(state.bounds, "ambiguous_display_path")
        names.add(normalized_name)
        state.entry_count += 1
        _admit_file(
            path,
            display_path,
            metadata,
            ImportSourceKind.SELECTED_FILE,
            state,
        )


def _scan_selected_directory(
    root_path: Path,
    root_metadata: os.stat_result,
    root_label: str,
    state: _DiscoveryState,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root_path, flags)
    except OSError:
        _reject(state.bounds, "selection_unreadable")
    try:
        if not _same_object(root_metadata, os.fstat(root_fd)):
            _reject(state.bounds, "selection_changed")
        _scan_directory_fd(
            directory_fd=root_fd,
            directory_path=root_path,
            relative_parts=(),
            parent_depth=0,
            root_label=root_label,
            state=state,
            is_selected_root=True,
        )
    finally:
        os.close(root_fd)


def _scan_directory_fd(
    *,
    directory_fd: int,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    state: _DiscoveryState,
    is_selected_root: bool = False,
) -> None:
    try:
        with os.scandir(directory_fd) as iterator:
            entries: list[os.DirEntry[str]] = []
            for entry in iterator:
                state.entry_count += 1
                if state.entry_count > state.bounds.max_entries:
                    _reject(state.bounds, "max_entries_exceeded")
                entries.append(entry)
            entries.sort(key=lambda entry: _display_sort_key(entry.name))
            for entry in entries:
                _scan_entry(
                    entry=entry,
                    directory_fd=directory_fd,
                    directory_path=directory_path,
                    relative_parts=relative_parts,
                    parent_depth=parent_depth,
                    root_label=root_label,
                    state=state,
                )
    except OSError:
        if is_selected_root:
            _reject(state.bounds, "selection_unreadable")
        else:
            _add_failure(
                state,
                directory_path,
                _display_path(root_label, relative_parts),
                "nested_unavailable",
            )


def _scan_entry(
    *,
    entry: os.DirEntry[str],
    directory_fd: int,
    directory_path: Path,
    relative_parts: tuple[str, ...],
    parent_depth: int,
    root_label: str,
    state: _DiscoveryState,
) -> None:
    entry_path = directory_path / entry.name
    entry_parts = (*relative_parts, entry.name)
    display_path = _display_path(root_label, entry_parts)
    if not _is_safe_display_path(display_path):
        safe_parts = tuple(quote(part, safe="-._~") for part in entry_parts)
        safe_display_path = _display_path(root_label, safe_parts)
        _add_failure(state, entry_path, safe_display_path, "nested_unavailable")
        return
    try:
        metadata = entry.stat(follow_symlinks=False)
    except OSError:
        _add_failure(state, entry_path, display_path, "nested_unavailable")
        return

    kind = _mode_kind(metadata.st_mode)
    if kind == "symlink":
        _add_failure(state, entry_path, display_path, "nested_symlink")
    elif kind == "other":
        _add_failure(state, entry_path, display_path, "nested_not_regular")
    elif kind == "file":
        _admit_file(
            entry_path,
            display_path,
            metadata,
            ImportSourceKind.DIRECTORY_MEMBER,
            state,
        )
    else:
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
    state: _DiscoveryState,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        child_fd = os.open(entry.name, flags, dir_fd=parent_fd)
    except OSError:
        _add_failure(state, entry_path, display_path, "nested_unavailable")
        return
    try:
        if not _same_object(metadata, os.fstat(child_fd)):
            _add_failure(state, entry_path, display_path, "nested_unavailable")
            return
        _scan_directory_fd(
            directory_fd=child_fd,
            directory_path=entry_path,
            relative_parts=entry_parts,
            parent_depth=child_depth,
            root_label=root_label,
            state=state,
        )
    finally:
        os.close(child_fd)


def _admit_file(
    path: Path,
    display_path: str,
    metadata: os.stat_result,
    kind: ImportSourceKind,
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
    return display_path.casefold(), display_path


def _bounded_message(bounds: ImportBounds, reason_code: str) -> str:
    return _MESSAGES[reason_code][: bounds.max_reason_length]


def _reject(bounds: ImportBounds, reason_code: str) -> None:
    raise ImportSelectionError(reason_code, _bounded_message(bounds, reason_code))
