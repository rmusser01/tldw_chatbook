"""Pure, read-only filesystem inspection under a revalidated workspace binding.

This module deliberately has no Textual, database, File Notes, Git, or logging
dependency.  A binding scope is an address captured at modal-open time, never a
capability: every public filesystem operation obtains and compares fresh
registry and filesystem facts before doing any work.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
import codecs
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Protocol

from .models import (
    DEFAULT_WORKSPACE_ID,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
)


DIRECTORY_PAGE_SIZE = 200
DIRECTORY_SCAN_LIMIT = 10_000
FILTER_DEBOUNCE_MS = 150
FILTER_VISIT_LIMIT = 50_000
FILTER_RESULT_LIMIT = 500
METADATA_ONLY_BYTES = 8 * 1024 * 1024
PAGED_TEXT_THRESHOLD = 200_000
TEXT_PAGE_SIZE = 100_000

_VCS_DIRECTORY_NAMES = frozenset({".git", ".hg", ".svn"})
_CACHE_DIRECTORY_NAMES = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
    }
)
_BIDI_CODEPOINTS = frozenset(
    {*range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x200E, 0x200F}
)


class WorkspaceRegistry(Protocol):
    """Small registry surface needed by the read-only inspector."""

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        """Get a workspace by stable ID."""

    def get_runtime_binding(
        self, binding_id: str
    ) -> WorkspaceRuntimeBinding | None:
        """Get a runtime binding by stable ID."""


class DirectoryStatus(str, Enum):
    """Directory-page outcome with explicit partial and failure states."""

    COMPLETE = "complete"
    EMPTY = "empty"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    FAILED = "failed"


class FilterStatus(str, Enum):
    """Bounded selected-binding filter outcomes."""

    COMPLETE = "complete"
    EMPTY = "empty"
    ONLY_EXCLUDED = "only_excluded"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FileReadKind(str, Enum):
    """Safe file-view classifications."""

    TEXT = "text"
    CONTROL_TEXT = "control_text"
    INVALID_UTF8 = "invalid_utf8"
    METADATA_ONLY = "metadata_only"
    PAGED = "paged"
    REVISION_CHANGED = "revision_changed"
    FAILED = "failed"


@dataclass(frozen=True)
class FileRevision:
    """Stable enough read identity for one regular-file observation."""

    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class DirectoryRevision:
    """Identity used to prevent combining different directory pages."""

    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True)
class BindingScope:
    """Immutable opening address for exactly one local-folder binding."""

    workspace_id: str
    binding_id: str
    binding_fingerprint: str
    canonical_root: str
    root_device: int
    root_inode: int


@dataclass(frozen=True)
class FileRef:
    """A raw root-relative path; display text is intentionally separate."""

    raw_parts: tuple[str, ...]
    display_path: str


@dataclass(frozen=True)
class DirectoryEntry:
    """One safe direct child visible in a directory page."""

    raw_parts: tuple[str, ...]
    display_name: str
    is_directory: bool


@dataclass(frozen=True)
class DirectoryContinuation:
    """Opaque page identity, bound to a scope and one directory revision."""

    binding_fingerprint: str
    directory_parts: tuple[str, ...]
    directory_revision: DirectoryRevision
    offset: int
    token: str


@dataclass(frozen=True)
class DirectoryPage:
    """A bounded direct-child page, never a recursive snapshot."""

    status: DirectoryStatus
    entries: tuple[DirectoryEntry, ...] = ()
    continuation: DirectoryContinuation | None = None
    error_code: str | None = None
    excluded_vcs_count: int = 0
    excluded_cache_count: int = 0
    scanned_entries: int = 0


@dataclass(frozen=True)
class FilterResult:
    """A literal selected-binding filter result with visible bounds."""

    status: FilterStatus
    matches: tuple[FileRef, ...] = ()
    visited_entries: int = 0
    excluded_count: int = 0
    status_copy: str = ""
    error_code: str | None = None
    progress: "FilterProgress" | None = None
    excluded_locations_unsearched: bool = False


@dataclass(frozen=True)
class FilterProgress:
    """Typed incremental filter observation for a caller-owned worker lane."""

    visited_entries: int
    matched_entries: int


@dataclass(frozen=True)
class FileReadResult:
    """Safe metadata, preview, or a revision-pinned decoded page."""

    kind: FileReadKind
    revision: FileRevision | None = None
    size_bytes: int = 0
    text: str = ""
    character_range: tuple[int, int] | None = None
    total_characters: int | None = None
    next_page_offset: int | None = None
    previous_page_offset: int | None = None
    error_code: str | None = None


class ScopeCaptureError(ValueError):
    """Raised when a modal cannot initially capture a safe binding address."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def safe_filesystem_text(value: str) -> str:
    """Render filesystem-derived text without terminal/markup control effects.

    The result is deliberately one-way.  Callers retain the original raw path
    components for later authority checks rather than attempting to parse this
    display string.
    """
    escaped: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\n":
            escaped.append("\\n")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\t":
            escaped.append("\\t")
        elif 0xDC80 <= codepoint <= 0xDCFF:
            escaped.append(f"\\x{codepoint - 0xDC00:02x}")
        elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            escaped.append(f"\\x{codepoint:02x}")
        elif codepoint in _BIDI_CODEPOINTS or 0xD800 <= codepoint <= 0xDFFF:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    return "".join(escaped)


class WorkspaceFileInspector:
    """Synchronous safe reads under scopes revalidated for every operation."""

    filter_debounce_ms = FILTER_DEBOUNCE_MS

    def __init__(self, registry: WorkspaceRegistry) -> None:
        self._registry = registry
        self._page_cache: dict[
            tuple[str, tuple[str, ...], FileRevision], OrderedDict[int, tuple[str, int]]
        ] = {}
        self._continuations: dict[str, DirectoryContinuation] = {}

    def capture_binding(self, workspace_id: str, binding_id: str) -> BindingScope:
        """Capture one immutable local-folder address for an inspector visit."""
        scope, error_code = self._current_scope(workspace_id, binding_id)
        if scope is None:
            raise ScopeCaptureError(error_code)
        return scope

    def list_directory(
        self,
        scope: BindingScope,
        directory_parts: tuple[str, ...] = (),
        *,
        continuation: DirectoryContinuation | None = None,
    ) -> DirectoryPage:
        """List one bounded direct-child page after fresh scope revalidation."""
        current, error_code = self._revalidate(scope)
        if current is None:
            return DirectoryPage(DirectoryStatus.FAILED, error_code=error_code)
        if continuation is not None:
            if (
                continuation.binding_fingerprint != scope.binding_fingerprint
                or continuation.directory_parts != directory_parts
                or continuation.offset < 0
                or continuation.offset % DIRECTORY_PAGE_SIZE
                or self._continuations.get(continuation.token) != continuation
            ):
                return DirectoryPage(DirectoryStatus.FAILED, error_code="invalid_page")
            del self._continuations[continuation.token]
        root_fd, error_code = _open_root_descriptor(current)
        if root_fd is None:
            return DirectoryPage(DirectoryStatus.FAILED, error_code=error_code)
        directory_fd, _directory_file_revision, error_code = _open_target_descriptor(
            root_fd, directory_parts, "dir"
        )
        os.close(root_fd)
        if directory_fd is None:
            return DirectoryPage(DirectoryStatus.FAILED, error_code=error_code)
        directory_revision = _directory_revision_from_stat(os.fstat(directory_fd))
        if (
            continuation is not None
            and continuation.directory_revision != directory_revision
        ):
            os.close(directory_fd)
            return DirectoryPage(DirectoryStatus.FAILED, error_code="directory_changed")

        entries: list[DirectoryEntry] = []
        vcs_count = 0
        cache_count = 0
        scanned = 0
        try:
            with os.scandir(directory_fd) as iterator:
                for entry in iterator:
                    scanned += 1
                    if scanned > DIRECTORY_SCAN_LIMIT:
                        break
                    kind = _entry_kind(entry)
                    if kind in {None, "link"}:
                        continue
                    if kind == "vcs":
                        vcs_count += 1
                        continue
                    if kind == "cache":
                        cache_count += 1
                        continue
                    entries.append(
                        DirectoryEntry(
                            raw_parts=directory_parts + (entry.name,),
                            display_name=safe_filesystem_text(entry.name),
                            is_directory=kind == "dir",
                        )
                    )
        except OSError:
            return DirectoryPage(DirectoryStatus.FAILED, error_code="directory_unavailable")
        finally:
            os.close(directory_fd)
        entries.sort(key=lambda item: (not item.is_directory, item.raw_parts[-1].casefold(), item.raw_parts[-1]))
        offset = continuation.offset if continuation is not None else 0
        page_entries = tuple(entries[offset : offset + DIRECTORY_PAGE_SIZE])
        more_entries = offset + len(page_entries) < len(entries)
        capped = scanned > DIRECTORY_SCAN_LIMIT
        next_page = (
            DirectoryContinuation(
                binding_fingerprint=scope.binding_fingerprint,
                directory_parts=directory_parts,
                directory_revision=directory_revision,
                offset=offset + len(page_entries),
                token=secrets.token_urlsafe(24),
            )
            if more_entries
            else None
        )
        if next_page is not None:
            self._continuations[next_page.token] = next_page
            while len(self._continuations) > 8:
                self._continuations.pop(next(iter(self._continuations)))
        if capped:
            status = DirectoryStatus.TRUNCATED
        elif next_page is not None:
            status = DirectoryStatus.PARTIAL
        elif not entries:
            status = DirectoryStatus.EMPTY
        else:
            status = DirectoryStatus.COMPLETE
        return DirectoryPage(
            status,
            page_entries,
            next_page,
            excluded_vcs_count=vcs_count,
            excluded_cache_count=cache_count,
            scanned_entries=min(scanned, DIRECTORY_SCAN_LIMIT),
        )

    def filter_paths(
        self,
        scope: BindingScope,
        query: str,
        *,
        is_cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[[FilterProgress], None] | None = None,
    ) -> FilterResult:
        """Literal case-insensitive recursive filter under exactly one binding."""
        current, error_code = self._revalidate(scope)
        if current is None:
            return FilterResult(FilterStatus.FAILED, error_code=error_code)
        if not isinstance(query, str):
            return FilterResult(FilterStatus.FAILED, error_code="invalid_query")
        needle = query.casefold()
        pending: list[tuple[str, ...]] = [()]
        matches: list[FileRef] = []
        visited = 0
        excluded = 0
        while pending:
            parent_parts = pending.pop()
            root_fd, error_code = _open_root_descriptor(current)
            if root_fd is None:
                return FilterResult(FilterStatus.FAILED, error_code=error_code)
            directory_fd, _revision, error_code = _open_target_descriptor(
                root_fd, parent_parts, "dir"
            )
            os.close(root_fd)
            if directory_fd is None:
                return FilterResult(FilterStatus.FAILED, error_code=error_code)
            try:
                iterator = os.scandir(directory_fd)
            except OSError:
                os.close(directory_fd)
                return FilterResult(
                    FilterStatus.FAILED,
                    tuple(matches),
                    visited,
                    excluded,
                    error_code="directory_unavailable",
                )
            try:
                for entry in iterator:
                    if is_cancelled is not None and is_cancelled():
                        return FilterResult(
                            FilterStatus.CANCELLED, tuple(matches), visited, excluded,
                            "Filter cancelled.", progress=FilterProgress(visited, len(matches)),
                        )
                    if visited >= FILTER_VISIT_LIMIT:
                        return FilterResult(
                            FilterStatus.PARTIAL, tuple(matches), visited, excluded,
                            "Filter stopped after 50,000 entries.", progress=FilterProgress(visited, len(matches)),
                        )
                    visited += 1
                    if on_progress is not None:
                        on_progress(FilterProgress(visited, len(matches)))
                    kind = _entry_kind(entry)
                    if kind in {None, "link"}:
                        continue
                    if kind in {"vcs", "cache"}:
                        excluded += 1
                        continue
                    raw_parts = parent_parts + (entry.name,)
                    relative_display = "/".join(raw_parts)
                    if needle in relative_display.casefold():
                        matches.append(FileRef(raw_parts, _display_parts(raw_parts)))
                        if len(matches) >= FILTER_RESULT_LIMIT:
                            return FilterResult(
                                FilterStatus.TRUNCATED, tuple(matches), visited, excluded,
                                "Showing the first 500 matching paths.",
                                progress=FilterProgress(visited, len(matches)),
                            )
                    if kind == "dir":
                        pending.append(raw_parts)
            finally:
                iterator.close()
                os.close(directory_fd)
        matches.sort(key=lambda item: (item.display_path.casefold(), item.display_path))
        if matches:
            status = FilterStatus.COMPLETE
            copy = f"{len(matches)} matching paths."
        elif excluded:
            status = FilterStatus.ONLY_EXCLUDED
            copy = "No visible matches; excluded folders were not searched."
        else:
            status = FilterStatus.EMPTY
            copy = "No matching paths."
        return FilterResult(
            status, tuple(matches), visited, excluded, copy,
            progress=FilterProgress(visited, len(matches)),
            excluded_locations_unsearched=bool(excluded),
        )

    def read_file(
        self,
        scope: BindingScope,
        raw_parts: tuple[str, ...],
        *,
        page_offset: int | None = None,
        expected_revision: FileRevision | None = None,
    ) -> FileReadResult:
        """Read a safe small preview or one revision-pinned text page."""
        current, error_code = self._revalidate(scope)
        if current is None:
            return FileReadResult(FileReadKind.FAILED, error_code=error_code)
        root_fd, error_code = _open_root_descriptor(current)
        if root_fd is None:
            return FileReadResult(FileReadKind.FAILED, error_code=error_code)
        descriptor, revision, error_code = _open_target_descriptor(root_fd, raw_parts, "file")
        os.close(root_fd)
        if descriptor is None or revision is None:
            return FileReadResult(FileReadKind.FAILED, error_code=error_code)
        try:
            if expected_revision is not None and revision != expected_revision:
                return FileReadResult(
                    FileReadKind.REVISION_CHANGED,
                    revision=revision,
                    size_bytes=revision.size,
                    error_code="revision_changed",
                )
            if revision.size > METADATA_ONLY_BYTES:
                return FileReadResult(
                    FileReadKind.METADATA_ONLY,
                    revision=revision,
                    size_bytes=revision.size,
                )
            offset = page_offset or 0
            cache_key = (scope.binding_fingerprint, raw_parts, revision)
            cached = self._page_cache.get(cache_key, {}).get(offset)
            if cached is not None:
                cached_text, cached_total = cached
                return FileReadResult(
                    FileReadKind.PAGED,
                    revision=revision,
                    size_bytes=revision.size,
                    text=safe_filesystem_text(cached_text),
                    character_range=(offset, min(offset + TEXT_PAGE_SIZE, cached_total)),
                    total_characters=cached_total,
                    next_page_offset=(offset + TEXT_PAGE_SIZE if offset + TEXT_PAGE_SIZE < cached_total else None),
                    previous_page_offset=max(0, offset - TEXT_PAGE_SIZE) if offset else None,
                )
            decoded_page = _decode_text_page(descriptor, page_offset or 0)
            final_revision = _revision_from_stat(os.fstat(descriptor))
            if final_revision != revision:
                return FileReadResult(
                    FileReadKind.REVISION_CHANGED,
                    revision=final_revision,
                    size_bytes=final_revision.size,
                    error_code="revision_changed",
                )
        except OSError:
            return FileReadResult(FileReadKind.FAILED, error_code="read_failed")
        finally:
            os.close(descriptor)
        if decoded_page is None:
            return FileReadResult(
                FileReadKind.INVALID_UTF8,
                revision=revision,
                size_bytes=revision.size,
                error_code="invalid_utf8",
            )
        decoded, page_text, total_characters = decoded_page
        if decoded is not None:
            kind = (
                FileReadKind.CONTROL_TEXT
                if _contains_control_text(decoded)
                else FileReadKind.TEXT
            )
            return FileReadResult(
                kind,
                revision=revision,
                size_bytes=revision.size,
                text=safe_filesystem_text(decoded),
                character_range=(0, len(decoded)),
                total_characters=len(decoded),
            )
        offset = page_offset or 0
        if offset < 0 or offset >= total_characters:
            return FileReadResult(FileReadKind.FAILED, error_code="invalid_page")
        end = min(offset + TEXT_PAGE_SIZE, total_characters)
        cache_key = (scope.binding_fingerprint, raw_parts, revision)
        cache = self._page_cache.setdefault(cache_key, OrderedDict())
        cache[offset] = (page_text, total_characters)
        for cached_offset in tuple(cache):
            if abs(cached_offset - offset) > TEXT_PAGE_SIZE:
                del cache[cached_offset]
        return FileReadResult(
            FileReadKind.PAGED,
            revision=revision,
            size_bytes=revision.size,
            text=safe_filesystem_text(page_text),
            character_range=(offset, end),
            total_characters=total_characters,
            next_page_offset=end if end < total_characters else None,
            previous_page_offset=max(0, offset - TEXT_PAGE_SIZE) if offset else None,
        )

    def cached_page_offsets(
        self, scope: BindingScope, raw_parts: tuple[str, ...]
    ) -> tuple[int, ...]:
        """Expose only sparse cached offsets for the modal's page contract."""
        offsets: list[int] = []
        for (fingerprint, candidate_parts, _revision), cache in self._page_cache.items():
            if fingerprint == scope.binding_fingerprint and candidate_parts == raw_parts:
                offsets.extend(cache)
        return tuple(sorted(offsets))

    @property
    def continuation_count(self) -> int:
        """Bounded number of outstanding service-issued directory tokens."""
        return len(self._continuations)

    def _current_scope(
        self, workspace_id: str, binding_id: str
    ) -> tuple[BindingScope | None, str]:
        try:
            workspace = self._registry.get_workspace(workspace_id)
            binding = self._registry.get_runtime_binding(binding_id)
        except Exception:  # Registry errors are intentionally not surfaced verbatim.
            return None, "registry_unavailable"
        if workspace is None or workspace.archived or workspace_id == DEFAULT_WORKSPACE_ID:
            return None, "workspace_unavailable"
        if (
            binding is None
            or binding.workspace_id != workspace_id
            or binding.binding_kind is not RuntimeBindingKind.LOCAL_FILESYSTEM
            or binding.status is not RuntimeBindingStatus.READY
        ):
            return None, "binding_changed"
        root = Path(binding.locator)
        try:
            root_stat = os.lstat(root)
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
                return None, "binding_changed"
            canonical_root = root.resolve(strict=True)
            if canonical_root != root:
                return None, "binding_changed"
        except OSError:
            return None, "binding_changed"
        return (
            BindingScope(
                workspace_id=workspace_id,
                binding_id=binding_id,
                binding_fingerprint=_binding_fingerprint(binding),
                canonical_root=str(canonical_root),
                root_device=root_stat.st_dev,
                root_inode=root_stat.st_ino,
            ),
            "",
        )

    def _revalidate(self, scope: BindingScope) -> tuple[BindingScope | None, str]:
        current, error_code = self._current_scope(scope.workspace_id, scope.binding_id)
        if current is None:
            return None, error_code
        if current != scope:
            return None, "binding_changed"
        return current, ""

    def _resolve_target(
        self,
        scope: BindingScope,
        raw_parts: tuple[str, ...],
        expected_kind: str,
    ) -> tuple[Path | None, str]:
        if not _safe_relative_parts(raw_parts):
            return None, "invalid_path"
        if any(part in _VCS_DIRECTORY_NAMES for part in raw_parts):
            return None, "excluded_path"
        target = Path(scope.canonical_root)
        try:
            for part in raw_parts:
                target = target / part
                target_stat = os.lstat(target)
                if stat.S_ISLNK(target_stat.st_mode):
                    return None, "unsafe_target"
            target_stat = os.lstat(target)
        except FileNotFoundError:
            return None, "missing_target"
        except OSError:
            return None, "target_unavailable"
        if expected_kind == "dir" and not stat.S_ISDIR(target_stat.st_mode):
            return None, "unsupported_target"
        if expected_kind == "file" and not stat.S_ISREG(target_stat.st_mode):
            return None, "unsafe_target"
        return target, ""


def _binding_fingerprint(binding: WorkspaceRuntimeBinding) -> str:
    """Derive a stable opaque fingerprint without rendering or logging metadata."""
    payload = {
        "workspace_id": binding.workspace_id,
        "binding_id": binding.binding_id,
        "kind": binding.binding_kind.value,
        "locator": binding.locator,
        "status": binding.status.value,
        "metadata": _json_safe_mapping(binding.metadata),
        "updated_at": binding.updated_at,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def _json_safe_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Convert registry metadata into deterministic JSON-compatible values."""
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_relative_parts(raw_parts: tuple[str, ...]) -> bool:
    if not isinstance(raw_parts, tuple):
        return False
    for part in raw_parts:
        if not isinstance(part, str) or not part or part in {".", ".."}:
            return False
        if Path(part).is_absolute() or "/" in part or "\\" in part:
            return False
    return True


def _directory_revision(path: Path) -> DirectoryRevision | None:
    try:
        observed = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        return None
    return DirectoryRevision(observed.st_dev, observed.st_ino, observed.st_mtime_ns)


def _directory_revision_from_stat(observed: os.stat_result) -> DirectoryRevision:
    return DirectoryRevision(observed.st_dev, observed.st_ino, observed.st_mtime_ns)


def _entry_kind(entry: os.DirEntry[str]) -> str | None:
    """Classify one direct entry without following symlinks."""
    try:
        if entry.is_symlink():
            return "link"
        observed = entry.stat(follow_symlinks=False)
    except OSError:
        return None
    if stat.S_ISLNK(observed.st_mode):
        return "link"
    if entry.name in _VCS_DIRECTORY_NAMES:
        return "vcs"
    if stat.S_ISDIR(observed.st_mode):
        return "cache" if entry.name in _CACHE_DIRECTORY_NAMES else "dir"
    if stat.S_ISREG(observed.st_mode):
        return "file"
    return None


def _display_parts(raw_parts: tuple[str, ...]) -> str:
    return "/".join(safe_filesystem_text(part) for part in raw_parts)


def _safe_open_flags(*, directory: bool) -> int | None:
    """Return no-follow descriptor flags or fail closed without support."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        return None
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _open_root_descriptor(scope: BindingScope) -> tuple[int | None, str]:
    flags = _safe_open_flags(directory=True)
    if flags is None:
        return None, "safe_descriptor_unavailable"
    try:
        descriptor = os.open(scope.canonical_root, flags)
        observed = os.fstat(descriptor)
    except OSError:
        return None, "binding_changed"
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != scope.root_device
        or observed.st_ino != scope.root_inode
    ):
        os.close(descriptor)
        return None, "binding_changed"
    return descriptor, ""


def _open_child_directory(parent_fd: int, name: str) -> tuple[int | None, str]:
    flags = _safe_open_flags(directory=True)
    if flags is None:
        return None, "safe_descriptor_unavailable"
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        observed = os.fstat(descriptor)
    except OSError:
        return None, "unsafe_target"
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        return None, "unsafe_target"
    return descriptor, ""


def _open_target_descriptor(
    root_fd: int, raw_parts: tuple[str, ...], expected_kind: str
) -> tuple[int | None, FileRevision | None, str]:
    """Traverse only root-anchored no-follow descriptors, never path strings."""
    if not _safe_relative_parts(raw_parts):
        return None, None, "invalid_path"
    if any(part in _VCS_DIRECTORY_NAMES for part in raw_parts):
        return None, None, "excluded_path"
    current_fd = os.dup(root_fd)
    success = False
    try:
        for index, part in enumerate(raw_parts):
            is_final = index == len(raw_parts) - 1
            flags = _safe_open_flags(directory=not is_final or expected_kind == "dir")
            if flags is None:
                return None, None, "safe_descriptor_unavailable"
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                return None, None, "missing_target"
            except OSError:
                return None, None, "unsafe_target"
            os.close(current_fd)
            current_fd = next_fd
        observed = os.fstat(current_fd)
        if expected_kind == "dir" and not stat.S_ISDIR(observed.st_mode):
            return None, None, "unsupported_target"
        if expected_kind == "file" and not stat.S_ISREG(observed.st_mode):
            return None, None, "unsafe_target"
        success = True
        return current_fd, _revision_from_stat(observed), ""
    finally:
        if not success:
            os.close(current_fd)


def _open_regular_file(path: Path) -> tuple[int | None, FileRevision | None, str]:
    """Compatibility helper retained for callers outside this module."""
    flags = _safe_open_flags(directory=False)
    if flags is None:
        return None, None, "safe_descriptor_unavailable"
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
    except FileNotFoundError:
        return None, None, "missing_target"
    except OSError:
        return None, None, "unsafe_target"
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        return None, None, "unsafe_target"
    return descriptor, _revision_from_stat(observed), ""


def _revision_from_stat(observed: os.stat_result) -> FileRevision:
    return FileRevision(
        device=observed.st_dev,
        inode=observed.st_ino,
        size=observed.st_size,
        modified_ns=observed.st_mtime_ns,
    )


def _decode_text_page(
    descriptor: int, page_offset: int
) -> tuple[str | None, str, int] | None:
    """Incrementally decode one page without retaining a large whole file.

    The function scans to establish a stable decoded-character total, but only
    retains a complete string while it remains under the normal-preview limit.
    This keeps large-file cache state sparse and page-oriented.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    small_parts: list[str] = []
    page_parts: list[str] = []
    total = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 64 * 1024):
            decoded = decoder.decode(chunk)
            _append_decoded_range(
                decoded, total, page_offset, small_parts, page_parts
            )
            total += len(decoded)
        decoded = decoder.decode(b"", final=True)
        _append_decoded_range(decoded, total, page_offset, small_parts, page_parts)
        total += len(decoded)
    except UnicodeDecodeError:
        return None
    if total <= PAGED_TEXT_THRESHOLD:
        return "".join(small_parts), "", total
    return None, "".join(page_parts), total


def _append_decoded_range(
    decoded: str,
    start: int,
    page_offset: int,
    small_parts: list[str],
    page_parts: list[str],
) -> None:
    """Keep a small preview or the requested character-page intersection."""
    if start < PAGED_TEXT_THRESHOLD:
        small_parts.append(decoded[: PAGED_TEXT_THRESHOLD - start])
    page_end = page_offset + TEXT_PAGE_SIZE
    start_index = max(0, page_offset - start)
    end_index = min(len(decoded), page_end - start)
    if start_index < end_index:
        page_parts.append(decoded[start_index:end_index])


def _contains_control_text(value: str) -> bool:
    return any(
        (ord(character) < 0x20 and character not in {"\n", "\r", "\t"})
        or 0x7F <= ord(character) <= 0x9F
        or ord(character) in _BIDI_CODEPOINTS
        for character in value
    )


__all__ = [
    "BindingScope",
    "DIRECTORY_PAGE_SIZE",
    "DIRECTORY_SCAN_LIMIT",
    "DirectoryContinuation",
    "DirectoryEntry",
    "DirectoryPage",
    "DirectoryRevision",
    "DirectoryStatus",
    "FILTER_DEBOUNCE_MS",
    "FILTER_RESULT_LIMIT",
    "FILTER_VISIT_LIMIT",
    "FileReadKind",
    "FileReadResult",
    "FileRef",
    "FileRevision",
    "FilterResult",
    "FilterStatus",
    "METADATA_ONLY_BYTES",
    "PAGED_TEXT_THRESHOLD",
    "ScopeCaptureError",
    "TEXT_PAGE_SIZE",
    "WorkspaceFileInspector",
    "WorkspaceRegistry",
    "safe_filesystem_text",
]
