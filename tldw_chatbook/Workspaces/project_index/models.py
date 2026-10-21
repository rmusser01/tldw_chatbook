"""Immutable folder-index contracts. No runtime/service construction on import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ContextManager

from tldw_chatbook.Tools.remote_root_types import LocalRoot

if TYPE_CHECKING:
    from tldw_chatbook.Utils.filesystem_identity import DirectoryChain
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
    from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

INDEX_STATES = frozenset(
    {"Off", "Building", "Partial", "Ready", "Paused", "Interrupted", "Error"}
)
MAX_PAGE_ENTRIES = 256
MAX_FILE_BYTES = 1024**2
MAX_QUEUED_BYTES = 4 * 1024**2
MAX_ENTRIES = 250_000
MAX_CACHE_BYTES = 1024**3
OWNER_ID = "db.workspace_project_index"


def relative_path(value: str, *, root: bool = False) -> str:
    """Normalize separators while rejecting absolute/escaping index paths."""
    if not isinstance(value, str) or "\x00" in value or ":" in value:
        raise ValueError("invalid_relative_path")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError("invalid_relative_path")
    normalized = path.as_posix()
    if normalized == ".":
        if root:
            return ""
        raise ValueError("invalid_relative_path")
    return normalized


def locator_fingerprint(root: Path) -> str:
    """Hash the canonical local locator; persist no raw locator."""
    return hashlib.sha256(os.path.normcase(os.path.abspath(root)).encode()).hexdigest()


def root_identity_fingerprint(chain: DirectoryChain) -> str:
    """Hash every pinned ancestor identity, not a matching path string."""
    values = [(i.device, i.inode, i.mode, i.reparse) for i in chain.identities]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectIndexKey:
    data_authority_id: str
    workspace_id: str
    binding_id: str
    locator_fingerprint: str
    root_identity_fingerprint: str

    def __post_init__(self):
        for value in (
            self.data_authority_id,
            self.workspace_id,
            self.binding_id,
            self.locator_fingerprint,
            self.root_identity_fingerprint,
        ):
            if (
                not isinstance(value, str)
                or not 0 < len(value) <= 256
                or "\x00" in value
            ):
                raise ValueError("invalid_index_key")

    @property
    def namespace(self) -> str:
        values = (
            self.data_authority_id,
            self.workspace_id,
            self.binding_id,
            self.locator_fingerprint,
            self.root_identity_fingerprint,
        )
        return hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectIndexPolicy:
    include_prefixes: tuple[str, ...]
    text_suffixes: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(
            self,
            "include_prefixes",
            tuple(
                sorted(set(relative_path(p, root=True) for p in self.include_prefixes))
            ),
        )
        suffixes = tuple(sorted(set(s.casefold() for s in self.text_suffixes)))
        if any(
            not s.startswith(".") or "/" in s or "\\" in s or "\x00" in s
            for s in suffixes
        ):
            raise ValueError("invalid_text_suffix")
        object.__setattr__(self, "text_suffixes", suffixes)


@dataclass(frozen=True, slots=True)
class ProjectIndexScope:
    key: ProjectIndexKey
    epoch: int
    processing_revision: int


@dataclass(frozen=True, slots=True)
class ProjectIndexAuthority:
    key: ProjectIndexKey
    root: LocalRoot = field(repr=False)
    chain: DirectoryChain = field(repr=False)
    exclusions: tuple[SensitiveExclusion, ...]
    authority_revision: int

    def __post_init__(self):
        if isinstance(self.root, Path):
            object.__setattr__(self, "root", LocalRoot(self.root))
        if type(self.root) is not LocalRoot:
            raise ValueError("local_root_required")
        if (
            locator_fingerprint(self.root.path) != self.key.locator_fingerprint
            or root_identity_fingerprint(self.chain)
            != self.key.root_identity_fingerprint
        ):
            raise ValueError("root_identity_mismatch")


IndexAuthorityReader = Callable[[ProjectIndexKey], ProjectIndexAuthority | None]
IndexAuthorityFence = Callable[
    [ProjectIndexKey], ContextManager[ProjectIndexAuthority | None]
]


@dataclass(frozen=True, slots=True)
class ProjectIndexStatus:
    scope: ProjectIndexScope
    state: str
    discovered: int
    indexed: int
    cache_bytes: int
    last_scan_at: str | None
    job_id: str | None
    job_revision: int
    manifest_revision: int

    def __post_init__(self):
        if self.state not in INDEX_STATES:
            raise ValueError("invalid_index_state")


@dataclass(frozen=True, slots=True)
class ScanEntry:
    relative_path: str
    kind: str
    identity: str
    mtime_ns: int
    size_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "relative_path", relative_path(self.relative_path))
        if self.kind not in {"file", "directory"} or self.size_bytes < 0:
            raise ValueError("invalid_scan_entry")


@dataclass(frozen=True, slots=True)
class FileCapture:
    relative_path: str
    identity: str
    mtime_ns: int
    size_bytes: int
    content_sha256: str
    text: str = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "relative_path", relative_path(self.relative_path))
        data = self.text.encode("utf-8")
        if (
            self.size_bytes != len(data)
            or len(data) > MAX_FILE_BYTES
            or hashlib.sha256(data).hexdigest() != self.content_sha256
        ):
            raise ValueError("invalid_file_capture")


@dataclass(frozen=True, slots=True)
class ScanPage:
    directory: str
    directory_identity: str
    entries: tuple[ScanEntry, ...]
    captures: tuple[FileCapture, ...]
    directory_complete: bool
    authority_revision: int

    def __post_init__(self):
        object.__setattr__(self, "directory", relative_path(self.directory, root=True))
        if (
            len(self.entries) > MAX_PAGE_ENTRIES
            or len(self.captures) > MAX_PAGE_ENTRIES
            or sum(c.size_bytes for c in self.captures) > MAX_QUEUED_BYTES
        ):
            raise ValueError("scan_page_limit")
        entries = {e.relative_path: e for e in self.entries}
        if len(entries) != len(self.entries):
            raise ValueError("duplicate_scan_entry")
        for capture in self.captures:
            entry = entries.get(capture.relative_path)
            if (
                entry is None
                or entry.kind != "file"
                or (entry.identity, entry.mtime_ns, entry.size_bytes)
                != (capture.identity, capture.mtime_ns, capture.size_bytes)
            ):
                raise ValueError("capture_entry_mismatch")


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    relative_path: str
    start_char: int
    end_char: int
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class PreparedScanPage:
    page: ScanPage
    chunks: tuple[PreparedChunk, ...] = field(repr=False)
    digest: str


@dataclass(frozen=True, slots=True)
class ProjectIndexJobClaim:
    scope: ProjectIndexScope
    job_id: str
    owner_token: str = field(repr=False)
    job_revision: int
    authority_revision: int


IndexRetirementVerifier = Callable[[ProjectIndexJobClaim], bool]


@dataclass(frozen=True, slots=True)
class IndexCandidate:
    relative_path: str
    file_identity: str
    content_sha256: str
    line_start: int
    line_end: int
    rank: int


ProjectIndexLiveReader = Callable[[IndexCandidate], FileCapture | None]


@dataclass(frozen=True, slots=True)
class ProjectIndexSearchAuthority:
    scope: ProjectIndexScope
    root: LocalRoot = field(repr=False)
    chain: DirectoryChain = field(repr=False)
    exclusions: tuple[SensitiveExclusion, ...]
    authority_revision: int
    guard: Callable[[], bool] = field(repr=False)
    executor: WorkspaceToolExecutor | None = field(repr=False)
    live_reader: ProjectIndexLiveReader = field(repr=False)

    def __post_init__(self):
        authority = ProjectIndexAuthority(
            self.scope.key,
            self.root,
            self.chain,
            self.exclusions,
            self.authority_revision,
        )
        object.__setattr__(self, "root", authority.root)


@dataclass(frozen=True, slots=True)
class ProjectIndexHit:
    relative_path: str
    line_start: int
    line_end: int
    excerpt: str
    origin: str = "workspace_file"

    def __post_init__(self):
        if self.origin != "workspace_file":
            raise ValueError("invalid_index_origin")


@dataclass(frozen=True, slots=True)
class ProjectSearchResult:
    status: str
    hits: tuple[ProjectIndexHit, ...]
    coverage: ProjectIndexStatus | None
    refresh_required: bool

    def __post_init__(self):
        if self.status not in {"ready", "partial", "unavailable"}:
            raise ValueError("invalid_search_status")


@dataclass(frozen=True, slots=True)
class ScanRunResult:
    """Truthful scan outcome; only completed receipts certify enumeration."""

    status: str
    directories_completed: int
    discovered: int
    issues: tuple[str, ...]

    def __post_init__(self):
        if self.status not in {"complete", "partial", "paused", "interrupted", "error"}:
            raise ValueError("invalid_scan_status")
