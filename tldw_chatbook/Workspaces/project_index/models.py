"""Immutable folder-index contracts. No runtime/service construction on import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
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
    def binding_namespace(self) -> str:
        """Stable owner across locator/root changes; never infer it from a path."""
        return hashlib.sha256(
            json.dumps(
                (self.workspace_id, self.binding_id),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @property
    def lifetime_name(self) -> str:
        """Authority-scoped native object, distinct from accounting ownership."""
        return (
            hashlib.sha256(self.data_authority_id.encode()).hexdigest()
            + ".lifetime.lock"
        )

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
class ProjectSemanticSettings:
    """Secret-free, restart-safe embedding destination selection."""

    provider_id: str
    model_id: str
    destination_fingerprint: str
    config_fingerprint: str
    approved_destination_fingerprint: str | None
    provider_ref: str
    config_ref: str

    def __post_init__(self):
        provider = (
            re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.provider_id)
            if isinstance(self.provider_id, str)
            else None
        )
        model = (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.model_id)
            if isinstance(self.model_id, str)
            else None
        )
        digest = re.compile(r"sha256:v1:[0-9a-f]{64}\Z")
        if not provider or not model:
            raise ValueError("invalid_semantic_identifier")
        if (
            self.provider_ref != "provider:v1:" + self.provider_id
            or self.config_ref != "model:v1:" + self.model_id
        ):
            raise ValueError("invalid_semantic_reference")
        if not all(
            isinstance(value, str) and digest.fullmatch(value)
            for value in (self.destination_fingerprint, self.config_fingerprint)
        ):
            raise ValueError("invalid_semantic_fingerprint")
        approved = self.approved_destination_fingerprint
        if approved is not None and (
            not isinstance(approved, str) or not digest.fullmatch(approved)
        ):
            raise ValueError("invalid_approved_destination")


@dataclass(frozen=True, slots=True)
class ProjectSemanticState:
    """Independent semantic consent, revision and safe readiness projection."""

    scope: ProjectIndexScope
    semantic_revision: int
    enabled: bool
    settings: ProjectSemanticSettings | None
    state: str
    reason: str | None = None
    ready_generation_id: str | None = None
    coverage: int | None = None
    close_pending: bool = False


@dataclass(frozen=True, slots=True)
class ProjectSemanticGeneration:
    """Immutable identity and durable build progress for one generation."""

    generation_id: str
    scope: ProjectIndexScope
    semantic_revision: int
    provider_id: str
    model_id: str
    destination_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    format_id: str
    dimension: int
    manifest_revision: int
    claim_token: str | None
    checkpoint_chunk_id: int | None
    state: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class ProjectSemanticMember:
    """Generation-owned live-validation metadata and float32 vector bytes."""

    generation_id: str
    ordinal: int
    relative_path: str
    identity: str
    content_sha256: str
    line_start: int
    line_end: int
    vector: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class SemanticSourceChunk:
    """One immutable keyword-source revision prepared for embedding."""

    chunk_id: int
    relative_path: str
    identity: str
    content_sha256: str
    line_start: int
    line_end: int
    text: str = field(repr=False)


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
class DirectoryReceipt:
    directory: str
    directory_identity: str
    entries: tuple[ScanEntry, ...]
    directory_complete: bool
    more: bool
    issues: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "directory", relative_path(self.directory, root=True))
        if self.directory_complete and (self.more or self.issues):
            raise ValueError("invalid_directory_receipt")
        if len(self.entries) > MAX_PAGE_ENTRIES or len(self.issues) > 16:
            raise ValueError("directory_receipt_limit")
        for entry in self.entries:
            parent = str(PurePosixPath(entry.relative_path).parent)
            if ("" if parent == "." else parent) != self.directory:
                raise ValueError("directory_entry_mismatch")


@dataclass(frozen=True, slots=True)
class ScanUnit:
    directories: tuple[DirectoryReceipt, ...]
    captures: tuple[FileCapture, ...]
    authority_revision: int
    inspections: int

    @property
    def entries(self) -> tuple[ScanEntry, ...]:
        return tuple(entry for receipt in self.directories for entry in receipt.entries)

    def __post_init__(self):
        if (
            not 1 <= len(self.directories) <= MAX_PAGE_ENTRIES
            or not 0 <= self.inspections <= MAX_PAGE_ENTRIES
        ):
            raise ValueError("scan_unit_limit")
        paths = [receipt.directory for receipt in self.directories]
        if (
            len(set(paths)) != len(paths)
            or sum(receipt.more for receipt in self.directories) > 1
        ):
            raise ValueError("duplicate_directory_receipt")
        if any(receipt.more for receipt in self.directories[:-1]):
            raise ValueError("invalid_unit_continuation")
        # Reuse the bounded entry/capture identity contract of the singleton API.
        ScanPage("", "", self.entries, self.captures, False, self.authority_revision)


@dataclass(frozen=True, slots=True)
class PreparedScanUnit:
    unit: ScanUnit
    chunks: tuple[PreparedChunk, ...] = field(repr=False)
    digest: str


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


class ProjectIndexRetirementUnproven(RuntimeError):
    """Code-owned Capture failure: its native reader has not proven retirement."""


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
    semantic_status: str = "unavailable"

    def __post_init__(self):
        if self.status not in {"ready", "partial", "unavailable"}:
            raise ValueError("invalid_search_status")
        if self.semantic_status not in {"ready", "unavailable"}:
            raise ValueError("invalid_semantic_search_status")


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
