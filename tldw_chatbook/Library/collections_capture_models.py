"""Source-neutral, authority-qualified contracts for Collections captures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping


CAPTURE_PAGE_SIZE = 20
CAPTURE_STATUSES = ("saved", "reading", "read", "archived")
CAPTURE_SORTS = (
    "saved_desc",
    "saved_asc",
    "updated_desc",
    "updated_asc",
    "title_asc",
    "title_desc",
    "relevance",
)
CAPTURE_PROCESSING_STATES = (
    "queued",
    "processing",
    "ready",
    "failed",
    "interrupted",
)
CAPTURE_CAPABILITY_NAMES = (
    "browse",
    "capture",
    "update",
    "highlights",
    "linked_notes",
    "summarize",
    "listen",
    "archive",
    "offline_copy",
    "hard_delete",
    "retry_extraction",
    "legacy_recovery",
)


class CollectionsCaptureError(RuntimeError):
    """Bounded, content-free failure exposed by the capture service boundary."""

    def __init__(self, reason: str, *, retryable: bool = False) -> None:
        normalized_reason = _nonempty(reason, "invalid_error_reason")
        self.reason = normalized_reason
        self.retryable = bool(retryable)
        super().__init__(normalized_reason)


class CapabilityState(str, Enum):
    """Discovery state for one authority-specific capture action."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


def _nonempty(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionsCaptureError(reason)
    return value.strip()


def _optional_text(value: Any, reason: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectionsCaptureError(reason)
    normalized = value.strip()
    return normalized or None


def _optional_content(value: Any, reason: str) -> str | None:
    """Validate optional content without changing meaningful whitespace."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectionsCaptureError(reason)
    return value


def _content(value: Any, reason: str) -> str:
    """Validate required string content without changing it."""
    if not isinstance(value, str):
        raise CollectionsCaptureError(reason)
    return value


def _positive_int(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CollectionsCaptureError(reason)
    return value


def _normalize_exact_values(
    values: Any,
    *,
    reason: str,
    allowed: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise CollectionsCaptureError(reason)
    normalized: set[str] = set()
    for value in values:
        item = _nonempty(value, reason).casefold()
        if allowed is not None and item not in allowed:
            raise CollectionsCaptureError(reason)
        normalized.add(item)
    return tuple(sorted(normalized))


def _normalize_display_tags(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise CollectionsCaptureError("invalid_tags")
    by_normalized_name: dict[str, str] = {}
    for value in values:
        display_name = _nonempty(value, "invalid_tags")
        by_normalized_name.setdefault(display_name.casefold(), display_name)
    return tuple(by_normalized_name[key] for key in sorted(by_normalized_name))


def _validate_page_shape(*, row_count: int, total: Any, page: int, size: int) -> int:
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise CollectionsCaptureError("invalid_total")
    if row_count > size:
        raise CollectionsCaptureError("oversized_page")
    offset = (page - 1) * size
    if row_count and offset + row_count > total:
        raise CollectionsCaptureError("impossible_total")
    if row_count < size and offset + row_count < total:
        raise CollectionsCaptureError("undersized_nonfinal_page")
    return total


@dataclass(frozen=True)
class CaptureAuthority:
    kind: Literal["local", "server"]
    key: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.kind not in {"local", "server"}:
            raise CollectionsCaptureError("invalid_authority_kind")
        object.__setattr__(self, "key", _nonempty(self.key, "invalid_authority_key"))
        object.__setattr__(
            self,
            "fingerprint",
            _nonempty(self.fingerprint, "invalid_authority_fingerprint"),
        )


@dataclass(frozen=True)
class CaptureIdentity:
    authority_key: str
    capture_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_key",
            _nonempty(self.authority_key, "invalid_authority_key"),
        )
        object.__setattr__(
            self, "capture_id", _nonempty(self.capture_id, "invalid_capture_id")
        )


@dataclass(frozen=True)
class CapturePageRequest:
    authority_key: str
    search: str = ""
    statuses: tuple[str, ...] = ()
    favorite: bool | None = None
    tags: tuple[str, ...] = ()
    domain: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sort: str = "saved_desc"
    page: int = 1
    size: int = CAPTURE_PAGE_SIZE

    _MAPPING_KEYS = frozenset(
        {
            "authority_key",
            "search",
            "statuses",
            "favorite",
            "tags",
            "domain",
            "date_from",
            "date_to",
            "sort",
            "page",
            "size",
        }
    )

    def __post_init__(self) -> None:
        authority_key = _nonempty(self.authority_key, "invalid_authority_key")
        if not isinstance(self.search, str):
            raise CollectionsCaptureError("invalid_search")
        search = " ".join(self.search.split())
        statuses = _normalize_exact_values(
            self.statuses,
            reason="invalid_statuses",
            allowed=CAPTURE_STATUSES,
        )
        tags = _normalize_exact_values(self.tags, reason="invalid_tags")
        if self.favorite is not None and not isinstance(self.favorite, bool):
            raise CollectionsCaptureError("invalid_favorite")
        domain = _optional_text(self.domain, "invalid_domain")
        if domain is not None:
            domain = domain.casefold()
        date_from = _optional_text(self.date_from, "invalid_date_from")
        date_to = _optional_text(self.date_to, "invalid_date_to")
        if date_from is not None and date_to is not None and date_from > date_to:
            raise CollectionsCaptureError("invalid_date_range")
        sort = _nonempty(self.sort, "invalid_sort").casefold()
        if sort not in CAPTURE_SORTS:
            raise CollectionsCaptureError("invalid_sort")
        if sort == "relevance" and not search:
            raise CollectionsCaptureError("relevance_requires_search")
        page = _positive_int(self.page, "invalid_page")
        if self.size != CAPTURE_PAGE_SIZE or isinstance(self.size, bool):
            raise CollectionsCaptureError("invalid_page_size")

        object.__setattr__(self, "authority_key", authority_key)
        object.__setattr__(self, "search", search)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "date_from", date_from)
        object.__setattr__(self, "date_to", date_to)
        object.__setattr__(self, "sort", sort)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "size", CAPTURE_PAGE_SIZE)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CapturePageRequest:
        """Return a validated copy of an untrusted request or saved query."""
        if not isinstance(values, Mapping):
            raise CollectionsCaptureError("invalid_page_request")
        if set(values) - cls._MAPPING_KEYS:
            raise CollectionsCaptureError("unknown_page_request_key")
        try:
            return cls(**dict(values))
        except CollectionsCaptureError:
            raise
        except (TypeError, ValueError) as exc:
            raise CollectionsCaptureError("invalid_page_request") from exc


@dataclass(frozen=True)
class CaptureSummary:
    identity: CaptureIdentity
    canonical_url: str
    title: str | None = None
    domain: str = ""
    summary: str | None = None
    published_at: str | None = None
    status: str = "saved"
    favorite: bool = False
    tags: tuple[str, ...] = ()
    processing_state: str = "ready"
    last_fetch_error: str | None = None
    created_at: str = ""
    updated_at: str = ""
    read_at: str | None = None
    revision: int = 1
    has_offline_copy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        object.__setattr__(
            self,
            "canonical_url",
            _nonempty(self.canonical_url, "invalid_canonical_url"),
        )
        object.__setattr__(self, "title", _optional_text(self.title, "invalid_title"))
        domain = _optional_text(self.domain, "invalid_domain") or ""
        object.__setattr__(self, "domain", domain.casefold())
        object.__setattr__(
            self, "summary", _optional_text(self.summary, "invalid_summary")
        )
        object.__setattr__(
            self,
            "published_at",
            _optional_text(self.published_at, "invalid_published_at"),
        )
        status = _nonempty(self.status, "invalid_status").casefold()
        if status not in CAPTURE_STATUSES:
            raise CollectionsCaptureError("invalid_status")
        processing_state = _nonempty(
            self.processing_state, "invalid_processing_state"
        ).casefold()
        if processing_state not in CAPTURE_PROCESSING_STATES:
            raise CollectionsCaptureError("invalid_processing_state")
        if not isinstance(self.favorite, bool):
            raise CollectionsCaptureError("invalid_favorite")
        if not isinstance(self.has_offline_copy, bool):
            raise CollectionsCaptureError("invalid_offline_copy_state")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "processing_state", processing_state)
        object.__setattr__(self, "tags", _normalize_display_tags(self.tags))
        object.__setattr__(
            self,
            "last_fetch_error",
            _optional_text(self.last_fetch_error, "invalid_fetch_error"),
        )
        object.__setattr__(
            self, "created_at", _content(self.created_at, "invalid_created_at")
        )
        object.__setattr__(
            self, "updated_at", _content(self.updated_at, "invalid_updated_at")
        )
        object.__setattr__(
            self, "read_at", _optional_text(self.read_at, "invalid_read_at")
        )
        object.__setattr__(self, "revision", _positive_int(self.revision, "invalid_revision"))


@dataclass(frozen=True)
class ExternalMediaReference:
    authority_key: str
    item_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_key",
            _nonempty(self.authority_key, "invalid_media_authority_key"),
        )
        object.__setattr__(self, "item_id", _nonempty(self.item_id, "invalid_media_item_id"))


@dataclass(frozen=True)
class CaptureOfflineCopy:
    identity: CaptureIdentity
    file_id: str
    state: Literal["staging", "ready", "failed", "purging"]
    content_hash: str | None = None
    size: int | None = None
    media_type: str | None = None
    failure_reason: str | None = None
    revision: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        object.__setattr__(self, "file_id", _nonempty(self.file_id, "invalid_file_id"))
        if self.state not in {"staging", "ready", "failed", "purging"}:
            raise CollectionsCaptureError("invalid_offline_copy_state")
        if self.size is not None and (
            isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0
        ):
            raise CollectionsCaptureError("invalid_offline_copy_size")
        object.__setattr__(
            self,
            "content_hash",
            _optional_text(self.content_hash, "invalid_content_hash"),
        )
        object.__setattr__(
            self, "media_type", _optional_text(self.media_type, "invalid_media_type")
        )
        object.__setattr__(
            self,
            "failure_reason",
            _optional_text(self.failure_reason, "invalid_failure_reason"),
        )
        object.__setattr__(self, "revision", _positive_int(self.revision, "invalid_revision"))


@dataclass(frozen=True)
class CaptureDetail(CaptureSummary):
    submitted_url: str = ""
    freeform_note: str | None = None
    text_content: str | None = None
    clean_html: str | None = None
    byline: str | None = None
    content_hash: str | None = None
    word_count: int | None = None
    media_reference: ExternalMediaReference | None = None
    offline_copy: CaptureOfflineCopy | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.submitted_url, str):
            raise CollectionsCaptureError("invalid_submitted_url")
        submitted_url = self.submitted_url.strip()
        object.__setattr__(self, "submitted_url", submitted_url or self.canonical_url)
        object.__setattr__(
            self,
            "freeform_note",
            _optional_content(self.freeform_note, "invalid_freeform_note"),
        )
        object.__setattr__(
            self,
            "text_content",
            _optional_content(self.text_content, "invalid_text_content"),
        )
        object.__setattr__(
            self,
            "clean_html",
            _optional_content(self.clean_html, "invalid_clean_html"),
        )
        object.__setattr__(self, "byline", _optional_text(self.byline, "invalid_byline"))
        object.__setattr__(
            self,
            "content_hash",
            _optional_text(self.content_hash, "invalid_content_hash"),
        )
        if self.word_count is not None and (
            isinstance(self.word_count, bool)
            or not isinstance(self.word_count, int)
            or self.word_count < 0
        ):
            raise CollectionsCaptureError("invalid_word_count")
        if self.media_reference is not None and not isinstance(
            self.media_reference, ExternalMediaReference
        ):
            raise CollectionsCaptureError("invalid_media_reference")
        if self.offline_copy is not None:
            if not isinstance(self.offline_copy, CaptureOfflineCopy):
                raise CollectionsCaptureError("invalid_offline_copy")
            if self.offline_copy.identity != self.identity:
                raise CollectionsCaptureError("offline_copy_identity_mismatch")


@dataclass(frozen=True)
class CapturePage:
    applied: CapturePageRequest
    items: tuple[CaptureSummary, ...]
    total: int
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.applied, CapturePageRequest):
            raise CollectionsCaptureError("invalid_applied_scope")
        if not isinstance(self.items, (tuple, list)):
            raise CollectionsCaptureError("invalid_page_items")
        items = tuple(self.items)
        if any(not isinstance(item, CaptureSummary) for item in items):
            raise CollectionsCaptureError("invalid_page_items")
        ids = [item.identity.capture_id for item in items]
        if len(ids) != len(set(ids)):
            raise CollectionsCaptureError("duplicate_capture_id")
        if any(
            item.identity.authority_key != self.applied.authority_key for item in items
        ):
            raise CollectionsCaptureError("page_authority_mismatch")
        total = _validate_page_shape(
            row_count=len(items),
            total=self.total,
            page=self.applied.page,
            size=self.applied.size,
        )
        source_revision = _optional_text(
            self.source_revision, "invalid_source_revision"
        )
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "source_revision", source_revision)


@dataclass(frozen=True)
class CaptureSaveRequest:
    authority_key: str
    submitted_url: str
    canonical_url: str | None = None
    title: str | None = None
    tags: tuple[str, ...] = ()
    status: str | None = None
    favorite: bool | None = None
    summary: str | None = None
    freeform_note: str | None = None
    text_content: str | None = None
    clean_html: str | None = None
    byline: str | None = None
    published_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_key",
            _nonempty(self.authority_key, "invalid_authority_key"),
        )
        object.__setattr__(
            self,
            "submitted_url",
            _nonempty(self.submitted_url, "invalid_submitted_url"),
        )
        object.__setattr__(
            self,
            "canonical_url",
            _optional_text(self.canonical_url, "invalid_canonical_url"),
        )
        object.__setattr__(self, "title", _optional_text(self.title, "invalid_title"))
        object.__setattr__(self, "tags", _normalize_display_tags(self.tags))
        if self.status is not None:
            status = _nonempty(self.status, "invalid_status").casefold()
            if status not in CAPTURE_STATUSES:
                raise CollectionsCaptureError("invalid_status")
            object.__setattr__(self, "status", status)
        if self.favorite is not None and not isinstance(self.favorite, bool):
            raise CollectionsCaptureError("invalid_favorite")
        object.__setattr__(
            self, "summary", _optional_text(self.summary, "invalid_summary")
        )
        object.__setattr__(
            self,
            "freeform_note",
            _optional_content(self.freeform_note, "invalid_freeform_note"),
        )
        object.__setattr__(
            self,
            "text_content",
            _optional_content(self.text_content, "invalid_text_content"),
        )
        object.__setattr__(
            self,
            "clean_html",
            _optional_content(self.clean_html, "invalid_clean_html"),
        )
        object.__setattr__(self, "byline", _optional_text(self.byline, "invalid_byline"))
        object.__setattr__(
            self,
            "published_at",
            _optional_text(self.published_at, "invalid_published_at"),
        )


@dataclass(frozen=True)
class CaptureSaveOutcome:
    capture: CaptureDetail | None
    created: bool | None
    extraction_pending: bool = False
    outcome_unknown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.extraction_pending, bool) or not isinstance(
            self.outcome_unknown, bool
        ):
            raise CollectionsCaptureError("invalid_save_outcome")
        if self.outcome_unknown:
            if (
                self.capture is not None
                or self.created is not None
                or self.extraction_pending
            ):
                raise CollectionsCaptureError("invalid_save_outcome")
            return
        if not isinstance(self.capture, CaptureDetail) or not isinstance(
            self.created, bool
        ):
            raise CollectionsCaptureError("invalid_save_outcome")


@dataclass(frozen=True)
class CaptureConflict:
    identity: CaptureIdentity
    expected_revision: int
    current: CaptureDetail

    def __post_init__(self) -> None:
        if self.current.identity != self.identity:
            raise CollectionsCaptureError("conflict_identity_mismatch")
        object.__setattr__(
            self,
            "expected_revision",
            _positive_int(self.expected_revision, "invalid_revision"),
        )

    @property
    def actual_revision(self) -> int:
        return self.current.revision


@dataclass(frozen=True)
class SavedCaptureSearch:
    authority_key: str
    search_id: str
    name: str
    request: CapturePageRequest
    created_at: str
    updated_at: str
    revision: int

    def __post_init__(self) -> None:
        authority_key = _nonempty(self.authority_key, "invalid_authority_key")
        object.__setattr__(self, "authority_key", authority_key)
        object.__setattr__(self, "search_id", _nonempty(self.search_id, "invalid_search_id"))
        object.__setattr__(self, "name", _nonempty(self.name, "invalid_saved_search_name"))
        if not isinstance(self.request, CapturePageRequest):
            raise CollectionsCaptureError("invalid_saved_search_request")
        if self.request.authority_key != authority_key:
            raise CollectionsCaptureError("saved_search_authority_mismatch")
        object.__setattr__(
            self, "created_at", _content(self.created_at, "invalid_created_at")
        )
        object.__setattr__(
            self, "updated_at", _content(self.updated_at, "invalid_updated_at")
        )
        object.__setattr__(self, "revision", _positive_int(self.revision, "invalid_revision"))


@dataclass(frozen=True)
class CaptureSavedSearchPage:
    items: tuple[SavedCaptureSearch, ...]
    total: int
    page: int
    size: int = CAPTURE_PAGE_SIZE

    def __post_init__(self) -> None:
        page = _positive_int(self.page, "invalid_page")
        if self.size != CAPTURE_PAGE_SIZE or isinstance(self.size, bool):
            raise CollectionsCaptureError("invalid_page_size")
        if not isinstance(self.items, (tuple, list)):
            raise CollectionsCaptureError("invalid_saved_search_items")
        items = tuple(self.items)
        if any(not isinstance(item, SavedCaptureSearch) for item in items):
            raise CollectionsCaptureError("invalid_saved_search_items")
        ids = [item.search_id for item in items]
        if len(ids) != len(set(ids)):
            raise CollectionsCaptureError("duplicate_saved_search_id")
        if len({item.authority_key for item in items}) > 1:
            raise CollectionsCaptureError("saved_search_authority_mismatch")
        total = _validate_page_shape(
            row_count=len(items), total=self.total, page=page, size=CAPTURE_PAGE_SIZE
        )
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "size", CAPTURE_PAGE_SIZE)
        object.__setattr__(self, "total", total)


@dataclass(frozen=True)
class CaptureHighlightDraft:
    quote: str
    note: str | None = None
    anchor_json: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote", _nonempty(self.quote, "invalid_highlight_quote"))
        object.__setattr__(
            self, "note", _optional_content(self.note, "invalid_highlight_note")
        )
        object.__setattr__(
            self,
            "anchor_json",
            _optional_content(self.anchor_json, "invalid_highlight_anchor"),
        )


@dataclass(frozen=True)
class CaptureHighlight:
    identity: CaptureIdentity
    highlight_id: str
    quote: str
    note: str | None
    anchor_json: str | None
    detached: bool
    created_at: str
    updated_at: str
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        object.__setattr__(
            self,
            "highlight_id",
            _nonempty(self.highlight_id, "invalid_highlight_id"),
        )
        object.__setattr__(self, "quote", _nonempty(self.quote, "invalid_highlight_quote"))
        object.__setattr__(
            self, "note", _optional_content(self.note, "invalid_highlight_note")
        )
        object.__setattr__(
            self,
            "anchor_json",
            _optional_content(self.anchor_json, "invalid_highlight_anchor"),
        )
        if not isinstance(self.detached, bool):
            raise CollectionsCaptureError("invalid_highlight_state")
        object.__setattr__(
            self, "created_at", _content(self.created_at, "invalid_created_at")
        )
        object.__setattr__(
            self, "updated_at", _content(self.updated_at, "invalid_updated_at")
        )
        object.__setattr__(self, "revision", _positive_int(self.revision, "invalid_revision"))


@dataclass(frozen=True)
class ExternalNoteReference:
    authority_key: str
    note_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_key",
            _nonempty(self.authority_key, "invalid_note_authority_key"),
        )
        object.__setattr__(self, "note_id", _nonempty(self.note_id, "invalid_note_id"))


@dataclass(frozen=True)
class CaptureNoteLink:
    identity: CaptureIdentity
    link_id: str
    note_reference: ExternalNoteReference
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        object.__setattr__(self, "link_id", _nonempty(self.link_id, "invalid_note_link_id"))
        if not isinstance(self.note_reference, ExternalNoteReference):
            raise CollectionsCaptureError("invalid_note_reference")
        object.__setattr__(
            self, "created_at", _content(self.created_at, "invalid_created_at")
        )


@dataclass(frozen=True)
class ExternalReferenceAvailability:
    state: Literal["available", "unavailable"]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"available", "unavailable"}:
            raise CollectionsCaptureError("invalid_reference_availability")
        object.__setattr__(
            self, "reason", _optional_text(self.reason, "invalid_availability_reason")
        )


@dataclass(frozen=True)
class ResolvedCaptureDetail:
    capture: CaptureDetail
    media: ExternalReferenceAvailability | None
    note_links: tuple[tuple[CaptureNoteLink, ExternalReferenceAvailability], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capture, CaptureDetail):
            raise CollectionsCaptureError("invalid_capture_detail")
        note_links = tuple(self.note_links)
        for link, availability in note_links:
            if link.identity != self.capture.identity:
                raise CollectionsCaptureError("note_link_identity_mismatch")
            if not isinstance(availability, ExternalReferenceAvailability):
                raise CollectionsCaptureError("invalid_reference_availability")
        if self.media is not None and not isinstance(
            self.media, ExternalReferenceAvailability
        ):
            raise CollectionsCaptureError("invalid_reference_availability")
        object.__setattr__(self, "note_links", note_links)


@dataclass(frozen=True)
class CaptureActionResult:
    identity: CaptureIdentity
    success: bool
    revision: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        if not isinstance(self.success, bool):
            raise CollectionsCaptureError("invalid_action_result")
        if self.revision is not None:
            object.__setattr__(
                self, "revision", _positive_int(self.revision, "invalid_revision")
            )
        object.__setattr__(
            self, "reason", _optional_text(self.reason, "invalid_action_reason")
        )


@dataclass(frozen=True)
class CaptureContentResult:
    identity: CaptureIdentity
    kind: Literal["summary", "audio"]
    text: str | None = None
    artifact_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CaptureIdentity):
            raise CollectionsCaptureError("invalid_capture_identity")
        if self.kind not in {"summary", "audio"}:
            raise CollectionsCaptureError("invalid_content_result_kind")
        object.__setattr__(
            self, "text", _optional_content(self.text, "invalid_content_text")
        )
        object.__setattr__(
            self,
            "artifact_reference",
            _optional_text(
                self.artifact_reference, "invalid_content_artifact_reference"
            ),
        )


@dataclass(frozen=True)
class CaptureCapability:
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            state = CapabilityState(self.state)
        except (TypeError, ValueError) as exc:
            raise CollectionsCaptureError("invalid_capability_state") from exc
        object.__setattr__(self, "state", state)
        reason = _optional_text(self.reason, "invalid_capability_reason")
        if state is CapabilityState.UNSUPPORTED and reason is None:
            raise CollectionsCaptureError("missing_capability_reason")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class CaptureCapabilities:
    values: Mapping[str, CaptureCapability]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise CollectionsCaptureError("invalid_capability_set")
        values = dict(self.values)
        if set(values) != set(CAPTURE_CAPABILITY_NAMES) or any(
            not isinstance(value, CaptureCapability) for value in values.values()
        ):
            raise CollectionsCaptureError("invalid_capability_set")
        object.__setattr__(self, "values", MappingProxyType(values))

    def for_action(self, action: str) -> CaptureCapability:
        """Return one known action state without an omission fallback."""
        try:
            return self.values[action]
        except KeyError as exc:
            raise CollectionsCaptureError("unknown_capability") from exc


__all__ = [
    "CAPTURE_CAPABILITY_NAMES",
    "CAPTURE_PAGE_SIZE",
    "CAPTURE_PROCESSING_STATES",
    "CAPTURE_SORTS",
    "CAPTURE_STATUSES",
    "CapabilityState",
    "CaptureActionResult",
    "CaptureAuthority",
    "CaptureCapabilities",
    "CaptureCapability",
    "CaptureConflict",
    "CaptureContentResult",
    "CaptureDetail",
    "CaptureHighlight",
    "CaptureHighlightDraft",
    "CaptureIdentity",
    "CaptureNoteLink",
    "CaptureOfflineCopy",
    "CapturePage",
    "CapturePageRequest",
    "CaptureSaveOutcome",
    "CaptureSaveRequest",
    "CaptureSavedSearchPage",
    "CaptureSummary",
    "CollectionsCaptureError",
    "ExternalMediaReference",
    "ExternalNoteReference",
    "ExternalReferenceAvailability",
    "ResolvedCaptureDetail",
    "SavedCaptureSearch",
]
