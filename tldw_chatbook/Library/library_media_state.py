"""Pure display-state contracts for the Library Browse ▸ Media canvas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)
from tldw_chatbook.Library.library_pager_state import PageFreshness

LIBRARY_MEDIA_EMPTY_COPY = (
    "No media in your Library yet. Import something to see it here."
)

# task-4025: the Trash view's honest empty state -- present tense, no
# promise of anything beyond what the surface does (items land here on
# delete and leave on restore).
LIBRARY_MEDIA_TRASH_EMPTY_COPY = (
    "Trash is empty. Items you delete from Media land here."
)

# task-4025 (F-018): every disabled Library action says why -- the Trash
# view's "Restore" action has three honest disabled reasons (still
# loading vs. the fetch failed vs. genuinely nothing there) plus its
# enabled description, mirroring the Export/Delete-selected tooltip pairs
# in library_shell_state.py. The error reason exists because a failed
# fetch also leaves zero rows -- without it the tooltip claimed "Nothing
# in Trash" for a Trash that merely could not be read (PR-1505 review).
LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP = "Restore the selected item to your Library."
LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP = "Nothing in Trash to restore."
LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP = "Trash is still loading."
LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP = "Trash could not be loaded."

LIBRARY_MEDIA_BROWSE_PAGE_SIZE = 20
_SQLITE_INTEGER_MAX = 2**63 - 1
_MEDIA_BROWSE_SORTS = frozenset(
    {
        "date_desc",
        "date_asc",
        "title_asc",
        "title_desc",
        "last_modified_asc",
        "last_modified_desc",
        "relevance",
    }
)

#: task-28013: the sort values the media browse chooser offers, in display
#: order, each mapped to its user-facing label. A deliberate subset of
#: ``_MEDIA_BROWSE_SORTS`` -- "relevance" is query-only (the scope validator
#: downgrades it without a query) so it is not a manual pick, and the
#: date_* pair duplicates last_modified_* for this local source.
MEDIA_SORT_CHOICES = (
    ("last_modified_desc", "Newest"),
    ("last_modified_asc", "Oldest"),
    ("title_asc", "Title A-Z"),
    ("title_desc", "Title Z-A"),
)
_MEDIA_SUMMARY_KEYS = frozenset(
    {"id", "backing_media_id", "title", "media_type", "updated_at"}
)
_MEDIA_TRASH_SUMMARY_KEYS = frozenset(
    {"id", "backing_media_id", "title", "media_type", "trash_date"}
)
_MEDIA_TRASH_ENVELOPE_KEYS = frozenset({"items", "total", "limit", "offset", "types"})

_ID_KEYS = ("id", "media_id", "uuid")
_TYPE_KEYS = ("type", "media_type")
_UPDATED_KEYS = ("last_modified", "ingestion_date", "date", "updated_at")


@dataclass(frozen=True)
class MediaBrowseScope:
    """One immutable exact page request for the local Library Media source."""

    query: str = ""
    media_type: str | None = None
    sort_by: str = "last_modified_desc"
    page: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string or None.")
        if not isinstance(self.sort_by, str):
            raise TypeError("sort_by must be a string.")
        if type(self.page) is not int or self.page < 1:
            raise ValueError("page must be a positive integer.")
        if (self.page - 1) * LIBRARY_MEDIA_BROWSE_PAGE_SIZE > _SQLITE_INTEGER_MAX:
            raise ValueError("page offset exceeds SQLite's integer range.")

        query = self.query.strip()
        media_type = self.media_type
        if media_type is not None and not media_type.strip():
            media_type = None
        sort_by = self.sort_by.strip().lower()
        if sort_by not in _MEDIA_BROWSE_SORTS:
            raise ValueError("sort_by is not supported for Media browsing.")
        if sort_by == "relevance" and not query:
            sort_by = "last_modified_desc"
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "sort_by", sort_by)

    @property
    def page_size(self) -> int:
        return LIBRARY_MEDIA_BROWSE_PAGE_SIZE

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            (self.query, self.media_type, self.sort_by, self.page),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_page(self, page: int) -> "MediaBrowseScope":
        return replace(self, page=page)

    def same_except_page(self, other: "MediaBrowseScope") -> bool:
        return isinstance(other, MediaBrowseScope) and self.with_page(
            1
        ) == other.with_page(1)


def _freeze_media_summary_value(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Media summary mappings require string keys.")
        return MappingProxyType(
            {key: _freeze_media_summary_value(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_media_summary_value(item) for item in value)
    raise TypeError("Media summary values must be JSON-like immutable data.")


def validate_media_browse_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate and detach exact five-key Library Media summary rows."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise TypeError("Media browse result items must be a sequence.")
    stable_ids: set[str] = set()
    backing_ids: set[int] = set()
    frozen: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("Media browse result items must be mappings.")
        if set(item) != _MEDIA_SUMMARY_KEYS:
            raise ValueError(
                "Media browse items must contain exactly five summary keys."
            )
        backing_id = item["backing_media_id"]
        if type(backing_id) is not int or backing_id < 1:
            raise ValueError("backing_media_id must be a positive integer.")
        stable_id = item["id"]
        if type(stable_id) is not str or stable_id != f"local:media:{backing_id}":
            raise ValueError("id must match the canonical backing_media_id.")
        if stable_id in stable_ids or backing_id in backing_ids:
            raise ValueError("Media browse identities must be page-unique.")
        stable_ids.add(stable_id)
        backing_ids.add(backing_id)
        frozen.append(
            MappingProxyType(
                {key: _freeze_media_summary_value(item[key]) for key in item}
            )
        )
    return tuple(frozen)


@dataclass(frozen=True)
class MediaBrowseResult:
    """One exact, immutable Library Media page returned by the service."""

    scope: MediaBrowseScope
    items: tuple[Mapping[str, Any], ...]
    total: int
    limit: int
    offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MediaBrowseScope):
            raise TypeError("scope must be a MediaBrowseScope.")
        for field_name, value, minimum in (
            ("total", self.total, 0),
            ("limit", self.limit, 1),
            ("offset", self.offset, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"{field_name} must be an integer of at least {minimum}."
                )
        if self.limit != self.scope.page_size:
            raise ValueError("limit must match the requested page size.")
        if self.offset != self.scope.offset:
            raise ValueError("offset must match the requested page offset.")
        frozen_items = validate_media_browse_items(self.items)
        expected_count = min(self.limit, max(self.total - self.offset, 0))
        if len(frozen_items) != expected_count:
            raise ValueError("Media browse result item count is invalid for this page.")
        object.__setattr__(self, "items", frozen_items)

    @property
    def last_page(self) -> int:
        return max(1, (self.total + self.limit - 1) // self.limit)

    @property
    def out_of_range(self) -> bool:
        return self.scope.page > self.last_page


def build_media_browse_result(
    scope: MediaBrowseScope, payload: Mapping[str, Any]
) -> MediaBrowseResult:
    """Build a fail-closed exact page from the Library summary envelope."""
    if not isinstance(payload, Mapping):
        raise TypeError("Media browse result must be a mapping.")
    for key in ("items", "total", "limit", "offset"):
        if key not in payload:
            raise ValueError(f"Media browse result is missing {key}.")
    raw_items = payload["items"]
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        raise TypeError("Media browse result items must be a sequence.")
    return MediaBrowseResult(
        scope=scope,
        items=tuple(raw_items),
        total=payload["total"],
        limit=payload["limit"],
        offset=payload["offset"],
    )


@dataclass(frozen=True)
class MediaTrashScope:
    """One immutable exact-page request for the local Media Trash source."""

    query: str = ""
    media_type: str | None = None
    page: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string or None.")
        if type(self.page) is not int or self.page < 1:
            raise ValueError("page must be a positive integer.")
        if (self.page - 1) * LIBRARY_MEDIA_BROWSE_PAGE_SIZE > _SQLITE_INTEGER_MAX:
            raise ValueError("page offset exceeds SQLite's integer range.")

        query = self.query.strip()
        if "\x00" in query:
            raise ValueError("query cannot contain NUL.")
        if len(query) > 200:
            raise ValueError("query is limited to 200 characters.")
        media_type = self.media_type.strip() if self.media_type is not None else None
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "media_type", media_type or None)

    @property
    def page_size(self) -> int:
        return LIBRARY_MEDIA_BROWSE_PAGE_SIZE

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    def with_page(self, page: int) -> "MediaTrashScope":
        return replace(self, page=page)

    def same_except_page(self, other: "MediaTrashScope") -> bool:
        return isinstance(other, MediaTrashScope) and self.with_page(
            1
        ) == other.with_page(1)


def _validate_media_trash_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise TypeError("Media Trash items must be a sequence.")
    stable_ids: set[str] = set()
    backing_ids: set[int] = set()
    frozen: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("Media Trash items must be mappings.")
        if set(item) != _MEDIA_TRASH_SUMMARY_KEYS:
            raise ValueError("Media Trash items must contain exactly five keys.")

        backing_id = item["backing_media_id"]
        if type(backing_id) is not int or backing_id < 1:
            raise ValueError("backing_media_id must be a positive integer.")
        stable_id = item["id"]
        if type(stable_id) is not str or stable_id != f"local:media:{backing_id}":
            raise ValueError("id must be the canonical backing_media_id identity.")
        if stable_id in stable_ids or backing_id in backing_ids:
            raise ValueError("Media Trash identities must be page-unique.")

        title = item["title"]
        if type(title) is not str or not title.strip() or title != title.strip():
            raise ValueError("title must be non-empty trimmed text.")
        media_type = item["media_type"]
        if media_type is not None and (
            type(media_type) is not str
            or not media_type
            or media_type != media_type.strip()
        ):
            raise ValueError("media_type must be trimmed non-empty text or None.")
        trash_date = item["trash_date"]
        if trash_date is not None:
            if type(trash_date) is not str or trash_date != trash_date.strip():
                raise TypeError("trash_date must be an ISO timestamp or None.")
            if len(trash_date) <= 10 or trash_date[10] not in {"T", " "}:
                raise ValueError("trash_date must be an ISO timestamp or None.")
            try:
                datetime.fromisoformat(trash_date.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "trash_date must be an ISO timestamp or None."
                ) from exc

        stable_ids.add(stable_id)
        backing_ids.add(backing_id)
        frozen.append(MappingProxyType(dict(item)))
    return tuple(frozen)


@dataclass(frozen=True)
class MediaTrashResult:
    """One exact immutable page returned by the local Media Trash service."""

    scope: MediaTrashScope
    items: tuple[Mapping[str, Any], ...]
    total: int
    limit: int
    offset: int
    types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MediaTrashScope):
            raise TypeError("scope must be a MediaTrashScope.")
        for field_name, value, minimum in (
            ("total", self.total, 0),
            ("limit", self.limit, 1),
            ("offset", self.offset, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"{field_name} must be an integer of at least {minimum}."
                )
        if self.limit != self.scope.page_size:
            raise ValueError("limit must match the requested page size.")
        if self.offset != self.scope.offset:
            raise ValueError("offset must match the requested page offset.")

        frozen_items = _validate_media_trash_items(self.items)
        expected_count = min(self.limit, max(self.total - self.offset, 0))
        if len(frozen_items) != expected_count:
            raise ValueError("Media Trash item count is invalid for this page.")
        if type(self.types) is not tuple or any(
            type(value) is not str or not value or value != value.strip()
            for value in self.types
        ):
            raise ValueError("types must be nonblank trimmed strings.")
        if self.types != tuple(sorted(set(self.types))):
            raise ValueError("types must be sorted and unique.")
        object.__setattr__(self, "items", frozen_items)

    @property
    def last_page(self) -> int:
        return max(1, (self.total + self.limit - 1) // self.limit)

    @property
    def out_of_range(self) -> bool:
        return self.scope.page > self.last_page


def build_media_trash_result(
    scope: MediaTrashScope, payload: Mapping[str, Any]
) -> MediaTrashResult:
    """Build a fail-closed exact Trash page from the canonical envelope."""
    if not isinstance(scope, MediaTrashScope):
        raise TypeError("scope must be a MediaTrashScope.")
    if not isinstance(payload, Mapping):
        raise TypeError("Media Trash result must be a mapping.")
    if set(payload) != _MEDIA_TRASH_ENVELOPE_KEYS:
        raise ValueError("Media Trash result must contain exactly five keys.")
    raw_items = payload["items"]
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        raise TypeError("Media Trash items must be a sequence.")
    raw_types = payload["types"]
    if not isinstance(raw_types, Sequence) or isinstance(
        raw_types, (str, bytes, bytearray)
    ):
        raise TypeError("Media Trash types must be a sequence.")
    return MediaTrashResult(
        scope=scope,
        items=tuple(raw_items),
        total=payload["total"],
        limit=payload["limit"],
        offset=payload["offset"],
        types=tuple(raw_types),
    )


MediaTrashRequestOrigin = Literal[
    "entry", "search", "type", "previous", "next", "retry", "mutation"
]
_MEDIA_TRASH_REQUEST_ORIGINS = frozenset(
    {"entry", "search", "type", "previous", "next", "retry", "mutation"}
)
_MEDIA_TRASH_MUTATION_STALE_COPY = "List may be out of date."


@dataclass(frozen=True)
class MediaTrashMutationTarget:
    """Full immutable identity captured before a Trash mutation."""

    stable_id: str
    backing_media_id: int
    title: str
    media_type: str | None
    trash_date: str | None
    page_index: int


@dataclass(frozen=True)
class MediaTrashBrowseState:
    """Complete immutable state owned by the Media Trash page source."""

    requested_scope: MediaTrashScope = MediaTrashScope()
    applied_result: MediaTrashResult | None = None
    retained_items: tuple[Mapping[str, Any], ...] = ()
    types: tuple[str, ...] = ()
    freshness: PageFreshness = "uninitialized"
    loading: bool = False
    error_copy: str = ""
    stale_copy: str = ""
    selected_id: str = ""
    confirmation_target: MediaTrashMutationTarget | None = None
    mutation_pending: bool = False
    request_origin: MediaTrashRequestOrigin = "entry"
    failed_scope: MediaTrashScope | None = None
    failed_origin: MediaTrashRequestOrigin | None = None
    committed_notice: str = ""


def _require_media_trash_state(state: MediaTrashBrowseState) -> None:
    if not isinstance(state, MediaTrashBrowseState):
        raise TypeError("state must be a MediaTrashBrowseState.")


def _require_media_trash_origin(origin: str) -> None:
    if origin not in _MEDIA_TRASH_REQUEST_ORIGINS:
        raise ValueError("origin is not a supported Media Trash request origin.")


def begin_media_trash_request(
    state: MediaTrashBrowseState,
    scope: MediaTrashScope,
    *,
    origin: MediaTrashRequestOrigin,
) -> MediaTrashBrowseState:
    """Begin one immutable exact-page request and clear page-local selection."""
    _require_media_trash_state(state)
    if not isinstance(scope, MediaTrashScope):
        raise TypeError("scope must be a MediaTrashScope.")
    _require_media_trash_origin(origin)
    if state.mutation_pending:
        raise ValueError("a Media Trash mutation is still pending.")
    return replace(
        state,
        requested_scope=scope,
        loading=True,
        error_copy="",
        selected_id="",
        confirmation_target=None,
        request_origin=origin,
        failed_scope=None,
        failed_origin=None,
    )


def apply_media_trash_result(
    state: MediaTrashBrowseState, result: MediaTrashResult
) -> MediaTrashBrowseState:
    """Apply one validated result for the state's current requested scope."""
    _require_media_trash_state(state)
    if not isinstance(result, MediaTrashResult):
        raise TypeError("result must be a MediaTrashResult.")
    if result.scope != state.requested_scope and not (
        result.scope.same_except_page(state.requested_scope)
        and result.scope.page < state.requested_scope.page
    ):
        raise ValueError("result scope must match the requested scope.")
    if result.out_of_range:
        raise ValueError("an out-of-range result cannot be applied.")
    selected_id = (
        str(result.items[0]["id"])
        if state.request_origin == "entry" and result.items
        else ""
    )
    return replace(
        state,
        applied_result=result,
        retained_items=result.items,
        types=result.types,
        freshness="fresh",
        loading=False,
        error_copy="",
        stale_copy="",
        selected_id=selected_id,
        confirmation_target=None,
        mutation_pending=False,
        failed_scope=None,
        failed_origin=None,
    )


def fail_media_trash_request(
    state: MediaTrashBrowseState,
    failed_scope: MediaTrashScope,
    *,
    copy: str,
) -> MediaTrashBrowseState:
    """Record a recoverable read failure without replacing retained authority."""
    _require_media_trash_state(state)
    if not isinstance(failed_scope, MediaTrashScope):
        raise TypeError("failed_scope must be a MediaTrashScope.")
    if not isinstance(copy, str) or not copy.strip():
        raise ValueError("copy must be non-empty text.")
    if state.freshness == "stale" or (
        state.applied_result is not None and failed_scope != state.requested_scope
    ):
        return replace(
            state,
            freshness="stale",
            loading=False,
            error_copy="",
            stale_copy=copy.strip(),
            confirmation_target=None,
            failed_scope=failed_scope,
            failed_origin=state.request_origin,
        )
    return replace(
        state,
        loading=False,
        error_copy=copy.strip(),
        confirmation_target=None,
        failed_scope=failed_scope,
        failed_origin=state.request_origin,
    )


def select_media_trash_item(
    state: MediaTrashBrowseState, stable_id: str
) -> MediaTrashBrowseState:
    """Select one visible item only while its page metadata is authoritative."""
    _require_media_trash_state(state)
    if not isinstance(stable_id, str):
        raise TypeError("stable_id must be a string.")
    visible_ids = {str(item["id"]) for item in state.retained_items}
    selected_id = (
        stable_id
        if state.freshness == "fresh"
        and not state.loading
        and not state.mutation_pending
        and stable_id in visible_ids
        else ""
    )
    return replace(
        state,
        selected_id=selected_id,
        confirmation_target=None,
        error_copy=state.error_copy if state.failed_scope is not None else "",
    )


def _media_trash_target_for_selected(
    state: MediaTrashBrowseState,
) -> MediaTrashMutationTarget | None:
    for page_index, item in enumerate(state.retained_items):
        if item["id"] == state.selected_id:
            return MediaTrashMutationTarget(
                stable_id=str(item["id"]),
                backing_media_id=int(item["backing_media_id"]),
                title=str(item["title"]),
                media_type=item["media_type"],
                trash_date=item["trash_date"],
                page_index=page_index,
            )
    return None


def open_media_trash_delete_confirmation(
    state: MediaTrashBrowseState,
) -> MediaTrashBrowseState:
    """Capture the full selected identity for irreversible confirmation."""
    _require_media_trash_state(state)
    target = (
        _media_trash_target_for_selected(state)
        if state.freshness == "fresh"
        and not state.loading
        and not state.mutation_pending
        else None
    )
    return replace(state, confirmation_target=target, error_copy="")


def cancel_media_trash_delete_confirmation(
    state: MediaTrashBrowseState,
) -> MediaTrashBrowseState:
    """Close permanent-delete confirmation without changing selection."""
    _require_media_trash_state(state)
    return replace(state, confirmation_target=None)


def begin_media_trash_mutation(
    state: MediaTrashBrowseState,
) -> MediaTrashBrowseState:
    """Claim the currently selected fresh identity for one mutation."""
    _require_media_trash_state(state)
    target = _media_trash_target_for_selected(state)
    if (
        target is None
        or state.freshness != "fresh"
        or state.loading
        or state.mutation_pending
        or (
            state.confirmation_target is not None
            and state.confirmation_target != target
        )
    ):
        return state
    return replace(
        state,
        mutation_pending=True,
        confirmation_target=None,
        error_copy="",
        failed_scope=None,
        failed_origin=None,
    )


def _require_pending_media_trash_target(
    state: MediaTrashBrowseState, target: MediaTrashMutationTarget
) -> None:
    if not isinstance(target, MediaTrashMutationTarget):
        raise TypeError("target must be a MediaTrashMutationTarget.")
    if not state.mutation_pending or _media_trash_target_for_selected(state) != target:
        raise ValueError("target does not own the pending Media Trash mutation.")


def fail_media_trash_mutation(
    state: MediaTrashBrowseState,
    target: MediaTrashMutationTarget,
    *,
    copy: str,
) -> MediaTrashBrowseState:
    """Release a pre-commit failure while retaining the authoritative row."""
    _require_media_trash_state(state)
    _require_pending_media_trash_target(state, target)
    if not isinstance(copy, str) or not copy.strip():
        raise ValueError("copy must be non-empty text.")
    return replace(
        state,
        mutation_pending=False,
        error_copy=copy.strip(),
        confirmation_target=None,
    )


def commit_media_trash_mutation(
    state: MediaTrashBrowseState,
    target: MediaTrashMutationTarget,
    *,
    notice: str,
) -> MediaTrashBrowseState:
    """Reconcile a committed removal and withdraw exact page authority."""
    _require_media_trash_state(state)
    _require_pending_media_trash_target(state, target)
    if not isinstance(notice, str) or not notice.strip():
        raise ValueError("notice must be non-empty text.")
    refresh_scope = (
        state.applied_result.scope
        if state.applied_result is not None
        else state.requested_scope
    )
    return replace(
        state,
        requested_scope=refresh_scope,
        retained_items=tuple(
            item for item in state.retained_items if item["id"] != target.stable_id
        ),
        freshness="stale",
        loading=True,
        error_copy="",
        stale_copy=_MEDIA_TRASH_MUTATION_STALE_COPY,
        selected_id="",
        confirmation_target=None,
        mutation_pending=False,
        request_origin="mutation",
        failed_scope=None,
        failed_origin=None,
        committed_notice=notice.strip(),
    )


@dataclass(frozen=True)
class LibraryMediaRow:
    """One selectable row in the Library Browse ▸ Media canvas."""

    media_id: str
    title: str
    media_type: str
    secondary: str
    selected: bool = False
    checked: bool = False
    loading: bool = False
    loaded: bool = False


@dataclass(frozen=True)
class LibraryMediaCanvasState:
    """Pure display state for the Library Browse ▸ Media canvas."""

    rows: tuple[LibraryMediaRow, ...]
    type_options: tuple[str | None, ...]
    active_type: str | None
    status_copy: str
    empty_copy: str
    selected_id: str
    preview_lines: tuple[str, ...]
    count: int
    select_mode: bool = False
    selected_count: int = 0
    # task-2853 AC3: True while the bulk-delete confirmation ("Delete N
    # selected items? ..." + Delete/Cancel) should render in place of the
    # normal select-mode toolbar row.
    confirming_bulk_delete: bool = False
    # task-4022 AC2: count of items from the most recently completed bulk
    # delete, shown as a "✓ deleted · N items" receipt with Undo/Dismiss
    # until acted on or replaced by a newer bulk-delete action. 0 means no
    # receipt to show (the normal state).
    delete_receipt_count: int = 0
    # task-14902: True while the type chooser's direct-pick strip replaces
    # the browse toolbar row (the Notes Sort choice-strip pattern).
    type_choices_visible: bool = False
    query: str = ""
    # task-28013: the browse sort chooser -- ``sort_by`` is the active
    # MediaBrowseScope sort, ``sort_choices_visible`` is True while its
    # direct-pick strip replaces the toolbar row (same choice-strip pattern
    # as the type chooser and the Prompts/Notes sort choosers).
    sort_by: str = "last_modified_desc"
    sort_choices_visible: bool = False
    # task-31236: name of the most recently dismissed review set, rendered
    # as a "✓ dismissed · <name>" receipt with Undo/Dismiss until acted on
    # or replaced. "" means no receipt (the normal state).
    review_dismiss_receipt_name: str = ""


@dataclass(frozen=True)
class LibraryMediaTrashRow:
    """One selectable row in the Library media Trash view (task-4025)."""

    media_id: str
    title: str
    media_type: str
    secondary: str
    selected: bool = False


@dataclass(frozen=True)
class LibraryMediaTrashState:
    """Pure display state for the Library media Trash view (task-4025).

    Attributes:
        rows: Trashed items in the seam's own trash_date-DESC order.
        count: Total trashed items reported by the seam (may exceed
            ``len(rows)`` when the fetch page was smaller than the trash).
        status_copy: Honest truncation line ("showing X of N") when the
            fetched rows undercount the total, else "".
        empty_copy: The empty-Trash copy -- only when the fetch has landed
            (``loading`` False), succeeded (``error`` empty), and found
            nothing; "" otherwise so a loading/error state never claims
            "Trash is empty".
        selected_id: Resolved selected row id ("" when there are no rows).
        loading: True while the trash fetch has not landed yet.
        error: Fetch-failure copy, "" on success.
        notice: Restore feedback line (e.g. "Restored 'Title'."), "" when
            nothing to report. Feedback only -- never a receipt: ADR-055's
            receipts accompany destruction, and restore is recovery.
    """

    rows: tuple[LibraryMediaTrashRow, ...]
    count: int
    status_copy: str
    empty_copy: str
    selected_id: str
    loading: bool = False
    error: str = ""
    notice: str = ""


def build_library_media_trash_state(
    records: Sequence[Any] | None,
    *,
    total: int = 0,
    selected_id: str = "",
    now: datetime | None = None,
    loading: bool | None = None,
    error: str = "",
    notice: str = "",
) -> LibraryMediaTrashState:
    """Build the Library media Trash view display state (task-4025).

    Args:
        records: Trash records from ``list_media_trash`` (id/title/type,
            plus ``trash_date`` where the seam selected it). ``None`` means
            the fetch has not landed yet (loading). Row order is preserved
            as given -- the seam already orders by ``trash_date DESC``, and
            re-sorting here would silently disagree with it for rows whose
            ``trash_date`` did not survive the projection.
        total: The seam's total trashed-item count (may exceed the fetched
            rows).
        selected_id: Requested selected media id; falls back to the first
            row when absent.
        now: Reference time for the "trashed <age>" secondary; defaults to
            current UTC time.
        loading: Explicit loading override; defaults to ``records is None``.
        error: Fetch-failure copy to surface instead of rows.
        notice: Restore feedback line to pass through.

    Returns:
        Immutable Trash view state.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    resolved_loading = (records is None) if loading is None else bool(loading)

    entries: list[tuple[str, str, str, str]] = []
    for record in records or ():
        if not isinstance(record, Mapping):
            continue
        media_id = _first_present_text(record, _ID_KEYS)
        if not media_id:
            continue
        entries.append(
            (
                media_id,
                _record_title(record),
                _first_present_text(record, _TYPE_KEYS),
                _first_present_text(record, ("trash_date",)),
            )
        )

    resolved_selected_id = str(selected_id or "")
    entry_ids = {media_id for media_id, _, _, _ in entries}
    if resolved_selected_id not in entry_ids:
        resolved_selected_id = entries[0][0] if entries else ""

    rows = []
    for media_id, title, media_type, trash_date in entries:
        age = format_console_relative_age(trash_date, now=reference_now)
        # The list's own secondary vocabulary ("{type} · {age}" / "{type}"
        # / "media"), with the age labelled for what it is here: when the
        # item was trashed, not when it was updated.
        trashed_age = f"trashed {age}" if age else ""
        rows.append(
            LibraryMediaTrashRow(
                media_id=media_id,
                title=title,
                media_type=media_type,
                secondary=_secondary_text(media_type, trashed_age),
                selected=media_id == resolved_selected_id,
            )
        )

    resolved_total = max(int(total or 0), len(rows))
    status_copy = (
        f"showing {len(rows)} of {resolved_total}"
        if rows and resolved_total > len(rows)
        else ""
    )
    empty_copy = (
        LIBRARY_MEDIA_TRASH_EMPTY_COPY
        if not rows and not resolved_loading and not error
        else ""
    )

    return LibraryMediaTrashState(
        rows=tuple(rows),
        count=resolved_total,
        status_copy=status_copy,
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        loading=resolved_loading,
        error=str(error or ""),
        notice=str(notice or ""),
    )


@dataclass(frozen=True)
class _MediaEntry:
    """Internal per-record fields used before rendering a display row."""

    media_id: str
    title: str
    media_type: str
    updated_raw: str
    sort_timestamp: datetime | None


def build_library_media_browse_state(
    result: MediaBrowseResult,
    *,
    type_options: tuple[str, ...],
    retained_items: tuple[Mapping[str, Any], ...] | None = None,
    selected_id: str = "",
    now: datetime | None = None,
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
    confirming_bulk_delete: bool = False,
    delete_receipt_count: int = 0,
    type_choices_visible: bool = False,
    sort_choices_visible: bool = False,
    loading_id: str = "",
    loaded_id: str = "",
    review_dismiss_receipt_name: str = "",
) -> LibraryMediaCanvasState:
    """Project one exact Media page without filtering, sorting, or slicing it.

    Args:
        review_dismiss_receipt_name: Name of the most recently dismissed
            review set, rendered as a "✓ dismissed · <name>" undo receipt
            until acted on or replaced (task-31236). "" (the default)
            means no receipt to show.
    """
    if not isinstance(result, MediaBrowseResult):
        raise TypeError("result must be a MediaBrowseResult.")
    if type(type_options) is not tuple or any(
        type(value) is not str or not value.strip() for value in type_options
    ):
        raise ValueError("type_options must be an exact tuple of non-empty strings.")
    normalized_types = tuple(sorted(set(type_options)))
    if len(normalized_types) != len(type_options):
        raise ValueError("type_options must contain unique source values.")
    items = (
        result.items
        if retained_items is None
        else validate_media_browse_items(retained_items)
    )
    reference_now = now if now is not None else datetime.now(timezone.utc)
    page_ids = {str(item["id"]) for item in items}
    resolved_selected_id = selected_id if selected_id in page_ids else ""
    if not resolved_selected_id and items:
        resolved_selected_id = str(items[0]["id"])
    rows = tuple(
        LibraryMediaRow(
            media_id=str(item["id"]),
            title=_record_title(item),
            media_type=_first_present_text(item, ("media_type",)),
            secondary=_secondary_text(
                _first_present_text(item, ("media_type",)),
                format_console_relative_age(
                    _first_present_text(item, ("updated_at",)), now=reference_now
                ),
            ),
            selected=item["id"] == resolved_selected_id,
            checked=item["id"] in selected_ids,
            loading=item["id"] == loading_id,
            loaded=item["id"] == loaded_id,
        )
        for item in items
    )
    selected = next(
        (item for item in items if item["id"] == resolved_selected_id), None
    )
    preview_lines = ()
    if selected is not None:
        media_type = _first_present_text(selected, ("media_type",)) or "unknown"
        age = format_console_relative_age(
            _first_present_text(selected, ("updated_at",)), now=reference_now
        )
        preview_lines = (
            _record_title(selected),
            f"Type: {media_type}",
            f"Updated: {age or 'unknown'}",
        )
    empty_copy = ""
    if not rows:
        if result.scope.query:
            empty_copy = f"No media matched ‘{result.scope.query}’."
        elif result.scope.media_type is not None:
            empty_copy = f"No media of type '{result.scope.media_type}'."
        else:
            empty_copy = LIBRARY_MEDIA_EMPTY_COPY
    return LibraryMediaCanvasState(
        rows=rows,
        type_options=(None, *normalized_types),
        active_type=result.scope.media_type,
        status_copy="",
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        preview_lines=preview_lines,
        count=result.total,
        select_mode=select_mode,
        selected_count=sum(row.checked for row in rows),
        confirming_bulk_delete=confirming_bulk_delete,
        delete_receipt_count=max(0, delete_receipt_count),
        type_choices_visible=type_choices_visible,
        query=result.scope.query,
        sort_by=result.scope.sort_by,
        sort_choices_visible=sort_choices_visible,
        review_dismiss_receipt_name=review_dismiss_receipt_name,
    )


def _first_present_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _record_title(record: Mapping[str, Any]) -> str:
    value = record.get("title")
    if value is None:
        return "Untitled media"
    text = str(value).strip()
    return text or "Untitled media"


def _parse_timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _secondary_text(media_type: str, age: str) -> str:
    """Return secondary display text: '{type} · {age}' or fallback.

    Rules:
    - If type and age both present: 'type · age'
    - If only type (no age): 'type'
    - If no type: 'media' (regardless of age)
    """
    has_type = bool(media_type)
    has_age = bool(age)

    if has_type and has_age:
        return f"{media_type} · {age}"
    elif has_type:
        return media_type
    else:
        # When no type, return 'media' regardless of age
        return "media"


def _sort_key(entry: _MediaEntry) -> tuple[int, float]:
    if entry.sort_timestamp is None:
        return (1, 0.0)
    return (0, -entry.sort_timestamp.timestamp())


def build_library_media_state(
    records: Sequence[Mapping[str, Any]],
    *,
    active_type: str = "All",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
    confirming_bulk_delete: bool = False,
    delete_receipt_count: int = 0,
    type_choices_visible: bool = False,
    review_dismiss_receipt_name: str = "",
) -> LibraryMediaCanvasState:
    """Build the Library Browse ▸ Media canvas display state.

    Args:
        records: Media records from the screen's media service.
            Tolerated to have missing/None fields.
        active_type: Filter rows to this media type, or "All" for no filter.
        selected_id: Requested selected media id; falls back to the
            first displayed row when absent from the filtered/limited rows.
        now: Reference time for relative age labels; defaults to current UTC time.
        limit: Maximum number of rows to display after sorting and filtering.
        delete_receipt_count: Count of items from the most recently
            completed bulk delete, rendered as a "✓ deleted · N items"
            receipt with Undo/Dismiss until acted on or replaced by a
            newer bulk-delete action. 0 (the default) means no receipt to
            show.
        review_dismiss_receipt_name: Name of the most recently dismissed
            review set, rendered as a "✓ dismissed · <name>" undo receipt
            until acted on or replaced (task-31236). "" (the default)
            means no receipt to show.

    Returns:
        Immutable canvas state: rows, type options, active type, status/empty copy,
        selection, preview lines, and total count.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)

    entries: list[_MediaEntry] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        media_id = _first_present_text(record, _ID_KEYS)
        if not media_id:
            continue
        updated_raw = _first_present_text(record, _UPDATED_KEYS)
        media_type = _first_present_text(record, _TYPE_KEYS)
        entries.append(
            _MediaEntry(
                media_id=media_id,
                title=_record_title(record),
                media_type=media_type,
                updated_raw=updated_raw,
                sort_timestamp=_parse_timestamp(updated_raw),
            )
        )

    # Calculate total count before filtering
    total_count = len(entries)

    # Filter by active_type if not "All"
    if active_type != "All":
        filtered_entries = [e for e in entries if e.media_type == active_type]
    else:
        filtered_entries = entries

    # Sort by updated timestamp desc (missing last)
    filtered_entries.sort(key=_sort_key)

    # Apply limit
    limited_entries = filtered_entries[: max(0, limit)]

    # Resolve selected_id
    resolved_selected_id = str(selected_id or "")
    displayed_ids = {entry.media_id for entry in limited_entries}
    if resolved_selected_id not in displayed_ids:
        resolved_selected_id = limited_entries[0].media_id if limited_entries else ""

    # Build rows
    rows = tuple(
        LibraryMediaRow(
            media_id=entry.media_id,
            title=entry.title,
            media_type=entry.media_type,
            secondary=_secondary_text(
                entry.media_type,
                format_console_relative_age(entry.updated_raw, now=reference_now),
            ),
            selected=entry.media_id == resolved_selected_id,
            checked=entry.media_id in selected_ids,
        )
        for entry in limited_entries
    )
    selected_count = sum(1 for r in rows if r.checked)

    # Build type_options: ("All",) + sorted distinct non-empty types
    distinct_types = {entry.media_type for entry in entries if entry.media_type}
    if active_type != "All":
        distinct_types.add(active_type)
    type_options = ("All",) + tuple(sorted(distinct_types))

    # Build status_copy and empty_copy
    if active_type != "All":
        # When filtering by type, report the count of all matches (pre-limit).
        status_copy = f"{len(filtered_entries)} of {total_count} · type: {active_type}"
        if not rows:
            empty_copy = f"No media of type '{active_type}'."
        else:
            empty_copy = ""
    else:
        # When showing all, no status copy
        status_copy = ""
        if not rows:
            empty_copy = LIBRARY_MEDIA_EMPTY_COPY
        else:
            empty_copy = ""

    # Build preview_lines for selected row
    selected_entry = next(
        (entry for entry in limited_entries if entry.media_id == resolved_selected_id),
        None,
    )
    if selected_entry is None:
        preview_lines: tuple[str, ...] = ()
    else:
        age = format_console_relative_age(selected_entry.updated_raw, now=reference_now)
        type_text = selected_entry.media_type or "unknown"
        age_text = age or "unknown"
        preview_lines = (
            selected_entry.title,
            f"Type: {type_text}",
            f"Updated: {age_text}",
        )

    return LibraryMediaCanvasState(
        rows=rows,
        type_options=type_options,
        active_type=active_type,
        status_copy=status_copy,
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        preview_lines=preview_lines,
        count=total_count,
        select_mode=select_mode,
        selected_count=selected_count,
        confirming_bulk_delete=confirming_bulk_delete,
        delete_receipt_count=max(0, delete_receipt_count),
        type_choices_visible=type_choices_visible,
        review_dismiss_receipt_name=review_dismiss_receipt_name,
    )
