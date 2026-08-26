"""Pure display-state contracts for the Library prompts canvas.

Consumes record mappings shaped like ``PromptsDatabase.fetch_prompt_details``
/ ``list_prompts`` rows (keys: ``id``, ``name``, ``author``, ``details``,
``system_prompt``, ``user_prompt``, ``keywords``, ``last_modified`` /
``created_at``, ``version``). No Textual imports; the only DB import is the
``ConflictError`` exception type used to classify save outcomes.
"""

from __future__ import annotations

import hashlib
import sqlite3
import json
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, cast

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    ArtifactDefinitionState,
    ArtifactType,
    BlockArtifactDefinition,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
    decode_prompt_artifact,
)
from tldw_chatbook.Prompt_Management.prompt_legacy_decomposer import (
    decompose_legacy_lanes,
)
from tldw_chatbook.Prompt_Management.prompt_restore_errors import (
    PromptRestoreError,
    PromptRestoreErrorCode,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    CANONICAL_JSON_UTF8_V1,
    PromptCapabilityError,
    PromptSourceCapabilities,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
    set_artifact_type,
)

from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

_TIMESTAMP_KEYS = ("last_modified", "created_at")

MAX_PROMPT_BROWSE_PAGE_SIZE = 100
DEFAULT_PROMPT_BROWSE_PAGE_SIZE = 20
_PROMPT_BROWSE_SORT_FIELDS = frozenset({"last_modified", "name"})
_PROMPT_BROWSE_SORT_ORDERS = frozenset({"asc", "desc"})
_PROMPT_BROWSE_STATUSES = frozenset(
    {
        "loading",
        "ready",
        "empty_library",
        "empty_collection",
        "no_matches",
        "error",
    }
)


PromptBrowseStatus = Literal[
    "loading",
    "ready",
    "empty_library",
    "empty_collection",
    "no_matches",
    "error",
]


@dataclass(frozen=True)
class PromptBrowseScope:
    """One normalized, local-only Library Prompt browse request."""

    backend: Literal["local"] = "local"
    query: str = ""
    collection_id: int | None = None
    sort_by: Literal["last_modified", "name"] = "last_modified"
    sort_order: Literal["asc", "desc"] = "desc"
    page: int = 1
    page_size: int = DEFAULT_PROMPT_BROWSE_PAGE_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend.strip().lower() != "local":
            raise ValueError("Prompt browsing is local-only.")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if self.collection_id is not None and (
            type(self.collection_id) is not int or self.collection_id <= 0
        ):
            raise ValueError("collection_id must be a positive integer or None.")
        if not isinstance(self.sort_by, str):
            raise TypeError("sort_by must be a string.")
        sort_by = self.sort_by.strip().lower()
        if sort_by not in _PROMPT_BROWSE_SORT_FIELDS:
            raise ValueError("sort_by must be 'last_modified' or 'name'.")
        if not isinstance(self.sort_order, str):
            raise TypeError("sort_order must be a string.")
        sort_order = self.sort_order.strip().lower()
        if sort_order not in _PROMPT_BROWSE_SORT_ORDERS:
            raise ValueError("sort_order must be 'asc' or 'desc'.")
        if type(self.page) is not int or self.page <= 0:
            raise ValueError("page must be a positive integer.")
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be a positive integer.")

        object.__setattr__(self, "backend", "local")
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "sort_by", sort_by)
        object.__setattr__(self, "sort_order", sort_order)
        object.__setattr__(
            self, "page_size", min(self.page_size, MAX_PROMPT_BROWSE_PAGE_SIZE)
        )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for every browse-affecting value."""
        encoded = json.dumps(
            (
                self.backend,
                self.query,
                self.collection_id,
                self.sort_by,
                self.sort_order,
                self.page,
                self.page_size,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _prompt_browse_request_token(request_token: int) -> int:
    if type(request_token) is not int or request_token <= 0:
        raise ValueError("request_token must be a positive integer.")
    return request_token


def _prompt_browse_integer(value: Any, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}.")
    return value


def _freeze_prompt_browse_value(value: Any) -> Any:
    """Detach one service value, normalizing timestamps into immutable text."""
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Prompt browse mappings must use JSON-like string keys.")
        return MappingProxyType(
            {key: _freeze_prompt_browse_value(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_prompt_browse_value(item) for item in value)
    raise TypeError("Prompt browse values must be JSON-like immutable data.")


def validate_prompt_browse_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate and detach stable Prompt page records for retained display."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("Prompt browse result items must be a sequence.")
    if any(not isinstance(item, Mapping) for item in items):
        raise TypeError("Prompt browse result items must be mappings.")
    stable_ids: set[str] = set()
    local_ids: set[int] = set()
    for item in items:
        stable_id = item.get("id")
        if type(stable_id) is not str or not stable_id.strip():
            raise ValueError("Prompt browse item id must be a non-blank string.")
        local_id = item.get("local_id")
        if type(local_id) is not int or local_id <= 0:
            raise ValueError("Prompt browse item local_id must be a positive integer.")
        if stable_id in stable_ids:
            raise ValueError("Prompt browse item id values must be unique.")
        if local_id in local_ids:
            raise ValueError("Prompt browse item local_id values must be unique.")
        stable_ids.add(stable_id)
        local_ids.add(local_id)
    return tuple(
        cast(Mapping[str, Any], _freeze_prompt_browse_value(item)) for item in items
    )


def _prompt_browse_settled_status(
    scope: PromptBrowseScope, *, has_items: bool
) -> PromptBrowseStatus:
    if has_items:
        return "ready"
    if scope.query:
        return "no_matches"
    if scope.collection_id is not None:
        return "empty_collection"
    return "empty_library"


@dataclass(frozen=True)
class PromptBrowseResult:
    """Immutable loading, result, or failure state for one browse request."""

    scope: PromptBrowseScope
    items: tuple[Mapping[str, Any], ...]
    total_items: int
    total_pages: int
    page: int
    status: PromptBrowseStatus
    request_fingerprint: str
    request_token: int
    error: str = ""
    requested_page: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PromptBrowseScope):
            raise TypeError("scope must be a PromptBrowseScope.")
        _prompt_browse_integer(self.total_items, field="total_items", minimum=0)
        _prompt_browse_integer(self.total_pages, field="total_pages", minimum=0)
        _prompt_browse_integer(self.page, field="page", minimum=1)
        _prompt_browse_request_token(self.request_token)
        if self.status not in _PROMPT_BROWSE_STATUSES:
            raise ValueError("status is not a supported Prompt browse status.")

        requested_page = (
            self.scope.page
            if self.requested_page is None
            else _prompt_browse_integer(
                self.requested_page, field="requested_page", minimum=1
            )
        )
        expected_request_scope = replace(self.scope, page=requested_page)
        if (
            not isinstance(self.request_fingerprint, str)
            or self.request_fingerprint != expected_request_scope.fingerprint
        ):
            raise ValueError("request_fingerprint does not match the request scope.")
        if not isinstance(self.error, str):
            raise TypeError("error must be a string.")
        normalized_error = self.error.strip()

        frozen_items = validate_prompt_browse_items(self.items)
        object.__setattr__(self, "items", frozen_items)
        object.__setattr__(self, "error", normalized_error)
        object.__setattr__(self, "requested_page", requested_page)

        if self.page != self.scope.page:
            raise ValueError("page must match scope.page.")
        if self.status in {"loading", "error"}:
            if requested_page != self.scope.page:
                raise ValueError("loading/error requested_page must match scope.page.")
            if frozen_items or self.total_items or self.total_pages:
                raise ValueError("loading/error status cannot include result items.")
            if self.status == "error" and not normalized_error:
                raise ValueError("error status requires non-empty error text.")
            if self.status == "loading" and normalized_error:
                raise ValueError("loading status cannot include error text.")
            return

        if normalized_error:
            raise ValueError("Settled result status cannot include error text.")
        expected_pages = (
            (self.total_items + self.scope.page_size - 1) // self.scope.page_size
            if self.total_items
            else 0
        )
        if self.total_pages != expected_pages:
            raise ValueError("total_pages does not match total_items and page_size.")
        if (
            clamp_prompt_browse_scope(
                expected_request_scope, total_pages=self.total_pages
            )
            != self.scope
        ):
            raise ValueError("page does not match the clamped requested page.")
        expected_items = min(
            self.scope.page_size,
            max(
                0,
                self.total_items - (self.page - 1) * self.scope.page_size,
            ),
        )
        if len(frozen_items) != expected_items:
            raise ValueError(
                "Prompt browse result item count is invalid for this page."
            )
        expected_status = _prompt_browse_settled_status(
            self.scope, has_items=bool(frozen_items)
        )
        if self.status != expected_status:
            raise ValueError("status does not match result items and scope.")

    @property
    def scope_fingerprint(self) -> str:
        return self.scope.fingerprint


def clamp_prompt_browse_scope(
    scope: PromptBrowseScope, *, total_pages: int
) -> PromptBrowseScope:
    """Clamp a requested page to the last exact page, or page one when empty."""
    if type(total_pages) is not int or total_pages < 0:
        raise ValueError("total_pages must be a non-negative integer.")
    last_page = max(1, total_pages)
    return scope if scope.page <= last_page else replace(scope, page=last_page)


def begin_prompt_browse(
    scope: PromptBrowseScope, *, request_token: int = 1
) -> PromptBrowseResult:
    """Build loading state bound to an exact scope fingerprint and token."""
    return PromptBrowseResult(
        scope=scope,
        items=(),
        total_items=0,
        total_pages=0,
        page=scope.page,
        status="loading",
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
    )


def _prompt_browse_int(record: Mapping[str, Any], key: str, *, minimum: int) -> int:
    return _prompt_browse_integer(record.get(key), field=key, minimum=minimum)


def build_prompt_browse_result(
    scope: PromptBrowseScope,
    record: Mapping[str, Any],
    *,
    request_token: int = 1,
) -> PromptBrowseResult:
    """Build one truthful result from the normalized exact-browse response."""
    if not isinstance(record, Mapping):
        raise TypeError("Prompt browse result must be a mapping.")
    raw_items = record.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise TypeError("Prompt browse result items must be a sequence.")
    items = tuple(raw_items)
    total_items = _prompt_browse_int(record, "total_items", minimum=0)
    total_pages = _prompt_browse_int(record, "total_pages", minimum=0)
    current_page = _prompt_browse_int(record, "current_page", minimum=1)
    page_alias = _prompt_browse_int(record, "page", minimum=1)
    per_page = _prompt_browse_int(record, "per_page", minimum=1)
    if page_alias != current_page:
        raise ValueError("page must match current_page.")
    if per_page != scope.page_size:
        raise ValueError("per_page must match the requested page_size.")
    resolved_scope = replace(scope, page=current_page)
    status = _prompt_browse_settled_status(resolved_scope, has_items=bool(items))

    return PromptBrowseResult(
        scope=resolved_scope,
        items=items,
        total_items=total_items,
        total_pages=total_pages,
        page=current_page,
        status=status,
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
        requested_page=scope.page,
    )


def build_prompt_browse_error(
    scope: PromptBrowseScope,
    *,
    request_token: int = 1,
    error: str = "Couldn't load prompts. Try again.",
) -> PromptBrowseResult:
    """Build failure state without misrepresenting it as an empty result."""
    return PromptBrowseResult(
        scope=scope,
        items=(),
        total_items=0,
        total_pages=0,
        page=scope.page,
        status="error",
        request_fingerprint=scope.fingerprint,
        request_token=request_token,
        error=error,
    )


def apply_prompt_browse_result(
    state: PromptBrowseResult, result: PromptBrowseResult
) -> PromptBrowseResult:
    """Settle only the matching in-flight scope fingerprint and request token."""
    if (
        state.status != "loading"
        or result.status == "loading"
        or state.request_fingerprint != result.request_fingerprint
        or state.request_token != result.request_token
        or state.scope.backend != result.scope.backend
        or state.scope.query != result.scope.query
        or state.scope.collection_id != result.scope.collection_id
        or state.scope.sort_by != result.scope.sort_by
        or state.scope.sort_order != result.scope.sort_order
        or state.scope.page_size != result.scope.page_size
        or state.scope.page != result.requested_page
    ):
        return state
    return result


PromptCollectionCatalogStatus = Literal["loading", "ready", "empty", "error"]
PromptMembershipStatus = Literal[
    "disabled",
    "loading",
    "ready",
    "applying",
    "success",
    "load_error",
    "apply_error",
]
PromptMembershipFailurePhase = Literal["load", "apply"]

_PROMPT_MEMBERSHIP_OUTCOME_MAX = 200


@dataclass(frozen=True)
class PromptCollectionOption:
    """One immutable local collection label keyed strictly by numeric ID."""

    collection_id: int
    name: str
    display_name: str

    def __post_init__(self) -> None:
        if type(self.collection_id) is not int or self.collection_id <= 0:
            raise ValueError("collection_id must be a positive integer.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Collection name is required.")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("Collection display_name is required.")


@dataclass(frozen=True)
class PromptCollectionCatalogState:
    """One exact bounded local collection catalog snapshot."""

    query: str
    items: tuple[PromptCollectionOption, ...]
    total: int
    limit: int
    offset: int
    status: PromptCollectionCatalogStatus
    request_token: int
    request_fingerprint: str
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if type(self.total) is not int or self.total < 0:
            raise ValueError("total must be a non-negative integer.")
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        _prompt_browse_request_token(self.request_token)
        if self.status not in {"loading", "ready", "empty", "error"}:
            raise ValueError("Unsupported collection catalog status.")
        if self.request_fingerprint != _prompt_collection_catalog_fingerprint(
            self.query
        ):
            raise ValueError("Collection catalog fingerprint does not match query.")
        if len({item.collection_id for item in self.items}) != len(self.items):
            raise ValueError("Collection catalog cannot contain duplicate IDs.")
        if len(self.items) > self.total:
            raise ValueError("Collection catalog items cannot exceed exact total.")
        if self.status == "error" and not self.error.strip():
            raise ValueError("Collection catalog error status requires error text.")
        if self.status != "error" and self.error:
            raise ValueError(
                "Only collection catalog error state may contain error text."
            )

    @property
    def has_more(self) -> bool:
        return len(self.items) < self.total

    @property
    def next_offset(self) -> int:
        return len(self.items)


def _prompt_collection_catalog_fingerprint(query: str) -> str:
    return hashlib.sha256(
        json.dumps(query, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def begin_prompt_collection_catalog(
    *,
    query: str,
    request_token: int,
    previous: PromptCollectionCatalogState | None = None,
    append: bool = False,
) -> PromptCollectionCatalogState:
    """Start one catalog request, retaining exact rows only for Load more."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    query = query.strip()
    _prompt_browse_request_token(request_token)
    if append:
        if previous is None or previous.query != query:
            raise ValueError("Load more requires the current matching catalog.")
        return replace(
            previous,
            offset=previous.next_offset,
            status="loading",
            request_token=request_token,
            error="",
        )
    return PromptCollectionCatalogState(
        query=query,
        items=(),
        total=0,
        limit=100,
        offset=0,
        status="loading",
        request_token=request_token,
        request_fingerprint=_prompt_collection_catalog_fingerprint(query),
    )


def _prompt_collection_option(record: Mapping[str, Any]) -> PromptCollectionOption:
    backend = record.get("backend", "local")
    if backend != "local":
        raise ValueError("Library Prompt collections are local-only.")
    collection_id = record.get("collection_id")
    name = record.get("name")
    display_name = record.get("display_name") or name
    return PromptCollectionOption(
        collection_id=collection_id,
        name=name,
        display_name=display_name,
    )


def apply_prompt_collection_catalog_page(
    state: PromptCollectionCatalogState,
    record: Mapping[str, Any],
    *,
    request_token: int,
    append: bool = False,
) -> PromptCollectionCatalogState:
    """Apply one exact service page, rejecting stale or divergent appends."""
    if request_token != state.request_token:
        return state
    if not isinstance(record, Mapping):
        raise TypeError("Collection catalog result must be a mapping.")
    raw_items = record.get("collections")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise TypeError("Collection catalog collections must be a sequence.")
    limit = _prompt_browse_integer(record.get("limit"), field="limit", minimum=1)
    if limit > 100:
        raise ValueError("limit must not exceed 100.")
    offset = _prompt_browse_integer(record.get("offset"), field="offset", minimum=0)
    total = _prompt_browse_integer(record.get("total"), field="total", minimum=0)
    if append and total != state.total:
        raise ValueError("Collection catalog exact total changed during append.")
    expected_offset = len(state.items) if append else 0
    if offset != expected_offset:
        raise ValueError("Collection catalog offset does not match the requested page.")
    page_items = tuple(_prompt_collection_option(item) for item in raw_items)
    if len(page_items) > limit:
        raise ValueError("Collection catalog page exceeds its limit.")
    items = state.items + page_items if append else page_items
    if len({item.collection_id for item in items}) != len(items):
        raise ValueError("Collection catalog cannot append duplicate IDs.")
    if len(items) > total:
        raise ValueError("Collection catalog exact total is smaller than loaded rows.")
    return PromptCollectionCatalogState(
        query=state.query,
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        status="ready" if items else "empty",
        request_token=request_token,
        request_fingerprint=state.request_fingerprint,
    )


def fail_prompt_collection_catalog(
    state: PromptCollectionCatalogState,
    *,
    request_token: int,
    error: str = "Couldn't load collections. Retry.",
) -> PromptCollectionCatalogState:
    """Settle only the current catalog request as a recoverable failure."""
    if request_token != state.request_token:
        return state
    return replace(state, status="error", error=error.strip())


@dataclass(frozen=True)
class PromptMembershipState:
    """Independent current/staged collection membership editor state."""

    prompt_id: int | None
    identity_fingerprint: str
    applied_ids: tuple[int, ...]
    staged_ids: tuple[int, ...]
    labels: tuple[tuple[int, str], ...]
    status: PromptMembershipStatus
    request_token: int
    outcome: str = ""
    disabled_reason: str = ""

    def __post_init__(self) -> None:
        if self.prompt_id is not None and (
            type(self.prompt_id) is not int or self.prompt_id <= 0
        ):
            raise ValueError("prompt_id must be a positive integer or None.")
        for field_name, values in (
            ("applied_ids", self.applied_ids),
            ("staged_ids", self.staged_ids),
        ):
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(f"{field_name} must contain positive integers.")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must be sorted and unique.")
        label_ids: list[int] = []
        for label in self.labels:
            if not isinstance(label, tuple) or len(label) != 2:
                raise ValueError("labels must contain ID and label pairs.")
            collection_id, display_label = label
            if type(collection_id) is not int or collection_id <= 0:
                raise ValueError("labels must use positive collection IDs.")
            if not isinstance(display_label, str):
                raise ValueError("membership labels must be strings.")
            label_ids.append(collection_id)
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("membership labels must use unique collection IDs.")
        if not set(label_ids).issubset(set(self.applied_ids) | set(self.staged_ids)):
            raise ValueError("membership labels must belong to applied or staged IDs.")
        if self.status not in {
            "disabled",
            "loading",
            "ready",
            "applying",
            "success",
            "load_error",
            "apply_error",
        }:
            raise ValueError("Unsupported Prompt membership status.")
        if not isinstance(self.outcome, str):
            raise ValueError("Membership outcome must be a string.")
        if self.status == "disabled":
            if not isinstance(self.disabled_reason, str) or not self.disabled_reason:
                raise ValueError("Disabled membership state requires a reason.")
            if (
                self.prompt_id is not None
                or self.identity_fingerprint
                or self.applied_ids
                or self.staged_ids
                or self.labels
                or self.request_token != 0
                or self.outcome
            ):
                raise ValueError("Disabled membership state cannot carry active data.")
            return
        if self.prompt_id is None:
            raise ValueError("Active membership state requires a Prompt identity.")
        if (
            not isinstance(self.identity_fingerprint, str)
            or not self.identity_fingerprint.strip()
        ):
            raise ValueError(
                "Active membership state requires an identity fingerprint."
            )
        _prompt_browse_request_token(self.request_token)
        if self.disabled_reason:
            raise ValueError("Active membership state cannot carry a disabled reason.")
        if self.status in {"load_error", "apply_error", "success"}:
            if not self.outcome.strip():
                raise ValueError(f"{self.status} membership state requires an outcome.")
            if len(self.outcome) > _PROMPT_MEMBERSHIP_OUTCOME_MAX:
                raise ValueError("Membership outcome is too long.")
        elif self.outcome:
            raise ValueError("Only settled membership state may contain an outcome.")

    @property
    def can_manage(self) -> bool:
        return self.status in {"ready", "success", "apply_error"}

    @property
    def can_retry_load(self) -> bool:
        return self.status == "load_error"

    @property
    def can_apply(self) -> bool:
        return self.can_manage and self.staged_ids != self.applied_ids


def disable_prompt_memberships(reason: str) -> PromptMembershipState:
    """Build a readable disabled state for an unsaved/foreign identity."""
    return PromptMembershipState(
        prompt_id=None,
        identity_fingerprint="",
        applied_ids=(),
        staged_ids=(),
        labels=(),
        status="disabled",
        request_token=0,
        disabled_reason=reason.strip() or "Memberships are unavailable.",
    )


def begin_prompt_memberships(
    *, prompt_id: int, identity_fingerprint: str, request_token: int
) -> PromptMembershipState:
    """Start membership loading for one exact persisted local Prompt."""
    _prompt_browse_request_token(request_token)
    if not isinstance(identity_fingerprint, str) or not identity_fingerprint:
        raise ValueError("identity_fingerprint is required.")
    return PromptMembershipState(
        prompt_id=prompt_id,
        identity_fingerprint=identity_fingerprint,
        applied_ids=(),
        staged_ids=(),
        labels=(),
        status="loading",
        request_token=request_token,
    )


def apply_prompt_memberships_loaded(
    state: PromptMembershipState,
    *,
    collection_ids: Sequence[int],
    labels: Mapping[int, str],
    request_token: int,
) -> PromptMembershipState:
    """Settle the current membership load without accepting a late result."""
    if state.status != "loading" or request_token != state.request_token:
        return state
    ids = tuple(sorted(set(collection_ids)))
    relevant_labels = tuple(
        (collection_id, str(labels[collection_id]))
        for collection_id in ids
        if collection_id in labels
    )
    return replace(
        state,
        applied_ids=ids,
        staged_ids=ids,
        labels=relevant_labels,
        status="ready",
        outcome="",
    )


def fail_prompt_memberships(
    state: PromptMembershipState,
    *,
    request_token: int,
    error: str,
    phase: PromptMembershipFailurePhase,
) -> PromptMembershipState:
    """Settle the exact current load or Apply as a recoverable failure."""
    expected_status = "loading" if phase == "load" else "applying"
    if phase not in {"load", "apply"}:
        raise ValueError("Unsupported membership failure phase.")
    if request_token != state.request_token or state.status != expected_status:
        return state
    return replace(
        state,
        status="load_error" if phase == "load" else "apply_error",
        outcome=error.strip(),
    )


def stage_prompt_memberships(
    state: PromptMembershipState, collection_ids: Sequence[int]
) -> PromptMembershipState:
    """Stage zero or more IDs without mutating the applied membership set."""
    if not state.can_manage:
        return state
    ids = tuple(sorted(set(collection_ids)))
    return replace(state, staged_ids=ids, status="ready", outcome="")


def begin_prompt_memberships_apply(
    state: PromptMembershipState, *, request_token: int
) -> PromptMembershipState:
    """Start one explicit Apply request; Prompt Save state is not part of this type."""
    _prompt_browse_request_token(request_token)
    if not state.can_apply:
        return state
    return replace(state, status="applying", request_token=request_token, outcome="")


def apply_prompt_memberships_saved(
    state: PromptMembershipState,
    *,
    collection_ids: Sequence[int],
    request_token: int,
) -> PromptMembershipState:
    """Settle only the current membership Apply request."""
    if state.status != "applying" or request_token != state.request_token:
        return state
    ids = tuple(sorted(set(collection_ids)))
    if ids != state.staged_ids:
        raise ValueError("Membership response does not match the staged IDs.")
    return replace(
        state,
        applied_ids=ids,
        staged_ids=ids,
        labels=tuple(label for label in state.labels if label[0] in ids),
        status="success",
        outcome="Memberships applied.",
    )


PromptHistoryPageStatus = Literal["closed", "loading", "loaded", "error"]
PromptHistoryCountStatus = Literal["idle", "loading", "loaded", "error"]
PromptHistoryRestoreKind = Literal[
    "restored",
    "no_change",
    "conflict",
    "snapshot_unavailable",
    "current_unavailable",
    "validation_error",
    "name_conflict",
    "error",
]


@dataclass(frozen=True)
class PromptHistoryRow:
    """One normalized retained Prompt/Recipe snapshot for a read-only preview."""

    prompt_uuid: str
    change_id: int
    version: int
    timestamp: str
    artifact_type: str
    artifact_type_raw: str
    name: str
    author: str
    details: str
    system_preview: str
    user_preview: str
    keywords: tuple[str, ...]
    keywords_captured: bool
    compatibility_state: str
    compatibility_reason: str
    restore_eligible: bool
    changed_fields: tuple[str, ...]
    change_summary: str


@dataclass(frozen=True)
class PromptHistoryPage:
    """A single normalized local retained-history page."""

    items: tuple[PromptHistoryRow, ...]
    total_count: int
    has_more: bool
    next_before_change_id: int | None


@dataclass(frozen=True)
class PromptHistoryRequest:
    """Prompt identity and scope token shared by one asynchronous history request."""

    prompt_uuid: str
    scope_token: int
    request_token: int


@dataclass(frozen=True)
class PromptHistoryPageRequest(PromptHistoryRequest):
    """One page request, including its exact retained-history cursor."""

    before_change_id: int | None


@dataclass(frozen=True)
class PromptHistoryPreviewRequest(PromptHistoryRequest):
    """One selected retained snapshot preview request."""

    change_id: int
    source_version: int


@dataclass(frozen=True)
class PromptHistorySelection:
    """The retained snapshot currently selected for inspection or restoration."""

    prompt_uuid: str
    change_id: int
    source_version: int
    row: PromptHistoryRow


@dataclass(frozen=True)
class PromptHistoryRestoreTarget:
    """The exact conditional-restore values captured from one selected row."""

    prompt_uuid: str
    change_id: int
    source_version: int
    expected_current_version: int


@dataclass(frozen=True)
class PromptHistoryRestoreRequest(PromptHistoryRestoreTarget):
    """A restore target bound to the editor scope and one worker request."""

    scope_token: int
    request_token: int


@dataclass(frozen=True)
class PromptHistoryRestoreGate:
    """Whether the selected snapshot may be restored from the current editor."""

    enabled: bool
    reason: str
    target: PromptHistoryRestoreTarget | None


@dataclass(frozen=True)
class PromptHistoryRestoreOutcome:
    """Stable user-facing interpretation of one local restore result."""

    kind: PromptHistoryRestoreKind
    message: str
    reload_required: bool
    keyword_disclosure: str = ""


@dataclass(frozen=True)
class PromptHistoryState:
    """Immutable retained-history display state for one open Library prompt."""

    prompt_uuid: str
    current_version: int | None
    scope_token: int
    is_open: bool = False
    page_status: PromptHistoryPageStatus = "closed"
    count_status: PromptHistoryCountStatus = "idle"
    retained_count: int | None = None
    rows: tuple[PromptHistoryRow, ...] = ()
    has_more: bool = False
    next_before_change_id: int | None = None
    selected: PromptHistorySelection | None = None
    count_request: PromptHistoryRequest | None = None
    page_request: PromptHistoryPageRequest | None = None
    preview_request: PromptHistoryPreviewRequest | None = None
    restore_request: PromptHistoryRestoreRequest | None = None
    restore_outcome: PromptHistoryRestoreOutcome | None = None
    restore_refresh_pending: bool = False
    error: str = ""
    last_request_token: int = -1


def _history_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _history_text(value: Any) -> str:
    """Return literal display text without stripping previews or metadata."""
    return value if isinstance(value, str) else ""


def _history_string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def build_prompt_history_row(record: Mapping[str, Any]) -> PromptHistoryRow:
    """Build an immutable row from the normalized retained-history contract."""
    if not isinstance(record, Mapping):
        raise TypeError("Retained history row must be a mapping.")
    prompt_uuid = record.get("prompt_uuid")
    if not isinstance(prompt_uuid, str) or not prompt_uuid:
        raise ValueError("Retained history row must include prompt_uuid.")
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("Retained history row must include timestamp.")
    restore_eligible = record.get("restore_eligible")
    if type(restore_eligible) is not bool:
        raise TypeError("Retained history row restore_eligible must be a bool.")
    keywords_captured = record.get("keywords_captured")
    if type(keywords_captured) is not bool:
        raise TypeError("Retained history row keywords_captured must be a bool.")
    return PromptHistoryRow(
        prompt_uuid=prompt_uuid,
        change_id=_history_positive_int(record.get("change_id"), field="change_id"),
        version=_history_positive_int(record.get("version"), field="version"),
        timestamp=timestamp,
        artifact_type=_history_text(record.get("artifact_type")) or "prompt",
        artifact_type_raw=_history_text(record.get("artifact_type_raw")),
        name=_history_text(record.get("name")),
        author=_history_text(record.get("author")),
        details=_history_text(record.get("details")),
        system_preview=_history_text(
            record["system_prompt"]
            if "system_prompt" in record
            else record.get("compiled_system_prompt")
        ),
        user_preview=_history_text(
            record["user_prompt"]
            if "user_prompt" in record
            else record.get("compiled_user_prompt")
        ),
        keywords=_history_string_tuple(record.get("keywords")),
        keywords_captured=keywords_captured,
        compatibility_state=_history_text(record.get("compatibility_state")),
        compatibility_reason=_history_text(record.get("compatibility_reason")),
        restore_eligible=restore_eligible,
        changed_fields=_history_string_tuple(record.get("changed_fields")),
        change_summary=_history_text(record.get("change_summary")),
    )


def build_prompt_history_page(record: Mapping[str, Any]) -> PromptHistoryPage:
    """Build a bounded retained-history page from the local normalized shape."""
    if not isinstance(record, Mapping):
        raise TypeError("Retained history page must be a mapping.")
    raw_items = record.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise TypeError("Retained history page items must be a sequence.")
    items = tuple(build_prompt_history_row(item) for item in raw_items)
    change_ids = tuple(row.change_id for row in items)
    if len(change_ids) != len(set(change_ids)):
        raise ValueError("Retained history page cannot contain duplicate change IDs.")
    if any(newer <= older for newer, older in zip(change_ids, change_ids[1:])):
        raise ValueError("Retained history page rows must be newest first.")
    prompt_uuids = {row.prompt_uuid for row in items}
    if len(prompt_uuids) > 1:
        raise ValueError("Retained history page rows must share one prompt UUID.")
    total_count = record.get("total_count")
    if type(total_count) is not int or total_count < len(items) or total_count < 0:
        raise ValueError("Retained history page total_count is invalid.")
    has_more = record.get("has_more")
    if type(has_more) is not bool:
        raise TypeError("Retained history page has_more must be a bool.")
    cursor = record.get("next_before_change_id")
    if has_more:
        if not items or type(cursor) is not int or cursor != items[-1].change_id:
            raise ValueError("Retained history page cursor is invalid.")
    elif cursor is not None:
        raise ValueError("Final retained history pages cannot have a cursor.")
    return PromptHistoryPage(
        items=items,
        total_count=total_count,
        has_more=has_more,
        next_before_change_id=cursor,
    )


def build_prompt_history_state(
    *, prompt_uuid: str, current_version: int | None, scope_token: int
) -> PromptHistoryState:
    """Start a closed retained-history scope for one selected local Prompt."""
    if not isinstance(prompt_uuid, str) or not prompt_uuid:
        raise ValueError("prompt_uuid is required for retained history.")
    if current_version is not None:
        _history_positive_int(current_version, field="current_version")
    _history_positive_int(scope_token, field="scope_token")
    return PromptHistoryState(
        prompt_uuid=prompt_uuid,
        current_version=current_version,
        scope_token=scope_token,
    )


def close_prompt_history(state: PromptHistoryState) -> PromptHistoryState:
    """Collapse page UI without cancelling an accepted conditional restore."""
    return replace(
        state,
        is_open=False,
        page_status="closed",
        rows=(),
        has_more=False,
        next_before_change_id=None,
        selected=None,
        page_request=None,
        preview_request=None,
        restore_outcome=None,
        error="",
    )


def reset_prompt_history_page(state: PromptHistoryState) -> PromptHistoryState:
    """Clear page-scoped work for an explicit first-page reload."""
    return replace(
        state,
        is_open=True,
        page_status="closed",
        rows=(),
        has_more=False,
        next_before_change_id=None,
        selected=None,
        page_request=None,
        preview_request=None,
        restore_request=None,
        restore_outcome=None,
        error="",
    )


def _next_history_request_token(state: PromptHistoryState, request_token: int) -> int:
    _history_positive_int(request_token, field="request_token")
    if request_token <= state.last_request_token:
        raise ValueError(
            "Retained history request tokens must increase within a scope."
        )
    return request_token


def begin_prompt_history_count(
    state: PromptHistoryState, *, request_token: int
) -> tuple[PromptHistoryState, PromptHistoryRequest]:
    """Mark the independent retained-count request loading and return its guard."""
    token = _next_history_request_token(state, request_token)
    request = PromptHistoryRequest(state.prompt_uuid, state.scope_token, token)
    return (
        replace(
            state,
            count_status="loading",
            count_request=request,
            error="",
            last_request_token=token,
        ),
        request,
    )


def apply_prompt_history_count(
    state: PromptHistoryState,
    request: PromptHistoryRequest,
    *,
    total_count: int | None = None,
    error: str = "",
) -> PromptHistoryState:
    """Settle a count only when its prompt identity, scope, and token still match."""
    if state.count_request != request:
        return state
    if error:
        return replace(state, count_status="error", count_request=None, error=error)
    if type(total_count) is not int or total_count < 0:
        return replace(
            state,
            count_status="error",
            count_request=None,
            error="Retained history count was invalid.",
        )
    return replace(
        state,
        count_status="loaded",
        retained_count=total_count,
        count_request=None,
    )


def prompt_history_count_label(state: PromptHistoryState) -> str:
    """Render the exact collapsed disclosure label for the settled retained count."""
    if state.retained_count is None:
        return "Retained history (…)"
    return f"Retained history ({state.retained_count})"


def begin_prompt_history_page(
    state: PromptHistoryState, *, request_token: int
) -> tuple[PromptHistoryState, PromptHistoryPageRequest]:
    """Open/retry/load an older bounded page and return its exact cursor guard."""
    if state.rows and not state.has_more:
        raise ValueError("No older retained history pages are available.")
    token = _next_history_request_token(state, request_token)
    before_change_id = state.next_before_change_id if state.rows else None
    request = PromptHistoryPageRequest(
        state.prompt_uuid, state.scope_token, token, before_change_id
    )
    return (
        replace(
            state,
            is_open=True,
            page_status="loading",
            page_request=request,
            error="",
            last_request_token=token,
        ),
        request,
    )


def apply_prompt_history_page(
    state: PromptHistoryState,
    request: PromptHistoryPageRequest,
    page: PromptHistoryPage | None,
    *,
    error: str = "",
) -> PromptHistoryState:
    """Merge one matching page append-only, refreshing count from its same read."""
    if state.page_request != request:
        return state
    if error:
        return replace(state, page_status="error", page_request=None, error=error)
    if page is None or any(row.prompt_uuid != state.prompt_uuid for row in page.items):
        return replace(
            state,
            page_status="error",
            page_request=None,
            error="Retained history page was invalid.",
        )
    existing_ids = {row.change_id for row in state.rows}
    incoming_ids = {row.change_id for row in page.items}
    if existing_ids.intersection(incoming_ids):
        return replace(
            state,
            page_status="error",
            page_request=None,
            error="Retained history page overlaps an already loaded row.",
        )
    if state.rows and (
        request.before_change_id != state.next_before_change_id
        or any(row.change_id >= request.before_change_id for row in page.items)
    ):
        return replace(
            state,
            page_status="error",
            page_request=None,
            error="Retained history page cursor no longer matches.",
        )
    rows = state.rows + page.items
    count_updates: dict[str, Any] = {}
    if (
        state.count_request is None
        or state.count_request.request_token < request.request_token
    ):
        count_updates = {
            "retained_count": page.total_count,
            "count_status": "loaded",
            "count_request": None,
        }
    return replace(
        state,
        page_status="loaded",
        page_request=None,
        rows=rows,
        has_more=page.has_more,
        next_before_change_id=page.next_before_change_id,
        **count_updates,
    )


def begin_prompt_history_preview(
    state: PromptHistoryState,
    *,
    change_id: int,
    source_version: int,
    request_token: int,
) -> tuple[PromptHistoryState, PromptHistoryPreviewRequest]:
    """Capture an existing row as a stale-guarded retained preview selection."""
    _history_positive_int(change_id, field="change_id")
    _history_positive_int(source_version, field="source_version")
    if not any(
        row.change_id == change_id and row.version == source_version
        for row in state.rows
    ):
        raise ValueError("Selected retained history row is not loaded.")
    token = _next_history_request_token(state, request_token)
    request = PromptHistoryPreviewRequest(
        state.prompt_uuid, state.scope_token, token, change_id, source_version
    )
    return replace(state, preview_request=request, last_request_token=token), request


def apply_prompt_history_preview(
    state: PromptHistoryState, request: PromptHistoryPreviewRequest
) -> PromptHistoryState:
    """Select a loaded row only if its identity and request token still match."""
    if state.preview_request != request:
        return state
    row = next(
        (
            candidate
            for candidate in state.rows
            if candidate.prompt_uuid == request.prompt_uuid
            and candidate.change_id == request.change_id
            and candidate.version == request.source_version
        ),
        None,
    )
    if row is None:
        return replace(
            state,
            preview_request=None,
            error="Selected retained version is no longer loaded.",
        )
    return replace(
        state,
        selected=PromptHistorySelection(
            prompt_uuid=request.prompt_uuid,
            change_id=request.change_id,
            source_version=request.source_version,
            row=row,
        ),
        preview_request=None,
    )


def history_restore_gate(
    state: PromptHistoryState, *, dirty: bool
) -> PromptHistoryRestoreGate:
    """Return the pure restore gate while leaving history viewing always available."""
    if state.restore_refresh_pending:
        return PromptHistoryRestoreGate(False, "Refreshing the restored Prompt…", None)
    if state.selected is None:
        return PromptHistoryRestoreGate(
            False, "Select a retained version to restore.", None
        )
    if dirty:
        return PromptHistoryRestoreGate(
            False,
            "Save or discard unsaved changes before restoring retained history.",
            None,
        )
    if state.current_version is None:
        return PromptHistoryRestoreGate(
            False, "Reload this Prompt before restoring retained history.", None
        )
    if not state.selected.row.restore_eligible:
        return PromptHistoryRestoreGate(
            False,
            state.selected.row.compatibility_reason
            or "This retained version is preview-only and cannot be restored.",
            None,
        )
    return PromptHistoryRestoreGate(
        True,
        "",
        PromptHistoryRestoreTarget(
            prompt_uuid=state.selected.prompt_uuid,
            change_id=state.selected.change_id,
            source_version=state.selected.source_version,
            expected_current_version=state.current_version,
        ),
    )


def begin_prompt_history_restore(
    state: PromptHistoryState, *, request_token: int, dirty: bool
) -> tuple[
    PromptHistoryState, PromptHistoryRestoreRequest | None, PromptHistoryRestoreGate
]:
    """Capture a gated conditional restore request for one selected snapshot."""
    gate = history_restore_gate(state, dirty=dirty)
    if not gate.enabled or gate.target is None:
        return state, None, gate
    token = _next_history_request_token(state, request_token)
    target = gate.target
    request = PromptHistoryRestoreRequest(
        prompt_uuid=target.prompt_uuid,
        change_id=target.change_id,
        source_version=target.source_version,
        expected_current_version=target.expected_current_version,
        scope_token=state.scope_token,
        request_token=token,
    )
    return (
        replace(
            state,
            restore_request=request,
            restore_outcome=None,
            error="",
            last_request_token=token,
        ),
        request,
        gate,
    )


def _outcome_versions(result: Mapping[str, Any]) -> tuple[int | None, int | None]:
    source_version = result.get("source_version")
    current_version = result.get("current_version")
    return (
        source_version if type(source_version) is int and source_version > 0 else None,
        current_version
        if type(current_version) is int and current_version > 0
        else None,
    )


def format_prompt_history_restore_outcome(
    result: Mapping[str, Any] | None = None, *, error: Exception | None = None
) -> PromptHistoryRestoreOutcome:
    """Classify local restore results into stable UI copy without side effects."""
    if isinstance(error, PromptRestoreError):
        if error.code == PromptRestoreErrorCode.EXPECTED_VERSION:
            return PromptHistoryRestoreOutcome(
                "conflict",
                "This Prompt changed elsewhere. Reload before restoring.",
                True,
            )
        if error.code == PromptRestoreErrorCode.NAME_CONFLICT:
            return PromptHistoryRestoreOutcome(
                "name_conflict",
                "Another active Prompt already uses this name. Rename it or choose another retained version, then retry.",
                False,
            )
        return PromptHistoryRestoreOutcome(
            "validation_error",
            "This retained version couldn't be validated for restore. Choose another retained version, then retry.",
            False,
        )
    if error is not None:
        return PromptHistoryRestoreOutcome(
            "error", "Couldn't restore retained history.", False
        )
    if not isinstance(result, Mapping):
        return PromptHistoryRestoreOutcome(
            "error", "Couldn't restore retained history.", False
        )
    kind = result.get("outcome")
    source_version, current_version = _outcome_versions(result)
    retained_keywords = result.get("retained_current_keywords") is True
    disclosure = (
        "Current keywords were retained because this older retained version did not capture keywords."
        if retained_keywords
        else ""
    )
    if kind == "restored" and source_version is not None:
        new_version = result.get("new_version")
        if type(new_version) is int and new_version > 0:
            return PromptHistoryRestoreOutcome(
                "restored",
                f"Restored v{source_version} as current v{new_version}.",
                False,
                disclosure,
            )
    if (
        kind == "no_change"
        and source_version is not None
        and current_version is not None
    ):
        return PromptHistoryRestoreOutcome(
            "no_change",
            f"Retained v{source_version} already matches current v{current_version}; no new version was created.",
            False,
            disclosure,
        )
    if kind == "snapshot_unavailable":
        return PromptHistoryRestoreOutcome(
            "snapshot_unavailable",
            "This retained version is no longer available. Reload retained history.",
            True,
        )
    if kind == "current_unavailable":
        return PromptHistoryRestoreOutcome(
            "current_unavailable",
            "This Prompt is no longer available. Reload the Library.",
            True,
        )
    return PromptHistoryRestoreOutcome(
        "error", "Couldn't restore retained history.", False
    )


def apply_prompt_history_restore(
    state: PromptHistoryState,
    request: PromptHistoryRestoreRequest,
    outcome: PromptHistoryRestoreOutcome,
) -> PromptHistoryState:
    """Apply an outcome only to its exact accepted conditional-write request."""
    if state.restore_request != request:
        return state
    return replace(
        state,
        restore_request=None,
        restore_outcome=outcome,
        restore_refresh_pending=outcome.kind == "restored",
    )


@dataclass(frozen=True)
class PromptArtifactDraft:
    """Exact structured payload measurements used by Library save gates."""

    artifact_type: ArtifactType
    definition: BlockArtifactDefinition
    system_prompt: str
    user_prompt: str
    definition_bytes: bytes
    request_bytes: bytes


def _definition_mapping(definition: BlockArtifactDefinition) -> dict[str, Any]:
    """Serialize one validated block definition without optional null fields."""
    return {
        "kind": definition.kind,
        "schema_version": definition.schema_version,
        "lanes": [
            {
                "id": lane.id,
                "blocks": [
                    {
                        "id": block.id,
                        "title": block.title,
                        "syntax": block.syntax,
                        "content": block.content,
                        **(
                            {"xml_tag": block.xml_tag}
                            if block.xml_tag is not None
                            else {}
                        ),
                        **(
                            {"mapping_hint": block.mapping_hint}
                            if block.mapping_hint is not None
                            else {}
                        ),
                    }
                    for block in lane.blocks
                ],
            }
            for lane in definition.lanes
        ],
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prepare_prompt_artifact_save(
    state: PromptBlockEditorState,
    *,
    artifact_type: ArtifactType,
    include_recipe_starter_content: bool,
    request_fields: Mapping[str, Any],
) -> tuple[PromptArtifactDraft, dict[str, Any], PromptBlockEditorState]:
    """Build the exact structured save mapping and its measured working copy.

    Recipe starter content is opt-in. Turning it off clears only block content;
    stable IDs, lane order, titles, syntax, XML tags, and mapping hints remain.
    """
    if state.issues:
        raise ValueError("Fix block validation errors before saving.")
    prepared = set_artifact_type(state, artifact_type)
    if artifact_type == "recipe" and not include_recipe_starter_content:
        definition = replace(
            prepared.definition,
            lanes=tuple(
                replace(
                    lane,
                    blocks=tuple(replace(block, content="") for block in lane.blocks),
                )
                for lane in prepared.definition.lanes
            ),
        )
        prepared = PromptBlockEditorState.from_definition(
            artifact_type="recipe",
            definition=definition,
            dirty_block_ids=prepared.dirty_block_ids,
        )

    definition_mapping = _definition_mapping(prepared.definition)
    payload = {key: value for key, value in request_fields.items() if value is not None}
    payload.update(
        {
            "artifact_type": artifact_type,
            "prompt_format": "structured",
            "prompt_schema_version": prepared.definition.schema_version,
            "prompt_definition": definition_mapping,
            "system_prompt": prepared.compiled_system,
            "user_prompt": prepared.compiled_user,
        }
    )
    draft = PromptArtifactDraft(
        artifact_type=artifact_type,
        definition=prepared.definition,
        system_prompt=prepared.compiled_system,
        user_prompt=prepared.compiled_user,
        definition_bytes=_canonical_json_bytes(definition_mapping),
        request_bytes=_canonical_json_bytes(payload),
    )
    return draft, payload, prepared


def require_artifact_save_supported(
    draft: PromptArtifactDraft,
    capabilities: PromptSourceCapabilities,
    *,
    update_original: bool = False,
    expected_version: int | None = None,
) -> None:
    """Reject unsupported or oversized artifact saves without truncation."""
    expected_kind = (
        "block_prompt" if draft.artifact_type == "prompt" else "block_recipe"
    )
    if draft.definition.kind != expected_kind:
        raise ValueError("artifact_type and prompt definition kind must agree.")

    pair = (draft.definition.schema_version, draft.definition.kind)
    if pair not in capabilities.structured_kinds:
        raise PromptCapabilityError(capabilities.backend, f"structured kind {pair!r}")
    if draft.artifact_type not in capabilities.artifact_types:
        raise PromptCapabilityError(
            capabilities.backend, f"artifact type {draft.artifact_type!r}"
        )
    if capabilities.json_byte_measurement != CANONICAL_JSON_UTF8_V1:
        raise PromptCapabilityError(
            capabilities.backend, "canonical JSON byte measurement"
        )

    for field, value in (
        ("system_prompt", draft.system_prompt),
        ("user_prompt", draft.user_prompt),
    ):
        if len(value) > capabilities.compiled_lane_limit:
            raise ValueError(
                f"{field} exceeds {capabilities.compiled_lane_limit} characters; "
                "shorten this lane or choose a source with a larger limit."
            )
    for field, value, limit in (
        ("prompt_definition", draft.definition_bytes, capabilities.definition_limit),
        ("request", draft.request_bytes, capabilities.request_limit),
    ):
        if len(value) > limit:
            raise ValueError(
                f"{field} exceeds {limit} UTF-8 bytes; reduce that field or "
                "choose a source with a larger limit."
            )

    if not update_original:
        return
    if not capabilities.conditional_update:
        raise ValueError(
            "This source does not support conditional update; save as new."
        )
    if type(expected_version) is not int or expected_version < 1:
        raise ValueError(
            "Update original requires the captured current version; Reload or save as new."
        )


_PROMPT_SELECTION_SQLITE_MAX_INTEGER = 2**63 - 1
_PROMPT_SELECTION_ARTIFACT_TYPES = frozenset({"prompt", "recipe"})


def _require_prompt_selection_integer(value: object, *, field_name: str) -> None:
    if type(value) is not int or not 1 <= value <= _PROMPT_SELECTION_SQLITE_MAX_INTEGER:
        raise ValueError(f"{field_name} must be a positive integer in SQLite range.")


def _require_prompt_selection_title(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("title must be non-empty exact text.")


def _require_prompt_selection_artifact_type(value: object) -> None:
    if type(value) is not str or value not in _PROMPT_SELECTION_ARTIFACT_TYPES:
        raise ValueError("artifact_type must be exactly 'prompt' or 'recipe'.")


@dataclass(frozen=True, slots=True)
class PromptSelectionEntry:
    """One Prompt identity and version captured at selection time.

    Attributes:
        local_id: Positive SQLite-range local Prompt row ID.
        expected_version: Positive version captured when selected.
        title: Non-empty literal Prompt or Recipe title captured when selected.
        artifact_type: Exact supported artifact type.
    """

    local_id: int = dataclass_field(repr=False)
    expected_version: int = dataclass_field(repr=False)
    title: str = dataclass_field(repr=False)
    artifact_type: ArtifactType = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        _require_prompt_selection_integer(self.local_id, field_name="local_id")
        _require_prompt_selection_integer(
            self.expected_version, field_name="expected_version"
        )
        _require_prompt_selection_title(self.title)
        _require_prompt_selection_artifact_type(self.artifact_type)


@dataclass(frozen=True, slots=True)
class PromptSelectionBasket:
    """Immutable cross-page Prompt selection with semantic generations.

    Attributes:
        entries: Selected entries in selection order.
        generation: Non-negative counter advanced once per semantic change.
    """

    entries: tuple[PromptSelectionEntry, ...] = dataclass_field(default=(), repr=False)
    generation: int = dataclass_field(default=0, repr=False)

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("entries must be an exact tuple.")
        if any(type(entry) is not PromptSelectionEntry for entry in self.entries):
            raise TypeError("entries must contain exact PromptSelectionEntry values.")
        local_ids = tuple(entry.local_id for entry in self.entries)
        if len(set(local_ids)) != len(local_ids):
            raise ValueError("entries must use unique local IDs.")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer.")

    @property
    def canonical_entries(self) -> tuple[PromptSelectionEntry, ...]:
        """Return selected entries in deterministic ascending local-ID order."""
        return tuple(sorted(self.entries, key=lambda entry: entry.local_id))

    def toggle(self, entry: PromptSelectionEntry) -> PromptSelectionBasket:
        """Toggle one exact entry, capturing it only when newly selected.

        Args:
            entry: Valid selection entry for the activated page row.

        Returns:
            New selection state with one semantic generation change.

        Raises:
            TypeError: If ``entry`` is not an exact selection entry.
        """
        if type(entry) is not PromptSelectionEntry:
            raise TypeError("entry must be an exact PromptSelectionEntry.")
        for index, selected in enumerate(self.entries):
            if selected.local_id == entry.local_id:
                return PromptSelectionBasket(
                    self.entries[:index] + self.entries[index + 1 :],
                    self.generation + 1,
                )
        return PromptSelectionBasket(self.entries + (entry,), self.generation + 1)

    def select_page(
        self, page: tuple[PromptSelectionEntry, ...]
    ) -> PromptSelectionBasket:
        """Add valid page entries without replacing prior captured values.

        Args:
            page: Exact tuple of selection entries from one settled page.

        Returns:
            New state when at least one identity is added, otherwise this state.

        Raises:
            TypeError: If the page tuple or any entry has the wrong exact type.
        """
        if type(page) is not tuple:
            raise TypeError("page must be an exact tuple.")
        if any(type(entry) is not PromptSelectionEntry for entry in page):
            raise TypeError("page must contain exact PromptSelectionEntry values.")
        selected_ids = {entry.local_id for entry in self.entries}
        additions: list[PromptSelectionEntry] = []
        for entry in page:
            if entry.local_id not in selected_ids:
                additions.append(entry)
                selected_ids.add(entry.local_id)
        if not additions:
            return self
        return PromptSelectionBasket(
            self.entries + tuple(additions), self.generation + 1
        )

    def clear(self) -> PromptSelectionBasket:
        """Clear all entries, advancing only when selection actually changes."""
        if not self.entries:
            return self
        return PromptSelectionBasket(generation=self.generation + 1)


@dataclass(frozen=True)
class PromptListRow:
    """One row in the Library prompts canvas's list view.

    Attributes:
        prompt_id: The prompt's id.
        name: Display name, raw (the canvas escapes markup at render time).
        secondary: ``"<details> · <age>"`` -- the prompt's purpose, not
            ``author``/``keywords`` (Task 8b D2/U1; see ``_matches_query``'s
            comment for why keywords are never shown here) -- with either
            part (no details, or no timestamp) omitted, along with its
            separator.
        version: Positive Prompt version exposed by the page row.
        checked: Whether this Prompt identity is in the current selection.
    """

    prompt_id: int
    name: str
    secondary: str
    artifact_type: ArtifactType = "prompt"
    type_label: str = "Prompt"
    lane_summary: str = "Empty"
    source_label: str = "Local"
    version: int = 1
    checked: bool = False

    def __post_init__(self) -> None:
        _require_prompt_selection_integer(self.version, field_name="version")
        if type(self.checked) is not bool:
            raise TypeError("checked must be a bool.")


@dataclass(frozen=True)
class LibraryPromptDeleteReceipt:
    """Exact Prompt/Recipe tombstone available to restore from the list."""

    prompt_id: int
    title: str
    artifact_type: ArtifactType
    expected_version: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt_id, int)
            or isinstance(self.prompt_id, bool)
            or self.prompt_id < 1
        ):
            raise ValueError("Prompt delete receipt id must be positive.")
        if not isinstance(self.title, str):
            raise TypeError("Prompt delete receipt title must be text.")
        if self.artifact_type not in {"prompt", "recipe"}:
            raise ValueError("Prompt delete receipt type must be prompt or recipe.")
        if (
            not isinstance(self.expected_version, int)
            or isinstance(self.expected_version, bool)
            or self.expected_version < 1
        ):
            raise ValueError("Prompt delete receipt version must be positive.")


@dataclass(frozen=True)
class PromptsListState:
    """Display state for the Library prompts canvas's list view.

    Attributes:
        rows: The prompts to render, already filtered/sorted.
        count: ``len(rows)``.
        sort: The sort mode used to build ``rows`` (``"newest"`` or
            ``"name"``), echoed back for the caller's toggle label.
        select_mode: Whether selection controls and checked rows are active.
        total_selected: Number selected across every page and search.
        selected_on_page: Number selected on the current projected page.
    """

    rows: tuple[PromptListRow, ...]
    count: int
    sort: str
    select_mode: bool = False
    total_selected: int = 0
    selected_on_page: int = 0

    def __post_init__(self) -> None:
        if type(self.select_mode) is not bool:
            raise TypeError("select_mode must be a bool.")
        for field_name, value in (
            ("total_selected", self.total_selected),
            ("selected_on_page", self.selected_on_page),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if self.selected_on_page > self.total_selected:
            raise ValueError("selected_on_page cannot exceed total_selected.")


@dataclass(frozen=True)
class PromptEditorState:
    """Display state for the Library prompts canvas's in-canvas editor.

    Attributes:
        prompt_id: The open prompt's id, or ``None`` when unknown/not yet
            saved.
        name: The prompt's name.
        author: The prompt's author.
        details: The prompt's description/details text.
        system_prompt: The prompt's system-prompt text.
        user_prompt: The prompt's user-prompt text.
        keywords_csv: The prompt's keywords as a single comma-separated
            string.
        version: The prompt's optimistic-lock version, or ``None`` when
            unknown.
        created: Raw ``created_at`` timestamp text, or ``""`` when absent.
        modified: Raw ``last_modified``/``created_at`` timestamp text, or
            ``""`` when absent.
    """

    prompt_id: int | None
    name: str
    author: str
    details: str
    system_prompt: str
    user_prompt: str
    keywords_csv: str
    version: int | None
    created: str
    modified: str
    artifact_type: ArtifactType = "prompt"
    definition_state: ArtifactDefinitionState = "legacy"
    block_editor_state: PromptBlockEditorState | None = None
    compiled_system_preview: str = ""
    compiled_user_preview: str = ""
    compatibility_stale: bool = False
    compatibility_reason: str = ""
    can_convert_as_new: bool = False
    source: str = "local"
    source_identity: str | None = None
    capabilities: PromptSourceCapabilities | None = None


PromptEditorMode = Literal["basic", "advanced"]


def coerce_prompt_editor_mode(value: object) -> PromptEditorMode:
    """Return the stored Prompt editor mode, defaulting invalid values to Basic."""
    return "advanced" if value == "advanced" else "basic"


def prompt_basic_unavailable_reason(
    state: PromptEditorState,
    *,
    conflict: bool = False,
    can_update_original: bool = True,
) -> str:
    """Explain why a Prompt must use the full structured editor."""
    if conflict:
        return "Resolve the version conflict in Advanced view."
    if state.artifact_type != "prompt":
        return "Recipes require Advanced view."
    if (
        state.definition_state not in {"legacy", "supported_v2"}
        or state.block_editor_state is None
        or state.compatibility_stale
    ):
        return "This prompt requires compatibility or conversion controls."
    if state.prompt_id is not None and not can_update_original:
        return "This saved prompt cannot be safely updated from Basic view."
    if any(len(lane.blocks) > 1 for lane in state.block_editor_state.definition.lanes):
        return "This prompt uses multiple structured blocks."
    return ""


#: task-2859 item 2: plain-language labels for ``ArtifactDefinitionState``
#: values shown in the prompt editor's artifact-status line. Every NEW
#: prompt starts life as ``"legacy"`` -- the internal name for the flat
#: system/user-text storage format, as opposed to the structured v2 block
#: format -- so a brand-new prompt was stamped "legacy" verbatim, reading as
#: "this is an old/deprecated prompt" rather than "this is the plain-text
#: format". Any state not listed here (there should be none) falls back to
#: the raw ``definition_state.replace('_', ' ')`` this dict replaces.
_DEFINITION_STATE_DISPLAY_LABELS: dict[str, str] = {
    "legacy": "text format",
    "supported_v2": "structured format",
    "foreign_v1": "external format",
    "unsupported": "unsupported format",
    "malformed": "malformed",
    "mismatched": "mismatched format",
}


def definition_state_display_label(definition_state: str) -> str:
    """Return the plain-language label for an ``ArtifactDefinitionState``.

    Args:
        definition_state: The raw internal state value (e.g. ``"legacy"``).

    Returns:
        The matching plain-language label, or ``definition_state`` with
        underscores replaced by spaces when the value is unrecognized.
    """
    return _DEFINITION_STATE_DISPLAY_LABELS.get(
        definition_state, definition_state.replace("_", " ")
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_text(value: Any) -> str:
    """Like ``_text`` but preserves body text verbatim (no stripping)."""
    return "" if value is None else str(value)


def _timestamp_raw(record: Mapping[str, Any]) -> str:
    for key in _TIMESTAMP_KEYS:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _csv_from_keywords(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        items = []
        for item in value:
            if isinstance(item, Mapping):
                item = item.get("keyword") or item.get("text") or item.get("label")
            text = _text(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return ""


def _matches_query(record: Mapping[str, Any], query_lower: str) -> bool:
    if not query_lower:
        return True
    if query_lower in _text(record.get("name")).lower():
        return True
    # Task 8b D2: matches `details`, NOT `keywords` -- the raw local
    # `list_prompts` DB query has no per-page-keyword-join seam (only a
    # per-single-id `fetch_keywords_for_prompt`, an N+1 shape for a whole
    # page), so real list-page records never carry `keywords` at all (see
    # `_prompts_page_records_or_empty`'s docstring). Matching on it here
    # would silently promise a capability the filter could never actually
    # deliver. `details` IS present on list rows (the DB query now selects
    # it too), so this is the honest, still-cheap (no extra query) fix.
    # Keyword-in-list filtering awaits a batched per-page keyword-join DB
    # seam (backlog) if that capability is ever wanted.
    return query_lower in _text(record.get("details")).lower()


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_editor_prompt_id(detail: Mapping[str, Any]) -> int | None:
    """Resolve a prompt detail mapping's raw numeric id, robustly.

    The REAL production seam (``PromptScopeService.get_prompt`` ->
    ``normalize_prompt_record``, see
    ``tldw_chatbook/Prompt_Management/prompt_normalizers.py``) returns
    ``detail["id"]`` as a COMPOSITE STRING (``"<backend>:prompt:<uuid>"``)
    -- the raw local numeric id lives under ``detail["local_id"]`` instead.
    Preferring ``local_id`` here (when it resolves to an int) fixes that
    seam. Falling back to ``id`` keeps the other, non-composite-shaped
    callers working: the raw ``PromptsDatabase.fetch_prompt_details``/
    ``list_prompts`` row shape (``id`` IS the raw int, no ``local_id`` key
    at all) used directly by a handful of call sites/tests, and the
    post-save ``patched_detail`` the screen builds itself (which always
    writes a raw int straight into ``id``). ``_to_int`` on a composite
    string (or ``None``) naturally returns ``None``, so this never
    resolves a truly blank/new-editor detail (neither key present, e.g.
    the D1 create/Duplicate-action shapes) to anything but ``None``.

    Args:
        detail: A prompt detail mapping.

    Returns:
        The resolved int id, or ``None`` when unresolvable.
    """
    local_id = _to_int(detail.get("local_id"))
    if local_id is not None:
        return local_id
    return _to_int(detail.get("id"))


def _row(
    record: Mapping[str, Any],
    *,
    now: datetime,
    checked_ids: frozenset[int] = frozenset(),
    strict_selection: bool = False,
) -> PromptListRow | None:
    if strict_selection:
        raw_prompt_id = (
            record.get("local_id") if "local_id" in record else record.get("id")
        )
        version = record.get("version")
        name = record.get("name")
        raw_artifact_type = record.get("artifact_type", "prompt")
        if (
            type(raw_prompt_id) is not int
            or not 1 <= raw_prompt_id <= _PROMPT_SELECTION_SQLITE_MAX_INTEGER
            or type(version) is not int
            or not 1 <= version <= _PROMPT_SELECTION_SQLITE_MAX_INTEGER
            or type(name) is not str
            or not name.strip()
            or type(raw_artifact_type) is not str
            or raw_artifact_type not in _PROMPT_SELECTION_ARTIFACT_TYPES
        ):
            return None
        prompt_id = raw_prompt_id
        row_version = version
        row_name = name.strip()
        artifact_type = cast(ArtifactType, raw_artifact_type)
    else:
        legacy_prompt_id = _to_int(record.get("local_id")) or _to_int(record.get("id"))
        if legacy_prompt_id is None:
            return None
        prompt_id = legacy_prompt_id
        legacy_version = _to_int(record.get("version"))
        row_version = (
            legacy_version
            if legacy_version is not None
            and 1 <= legacy_version <= _PROMPT_SELECTION_SQLITE_MAX_INTEGER
            else 1
        )
        row_name = _text(record.get("name"))
        artifact_type = cast(
            ArtifactType,
            "recipe" if record.get("artifact_type") == "recipe" else "prompt",
        )
    # Task 8b D2/U1: surfaces the prompt's PURPOSE (details) instead of
    # `author · age` -- author (and keywords, never present on list rows
    # anyway -- see `_matches_query`'s comment) are dropped from the
    # secondary line entirely.
    details = _text(record.get("details"))
    raw_timestamp = _timestamp_raw(record)
    age = format_console_relative_age(raw_timestamp, now=now) if raw_timestamp else ""
    secondary = " · ".join(part for part in (details, age) if part)
    has_system = record.get("has_system_prompt")
    if not isinstance(has_system, bool):
        has_system = bool(_raw_text(record.get("system_prompt")).strip())
    has_user = record.get("has_user_prompt")
    if not isinstance(has_user, bool):
        has_user = bool(_raw_text(record.get("user_prompt")).strip())
    if has_system and has_user:
        lane_summary = "System + User"
    elif has_system:
        lane_summary = "System only"
    elif has_user:
        lane_summary = "User only"
    else:
        lane_summary = "Empty"
    source = _text(record.get("backend")) or "local"
    return PromptListRow(
        prompt_id=prompt_id,
        name=row_name,
        secondary=secondary,
        artifact_type=artifact_type,
        type_label=artifact_type.title(),
        lane_summary=lane_summary,
        source_label=source.title(),
        version=row_version,
        checked=prompt_id in checked_ids,
    )


def build_prompts_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    query: str,
    sort: str,
    now: datetime,
) -> PromptsListState:
    """Build the Library prompts canvas's list-view display state.

    Records missing a mapping shape or a convertible ``id`` are silently
    dropped rather than raising, matching the Library notes state module's
    degrade-don't-crash behavior for malformed source records.

    Args:
        records: The prompts to render.
        query: Filter text, matched case-insensitively against name and
            details (Task 8b D2 -- not keywords, a field list-page records
            never actually carry; see ``_matches_query``); ``""`` disables
            filtering.
        sort: ``"name"`` sorts alphabetically case-insensitively; any other
            value (including ``"newest"``) sorts by most-recent
            modified/created timestamp, newest first.
        now: Reference time for the secondary line's relative-age part.

    Returns:
        The list view's display state.
    """
    query_lower = _text(query).lower()
    items = [
        record
        for record in (records or ())
        if isinstance(record, Mapping) and _matches_query(record, query_lower)
    ]
    if sort == "name":
        items.sort(key=lambda record: _text(record.get("name")).lower())
    else:
        items.sort(key=_timestamp_raw, reverse=True)
    rows = tuple(
        row for row in (_row(record, now=now) for record in items) if row is not None
    )
    return PromptsListState(rows=rows, count=len(rows), sort=sort)


def build_prompt_browse_list_state(
    result: PromptBrowseResult,
    *,
    now: datetime,
    retained_items: tuple[Mapping[str, Any], ...] | None = None,
    selection: PromptSelectionBasket | None = None,
    select_mode: bool = False,
) -> PromptsListState:
    """Project an exact service-backed page without filtering or sorting it.

    Args:
        result: Immutable browse result whose page order is authoritative.
        now: Reference time used for row-relative timestamps.
        retained_items: Validated last-good records to project when they differ
            from the exact result after a committed mutation.
        selection: Immutable cross-page selection to project into page rows.
        select_mode: Whether the list should expose selection-mode controls.

    Returns:
        Display rows in service order with exact versions and selection counts.

    Raises:
        TypeError: If ``selection`` or ``select_mode`` has the wrong type.
        ValueError: If a validated source item cannot form a selection row.
    """
    if selection is None:
        selected_entries: tuple[PromptSelectionEntry, ...] = ()
    elif type(selection) is PromptSelectionBasket:
        selected_entries = selection.entries
    else:
        raise TypeError("selection must be a PromptSelectionBasket or None.")
    if type(select_mode) is not bool:
        raise TypeError("select_mode must be a bool.")
    if retained_items is not None and type(retained_items) is not tuple:
        raise TypeError("retained_items must be an exact tuple or None.")
    records = (
        result.items
        if retained_items is None
        else validate_prompt_browse_items(retained_items)
    )
    selected_ids = frozenset(entry.local_id for entry in selected_entries)
    rows_list: list[PromptListRow] = []
    for record in records:
        row = _row(
            record,
            now=now,
            checked_ids=selected_ids,
            strict_selection=True,
        )
        if row is None:
            raise ValueError("Validated Prompt browse item could not be projected.")
        rows_list.append(row)
    rows = tuple(rows_list)
    return PromptsListState(
        rows=rows,
        count=len(rows),
        sort="name" if result.scope.sort_by == "name" else "newest",
        select_mode=select_mode,
        total_selected=len(selected_entries),
        selected_on_page=sum(row.checked for row in rows),
    )


def build_prompt_editor_state(
    detail: Mapping[str, Any],
    *,
    capabilities: PromptSourceCapabilities | None = None,
) -> PromptEditorState:
    """Build the prompt editor's display state from a prompt detail mapping.

    Args:
        detail: A prompt detail mapping -- either the raw
            ``fetch_prompt_details`` row shape (``id`` IS the raw int,
            ``keywords`` a list of strings), or the normalized
            ``PromptScopeService.get_prompt``/``normalize_prompt_record``
            shape (``id`` a composite ``"<backend>:prompt:<uuid>"``
            string, the raw int under ``local_id`` instead -- see
            ``_resolve_editor_prompt_id``), or a malformed/empty mapping.
            Tolerated to have missing/None fields.

    Returns:
        Immutable editor state, with keywords joined into a single
        comma-separated string.
    """
    if not isinstance(detail, Mapping):
        detail = {}
    try:
        decoded = decode_prompt_artifact(detail)
    except (TypeError, ValueError):
        decoded = None

    artifact_type: ArtifactType = (
        decoded.artifact_type
        if decoded is not None
        else cast(
            ArtifactType,
            "recipe" if detail.get("artifact_type") == "recipe" else "prompt",
        )
    )
    definition_state: ArtifactDefinitionState = (
        decoded.state if decoded is not None else "malformed"
    )
    compiled_system = (
        decoded.compiled_system
        if decoded is not None
        else _raw_text(detail.get("system_prompt"))
    )
    compiled_user = (
        decoded.compiled_user
        if decoded is not None
        else _raw_text(detail.get("user_prompt"))
    )
    block_state: PromptBlockEditorState | None = None
    if decoded is not None and decoded.state == "supported_v2":
        try:
            block_state = PromptBlockEditorState.from_definition(
                artifact_type=decoded.artifact_type,
                definition=decoded.definition,
            )
        except (TypeError, ValueError):
            definition_state = "malformed"
    elif (
        decoded is not None and decoded.state == "legacy" and artifact_type == "prompt"
    ):
        decomposition = decompose_legacy_lanes(compiled_system, compiled_user)
        block_state = PromptBlockEditorState.from_definition(
            artifact_type="prompt",
            definition=decomposition.definition,
            system_origin=decomposition.system_origin,
            user_origin=decomposition.user_origin,
        )

    compatibility_reason = ""
    if block_state is None:
        compatibility_reason = (
            f"{definition_state.replace('_', ' ')} artifact is read-only; "
            "use compatibility text and convert only as a new Prompt."
        )
    source = _text(detail.get("backend")) or "local"
    source_identity_value = detail.get("id", detail.get("uuid"))
    source_identity = (
        str(source_identity_value) if source_identity_value not in (None, "") else None
    )
    return PromptEditorState(
        prompt_id=_resolve_editor_prompt_id(detail),
        name=_text(detail.get("name")),
        author=_text(detail.get("author")),
        details=_raw_text(detail.get("details")),
        system_prompt=_raw_text(detail.get("system_prompt")),
        user_prompt=_raw_text(detail.get("user_prompt")),
        keywords_csv=_csv_from_keywords(detail.get("keywords")),
        version=_to_int(detail.get("version")),
        created=_text(detail.get("created_at")),
        modified=_timestamp_raw(detail),
        artifact_type=artifact_type,
        definition_state=definition_state,
        block_editor_state=block_state,
        compiled_system_preview=compiled_system,
        compiled_user_preview=compiled_user,
        compatibility_stale=bool(
            decoded.compatibility_stale if decoded is not None else False
        ),
        compatibility_reason=compatibility_reason,
        can_convert_as_new=bool(
            block_state is None and (compiled_system or compiled_user)
        ),
        source=source,
        source_identity=source_identity,
        capabilities=capabilities,
    )


def prompt_editor_meta_line(
    editor_state: PromptEditorState, *, now: datetime | None = None, dirty: bool = False
) -> str:
    """Render the prompt editor's muted meta line.

    Unlike the notes editor's ``meta_line`` (precomputed as part of
    ``LibraryNoteEditorState`` by ``build_library_note_editor_state``),
    ``PromptEditorState`` carries only raw ``modified``/``version`` fields
    -- shared here (rather than duplicated) so both the editor canvas's
    initial render and the screen's post-save targeted Static update agree
    on the exact same text. The Prompts table has no ``created_at`` column
    at all, so this renders only ``Modified <age>`` (never a "Created"
    part, and never a fake one) plus ``vN``.

    Args:
        editor_state: The prompt editor's current display state.
        now: Reference time for the relative-age part; defaults to the
            current UTC time.
        dirty: Task 8c U6: whether the editor has unsaved in-progress
            edits. A plain pure-function input (never derived from
            ``editor_state`` itself, which only ever reflects the
            last-saved record) -- callers thread the screen's own
            ``_library_prompt_dirty`` flag through. Defaults to ``False``
            so every pre-existing call site is unaffected.

    Returns:
        ``"New prompt"`` when ``editor_state.prompt_id`` is ``None`` (the
        Task 8b D1 create-flow sentinel: a blank, not-yet-saved record --
        see ``library_screen.py``'s ``_enter_library_prompt_create_editor``
        and the Duplicate action). Otherwise ``"Modified <age> · vN"``,
        with either part omitted (and its separator) when unknown. Either
        form gets a trailing ``"· • Unsaved changes"`` appended when
        ``dirty`` is ``True`` -- the only visible cue today that explicit
        Save/the nav-away dirty veto (``flush_pending_work``) has anything
        to act on.
    """
    if editor_state.prompt_id is None:
        base = "New prompt"
    else:
        reference_now = now if now is not None else datetime.now(timezone.utc)
        parts: list[str] = []
        if editor_state.modified:
            age = format_console_relative_age(editor_state.modified, now=reference_now)
            parts.append(f"Modified {age}")
        if editor_state.version is not None:
            parts.append(f"v{editor_state.version}")
        base = " · ".join(parts)
    if not dirty:
        return base
    return f"{base} · • Unsaved changes" if base else "• Unsaved changes"


def _is_name_conflict(exc: Exception | None, message_lower: str) -> bool:
    if isinstance(exc, sqlite3.IntegrityError) and "unique" in str(exc).lower():
        return True
    return "unique" in message_lower or "already exists" in message_lower


def classify_prompt_save_error(
    result_id: Any, message: str, exc: Exception | None
) -> str:
    """Classify the outcome of a prompt save (add/update) call.

    Args:
        result_id: The id the save call returned, or ``None`` when it did
            not produce a fresh saved row.
        message: Any accompanying human-readable message from the save
            call (e.g. the ``add_prompt`` tuple's message slot).
        exc: The exception raised by the save call, if any.

    Returns:
        One of ``"soft-deleted-name"``, ``"conflict"``, ``"name-in-use"``,
        ``"ok"``, or ``"error"``.
    """
    message_lower = _text(message).lower()
    if result_id is None and "soft-deleted" in message_lower:
        return "soft-deleted-name"
    if isinstance(exc, ConflictError):
        return "conflict"
    if _is_name_conflict(exc, message_lower):
        return "name-in-use"
    if exc is None and result_id is not None:
        return "ok"
    return "error"
