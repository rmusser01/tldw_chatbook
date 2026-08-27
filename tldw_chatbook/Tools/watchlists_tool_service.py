"""Shared synchronous service for read-only Watchlists agent tools."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import logging
import math
import re
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tldw_chatbook.DB.Subscriptions_DB import (
    SubscriptionsDBReadError,
    SubscriptionsDBUnavailableError,
)
from tldw_chatbook.Subscriptions.html_text import (
    readable_body_text,
    strip_control_characters,
)

_SEARCH_KEYS = frozenset(
    {"query", "collection", "source", "statuses", "since", "limit", "cursor"}
)
_STATUSES = frozenset({"new", "reviewed", "ingested", "ignored", "error"})
_ORDERING = "effective_date_desc_item_id_asc"
_SERVER_UNSUPPORTED_MESSAGE = (
    "server Watchlists search is not supported; switch Watchlists to Local "
    "before retrying"
)
_PERMANENT_UNAVAILABLE_MESSAGE = (
    "local Watchlists data is unavailable; open Watchlists in Local mode "
    "to initialize or migrate it, then retry"
)
_TRANSIENT_UNAVAILABLE_MESSAGE = (
    "local Watchlists data is temporarily unavailable; retry later"
)
_RFC3339_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z",
    re.IGNORECASE,
)
_CANONICAL_SCOPE_RE = re.compile(
    r"local:(?P<kind>subscription|watchlist):(?P<id>[1-9][0-9]*)\Z"
)
_CANONICAL_ITEM_RE = re.compile(r"local:watchlist_item:(?P<id>[1-9][0-9]*)\Z")
_CANONICAL_BRIEFING_RE = re.compile(r"local:briefing:(?P<id>[1-9][0-9]*)\Z")
_CANONICAL_OPERATION_RE = re.compile(
    r"local:(?P<kind>watchlist_run|briefing):(?P<id>[1-9][0-9]*)\Z"
)
_COMPOSITE_ID_RE = re.compile(r"[^:\s]+:(?:subscription|watchlist|watchlist_item):.*\Z")
_CANDIDATE_LIMIT = 10
_MAX_SQLITE_ROW_ID = 2**63 - 1
_MAX_RESULT_BYTES = 30 * 1024
_TRUNCATION_SUFFIX = "…[truncated]"
_MAX_TITLE_BYTES = 1_024
_MAX_AUTHOR_BYTES = 512
_MAX_NAME_BYTES = 512
_MAX_URL_BYTES = 1_024
_MAX_SNIPPET_BYTES = 4_096
_MAX_CHANGE_SUMMARY_BYTES = 8_192
_BRIEFING_BODY_BUDGET = 12 * 1024
_PROVENANCE_ARRAY_BUDGET = 6 * 1024
_PROVENANCE_LIMIT = 50
_PUBLIC_EXECUTION_ERROR = "Watchlists tool execution error"
_CURSOR_VERSION = 1
_CURSOR_KEYS = frozenset(
    {
        "version",
        "as_of",
        "snapshot_max_item_id",
        "last_effective_date",
        "last_effective_date_is_null",
        "last_item_id",
        "filter_fingerprint",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SQLITE_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\Z")
_LOGGER = logging.getLogger(__name__)


class _InvalidArgument(ValueError):
    """Internal sentinel carrying one fixed public validation message."""


@dataclass(frozen=True, slots=True)
class _SearchRequest:
    query: str | None
    collection: int | str | None
    source: int | str | None
    statuses: tuple[str, ...] | None
    since: str | None
    limit: int
    cursor: _Cursor | None


@dataclass(frozen=True, slots=True)
class _Cursor:
    as_of: str
    snapshot_max_item_id: int
    last_effective_date: str | None
    last_item_id: int
    filter_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PageCursor:
    """Opaque keyset continuation for metadata pages."""

    kind: str
    position: dict[str, Any]
    filter_fingerprint: str


class WatchlistsToolService:
    """Validate and orchestrate the shared read-only Watchlists tool core.

    Args:
        db_resolver: Synchronous callable returning the current database owner.
        runtime_source_loader: Synchronous callable returning current runtime state.
        clock: Optional UTC-aware clock used for traversal context timestamps.
        operational_state_loader: Optional scheduler/gate snapshot loader.
    """

    def __init__(
        self,
        *,
        db_resolver: Callable[[], Any],
        runtime_source_loader: Callable[[], Any],
        clock: Callable[[], datetime] | None = None,
        operational_state_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._db_resolver = db_resolver
        self._runtime_source_loader = runtime_source_loader
        self._clock = clock or (lambda: datetime.now(UTC))
        self._operational_state_loader = (
            operational_state_loader or self._default_operational_state
        )

    def search_items(self, arguments: object) -> str:
        """Search or browse Watchlists items and return one JSON object.

        Args:
            arguments: Raw tool argument object.

        Returns:
            JSON text containing a structured domain outcome.
        """
        try:
            return self._search_items(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _search_items(self, arguments: object) -> str:
        """Execute a validated search while preserving expected outcomes."""
        request = self._validate_search(arguments)

        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )

        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable

        collection, outcome = self._resolve_scope(
            database, "collection", request.collection
        )
        if outcome is not None:
            return outcome
        source, outcome = self._resolve_scope(database, "source", request.source)
        if outcome is not None:
            return outcome

        collection_id = int(collection["id"]) if collection is not None else None
        source_id = int(source["id"]) if source is not None else None
        fingerprint = self._filter_fingerprint(
            query=request.query,
            collection_id=collection_id,
            source_id=source_id,
            statuses=request.statuses,
            since=request.since,
        )
        if (
            request.cursor is not None
            and request.cursor.filter_fingerprint != fingerprint
        ):
            raise _InvalidArgument("cursor does not match the search filters")

        as_of = (
            request.cursor.as_of
            if request.cursor is not None
            else self._format_utc(self._clock())
        )
        page = database.search_items_for_agent(
            query=request.query,
            subscription_id=source_id,
            watchlist_id=collection_id,
            statuses=request.statuses,
            since=request.since,
            limit=request.limit,
            snapshot_max_item_id=(
                request.cursor.snapshot_max_item_id
                if request.cursor is not None
                else None
            ),
            after_effective_date=(
                request.cursor.last_effective_date
                if request.cursor is not None
                else None
            ),
            after_item_id=(
                request.cursor.last_item_id if request.cursor is not None else None
            ),
        )
        rows = page["items"]
        memberships = database.get_source_collection_memberships(
            [int(row["subscription_id"]) for row in rows]
        )
        scope = {
            "collection": self._shape_collection(collection),
            "source": self._shape_selected_source(source),
        }
        items: list[dict[str, Any]] = []
        final_payload = self._search_response(
            request=request,
            as_of=as_of,
            snapshot_max_item_id=int(page["snapshot_max_item_id"]),
            scope=scope,
            items=items,
            has_more=False,
            next_cursor=None,
        )
        for index, row in enumerate(rows):
            candidate = self._shape_search_item(
                row,
                memberships[int(row["subscription_id"])],
                request.query,
            )
            candidate_items = [*items, candidate]
            has_more = bool(page["has_more"]) or index < len(rows) - 1
            next_cursor = (
                self._encode_cursor(
                    as_of=as_of,
                    snapshot_max_item_id=int(page["snapshot_max_item_id"]),
                    last_effective_date=row["effective_date"],
                    last_item_id=int(row["id"]),
                    filter_fingerprint=fingerprint,
                )
                if has_more
                else None
            )
            candidate_payload = self._search_response(
                request=request,
                as_of=as_of,
                snapshot_max_item_id=int(page["snapshot_max_item_id"]),
                scope=scope,
                items=candidate_items,
                has_more=has_more,
                next_cursor=next_cursor,
            )
            if self._json_size(candidate_payload) >= _MAX_RESULT_BYTES:
                break
            items = candidate_items
            final_payload = candidate_payload

        if rows and not items:
            raise RuntimeError("bounded Watchlists item did not fit")
        return self._finalize(final_payload)

    def get_item(self, arguments: object) -> str:
        """Return authoritative detail for one canonical Watchlists item ID.

        Args:
            arguments: Raw tool argument object containing ``item_id``.

        Returns:
            JSON text containing a structured domain outcome.
        """
        try:
            return self._get_item(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _get_item(self, arguments: object) -> str:
        """Execute one validated detail read with bounded evidence."""
        item_id = self._validate_detail(arguments)

        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )

        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable

        row = database.get_item_detail_for_agent(item_id)
        if row is None:
            return self._outcome(
                "not_found", "Watchlists item was not found", retryable=False
            )
        membership = database.get_source_collection_memberships(
            [int(row["subscription_id"])]
        )[int(row["subscription_id"])]
        item = self._shape_item_metadata(row, membership)
        if row["content"] is not None:
            normalized_content = readable_body_text(row["content"])
            if normalized_content.strip():
                item["evidence"] = {
                    "content_is_untrusted": True,
                    "content": normalized_content,
                    "content_normalized": True,
                    "content_truncated": False,
                }
                return self._fit_detail_content(item)
        if row["diff_summary"] is not None:
            summary, summary_truncated = self._bounded_text(
                row["diff_summary"], _MAX_CHANGE_SUMMARY_BYTES
            )
            percentage, percentage_invalid = self._finite_number(
                row["change_percentage"]
            )
            item["evidence"] = {
                "content_is_untrusted": True,
                "change_summary": summary,
                "change_summary_truncated": summary_truncated,
                "change_type": row["change_type"],
                "change_percentage": percentage,
                "change_percentage_invalid": percentage_invalid,
            }
        else:
            item["evidence"] = {
                "content_is_untrusted": True,
                "content": None,
                "content_normalized": True,
                "content_truncated": False,
            }
        return self._finalize({"status": "ok", "item": item})

    def list_sources(self, arguments: object) -> str:
        """List bounded local Watchlists source metadata."""
        try:
            return self._list_sources(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _list_sources(self, arguments: object) -> str:
        values = self._exact_arguments(
            arguments,
            frozenset({"name", "type", "state", "collection", "limit", "cursor"}),
        )
        name = self._optional_string(values, "name", maximum=512)
        source_type = self._optional_string(values, "type", maximum=32)
        state = self._optional_enum(
            values, "state", frozenset({"active", "paused", "disabled", "all"})
        )
        collection_value = self._validate_scope_value(
            "collection",
            values.get("collection"),
            maximum=256,
            supplied="collection" in values,
        )
        limit = self._validate_limit(values.get("limit", 10))
        cursor = self._validate_page_cursor(
            values.get("cursor"), supplied="cursor" in values, kind="sources"
        )
        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        collection, outcome = self._resolve_scope(
            database, "collection", collection_value
        )
        if outcome is not None:
            return outcome
        collection_id = int(collection["id"]) if collection is not None else None
        filters = {
            "name": name,
            "type": source_type,
            "state": state,
            "collection_id": collection_id,
            "ordering": "casefolded_name_asc_name_asc_id_asc",
        }
        fingerprint = self._metadata_fingerprint(filters)
        self._require_matching_page_cursor(cursor, fingerprint)
        position = cursor.position if cursor is not None else {}
        page = database.list_sources_for_agent(
            name_query=name,
            source_type=source_type,
            is_active=False if state == "disabled" else (True if state == "active" else None),
            is_paused=True if state == "paused" else (False if state == "active" else None),
            watchlist_id=collection_id,
            limit=limit,
            after_name_casefold=position.get("name_casefold"),
            after_name=position.get("name"),
            after_id=position.get("id"),
        )
        rows = page["items"]
        memberships = database.get_source_collection_memberships(
            [int(row["id"]) for row in rows]
        )
        sources = [
            self._shape_source_metadata(row, memberships[int(row["id"])])
            for row in rows
        ]
        return self._finalize_metadata_page(
            base={
                "status": "ok",
                "ordering": "casefolded_name_asc_name_asc_id_asc",
            },
            item_key="sources",
            rows=rows,
            shaped=sources,
            storage_has_more=bool(page["has_more"]),
            cursor_for=lambda row: self._encode_page_cursor(
                "sources",
                {
                    "name_casefold": row["name_casefold"],
                    "name": row["name"],
                    "id": int(row["id"]),
                },
                fingerprint,
            ),
        )

    def list_collections(self, arguments: object) -> str:
        """List bounded local Watchlists collection metadata."""
        try:
            return self._list_collections(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _list_collections(self, arguments: object) -> str:
        values = self._exact_arguments(
            arguments, frozenset({"name", "limit", "cursor"})
        )
        name = self._optional_string(values, "name", maximum=512)
        limit = self._validate_limit(values.get("limit", 10))
        cursor = self._validate_page_cursor(
            values.get("cursor"), supplied="cursor" in values, kind="collections"
        )
        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        fingerprint = self._metadata_fingerprint(
            {
                "name": name,
                "ordering": "casefolded_name_asc_name_asc_id_asc",
            }
        )
        self._require_matching_page_cursor(cursor, fingerprint)
        position = cursor.position if cursor is not None else {}
        page = database.list_collections_for_agent(
            name_query=name,
            limit=limit,
            after_name_casefold=position.get("name_casefold"),
            after_name=position.get("name"),
            after_id=position.get("id"),
        )
        rows = page["items"]
        operational_state = self._operational_state()
        collections = [
            self._shape_collection_metadata(row, operational_state, self._clock())
            for row in rows
        ]
        return self._finalize_metadata_page(
            base={
                "status": "ok",
                "ordering": "casefolded_name_asc_name_asc_id_asc",
            },
            item_key="collections",
            rows=rows,
            shaped=collections,
            storage_has_more=bool(page["has_more"]),
            cursor_for=lambda row: self._encode_page_cursor(
                "collections",
                {
                    "name_casefold": row["name_casefold"],
                    "name": row["name"],
                    "id": int(row["id"]),
                },
                fingerprint,
            ),
        )

    def list_briefings(self, arguments: object) -> str:
        """List bounded briefing receipts without briefing content."""
        try:
            return self._list_briefings(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _list_briefings(self, arguments: object) -> str:
        values = self._exact_arguments(
            arguments,
            frozenset({"collection", "statuses", "since", "limit", "cursor"}),
        )
        collection_value = self._validate_scope_value(
            "collection",
            values.get("collection"),
            maximum=256,
            supplied="collection" in values,
        )
        statuses = self._validate_briefing_statuses(
            values.get("statuses"), supplied="statuses" in values
        )
        since = self._validate_since(values.get("since"), supplied="since" in values)
        limit = self._validate_limit(values.get("limit", 10))
        cursor = self._validate_page_cursor(
            values.get("cursor"), supplied="cursor" in values, kind="briefings"
        )
        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        collection, outcome = self._resolve_scope(
            database, "collection", collection_value
        )
        if outcome is not None:
            return outcome
        collection_id = int(collection["id"]) if collection is not None else None
        fingerprint = self._metadata_fingerprint(
            {
                "collection_id": collection_id,
                "statuses": sorted(statuses or ()),
                "since": since,
                "ordering": "created_at_desc_id_desc",
            }
        )
        self._require_matching_page_cursor(cursor, fingerprint)
        position = cursor.position if cursor is not None else {}
        page = database.list_briefings_for_agent(
            watchlist_id=collection_id,
            statuses=statuses,
            since=since,
            limit=limit,
            after_created_at=position.get("created_at"),
            after_id=position.get("id"),
        )
        rows = page["items"]
        receipts = [self._shape_briefing_receipt(row) for row in rows]
        payload: dict[str, Any] = {
            "status": "ok",
            "ordering": "created_at_desc_id_desc",
        }
        if collection_id is not None and cursor is None:
            latest = database.get_latest_completed_briefing_for_agent(
                collection_id, context_limit=3
            )
            payload["latest_readable"] = (
                self._shape_briefing_receipt(latest["briefing"])
                if latest is not None
                else None
            )
            payload["newer_operational_context"] = (
                [
                    self._shape_briefing_receipt(row)
                    for row in latest["newer_attempts"]
                ]
                if latest is not None
                else []
            )
        return self._finalize_metadata_page(
            base=payload,
            item_key="briefings",
            rows=rows,
            shaped=receipts,
            storage_has_more=bool(page["has_more"]),
            cursor_for=lambda row: self._encode_page_cursor(
                "briefings",
                {"created_at": row["created_at"], "id": int(row["id"])},
                fingerprint,
            ),
        )

    def get_briefing(self, arguments: object) -> str:
        """Return one bounded generated briefing with immutable provenance."""
        try:
            return self._get_briefing(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _get_briefing(self, arguments: object) -> str:
        values = self._exact_arguments(
            arguments,
            frozenset({"briefing_id", "selected_cursor", "cited_cursor"}),
        )
        briefing_id = self._validate_canonical_id(
            {"briefing_id": values.get("briefing_id")},
            field="briefing_id",
            pattern=_CANONICAL_BRIEFING_RE,
            label="briefing",
        )
        selected_cursor = self._validate_page_cursor(
            values.get("selected_cursor"),
            supplied="selected_cursor" in values,
            kind="briefing_selected",
        )
        cited_cursor = self._validate_page_cursor(
            values.get("cited_cursor"),
            supplied="cited_cursor" in values,
            kind="briefing_cited",
        )
        selected_fingerprint = self._metadata_fingerprint(
            {
                "briefing_id": briefing_id,
                "stream": "selected",
                "ordering": "position_nulls_last_position_asc_item_id_asc",
            }
        )
        cited_fingerprint = self._metadata_fingerprint(
            {
                "briefing_id": briefing_id,
                "stream": "cited",
                "ordering": "position_nulls_last_position_asc_item_id_asc",
            }
        )
        self._require_matching_page_cursor(selected_cursor, selected_fingerprint)
        self._require_matching_page_cursor(cited_cursor, cited_fingerprint)
        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        row = database.get_briefing_for_agent(briefing_id)
        if row is None:
            return self._outcome(
                "not_found", "briefing was not found", retryable=False
            )
        provenance = database.get_briefing_provenance_for_agent(
            briefing_id,
            limit=_PROVENANCE_LIMIT,
            selected_after=self._provenance_after(selected_cursor),
            cited_after=self._provenance_after(cited_cursor),
        )
        selected = [self._shape_provenance(item) for item in provenance["selected"]]
        cited = [self._shape_provenance(item) for item in provenance["cited"]]
        body = strip_control_characters(row.get("body_markdown") or "")
        bounded_body, body_truncated = self._bounded_text(
            body, _BRIEFING_BODY_BUDGET
        )
        briefing = self._shape_briefing_receipt(row)
        briefing["content"] = {
            "body_markdown": bounded_body or "",
            "body_byte_count": len(body.encode("utf-8")),
            "returned_body_byte_count": len((bounded_body or "").encode("utf-8")),
            "content_is_generated": True,
            "content_is_untrusted": True,
            "content_truncated": body_truncated,
        }
        briefing["selected_items"] = []
        briefing["cited_items"] = []
        briefing["selected_items_truncated"] = False
        briefing["cited_items_truncated"] = False
        briefing["selected_items_next_cursor"] = None
        briefing["cited_items_next_cursor"] = None
        payload = {"status": "ok", "briefing": briefing}
        selected_count = self._pack_provenance(briefing["selected_items"], selected)
        cited_count = self._pack_provenance(briefing["cited_items"], cited)
        self._finish_provenance_page(
            briefing,
            rows=provenance["selected"],
            accepted_count=selected_count,
            storage_has_more=bool(provenance["selected_has_more"]),
            stream="selected",
            fingerprint=selected_fingerprint,
        )
        self._finish_provenance_page(
            briefing,
            rows=provenance["cited"],
            accepted_count=cited_count,
            storage_has_more=bool(provenance["cited_has_more"]),
            stream="cited",
            fingerprint=cited_fingerprint,
        )
        return self._finalize(payload)

    def get_operations_status(self, arguments: object) -> str:
        """Return a bounded snapshot of source-check and briefing receipts."""
        try:
            return self._get_operations_status(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _get_operations_status(self, arguments: object) -> str:
        values = self._exact_arguments(
            arguments, frozenset({"source", "collection", "limit", "cursor"})
        )
        source_value = self._validate_scope_value(
            "source", values.get("source"), maximum=2_048, supplied="source" in values
        )
        collection_value = self._validate_scope_value(
            "collection",
            values.get("collection"),
            maximum=256,
            supplied="collection" in values,
        )
        limit = self._validate_limit(values.get("limit", 10))
        cursor = self._validate_page_cursor(
            values.get("cursor"), supplied="cursor" in values, kind="operations"
        )
        runtime_source = self._runtime_source()
        if runtime_source == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        source, outcome = self._resolve_scope(database, "source", source_value)
        if outcome is not None:
            return outcome
        collection, outcome = self._resolve_scope(
            database, "collection", collection_value
        )
        if outcome is not None:
            return outcome
        source_id = int(source["id"]) if source is not None else None
        collection_id = int(collection["id"]) if collection is not None else None
        fingerprint = self._metadata_fingerprint(
            {
                "source_id": source_id,
                "collection_id": collection_id,
                "ordering": "created_at_desc_kind_asc_id_desc",
            }
        )
        self._require_matching_page_cursor(cursor, fingerprint)
        position = cursor.position if cursor is not None else {}
        rows = database.list_operations_for_agent(
            source_id=source_id,
            watchlist_id=collection_id,
            limit=limit,
            after_created_at=position.get("created_at"),
            after_kind=position.get("kind"),
            after_id=position.get("id"),
        )
        operations = [
            (
                self._shape_run_operation(item["row"])
                if item["kind"] == "source_check"
                else self._shape_briefing_operation(item["row"])
            )
            for item in rows["operations"]
        ]
        operational_state = self._operational_state()
        return self._finalize_metadata_page(
            base={
                "status": "ok",
                "runtime_source": runtime_source,
                "app_gates": {
                    "watchlist_checks_enabled": operational_state[
                        "watchlist_checks_enabled"
                    ],
                    "briefing_schedules_enabled": operational_state[
                        "briefing_schedules_enabled"
                    ],
                },
                "scheduler_running": operational_state["scheduler_running"],
                "queue_reload_state": operational_state["queue_reload_state"],
                "ordering": "created_at_desc_kind_asc_id_desc",
            },
            item_key="operations",
            rows=rows["operations"],
            shaped=operations,
            storage_has_more=bool(rows["has_more"]),
            cursor_for=lambda item: self._encode_page_cursor(
                "operations",
                {
                    "created_at": item["row"]["created_at"],
                    "kind": item["kind"],
                    "id": int(item["row"]["id"]),
                },
                fingerprint,
            ),
        )

    def get_operation_status(self, arguments: object) -> str:
        """Return one exact source-check or briefing operation receipt."""
        try:
            return self._get_operation_status(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)
        except Exception as exc:
            self._raise_unexpected(exc)

    def _get_operation_status(self, arguments: object) -> str:
        values = self._exact_arguments(arguments, frozenset({"operation_id"}))
        if set(values) != {"operation_id"} or type(values["operation_id"]) is not str:
            raise _InvalidArgument("operation_id must be an exact canonical operation id")
        match = _CANONICAL_OPERATION_RE.fullmatch(values["operation_id"])
        if match is None:
            raise _InvalidArgument("operation_id must be an exact canonical operation id")
        operation_id = int(match.group("id"))
        if operation_id > _MAX_SQLITE_ROW_ID:
            raise _InvalidArgument("operation_id is outside the supported id range")
        if self._runtime_source() == "server":
            return self._outcome(
                "unsupported", _SERVER_UNSUPPORTED_MESSAGE, retryable=False
            )
        database, unavailable = self._database()
        if unavailable is not None:
            return unavailable
        if match.group("kind") == "watchlist_run":
            row = database.get_watchlist_run_for_agent(operation_id)
            operation = self._shape_run_operation(row) if row is not None else None
        else:
            row = database.get_briefing_for_agent(operation_id)
            operation = self._shape_briefing_operation(row) if row is not None else None
        if operation is None:
            return self._outcome(
                "not_found", "operation receipt was not found", retryable=False
            )
        return self._finalize({"status": "ok", "operation": operation})

    @staticmethod
    def _validate_search(arguments: object) -> _SearchRequest:
        values = WatchlistsToolService._exact_arguments(arguments, _SEARCH_KEYS)
        query = WatchlistsToolService._validate_query(
            values.get("query"), supplied="query" in values
        )
        collection = WatchlistsToolService._validate_scope_value(
            "collection",
            values.get("collection"),
            maximum=256,
            supplied="collection" in values,
        )
        source = WatchlistsToolService._validate_scope_value(
            "source",
            values.get("source"),
            maximum=2_048,
            supplied="source" in values,
        )
        statuses = WatchlistsToolService._validate_statuses(
            values.get("statuses"), supplied="statuses" in values
        )
        since = WatchlistsToolService._validate_since(
            values.get("since"), supplied="since" in values
        )
        limit = WatchlistsToolService._validate_limit(values.get("limit", 10))
        cursor = WatchlistsToolService._validate_cursor(
            values.get("cursor"), supplied="cursor" in values
        )
        return _SearchRequest(
            query=query,
            collection=collection,
            source=source,
            statuses=statuses,
            since=since,
            limit=limit,
            cursor=cursor,
        )

    @staticmethod
    def _validate_detail(arguments: object) -> int:
        values = WatchlistsToolService._exact_arguments(
            arguments, frozenset({"item_id"})
        )
        if set(values) != {"item_id"}:
            raise _InvalidArgument("item_id is required")
        item_id = values["item_id"]
        if type(item_id) is not str:
            raise _InvalidArgument("item_id must be a canonical Watchlists item id")
        match = _CANONICAL_ITEM_RE.fullmatch(item_id)
        if match is None:
            raise _InvalidArgument("item_id must be a canonical Watchlists item id")
        parsed_id = int(match.group("id"))
        if parsed_id > _MAX_SQLITE_ROW_ID:
            raise _InvalidArgument("item_id is outside the supported id range")
        return parsed_id

    @staticmethod
    def _validate_canonical_id(
        arguments: object,
        *,
        field: str,
        pattern: re.Pattern[str],
        label: str,
    ) -> int:
        values = WatchlistsToolService._exact_arguments(
            arguments, frozenset({field})
        )
        value = values.get(field)
        if set(values) != {field} or type(value) is not str:
            raise _InvalidArgument(f"{field} must be a canonical {label} id")
        match = pattern.fullmatch(value)
        if match is None:
            raise _InvalidArgument(f"{field} must be a canonical {label} id")
        parsed_id = int(match.group("id"))
        if parsed_id > _MAX_SQLITE_ROW_ID:
            raise _InvalidArgument(f"{field} is outside the supported id range")
        return parsed_id

    @staticmethod
    def _exact_arguments(arguments: object, allowed: frozenset[str]) -> dict[str, Any]:
        if type(arguments) is not dict:
            raise _InvalidArgument("arguments must be an object")
        unknown = set(arguments) - allowed
        if unknown:
            raise _InvalidArgument("arguments contain unknown properties")
        return arguments

    @staticmethod
    def _optional_string(
        values: Mapping[str, Any], field: str, *, maximum: int
    ) -> str | None:
        if field not in values:
            return None
        value = values[field]
        if type(value) is not str or not value.strip():
            raise _InvalidArgument(f"{field} must be a nonblank string")
        if len(value) > maximum:
            raise _InvalidArgument(f"{field} must be at most {maximum} characters")
        return value.strip()

    @staticmethod
    def _optional_enum(
        values: Mapping[str, Any], field: str, allowed: frozenset[str]
    ) -> str | None:
        if field not in values:
            return None
        value = values[field]
        if type(value) is not str or value not in allowed:
            raise _InvalidArgument(f"{field} contains an unsupported value")
        return value

    @staticmethod
    def _validate_briefing_statuses(
        value: object, *, supplied: bool
    ) -> tuple[str, ...] | None:
        if not supplied:
            return None
        allowed = frozenset({"generating", "complete", "empty", "failed"})
        if type(value) is not list or not value or len(value) > len(allowed):
            raise _InvalidArgument("statuses must be a nonempty bounded array")
        if any(type(status) is not str or status not in allowed for status in value):
            raise _InvalidArgument("statuses contain an unsupported value")
        if len(set(value)) != len(value):
            raise _InvalidArgument("statuses must contain unique values")
        return tuple(value)

    @staticmethod
    def _metadata_fingerprint(filters: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            filters,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _encode_page_cursor(
        kind: str, position: Mapping[str, Any], filter_fingerprint: str
    ) -> str:
        raw = WatchlistsToolService._json(
            {
                "version": _CURSOR_VERSION,
                "kind": kind,
                "position": dict(position),
                "filter_fingerprint": filter_fingerprint,
            }
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        if len(encoded) <= 2_048:
            return encoded
        compressed = base64.urlsafe_b64encode(zlib.compress(raw)).decode("ascii")
        return "z." + compressed.rstrip("=")

    @staticmethod
    def _validate_page_cursor(
        value: object, *, supplied: bool, kind: str
    ) -> _PageCursor | None:
        if not supplied:
            return None
        if type(value) is not str or not value or len(value) > 2_048:
            raise _InvalidArgument("cursor is invalid")
        try:
            compressed = value.startswith("z.")
            encoded = value[2:] if compressed else value
            padding = b"=" * (-len(encoded) % 4)
            raw = base64.b64decode(
                encoded.encode() + padding, altchars=b"-_", validate=True
            )
            if compressed:
                decompressor = zlib.decompressobj()
                raw = decompressor.decompress(raw, 32_769)
                if len(raw) > 32_768 or not decompressor.eof:
                    raise ValueError
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=WatchlistsToolService._unique_json_object,
            )
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise _InvalidArgument("cursor is invalid") from None
        if (
            type(payload) is not dict
            or set(payload)
            != {"version", "kind", "position", "filter_fingerprint"}
            or payload["version"] != _CURSOR_VERSION
            or type(payload["version"]) is not int
            or payload["kind"] != kind
            or type(payload["position"]) is not dict
            or type(payload["filter_fingerprint"]) is not str
            or _SHA256_RE.fullmatch(payload["filter_fingerprint"]) is None
        ):
            raise _InvalidArgument("cursor is invalid")
        position = payload["position"]
        if kind in {"sources", "collections"}:
            expected = {"name_casefold", "name", "id"}
        elif kind in {"briefing_selected", "briefing_cited"}:
            expected = {"position_is_null", "position", "item_id"}
        elif kind == "operations":
            expected = {"created_at", "kind", "id"}
        else:
            expected = {"created_at", "id"}
        if set(position) != expected:
            raise _InvalidArgument("cursor is invalid")
        if kind in {"briefing_selected", "briefing_cited"}:
            null_state = position["position_is_null"]
            item_position = position["position"]
            item_id = position["item_id"]
            if (
                type(null_state) is not bool
                or null_state is not (item_position is None)
                or (
                    item_position is not None
                    and (
                        type(item_position) is not int
                        or not -_MAX_SQLITE_ROW_ID <= item_position <= _MAX_SQLITE_ROW_ID
                    )
                )
                or type(item_id) is not int
                or not 1 <= item_id <= _MAX_SQLITE_ROW_ID
            ):
                raise _InvalidArgument("cursor is invalid")
            return _PageCursor(
                kind=kind,
                position=position,
                filter_fingerprint=payload["filter_fingerprint"],
            )
        if type(position["id"]) is not int or not 1 <= position["id"] <= _MAX_SQLITE_ROW_ID:
            raise _InvalidArgument("cursor is invalid")
        for key, item in position.items():
            maximum = 16_384 if key in {"name", "name_casefold"} else 512
            if key != "id" and (type(item) is not str or len(item) > maximum):
                raise _InvalidArgument("cursor is invalid")
        return _PageCursor(
            kind=kind,
            position=position,
            filter_fingerprint=payload["filter_fingerprint"],
        )

    @staticmethod
    def _require_matching_page_cursor(
        cursor: _PageCursor | None, fingerprint: str
    ) -> None:
        if cursor is not None and cursor.filter_fingerprint != fingerprint:
            raise _InvalidArgument("cursor does not match the list filters")

    @staticmethod
    def _validate_query(value: object, *, supplied: bool) -> str | None:
        if not supplied:
            return None
        if type(value) is not str:
            raise _InvalidArgument("query must be a string")
        if len(value) > 512:
            raise _InvalidArgument("query must be at most 512 characters")
        terms = value.split()
        if len(terms) > 32:
            raise _InvalidArgument("query must contain at most 32 terms")
        return " ".join(terms) if terms else None

    @staticmethod
    def _validate_scope_value(
        field: str, value: object, *, maximum: int, supplied: bool
    ) -> int | str | None:
        if not supplied:
            return None
        if type(value) is int:
            if not 1 <= value <= _MAX_SQLITE_ROW_ID:
                raise _InvalidArgument(f"{field} id must be a positive integer")
            return value
        if type(value) is not str:
            raise _InvalidArgument(f"{field} must be a name or canonical id")
        if len(value) > maximum:
            raise _InvalidArgument(f"{field} must be at most {maximum} characters")
        stripped = value.strip()
        if not stripped:
            raise _InvalidArgument(f"{field} must not be blank")

        match = _CANONICAL_SCOPE_RE.fullmatch(stripped)
        expected_kind = "subscription" if field == "source" else "watchlist"
        if match is not None:
            if match.group("kind") != expected_kind:
                raise _InvalidArgument(f"{field} has the wrong canonical id type")
            parsed_id = int(match.group("id"))
            if parsed_id > _MAX_SQLITE_ROW_ID:
                raise _InvalidArgument(f"{field} id is outside the supported range")
            return parsed_id
        if stripped.startswith("local:") or _COMPOSITE_ID_RE.fullmatch(stripped):
            raise _InvalidArgument(f"{field} has an invalid canonical id")
        return stripped

    @staticmethod
    def _validate_statuses(value: object, *, supplied: bool) -> tuple[str, ...] | None:
        if not supplied:
            return None
        if type(value) is not list or not value:
            raise _InvalidArgument("statuses must be a nonempty array")
        if any(type(status) is not str or status not in _STATUSES for status in value):
            raise _InvalidArgument("statuses contain an unsupported value")
        if len(set(value)) != len(value):
            raise _InvalidArgument("statuses must contain unique values")
        return tuple(value)

    @staticmethod
    def _validate_since(value: object, *, supplied: bool) -> str | None:
        if not supplied:
            return None
        if type(value) is not str or not value:
            raise _InvalidArgument("since must be YYYY-MM-DD or RFC3339")
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                parsed = datetime.combine(date.fromisoformat(value), time(), tzinfo=UTC)
            elif _RFC3339_RE.fullmatch(value):
                parsed = datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
            else:
                raise ValueError
        except ValueError:
            raise _InvalidArgument("since must be YYYY-MM-DD or RFC3339") from None
        return WatchlistsToolService._format_utc(parsed)

    @staticmethod
    def _validate_limit(value: object) -> int:
        if type(value) is not int or not 1 <= value <= 50:
            raise _InvalidArgument("limit must be an integer from 1 through 50")
        return value

    @staticmethod
    def _validate_cursor(value: object, *, supplied: bool) -> _Cursor | None:
        if not supplied:
            return None
        if type(value) is not str or not value or len(value) > 2_048:
            raise _InvalidArgument("cursor is invalid")
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            raise _InvalidArgument("cursor is invalid")
        try:
            padding = b"=" * (-len(value) % 4)
            raw = base64.b64decode(
                value.encode() + padding, altchars=b"-_", validate=True
            )
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=WatchlistsToolService._unique_json_object,
            )
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise _InvalidArgument("cursor is invalid") from None
        if type(payload) is not dict or set(payload) != _CURSOR_KEYS:
            raise _InvalidArgument("cursor is invalid")
        if payload["version"] != _CURSOR_VERSION or type(payload["version"]) is not int:
            raise _InvalidArgument("cursor is invalid")

        as_of = payload["as_of"]
        if type(as_of) is not str or _RFC3339_RE.fullmatch(as_of) is None:
            raise _InvalidArgument("cursor is invalid")
        try:
            parsed_as_of = datetime.fromisoformat(as_of.upper().replace("Z", "+00:00"))
        except ValueError:
            raise _InvalidArgument("cursor is invalid") from None
        if WatchlistsToolService._format_utc(parsed_as_of) != as_of:
            raise _InvalidArgument("cursor is invalid")

        snapshot_max_item_id = payload["snapshot_max_item_id"]
        last_item_id = payload["last_item_id"]
        if (
            type(snapshot_max_item_id) is not int
            or not 1 <= snapshot_max_item_id <= _MAX_SQLITE_ROW_ID
            or type(last_item_id) is not int
            or not 1 <= last_item_id <= snapshot_max_item_id
        ):
            raise _InvalidArgument("cursor is invalid")

        null_state = payload["last_effective_date_is_null"]
        last_effective_date = payload["last_effective_date"]
        if type(null_state) is not bool or null_state is not (
            last_effective_date is None
        ):
            raise _InvalidArgument("cursor is invalid")
        if last_effective_date is not None:
            if (
                type(last_effective_date) is not str
                or _SQLITE_DATETIME_RE.fullmatch(last_effective_date) is None
            ):
                raise _InvalidArgument("cursor is invalid")
            try:
                datetime.fromisoformat(last_effective_date)
            except ValueError:
                raise _InvalidArgument("cursor is invalid") from None

        fingerprint = payload["filter_fingerprint"]
        if type(fingerprint) is not str or _SHA256_RE.fullmatch(fingerprint) is None:
            raise _InvalidArgument("cursor is invalid")
        return _Cursor(
            as_of=as_of,
            snapshot_max_item_id=snapshot_max_item_id,
            last_effective_date=last_effective_date,
            last_item_id=last_item_id,
            filter_fingerprint=fingerprint,
        )

    @staticmethod
    def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate cursor key")
            result[key] = value
        return result

    @staticmethod
    def _filter_fingerprint(
        *,
        query: str | None,
        collection_id: int | None,
        source_id: int | None,
        statuses: tuple[str, ...] | None,
        since: str | None,
        ordering: str = _ORDERING,
    ) -> str:
        normalized = {
            "query": query,
            "collection_id": collection_id,
            "source_id": source_id,
            "statuses": sorted(set(statuses or ())),
            "since": since,
            "ordering": ordering,
        }
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _encode_cursor(
        *,
        as_of: str,
        snapshot_max_item_id: int,
        last_effective_date: str | None,
        last_item_id: int,
        filter_fingerprint: str,
    ) -> str:
        payload = {
            "version": _CURSOR_VERSION,
            "as_of": as_of,
            "snapshot_max_item_id": snapshot_max_item_id,
            "last_effective_date": last_effective_date,
            "last_effective_date_is_null": last_effective_date is None,
            "last_item_id": last_item_id,
            "filter_fingerprint": filter_fingerprint,
        }
        raw = WatchlistsToolService._json(payload).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _runtime_source(self) -> str:
        state = self._runtime_source_loader()
        if isinstance(state, str) and state in {"local", "server"}:
            return state
        if isinstance(state, Mapping):
            source = state.get("active_source")
            return (
                source
                if isinstance(source, str) and source in {"local", "server"}
                else "local"
            )
        source = getattr(state, "active_source", None)
        return (
            source
            if isinstance(source, str) and source in {"local", "server"}
            else "local"
        )

    @staticmethod
    def _default_operational_state() -> Mapping[str, Any]:
        """Read configuration gates without inventing live scheduler state."""
        try:
            from tldw_chatbook.config import get_cli_setting

            checks_enabled = bool(
                get_cli_setting("scheduling", "watchlist_checks_enabled", True)
            )
            briefings_enabled = bool(
                get_cli_setting("scheduling", "briefing_schedules_enabled", True)
            )
        except Exception:  # noqa: BLE001 -- unavailable config fails closed
            checks_enabled = False
            briefings_enabled = False
        return {
            "watchlist_checks_enabled": checks_enabled,
            "briefing_schedules_enabled": briefings_enabled,
            "scheduler_running": None,
            "queue_reload_state": "not_observed",
        }

    def _operational_state(self) -> dict[str, Any]:
        """Return one validated, bounded operational-state snapshot."""
        value = self._operational_state_loader()
        if not isinstance(value, Mapping):
            raise RuntimeError("invalid Watchlists operational state")
        checks = value.get("watchlist_checks_enabled", False)
        briefings = value.get("briefing_schedules_enabled", False)
        scheduler = value.get("scheduler_running")
        if type(checks) is not bool or type(briefings) is not bool:
            raise RuntimeError("invalid Watchlists operational gates")
        if scheduler is not None and type(scheduler) is not bool:
            raise RuntimeError("invalid Watchlists scheduler state")
        return {
            "watchlist_checks_enabled": checks,
            "briefing_schedules_enabled": briefings,
            "scheduler_running": scheduler,
            "queue_reload_state": self._safe_text(
                value.get("queue_reload_state", "not_observed"), 128
            ),
        }

    def _database(self) -> tuple[Any | None, str | None]:
        try:
            database = self._db_resolver()
        except (SubscriptionsDBUnavailableError, FileNotFoundError, ImportError):
            return None, self._feature_unavailable(retryable=False)
        except SubscriptionsDBReadError:
            return None, self._feature_unavailable(retryable=True)

        if database is None:
            return None, self._feature_unavailable(retryable=False)
        readiness_check = getattr(database, "assert_agent_read_ready", None)
        if callable(readiness_check):
            try:
                readiness_check()
            except SubscriptionsDBUnavailableError:
                return None, self._feature_unavailable(retryable=False)
            except SubscriptionsDBReadError:
                return None, self._feature_unavailable(retryable=True)
        return database, None

    @staticmethod
    def _resolve_scope(
        database: Any, field: str, value: int | str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        if value is None:
            return None, None
        if field == "source":
            candidates = database.resolve_source_candidates(
                value, limit=_CANDIDATE_LIMIT
            )
        else:
            candidates = database.resolve_collection_candidates(
                value, limit=_CANDIDATE_LIMIT
            )
        if not candidates:
            return None, WatchlistsToolService._outcome(
                "not_found", f"{field} was not found", retryable=False
            )
        if len(candidates) > 1:
            return None, WatchlistsToolService._finalize(
                {
                    "status": "needs_disambiguation",
                    "retryable": False,
                    "message": (f"{field} is ambiguous; retry with one candidate id"),
                    "candidates": [
                        WatchlistsToolService._shape_candidate(field, candidate)
                        for candidate in candidates
                    ],
                }
            )
        return candidates[0], None

    @staticmethod
    def _shape_candidate(field: str, row: Mapping[str, Any]) -> dict[str, Any]:
        name, name_truncated = WatchlistsToolService._bounded_text(
            row["name"], _MAX_NAME_BYTES
        )
        if field == "source":
            url, url_redacted, url_truncated = WatchlistsToolService._sanitize_url(
                row["source"]
            )
            return {
                "id": f"local:subscription:{row['id']}",
                "name": name,
                "name_truncated": name_truncated,
                "url": url,
                "url_redacted": url_redacted,
                "url_truncated": url_truncated,
            }
        return {
            "id": f"local:watchlist:{row['id']}",
            "name": name,
            "name_truncated": name_truncated,
        }

    @staticmethod
    def _shape_source_metadata(
        row: Mapping[str, Any], membership: Mapping[str, Any]
    ) -> dict[str, Any]:
        name, name_truncated = WatchlistsToolService._bounded_text(
            row["name"], _MAX_NAME_BYTES
        )
        url, url_redacted, url_truncated = WatchlistsToolService._sanitize_url(
            row["source"]
        )
        failures = row["consecutive_failures"]
        return {
            "id": f"local:subscription:{row['id']}",
            "name": name,
            "name_truncated": name_truncated,
            "type": WatchlistsToolService._safe_text(row["type"], 128),
            "url": url,
            "url_redacted": url_redacted,
            "url_truncated": url_truncated,
            "is_active": bool(row["is_active"]),
            "is_paused": bool(row["is_paused"]),
            "check_frequency_seconds": row["check_frequency"],
            "last_checked": WatchlistsToolService._safe_text(
                row["last_checked"], 128
            ),
            "last_successful_check": WatchlistsToolService._safe_text(
                row["last_successful_check"], 128
            ),
            "consecutive_failures": failures,
            "attention_state": (
                "needs_attention"
                if bool(row["is_paused"]) or int(failures or 0) > 0
                else "ok"
            ),
            "collections": [
                WatchlistsToolService._shape_collection(item)
                for item in membership["collections"]
            ],
            "collections_truncated": bool(membership["has_more"]),
            "created_at": WatchlistsToolService._safe_text(row["created_at"], 128),
            "updated_at": WatchlistsToolService._safe_text(row["updated_at"], 128),
        }

    @staticmethod
    def _shape_collection_metadata(
        row: Mapping[str, Any],
        operational_state: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        name, name_truncated = WatchlistsToolService._bounded_text(
            row["name"], _MAX_NAME_BYTES
        )
        preset_name, preset_name_truncated = WatchlistsToolService._bounded_text(
            row["default_preset_name"], _MAX_NAME_BYTES
        )
        cadence = row["briefing_cadence_seconds"]
        schedule_state, next_eligible_at = WatchlistsToolService._schedule_state(
            row, operational_state, now
        )
        return {
            "id": f"local:watchlist:{row['id']}",
            "name": name,
            "name_truncated": name_truncated,
            "is_active": bool(row["is_active"]),
            "source_count": int(row["source_count"]),
            "briefing_selection_mode": WatchlistsToolService._safe_text(
                row["briefing_selection_mode"], 64
            ),
            "default_preset": (
                {
                    "id": row["default_briefing_preset_id"],
                    "name": preset_name,
                    "name_truncated": preset_name_truncated,
                }
                if row["default_briefing_preset_id"] is not None
                else None
            ),
            "briefing_cadence_seconds": cadence,
            "effective_scheduler_state": schedule_state,
            "next_eligible_at": next_eligible_at,
            "last_briefing_attempt_at": WatchlistsToolService._safe_text(
                row["last_briefing_attempt_at"], 128
            ),
            "last_briefing_success_at": WatchlistsToolService._safe_text(
                row["last_briefing_success_at"], 128
            ),
            "attention_state": "ok" if bool(row["is_active"]) else "disabled",
            "created_at": WatchlistsToolService._safe_text(row["created_at"], 128),
            "updated_at": WatchlistsToolService._safe_text(row["updated_at"], 128),
        }

    @staticmethod
    def _schedule_state(
        row: Mapping[str, Any],
        operational_state: Mapping[str, Any],
        now: datetime,
    ) -> tuple[str, str | None]:
        cadence = row["briefing_cadence_seconds"]
        if cadence is None:
            return "never_scheduled", None
        if not operational_state["briefing_schedules_enabled"]:
            return "app_scheduling_disabled", None
        if operational_state["scheduler_running"] is not True:
            return "scheduler_not_running", None
        latest_status = row["last_briefing_status"]
        if latest_status == "generating":
            return "generation_in_progress", None
        anchor = row["last_briefing_attempt_at"]
        next_eligible: datetime | None = None
        if isinstance(anchor, str):
            try:
                parsed = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                next_eligible = parsed.astimezone(UTC) + timedelta(
                    seconds=int(cadence)
                )
            except (TypeError, ValueError, OverflowError):
                next_eligible = None
        if latest_status == "failed":
            state = "last_attempt_failed"
        elif next_eligible is not None and next_eligible <= (
            now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        ):
            state = "due_or_queued"
        elif latest_status in {"complete", "empty"}:
            state = "completed_or_empty"
        else:
            state = "scheduled_waiting"
        return (
            state,
            WatchlistsToolService._format_utc(next_eligible)
            if next_eligible is not None
            else None,
        )

    @staticmethod
    def _shape_briefing_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
        collection_name, collection_name_truncated = (
            WatchlistsToolService._bounded_text(row["watchlist_name"], _MAX_NAME_BYTES)
        )
        preset_name, preset_name_truncated = WatchlistsToolService._bounded_text(
            row["preset_name"], _MAX_NAME_BYTES
        )
        status = WatchlistsToolService._safe_text(row["status"], 64)
        return {
            "id": f"local:briefing:{row['id']}",
            "collection": {
                "id": f"local:watchlist:{row['watchlist_id']}",
                "name": collection_name,
                "name_truncated": collection_name_truncated,
            },
            "status": status,
            "created_at": WatchlistsToolService._safe_text(row["created_at"], 128),
            "updated_at": WatchlistsToolService._safe_text(row["updated_at"], 128),
            "selection_mode": WatchlistsToolService._safe_text(
                row["selection_mode"], 64
            ),
            "preset": (
                {
                    "id": row["preset_id"],
                    "name": preset_name,
                    "name_truncated": preset_name_truncated,
                }
                if row["preset_id"] is not None
                else None
            ),
            "model_used": WatchlistsToolService._safe_text(row["model_used"], 512),
            "coverage": {
                "from": WatchlistsToolService._safe_text(
                    row["covers_from_ts"], 128
                ),
                "through_item_id": (
                    f"local:watchlist_item:{row['covers_through_item_id']}"
                    if row["covers_through_item_id"] is not None
                    else None
                ),
            },
            "counts": {
                "items": int(row["item_count"] or 0),
                "featured": int(row["featured_count"] or 0),
                "overflow": int(row["overflow_count"] or 0),
            },
            "body_available": bool(row["body_available"]),
            "body_byte_count": int(row["body_byte_count"] or 0),
            "state": WatchlistsToolService._normalize_briefing_state(status),
            "retryable": status == "failed",
            "attention_state": (
                "needs_attention" if status == "failed" else "ok"
            ),
        }

    @staticmethod
    def _shape_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
        title, title_truncated = WatchlistsToolService._bounded_text(
            row["item_title"], _MAX_TITLE_BYTES
        )
        source_name, source_name_truncated = WatchlistsToolService._bounded_text(
            row["source_name"], _MAX_NAME_BYTES
        )
        item_url, item_url_redacted, item_url_truncated = (
            WatchlistsToolService._sanitize_url(row["item_url"])
        )
        source_url, source_url_redacted, source_url_truncated = (
            WatchlistsToolService._sanitize_url(row["source_url"])
        )
        legacy = int(row["provenance_version"] or 0) < 2
        missing = row["item_title"] is None or row["source_name"] is None
        return {
            "id": f"local:watchlist_item:{row['item_id']}",
            "selection_position": row["selection_position"],
            "citation_position": row["citation_position"],
            "featured": bool(row["featured"]),
            "cited": bool(row["cited"]),
            "title": title,
            "title_truncated": title_truncated,
            "url": item_url,
            "url_redacted": item_url_redacted,
            "url_truncated": item_url_truncated,
            "published_date": WatchlistsToolService._safe_text(
                row["item_published_date"], 128
            ),
            "effective_date": WatchlistsToolService._safe_text(
                row["item_effective_date"], 128
            ),
            "source": (
                {
                    "id": (
                        f"local:subscription:{row['source_id']}"
                        if row["source_id"] is not None
                        else None
                    ),
                    "name": source_name,
                    "name_truncated": source_name_truncated,
                    "type": WatchlistsToolService._safe_text(
                        row["source_type"], 128
                    ),
                    "url": source_url,
                    "url_redacted": source_url_redacted,
                    "url_truncated": source_url_truncated,
                }
                if row["source_id"] is not None or row["source_name"] is not None
                else None
            ),
            "provenance_quality": (
                "legacy_best_effort" if legacy else "ordered_snapshot"
            ),
            "legacy_provenance": legacy,
            "missing_reference": missing,
            "content_is_untrusted": True,
        }

    @staticmethod
    def _normalize_briefing_state(status: object) -> str:
        if status == "generating":
            return "running"
        if status == "failed":
            return "needs_attention"
        return "ok"

    @staticmethod
    def _normalize_run_state(status: object) -> str:
        if status == "queued":
            return "waiting"
        if status == "running":
            return "running"
        if status in {"failed", "error"}:
            return "needs_attention"
        if status in {"disabled", "cancelled", "canceled"}:
            return "disabled"
        return "ok"

    @staticmethod
    def _shape_run_operation(row: Mapping[str, Any]) -> dict[str, Any]:
        source_name, source_name_truncated = WatchlistsToolService._bounded_text(
            row["source_name"], _MAX_NAME_BYTES
        )
        status = WatchlistsToolService._safe_text(row["status"], 64)
        return {
            "id": f"local:watchlist_run:{row['id']}",
            "kind": "source_check",
            "state": WatchlistsToolService._normalize_run_state(status),
            "status_detail": status,
            "source": {
                "id": f"local:subscription:{row['source_id']}",
                "name": source_name,
                "name_truncated": source_name_truncated,
            },
            "created_at": WatchlistsToolService._safe_text(row["created_at"], 128),
            "updated_at": WatchlistsToolService._safe_text(row["updated_at"], 128),
            "started_at": WatchlistsToolService._safe_text(row["started_at"], 128),
            "finished_at": WatchlistsToolService._safe_text(
                row["finished_at"], 128
            ),
            "result_available": row["stats_json"] is not None,
            "error_category": "source_check_failed" if row["has_error"] else None,
            "retry_capable": status in {"failed", "error"},
            "cancel_capable": status in {"queued", "running"},
            "destination": "runs",
        }

    @staticmethod
    def _shape_briefing_operation(row: Mapping[str, Any]) -> dict[str, Any]:
        receipt = WatchlistsToolService._shape_briefing_receipt(row)
        status = WatchlistsToolService._safe_text(row["status"], 64)
        return {
            "id": receipt["id"],
            "kind": "briefing_generation",
            "state": WatchlistsToolService._normalize_briefing_state(status),
            "status_detail": status,
            "collection": receipt["collection"],
            "created_at": receipt["created_at"],
            "updated_at": receipt["updated_at"],
            "started_at": receipt["created_at"],
            "finished_at": receipt["updated_at"] if status != "generating" else None,
            "result_available": bool(row["body_available"]),
            "error_category": "briefing_failed" if status == "failed" else None,
            "retry_capable": status == "failed",
            "cancel_capable": False,
            "destination": "artifacts",
        }

    @staticmethod
    def _pack_provenance(
        destination: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> int:
        for candidate in candidates:
            destination.append(candidate)
            if WatchlistsToolService._json_size(destination) >= _PROVENANCE_ARRAY_BUDGET:
                destination.pop()
                break
        if candidates and not destination:
            raise RuntimeError("bounded Watchlists provenance row did not fit")
        return len(destination)

    @staticmethod
    def _provenance_after(
        cursor: _PageCursor | None,
    ) -> tuple[int, int, int] | None:
        if cursor is None:
            return None
        position = cursor.position
        return (
            int(position["position_is_null"]),
            int(position["position"] or 0),
            int(position["item_id"]),
        )

    @staticmethod
    def _finish_provenance_page(
        briefing: dict[str, Any],
        *,
        rows: list[Mapping[str, Any]],
        accepted_count: int,
        storage_has_more: bool,
        stream: str,
        fingerprint: str,
    ) -> None:
        has_more = storage_has_more or accepted_count < len(rows)
        briefing[f"{stream}_items_truncated"] = has_more
        if not has_more:
            briefing[f"{stream}_items_next_cursor"] = None
            return
        row = rows[accepted_count - 1]
        position_field = (
            "selection_position" if stream == "selected" else "citation_position"
        )
        item_position = row[position_field]
        briefing[f"{stream}_items_next_cursor"] = (
            WatchlistsToolService._encode_page_cursor(
                f"briefing_{stream}",
                {
                    "position_is_null": item_position is None,
                    "position": item_position,
                    "item_id": int(row["item_id"]),
                },
                fingerprint,
            )
        )

    @staticmethod
    def _finalize_metadata_page(
        *,
        base: Mapping[str, Any],
        item_key: str,
        rows: list[Any],
        shaped: list[dict[str, Any]],
        storage_has_more: bool,
        cursor_for: Callable[[Any], str],
    ) -> str:
        """Pack complete metadata rows while preserving a usable continuation."""
        payload = dict(base)
        accepted: list[dict[str, Any]] = []
        payload[item_key] = accepted
        payload["returned_count"] = 0
        payload["has_more"] = False
        payload["next_cursor"] = None
        for index, (row, item) in enumerate(zip(rows, shaped, strict=True)):
            accepted.append(item)
            has_more = storage_has_more or index < len(rows) - 1
            payload["returned_count"] = len(accepted)
            payload["has_more"] = has_more
            payload["next_cursor"] = cursor_for(row) if has_more else None
            if WatchlistsToolService._json_size(payload) >= _MAX_RESULT_BYTES:
                accepted.pop()
                if not accepted:
                    raise RuntimeError("bounded Watchlists metadata row did not fit")
                payload["returned_count"] = len(accepted)
                payload["has_more"] = True
                payload["next_cursor"] = cursor_for(rows[index - 1])
                break
        return WatchlistsToolService._finalize(payload)

    @staticmethod
    def _shape_collection(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        name, name_truncated = WatchlistsToolService._bounded_text(
            row["name"], _MAX_NAME_BYTES
        )
        return {
            "id": f"local:watchlist:{row['id']}",
            "name": name,
            "name_truncated": name_truncated,
        }

    @staticmethod
    def _shape_selected_source(
        row: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        name, name_truncated = WatchlistsToolService._bounded_text(
            row["name"], _MAX_NAME_BYTES
        )
        url, url_redacted, url_truncated = WatchlistsToolService._sanitize_url(
            row["source"]
        )
        return {
            "id": f"local:subscription:{row['id']}",
            "name": name,
            "name_truncated": name_truncated,
            "type": row["type"],
            "url": url,
            "url_redacted": url_redacted,
            "url_truncated": url_truncated,
            "is_active": bool(row["is_active"]),
            "is_paused": bool(row["is_paused"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_checked": row["last_checked"],
            "last_successful_check": row["last_successful_check"],
        }

    @staticmethod
    def _shape_item_metadata(
        row: Mapping[str, Any], membership: Mapping[str, Any]
    ) -> dict[str, Any]:
        title, title_truncated = WatchlistsToolService._bounded_text(
            row["title"], _MAX_TITLE_BYTES
        )
        author, author_truncated = WatchlistsToolService._bounded_text(
            row["author"], _MAX_AUTHOR_BYTES
        )
        item_url, item_url_redacted, item_url_truncated = (
            WatchlistsToolService._sanitize_url(row["url"])
        )
        source_name, source_name_truncated = WatchlistsToolService._bounded_text(
            row["subscription_name"], _MAX_NAME_BYTES
        )
        source_url, source_url_redacted, source_url_truncated = (
            WatchlistsToolService._sanitize_url(row["subscription_source"])
        )
        return {
            "id": f"local:watchlist_item:{row['id']}",
            "title": title,
            "title_truncated": title_truncated,
            "url": item_url,
            "url_redacted": item_url_redacted,
            "url_truncated": item_url_truncated,
            "author": author,
            "author_truncated": author_truncated,
            "status": row["status"],
            "effective_date": row["effective_date"],
            "published_date": row["published_date"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "content_format": row["content_format"],
            "content_kind": row["content_kind"],
            "source": {
                "id": f"local:subscription:{row['subscription_id']}",
                "name": source_name,
                "name_truncated": source_name_truncated,
                "type": row["subscription_type"],
                "url": source_url,
                "url_redacted": source_url_redacted,
                "url_truncated": source_url_truncated,
                "is_active": bool(row["subscription_is_active"]),
                "is_paused": bool(row["subscription_is_paused"]),
            },
            "collections": [
                WatchlistsToolService._shape_collection(collection)
                for collection in membership["collections"]
            ],
            "collections_truncated": bool(membership["has_more"]),
        }

    @staticmethod
    def _shape_search_item(
        row: Mapping[str, Any], membership: Mapping[str, Any], query: str | None
    ) -> dict[str, Any]:
        item = WatchlistsToolService._shape_item_metadata(row, membership)
        snippet, snippet_truncated = WatchlistsToolService._search_excerpt(row, query)
        item["evidence"] = {
            "content_is_untrusted": True,
            "snippet": snippet,
            "snippet_truncated": snippet_truncated,
        }
        return item

    @staticmethod
    def _search_response(
        *,
        request: _SearchRequest,
        as_of: str,
        snapshot_max_item_id: int,
        scope: dict[str, Any],
        items: list[dict[str, Any]],
        has_more: bool,
        next_cursor: str | None,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "query_mode": (
                "literal_full_text" if request.query is not None else "browse"
            ),
            "ordering": _ORDERING,
            "as_of": as_of,
            "snapshot_max_item_id": snapshot_max_item_id,
            "returned_count": len(items),
            "has_more": has_more,
            "next_cursor": next_cursor,
            "scope": scope,
            "items": items,
        }

    @staticmethod
    def _bounded_text(value: object, maximum_bytes: int) -> tuple[str | None, bool]:
        if value is None:
            return None, False
        text = strip_control_characters(value)
        if WatchlistsToolService._json_size(text) <= maximum_bytes:
            return text, False
        low = 0
        high = len(text)
        fitted = _TRUNCATION_SUFFIX
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle] + _TRUNCATION_SUFFIX
            if WatchlistsToolService._json_size(candidate) <= maximum_bytes:
                fitted = candidate
                low = middle + 1
            else:
                high = middle - 1
        return fitted, True

    @staticmethod
    def _safe_text(value: object, maximum_bytes: int) -> str | None:
        """Return control-free bounded stored text without its truncation flag."""
        return WatchlistsToolService._bounded_text(value, maximum_bytes)[0]

    @staticmethod
    def _sanitize_url(value: object) -> tuple[str | None, bool, bool]:
        if value is None:
            return None, False, False
        if not isinstance(value, str) or not value:
            return None, True, False
        if re.search(r"[\x00-\x20\x7f-\x9f]", value):
            return None, True, False
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            port = parsed.port
            if parsed.scheme.casefold() not in {"http", "https"} or not hostname:
                return None, True, False
            if any(
                character.isspace() or ord(character) < 0x20 for character in hostname
            ):
                return None, True, False
            ascii_hostname = hostname.encode("idna").decode("ascii")
            if ":" in ascii_hostname:
                ipaddress.IPv6Address(ascii_hostname)
                ascii_hostname = f"[{ascii_hostname}]"
            else:
                labels = ascii_hostname.rstrip(".").split(".")
                if not labels or any(
                    not label
                    or len(label) > 63
                    or re.fullmatch(r"[A-Za-z0-9-]+", label) is None
                    or label.startswith("-")
                    or label.endswith("-")
                    for label in labels
                ):
                    return None, True, False
            netloc = ascii_hostname if port is None else f"{ascii_hostname}:{port}"
            sanitized = urlunsplit(
                (parsed.scheme.casefold(), netloc, parsed.path, "", "")
            )
        except (UnicodeError, ValueError):
            return None, True, False
        bounded, truncated = WatchlistsToolService._bounded_text(
            sanitized, _MAX_URL_BYTES
        )
        return bounded, sanitized != value or truncated, truncated

    @staticmethod
    def _search_excerpt(
        row: Mapping[str, Any], query: str | None
    ) -> tuple[str | None, bool]:
        if query is None:
            return WatchlistsToolService._bounded_text(
                row["content_match_context"], _MAX_SNIPPET_BYTES
            )
        needles = [query, *query.split()]
        for field in ("title", "author", "content_match_context"):
            value = row[field]
            if value is None:
                continue
            text = str(value)
            folded_parts: list[str] = []
            original_positions: list[int] = []
            for position, character in enumerate(text):
                folded_character = character.casefold()
                folded_parts.append(folded_character)
                original_positions.extend([position] * len(folded_character))
            folded_text = "".join(folded_parts)
            for needle in needles:
                folded_needle = needle.casefold()
                folded_index = folded_text.find(folded_needle)
                if folded_index >= 0:
                    start = original_positions[folded_index]
                    end = original_positions[folded_index + len(folded_needle) - 1] + 1
                    return WatchlistsToolService._centered_excerpt(
                        text, start, end - start
                    )
        return WatchlistsToolService._bounded_text(
            row["content_match_context"], _MAX_SNIPPET_BYTES
        )

    @staticmethod
    def _centered_excerpt(
        text: str, match_index: int, match_length: int
    ) -> tuple[str, bool]:
        maximum_chars = _MAX_SNIPPET_BYTES // 4
        bounded, truncated = WatchlistsToolService._bounded_text(
            text, _MAX_SNIPPET_BYTES
        )
        if not truncated:
            return bounded or "", False
        half_window = max(1, (maximum_chars - match_length) // 2)
        start = max(0, match_index - half_window)
        end = min(len(text), match_index + match_length + half_window)
        excerpt = text[start:end]
        if start:
            excerpt = "…" + excerpt
        if end < len(text):
            excerpt += _TRUNCATION_SUFFIX
        bounded, _ = WatchlistsToolService._bounded_text(excerpt, _MAX_SNIPPET_BYTES)
        return bounded or "", True

    @staticmethod
    def _finite_number(value: object) -> tuple[int | float | None, bool]:
        if value is None:
            return None, False
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            return None, True
        return value, False

    @staticmethod
    def _fit_detail_content(item: dict[str, Any]) -> str:
        payload = {"status": "ok", "item": item}
        rendered = WatchlistsToolService._json(payload)
        if len(rendered.encode("utf-8")) < _MAX_RESULT_BYTES:
            return rendered

        evidence = item["evidence"]
        content = evidence["content"]
        if not isinstance(content, str):
            raise RuntimeError("invalid detail content contract")
        evidence["content_truncated"] = True
        low = 0
        high = len(content)
        fitted: str | None = None
        while low <= high:
            middle = (low + high) // 2
            evidence["content"] = content[:middle] + _TRUNCATION_SUFFIX
            candidate = WatchlistsToolService._json(payload)
            if len(candidate.encode("utf-8")) < _MAX_RESULT_BYTES:
                fitted = candidate
                low = middle + 1
            else:
                high = middle - 1
        if fitted is None:
            raise RuntimeError("bounded Watchlists detail did not fit")
        return fitted

    @staticmethod
    def _json_size(payload: object) -> int:
        return len(WatchlistsToolService._json(payload).encode("utf-8"))

    @staticmethod
    def _finalize(payload: object) -> str:
        rendered = WatchlistsToolService._json(payload)
        if len(rendered.encode("utf-8")) >= _MAX_RESULT_BYTES:
            raise RuntimeError("bounded Watchlists result did not fit")
        return rendered

    @staticmethod
    def _raise_unexpected(exc: Exception) -> None:
        category = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:64]
        _LOGGER.error(
            "Watchlists tool execution failed category=%s",
            category or "Exception",
        )
        raise RuntimeError(_PUBLIC_EXECUTION_ERROR) from None

    @staticmethod
    def _feature_unavailable(*, retryable: bool) -> str:
        return WatchlistsToolService._outcome(
            "feature_unavailable",
            (
                _TRANSIENT_UNAVAILABLE_MESSAGE
                if retryable
                else _PERMANENT_UNAVAILABLE_MESSAGE
            ),
            retryable=retryable,
        )

    @staticmethod
    def _outcome(status: str, message: str, *, retryable: bool) -> str:
        return WatchlistsToolService._finalize(
            {"status": status, "retryable": retryable, "message": message}
        )

    @staticmethod
    def _format_utc(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        normalized = value.astimezone(UTC).isoformat()
        return normalized.replace("+00:00", "Z")

    @staticmethod
    def _json(payload: object) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
