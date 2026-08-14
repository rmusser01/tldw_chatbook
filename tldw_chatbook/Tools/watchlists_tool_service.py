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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
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


class WatchlistsToolService:
    """Validate and orchestrate the shared read-only Watchlists tool core.

    Args:
        db_resolver: Synchronous callable returning the current database owner.
        runtime_source_loader: Synchronous callable returning current runtime state.
        clock: Optional UTC-aware clock used for traversal context timestamps.
    """

    def __init__(
        self,
        *,
        db_resolver: Callable[[], Any],
        runtime_source_loader: Callable[[], Any],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db_resolver = db_resolver
        self._runtime_source_loader = runtime_source_loader
        self._clock = clock or (lambda: datetime.now(UTC))

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
    def _exact_arguments(arguments: object, allowed: frozenset[str]) -> dict[str, Any]:
        if type(arguments) is not dict:
            raise _InvalidArgument("arguments must be an object")
        unknown = set(arguments) - allowed
        if unknown:
            raise _InvalidArgument("arguments contain unknown properties")
        return arguments

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
