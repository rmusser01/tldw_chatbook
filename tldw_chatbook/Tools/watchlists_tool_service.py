"""Shared synchronous service for read-only Watchlists agent tools."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDBUnavailableError

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
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_CANONICAL_SCOPE_RE = re.compile(
    r"local:(?P<kind>subscription|watchlist):(?P<id>[1-9][0-9]*)\Z"
)
_CANONICAL_ITEM_RE = re.compile(r"local:watchlist_item:(?P<id>[1-9][0-9]*)\Z")
_COMPOSITE_ID_RE = re.compile(r"[^:\s]+:(?:subscription|watchlist|watchlist_item):.*\Z")
_CANDIDATE_LIMIT = 10
_MAX_SQLITE_ROW_ID = 2**63 - 1


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
            request = self._validate_search(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)

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

        as_of = self._format_utc(self._clock())
        page = database.search_items_for_agent(
            query=request.query,
            subscription_id=int(source["id"]) if source is not None else None,
            watchlist_id=(int(collection["id"]) if collection is not None else None),
            statuses=request.statuses,
            since=request.since,
            limit=request.limit,
        )
        rows = page["items"]
        memberships = database.get_source_collection_memberships(
            [int(row["subscription_id"]) for row in rows]
        )
        items = [
            self._shape_search_item(
                row,
                memberships.get(
                    int(row["subscription_id"]),
                    {"collections": [], "has_more": False},
                ),
            )
            for row in rows
        ]
        return self._json(
            {
                "status": "ok",
                "query_mode": (
                    "literal_full_text" if request.query is not None else "browse"
                ),
                "ordering": _ORDERING,
                "as_of": as_of,
                "snapshot_max_item_id": page["snapshot_max_item_id"],
                "returned_count": len(items),
                "has_more": bool(page["has_more"]),
                "next_cursor": None,
                "scope": {
                    "collection": self._shape_collection(collection),
                    "source": self._shape_selected_source(source),
                },
                "items": items,
            }
        )

    def get_item(self, arguments: object) -> str:
        """Return authoritative detail for one canonical Watchlists item ID.

        Args:
            arguments: Raw tool argument object containing ``item_id``.

        Returns:
            JSON text containing a structured domain outcome.
        """
        try:
            item_id = self._validate_detail(arguments)
        except _InvalidArgument as exc:
            return self._outcome("invalid_argument", str(exc), retryable=False)

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
        item["evidence"] = {
            "content_is_untrusted": True,
            "content": row["content"],
        }
        return self._json({"status": "ok", "item": item})

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
        WatchlistsToolService._validate_cursor(
            values.get("cursor"), supplied="cursor" in values
        )
        return _SearchRequest(
            query=query,
            collection=collection,
            source=source,
            statuses=statuses,
            since=since,
            limit=limit,
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
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
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
    def _validate_cursor(value: object, *, supplied: bool) -> None:
        if not supplied:
            return
        if type(value) is not str:
            raise _InvalidArgument("cursor must be a string")
        if value.strip():
            raise _InvalidArgument("cursor support is unavailable until pagination")

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
        except Exception:  # noqa: BLE001 - resolver failures are scrubbed outcomes
            return None, self._feature_unavailable(retryable=True)

        if database is None:
            return None, self._feature_unavailable(retryable=False)
        readiness_check = getattr(database, "assert_agent_read_ready", None)
        if callable(readiness_check):
            try:
                readiness_check()
            except SubscriptionsDBUnavailableError:
                return None, self._feature_unavailable(retryable=False)
            except Exception:  # noqa: BLE001 - readiness may recover on another call
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
            return None, WatchlistsToolService._json(
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
        if field == "source":
            return {
                "id": f"local:subscription:{row['id']}",
                "name": row["name"],
                "url": row["source"],
            }
        return {
            "id": f"local:watchlist:{row['id']}",
            "name": row["name"],
        }

    @staticmethod
    def _shape_collection(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": f"local:watchlist:{row['id']}",
            "name": row["name"],
        }

    @staticmethod
    def _shape_selected_source(
        row: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": f"local:subscription:{row['id']}",
            "name": row["name"],
            "type": row["type"],
            "url": row["source"],
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
        return {
            "id": f"local:watchlist_item:{row['id']}",
            "title": row["title"],
            "url": row["url"],
            "author": row["author"],
            "status": row["status"],
            "effective_date": row["effective_date"],
            "published_date": row["published_date"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "source": {
                "id": f"local:subscription:{row['subscription_id']}",
                "name": row["subscription_name"],
                "type": row["subscription_type"],
                "url": row["subscription_source"],
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
        row: Mapping[str, Any], membership: Mapping[str, Any]
    ) -> dict[str, Any]:
        item = WatchlistsToolService._shape_item_metadata(row, membership)
        item["evidence"] = {
            "content_is_untrusted": True,
            "snippet": row["content_match_context"],
        }
        return item

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
        return WatchlistsToolService._json(
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
