"""Stable, user-safe Watchlists source-check failure outcomes."""

from __future__ import annotations

import json
import math
import socket
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

import httpx

from ..DB.Subscriptions_DB import AuthenticationError, RateLimitError
from ..Utils.egress import EgressBlockedError
from .security import SecurityError


_MAX_RETRY_AFTER_SECONDS = 86_400
_MAX_RETRY_AFTER_DIGITS = len(str(_MAX_RETRY_AFTER_SECONDS))
_MAX_FAILED_RUN_COUNTER = (1 << 63) - 1
_FAILED_RUN_COUNTER_FIELDS = frozenset(
    {
        "items_found",
        "found_count",
        "found",
        "items_ingested",
        "processed_count",
        "new_items_found",
        "items_filtered",
        "filtered_count",
        "error_count",
        "items_errored",
        "response_time_ms",
        "bytes_transferred",
    }
)
_DISPOSITION_COUNTER_FIELDS = (
    "changed",
    "unchanged",
    "withheld",
    "baseline",
    "rebaselined",
    "error",
    "skipped",
)
LEGACY_FAILURE_MESSAGE = "Watchlists source check failed."
LEGACY_FAILURE_NEXT_ACTION = "Review the source configuration before trying again."


class WatchlistFailureCategory(StrEnum):
    """Stable machine categories for source-check failures."""

    ACCESS_DENIED = "access_denied"
    AUTHENTICATION_REQUIRED = "authentication_required"
    RATE_LIMITED = "rate_limited"
    INVALID_FEED = "invalid_feed"
    CONNECTION_FAILURE = "connection_failure"
    TEMPORARY_SERVER_ERROR = "temporary_server_error"
    POLICY_BLOCKED = "policy_blocked"


@dataclass(frozen=True)
class WatchlistFailure:
    """One bounded source-check failure suitable for persistence and display."""

    category: WatchlistFailureCategory
    message: str
    retryable: bool
    http_status: int | None
    retry_after_seconds: int | None
    next_action: str


class InvalidFeedError(ValueError):
    """A fetched payload is not a supported feed."""


class WatchlistPolicyFailure:
    """Marker for an owned wrapper around a concrete network-policy block."""


_COPY: dict[WatchlistFailureCategory, tuple[str, bool, str]] = {
    WatchlistFailureCategory.ACCESS_DENIED: (
        "The source denied access.",
        False,
        "Check whether this source permits automated access.",
    ),
    WatchlistFailureCategory.AUTHENTICATION_REQUIRED: (
        "The source requires authentication.",
        False,
        "Check the source credentials and authentication settings.",
    ),
    WatchlistFailureCategory.RATE_LIMITED: (
        "The source is rate limiting checks.",
        True,
        "Retry after the source's wait period.",
    ),
    WatchlistFailureCategory.INVALID_FEED: (
        "The source did not return a valid feed.",
        False,
        "Check the source URL and feed format.",
    ),
    WatchlistFailureCategory.CONNECTION_FAILURE: (
        "The source could not be reached.",
        True,
        "Retry when the network or source is available.",
    ),
    WatchlistFailureCategory.TEMPORARY_SERVER_ERROR: (
        "The source is temporarily unavailable.",
        True,
        "Retry later.",
    ),
    WatchlistFailureCategory.POLICY_BLOCKED: (
        "The source was blocked by the network safety policy.",
        False,
        "Choose a public HTTP(S) source allowed by the network safety policy.",
    ),
}


def _error_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 16:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        return status
    status = getattr(error, "http_status", None)
    if type(status) is int and isinstance(error, AuthenticationError) and status == 401:
        return 401
    if type(status) is int and isinstance(error, RateLimitError) and status == 429:
        return 429
    return None


def _bounded_retry_after(value: Any) -> int | None:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        if len(value) > _MAX_RETRY_AFTER_DIGITS:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if 0 <= parsed <= _MAX_RETRY_AFTER_SECONDS else None


def _retry_after(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value: Any = None
    if isinstance(headers, Mapping):
        value = headers.get("Retry-After")
        if value is None:
            value = headers.get("retry-after")
    if value is None:
        value = getattr(error, "retry_after_seconds", None)
    return _bounded_retry_after(value)


def _category_for(
    error: BaseException, status: int | None
) -> WatchlistFailureCategory | None:
    if _is_policy_error(error):
        return WatchlistFailureCategory.POLICY_BLOCKED
    if isinstance(error, (InvalidFeedError, json.JSONDecodeError, ElementTree.ParseError)):
        return WatchlistFailureCategory.INVALID_FEED
    if status == 401 or isinstance(error, AuthenticationError):
        return WatchlistFailureCategory.AUTHENTICATION_REQUIRED
    if status == 403:
        return WatchlistFailureCategory.ACCESS_DENIED
    if status == 429 or isinstance(error, RateLimitError):
        return WatchlistFailureCategory.RATE_LIMITED
    if status in {500, 502, 503, 504}:
        return WatchlistFailureCategory.TEMPORARY_SERVER_ERROR
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            socket.gaierror,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RequestError,
        ),
    ):
        return WatchlistFailureCategory.CONNECTION_FAILURE
    return None


def _is_policy_error(error: BaseException) -> bool:
    """Recognize egress policy errors without importing the monitor wrapper."""
    return isinstance(
        error,
        (
            EgressBlockedError,
            SecurityError,
            WatchlistPolicyFailure,
        ),
    )


def classify_watchlist_failure(error: BaseException) -> WatchlistFailure:
    """Map one internal failure to a bounded user-safe domain outcome."""
    chain = _error_chain(error)
    policy_error = next(
        (
            candidate
            for candidate in reversed(chain)
            if _is_policy_error(candidate)
        ),
        None,
    )
    category: WatchlistFailureCategory | None = None
    status: int | None = None
    if policy_error is not None:
        category = WatchlistFailureCategory.POLICY_BLOCKED
    else:
        for candidate in chain:
            candidate_status = _http_status(candidate)
            candidate_category = _category_for(candidate, candidate_status)
            if candidate_category is not None:
                category = candidate_category
                status = candidate_status
                break
    if category is None:
        category = WatchlistFailureCategory.CONNECTION_FAILURE
    message, retryable, next_action = _COPY[category]
    retry_after = None
    if category in {
        WatchlistFailureCategory.RATE_LIMITED,
        WatchlistFailureCategory.TEMPORARY_SERVER_ERROR,
    }:
        retry_after = next(
            (
                candidate_retry
                for candidate in chain
                if (candidate_retry := _retry_after(candidate)) is not None
            ),
            None,
        )
    return WatchlistFailure(
        category=category,
        message=message,
        retryable=retryable,
        http_status=status,
        retry_after_seconds=retry_after,
        next_action=next_action,
    )


def watchlist_failure_stats(failure: WatchlistFailure) -> dict[str, Any]:
    """Return the bounded machine fields persisted in a run's ``stats_json``."""
    return {
        "failure_category": failure.category.value,
        "retryable": failure.retryable,
        "http_status": failure.http_status,
        "retry_after_seconds": failure.retry_after_seconds,
        "next_action": failure.next_action,
    }


def watchlist_failure_from_stats(
    stats: Mapping[str, Any] | None,
) -> WatchlistFailure | None:
    """Validate stored fields and rebuild their fixed safe presentation."""
    if not isinstance(stats, Mapping):
        return None
    try:
        category = WatchlistFailureCategory(stats.get("failure_category"))
    except (TypeError, ValueError):
        return None
    message, retryable, next_action = _COPY[category]
    status = stats.get("http_status")
    allowed_statuses = {
        WatchlistFailureCategory.AUTHENTICATION_REQUIRED: {401},
        WatchlistFailureCategory.ACCESS_DENIED: {403},
        WatchlistFailureCategory.RATE_LIMITED: {429},
        WatchlistFailureCategory.TEMPORARY_SERVER_ERROR: {500, 502, 503, 504},
    }.get(category, set())
    if type(status) is not int or status not in allowed_statuses:
        status = None
    retry_after = _bounded_retry_after(stats.get("retry_after_seconds"))
    if category not in {
        WatchlistFailureCategory.RATE_LIMITED,
        WatchlistFailureCategory.TEMPORARY_SERVER_ERROR,
    }:
        retry_after = None
    return WatchlistFailure(
        category=category,
        message=message,
        retryable=retryable,
        http_status=status,
        retry_after_seconds=retry_after,
        next_action=next_action,
    )


def sanitize_watchlist_failure_stats(
    stats: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], WatchlistFailure | None]:
    """Allowlist failed-run counters and canonical recovery machine fields."""
    if not isinstance(stats, Mapping):
        return {}, None
    failure = watchlist_failure_from_stats(stats)
    safe_stats: dict[str, Any] = {}
    for key in _FAILED_RUN_COUNTER_FIELDS:
        value = stats.get(key)
        if type(value) is int and 0 <= value <= _MAX_FAILED_RUN_COUNTER:
            safe_stats[key] = value

    dispositions = stats.get("dispositions")
    if isinstance(dispositions, Mapping):
        safe_dispositions = {
            key: value
            for key in _DISPOSITION_COUNTER_FIELDS
            if type(value := dispositions.get(key)) is int
            and 0 <= value <= _MAX_FAILED_RUN_COUNTER
        }
        if safe_dispositions:
            safe_stats["dispositions"] = safe_dispositions

    max_withheld = stats.get("max_withheld_pct")
    if type(max_withheld) is int:
        valid_max_withheld = 0 <= max_withheld <= 100
    elif type(max_withheld) is float:
        valid_max_withheld = math.isfinite(max_withheld) and 0 <= max_withheld <= 100
    else:
        valid_max_withheld = False
    if valid_max_withheld:
        safe_stats["max_withheld_pct"] = max_withheld

    if failure is not None:
        safe_stats.update(watchlist_failure_stats(failure))
    return safe_stats, failure


def project_watchlist_failure(
    value: Mapping[str, Any] | None, *, failed: bool
) -> dict[str, Any] | None:
    """Project persisted failure metadata into fixed public recovery fields."""
    if not failed:
        return None
    candidates: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        stats = value.get("stats")
        if isinstance(stats, Mapping):
            candidates.append(stats)
        stats_json = value.get("stats_json")
        if isinstance(stats_json, Mapping):
            candidates.append(stats_json)
        elif isinstance(stats_json, str):
            try:
                parsed = json.loads(stats_json)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                candidates.append(parsed)
        candidates.append(value)
    failure = next(
        (
            candidate_failure
            for candidate in candidates
            if (candidate_failure := watchlist_failure_from_stats(candidate))
            is not None
        ),
        None,
    )
    if failure is None:
        return {
            "error_category": "source_check_failed",
            "error_message": LEGACY_FAILURE_MESSAGE,
            "retry_capable": False,
            "http_status": None,
            "retry_after_seconds": None,
            "next_action": LEGACY_FAILURE_NEXT_ACTION,
        }
    return {
        "error_category": failure.category.value,
        "error_message": failure.message,
        "retry_capable": failure.retryable,
        "http_status": failure.http_status,
        "retry_after_seconds": failure.retry_after_seconds,
        "next_action": failure.next_action,
    }
