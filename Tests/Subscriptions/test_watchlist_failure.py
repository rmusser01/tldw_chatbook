"""Stable, redacted Watchlists source-check failure outcomes."""

from __future__ import annotations

import json
import socket
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace

import httpx
import pytest
from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import AuthenticationError, RateLimitError
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.security import SSRFError
from tldw_chatbook.Subscriptions.watchlist_failure import (
    InvalidFeedError,
    WatchlistFailureCategory,
    classify_watchlist_failure,
)
from tldw_chatbook.Utils.egress import EgressBlockedError, EgressFetchError


PRODUCT_USER_AGENT = "tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)"


EXPECTED_COPY = {
    WatchlistFailureCategory.ACCESS_DENIED: (
        "The source denied access.",
        "Check whether this source permits automated access.",
    ),
    WatchlistFailureCategory.AUTHENTICATION_REQUIRED: (
        "The source requires authentication.",
        "Check the source credentials and authentication settings.",
    ),
    WatchlistFailureCategory.RATE_LIMITED: (
        "The source is rate limiting checks.",
        "Retry after the source's wait period.",
    ),
    WatchlistFailureCategory.INVALID_FEED: (
        "The source did not return a valid feed.",
        "Check the source URL and feed format.",
    ),
    WatchlistFailureCategory.CONNECTION_FAILURE: (
        "The source could not be reached.",
        "Retry when the network or source is available.",
    ),
    WatchlistFailureCategory.TEMPORARY_SERVER_ERROR: (
        "The source is temporarily unavailable.",
        "Retry later.",
    ),
    WatchlistFailureCategory.POLICY_BLOCKED: (
        "The source was blocked by the network safety policy.",
        "Choose a public HTTP(S) source allowed by the network safety policy.",
    ),
}


def _http_error(
    status: int,
    *,
    retry_after: str | None = None,
    message: str = "remote failure",
) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://source.example/feed")
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError(message, request=request, response=response)


@pytest.mark.parametrize(
    ("error", "category", "status", "retryable"),
    [
        (AuthenticationError("raw auth detail"), "authentication_required", None, False),
        (_http_error(401), "authentication_required", 401, False),
        (_http_error(403), "access_denied", 403, False),
        (RateLimitError("raw retry detail"), "rate_limited", None, True),
        (_http_error(429, retry_after="17"), "rate_limited", 429, True),
        (_http_error(500), "temporary_server_error", 500, True),
        (_http_error(502), "temporary_server_error", 502, True),
        (_http_error(503), "temporary_server_error", 503, True),
        (_http_error(504), "temporary_server_error", 504, True),
        (_http_error(404), "connection_failure", None, True),
        (TimeoutError("signed query and certificate detail"), "connection_failure", None, True),
        (ConnectionError("dns failed for secret.example"), "connection_failure", None, True),
        (socket.gaierror("private DNS detail"), "connection_failure", None, True),
        (
            httpx.ConnectError(
                "private certificate detail",
                request=httpx.Request("GET", "https://source.example"),
            ),
            "connection_failure",
            None,
            True,
        ),
        (
            InvalidFeedError("raw response body"),
            "invalid_feed",
            None,
            False,
        ),
        (
            json.JSONDecodeError("raw JSON body", "secret payload", 0),
            "invalid_feed",
            None,
            False,
        ),
        (ElementTree.ParseError("raw XML body"), "invalid_feed", None, False),
        (
            EgressBlockedError(
                "http://127.0.0.1/feed?token=SIGNED-QUERY", "private"
            ),
            "policy_blocked",
            None,
            False,
        ),
        (
            EgressFetchError(
                "redirect target blocked",
                url="https://source.example/feed?token=SIGNED-QUERY",
            ),
            "connection_failure",
            None,
            True,
        ),
        (SSRFError("private path and host"), "policy_blocked", None, False),
        (RuntimeError("unknown secret detail"), "connection_failure", None, True),
    ],
)
def test_classifier_maps_failures_to_the_exact_safe_vocabulary(
    error: BaseException,
    category: str,
    status: int | None,
    retryable: bool,
) -> None:
    failure = classify_watchlist_failure(error)

    assert failure.category.value == category
    assert failure.http_status == status
    assert failure.retryable is retryable
    assert (failure.message, failure.next_action) == EXPECTED_COPY[failure.category]
    assert len(failure.message.encode("utf-8")) <= 256
    assert len(failure.next_action.encode("utf-8")) <= 256


@pytest.mark.parametrize(
    ("error_type", "status", "category", "retry_after"),
    [
        (AuthenticationError, 401, "authentication_required", None),
        (RateLimitError, 429, "rate_limited", 23),
    ],
)
def test_owned_domain_errors_retain_structured_http_status_direct_and_wrapped(
    error_type, status: int, category: str, retry_after: int | None
) -> None:
    error = error_type("RAW-DOMAIN-ERROR-CANARY-22865")
    error.http_status = status
    error.retry_after_seconds = "23"
    wrapped = RuntimeError("RAW-WRAPPER-CANARY-22865")
    wrapped.__cause__ = error

    direct_failure = classify_watchlist_failure(error)
    wrapped_failure = classify_watchlist_failure(wrapped)

    for failure in (direct_failure, wrapped_failure):
        assert failure.category.value == category
        assert failure.http_status == status
        assert failure.retry_after_seconds == retry_after
        public = json.dumps(failure.__dict__, default=str)
        assert "RAW-" not in public


@pytest.mark.parametrize(
    ("error_type", "status"),
    [(AuthenticationError, 401.0), (RateLimitError, 429.0)],
)
def test_owned_domain_http_status_requires_an_exact_integer_type(
    error_type, status: object
) -> None:
    error = error_type("RAW-NONINTEGER-STATUS-CANARY-22865")
    error.http_status = status

    failure = classify_watchlist_failure(error)

    assert failure.http_status is None
    assert "RAW-" not in json.dumps(failure.__dict__, default=str)


def test_public_category_vocabulary_is_exact() -> None:
    assert {category.value for category in WatchlistFailureCategory} == {
        "access_denied",
        "authentication_required",
        "rate_limited",
        "invalid_feed",
        "connection_failure",
        "temporary_server_error",
        "policy_blocked",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("17", 17),
        (86_400, 86_400),
        ("Wed, 21 Oct 2015 07:28:00 GMT", None),
        ("-1", None),
        ("86401", None),
        ("9" * 5_000, None),
        (True, None),
        ("12\n13", None),
        (" 12", None),
        (1.5, None),
    ],
)
def test_retry_after_accepts_only_bounded_plain_integer_delays(
    value: object, expected: int | None
) -> None:
    error = RuntimeError("response canary")
    error.response = SimpleNamespace(  # type: ignore[attr-defined]
        status_code=429,
        headers={"Retry-After": value},
    )

    failure = classify_watchlist_failure(error)

    assert failure.category is WatchlistFailureCategory.RATE_LIMITED
    assert failure.retry_after_seconds == expected


def test_retry_after_is_discarded_for_a_non_retryable_http_category() -> None:
    failure = classify_watchlist_failure(_http_error(403, retry_after="17"))

    assert failure.category is WatchlistFailureCategory.ACCESS_DENIED
    assert failure.retry_after_seconds is None


def test_direct_fetch_block_wrapper_is_a_non_retryable_policy_failure() -> None:
    from tldw_chatbook.Subscriptions.monitoring_engine import FetchBlockedError

    failure = classify_watchlist_failure(
        FetchBlockedError("DIRECT-POLICY-CANARY-22865")
    )

    assert failure.category is WatchlistFailureCategory.POLICY_BLOCKED
    assert failure.retryable is False
    assert "DIRECT-POLICY-CANARY" not in json.dumps(failure.__dict__)


def test_wrapped_fetch_transport_error_is_connection_failure_without_raw_data() -> None:
    inner = EgressFetchError(
        "redirect without Location",
        url="https://source.example/feed?token=WRAPPED-FETCH-CANARY-22865",
    )
    outer = RuntimeError("WRAPPED-OUTER-FETCH-CANARY-22865")
    outer.__cause__ = inner

    failure = classify_watchlist_failure(outer)

    assert failure.category is WatchlistFailureCategory.CONNECTION_FAILURE
    assert failure.retryable is True
    assert "CANARY" not in json.dumps(failure.__dict__)


def test_wrapped_real_policy_error_remains_non_retryable_policy_blocked() -> None:
    from tldw_chatbook.Subscriptions.monitoring_engine import FetchBlockedError

    inner = FetchBlockedError("WRAPPED-POLICY-CANARY-22865")
    outer = RuntimeError("WRAPPED-POLICY-OUTER-CANARY-22865")
    outer.__cause__ = inner

    failure = classify_watchlist_failure(outer)

    assert failure.category is WatchlistFailureCategory.POLICY_BLOCKED
    assert failure.retryable is False
    assert "CANARY" not in json.dumps(failure.__dict__)


def test_exception_class_name_lookalikes_use_the_unknown_fallback() -> None:
    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class FetchBlockedError(Exception):
        pass

    for error in (
        AuthenticationError("SPOOF-AUTH-CANARY-22865"),
        RateLimitError("SPOOF-RATE-CANARY-22865"),
        FetchBlockedError("SPOOF-POLICY-CANARY-22865"),
    ):
        failure = classify_watchlist_failure(error)
        assert failure.category is WatchlistFailureCategory.CONNECTION_FAILURE
        assert failure.retryable is True
        assert "CANARY" not in json.dumps(failure.__dict__)


def test_classifier_finds_a_supported_failure_inside_a_safe_wrapper() -> None:
    inner = InvalidFeedError("WRAPPED-INNER-CANARY-22865")
    outer = RuntimeError("WRAPPED-OUTER-CANARY-22865")
    outer.__cause__ = inner

    failure = classify_watchlist_failure(outer)

    assert failure.category is WatchlistFailureCategory.INVALID_FEED
    assert "CANARY" not in json.dumps(failure.__dict__)


def test_classifier_redaction_is_anti_vacuous() -> None:
    query_canary = "SIGNED-QUERY-CANARY-22865"
    auth_canary = "AUTH-CANARY-22865"
    custom_header_canary = "CUSTOM-HEADER-CANARY-22865"
    body_canary = "BODY-CANARY-22865"
    path_canary = "/private/watchlists-CANARY-22865.db"
    certificate_canary = "CERTIFICATE-CANARY-22865"
    request = httpx.Request(
        "GET",
        f"https://source.example/feed?token={query_canary}",
        headers={
            "Authorization": f"Bearer {auth_canary}",
            "X-Feed-Token": custom_header_canary,
        },
    )
    response = httpx.Response(403, request=request, text=body_canary)
    error = httpx.HTTPStatusError(
        f"{path_canary} {certificate_canary}", request=request, response=response
    )

    # Anti-vacuity controls: every canary is present at the classifier boundary.
    assert query_canary in str(error.request.url)
    assert auth_canary in error.request.headers["Authorization"]
    assert custom_header_canary in error.request.headers["X-Feed-Token"]
    assert body_canary in error.response.text
    assert path_canary in str(error)
    assert certificate_canary in str(error)

    failure = classify_watchlist_failure(error)
    rendered = " ".join(
        (
            failure.category.value,
            failure.message,
            failure.next_action,
            str(failure.http_status),
            str(failure.retry_after_seconds),
        )
    )
    for canary in (
        query_canary,
        auth_canary,
        custom_header_canary,
        body_canary,
        path_canary,
        certificate_canary,
    ):
        assert canary not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_factory", "category", "retryable", "status", "retry_after"),
    [
        (lambda: _http_error(403, message="DURABLE-CANARY"), "access_denied", False, 403, None),
        (
            lambda: _http_error(401, message="DURABLE-CANARY"),
            "authentication_required",
            False,
            401,
            None,
        ),
        (
            lambda: _http_error(429, retry_after="23", message="DURABLE-CANARY"),
            "rate_limited",
            True,
            429,
            23,
        ),
        (lambda: InvalidFeedError("DURABLE-CANARY"), "invalid_feed", False, None, None),
        (
            lambda: ConnectionError("DURABLE-CANARY"),
            "connection_failure",
            True,
            None,
            None,
        ),
        (
            lambda: _http_error(503, message="DURABLE-CANARY"),
            "temporary_server_error",
            True,
            503,
            None,
        ),
        (
            lambda: EgressBlockedError(
                "https://source.example/feed?token=DURABLE-CANARY", "private"
            ),
            "policy_blocked",
            False,
            None,
            None,
        ),
    ],
)
async def test_service_persists_each_classification_without_raw_failure_data(
    tmp_path,
    error_factory,
    category: str,
    retryable: bool,
    status: int | None,
    retry_after: int | None,
) -> None:
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    error = error_factory()
    assert "DURABLE-CANARY" in (
        str(error)
        + str(getattr(error, "url", ""))
        + str(getattr(getattr(error, "request", None), "url", ""))
    ), "anti-vacuity: the sensitive canary must reach the service failure input"

    async def fail(_subscription):
        raise error

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=fail)
    source = await service.create_source(
        {
            "name": "Failure fixture",
            "url": "https://source.example/feed",
            "source_type": "rss",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_run = dict(
        db.conn.execute(
            "SELECT stats_json, error_msg, log_text FROM local_watchlist_runs WHERE id = ?",
            (launched["run_id"],),
        ).fetchone()
    )
    stats = json.loads(stored_run["stats_json"])
    expected_message, expected_action = EXPECTED_COPY[WatchlistFailureCategory(category)]
    assert {
        "failure_category": stats["failure_category"],
        "retryable": stats["retryable"],
        "http_status": stats["http_status"],
        "retry_after_seconds": stats["retry_after_seconds"],
        "next_action": stats["next_action"],
    } == {
        "failure_category": category,
        "retryable": retryable,
        "http_status": status,
        "retry_after_seconds": retry_after,
        "next_action": expected_action,
    }
    assert stored_run["error_msg"] == expected_message
    assert db.get_subscription(source["source_id"])["last_error"] == expected_message
    assert completed["failure_category"] == category
    assert completed["retryable"] is retryable
    assert completed["next_action"] == expected_action
    durable_and_public = json.dumps(
        {
            "stored": stored_run,
            "source": dict(db.get_subscription(source["source_id"])),
            "completed": completed,
        },
        default=str,
    )
    assert "DURABLE-CANARY" not in durable_and_public


@pytest.mark.asyncio
async def test_manual_and_accepted_checks_share_the_same_failure_projection(tmp_path) -> None:
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")

    async def fail(_subscription):
        raise RuntimeError("PARITY-CANARY /private/source.db?token=secret")

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=fail)
    source = await service.create_source(
        {
            "name": "Parity fixture",
            "url": "https://source.example/feed",
            "source_type": "rss",
        }
    )
    manual = await service.launch_run(source_id=source["source_id"])

    manual_result = await service.execute_run(manual["run_id"])
    accepted = await service.launch_run(source_id=source["source_id"])
    accepted_result = await service.execute_accepted_run(accepted["run_id"])

    fields = (
        "failure_category",
        "retryable",
        "http_status",
        "retry_after_seconds",
        "next_action",
        "error_msg",
    )
    assert {field: manual_result.get(field) for field in fields} == {
        field: accepted_result.get(field) for field in fields
    }
    assert "PARITY-CANARY" not in json.dumps(manual_result, default=str)
    assert "PARITY-CANARY" not in json.dumps(accepted_result, default=str)


@pytest.mark.asyncio
async def test_failed_executor_payload_is_sanitized_before_source_and_run_writes(
    tmp_path,
) -> None:
    canary = "EXECUTOR-PAYLOAD-CANARY-22865 /private/source.db?token=secret"

    async def fail_without_raising(_subscription):
        return {
            "status": "failed",
            "items": [],
            "error_msg": canary,
            "log_text": canary,
            "stats": {
                "failure_category": "invalid_feed",
                "retryable": True,
                "next_action": canary,
                "debug_response_body": canary,
                "items_found": 0,
                "items_ingested": 0,
                "max_withheld_pct": 10**1_000,
            },
        }

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(
        db_factory=lambda: db, run_executor=fail_without_raising
    )
    source = await service.create_source(
        {
            "name": "Injected failure fixture",
            "url": "https://source.example/feed",
            "source_type": "rss",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    stored_source = dict(db.get_subscription(source["source_id"]))
    stored_run = dict(
        db.conn.execute(
            "SELECT stats_json, error_msg, log_text FROM local_watchlist_runs "
            "WHERE id = ?",
            (launched["run_id"],),
        ).fetchone()
    )
    assert stored_source["last_error"] == "The source did not return a valid feed."
    assert completed["failure_category"] == "invalid_feed"
    assert completed["stats"]["items_found"] == 0
    assert "debug_response_body" not in completed["stats"]
    assert canary not in json.dumps(
        {"source": stored_source, "run": stored_run, "completed": completed},
        default=str,
    )


@pytest.mark.asyncio
async def test_legacy_and_tampered_failure_rows_normalize_to_fixed_safe_copy(tmp_path) -> None:
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Legacy fixture",
            "url": "https://source.example/feed",
            "source_type": "rss",
        }
    )
    with db.transaction() as conn:
        legacy_id = int(
            conn.execute(
                """
                INSERT INTO local_watchlist_runs (
                    source_id, status, error_msg, log_text, created_at, updated_at
                ) VALUES (?, 'failed', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    source["source_id"],
                    "LEGACY-RAW-CANARY /private/legacy.db",
                    "LEGACY-LOG-CANARY token=secret",
                ),
            ).lastrowid
        )
        tampered_id = int(
            conn.execute(
                """
                INSERT INTO local_watchlist_runs (
                    source_id, status, stats_json, error_msg, log_text,
                    created_at, updated_at
                ) VALUES (?, 'failed', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    source["source_id"],
                    json.dumps(
                        {
                            "failure_category": "policy_blocked",
                            "retryable": True,
                            "http_status": 999,
                            "retry_after_seconds": 999_999_999,
                            "next_action": "TAMPERED-ACTION-CANARY token=secret",
                            "response_body": "BODY-STATS-CANARY-22865",
                            "request_url": (
                                "https://source.example/feed?token="
                                "SIGNED-STATS-CANARY-22865"
                            ),
                            "authorization": "HEADER-STATS-CANARY-22865",
                            "database_path": "/private/PATH-STATS-CANARY-22865.db",
                            "items_found": 3,
                            "dispositions": {
                                "error": 1,
                                "raw_detail": "DISPOSITION-STATS-CANARY-22865",
                            },
                        }
                    ),
                    "TAMPERED-ERROR-CANARY /private/tampered.db",
                    "TAMPERED-LOG-CANARY",
                ),
            ).lastrowid
        )

    raw_tampered_stats = db.conn.execute(
        "SELECT stats_json FROM local_watchlist_runs WHERE id = ?", (tampered_id,)
    ).fetchone()["stats_json"]
    for canary in (
        "BODY-STATS-CANARY",
        "SIGNED-STATS-CANARY",
        "HEADER-STATS-CANARY",
        "PATH-STATS-CANARY",
        "DISPOSITION-STATS-CANARY",
    ):
        assert canary in raw_tampered_stats, (
            "anti-vacuity: the nested canary must reach the normalizer input"
        )

    legacy = await service.get_run(legacy_id)
    tampered = await service.get_run(tampered_id)

    assert legacy["failure_category"] is None
    assert legacy["retryable"] is False
    assert legacy["error_msg"] == "Watchlists source check failed."
    assert legacy["next_action"] == "Review the source configuration before trying again."
    assert tampered["failure_category"] == "policy_blocked"
    assert tampered["retryable"] is False
    assert tampered["http_status"] is None
    assert tampered["retry_after_seconds"] is None
    assert tampered["stats"]["items_found"] == 3
    assert tampered["stats"]["dispositions"] == {"error": 1}
    assert tampered["next_action"] == EXPECTED_COPY[
        WatchlistFailureCategory.POLICY_BLOCKED
    ][1]
    public = json.dumps({"legacy": legacy, "tampered": tampered}, default=str)
    for canary in (
        "LEGACY-RAW",
        "LEGACY-LOG",
        "TAMPERED-ACTION",
        "TAMPERED-ERROR",
        "TAMPERED-LOG",
        "BODY-STATS-CANARY",
        "SIGNED-STATS-CANARY",
        "HEADER-STATS-CANARY",
        "PATH-STATS-CANARY",
        "DISPOSITION-STATS-CANARY",
    ):
        assert canary not in public


def _install_mock_transport(monkeypatch, handler):
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    return real_async_client, transport


def _allow_mock_hosts(monkeypatch) -> None:
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(egress, "_resolve", lambda _host: ["93.184.216.34"])

    async def resolve(_host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    monkeypatch.setattr(egress, "get_cli_setting", lambda _s, _k=None, d=None: d)


@pytest.mark.asyncio
async def test_mock_transport_429_persists_bounded_retry_after_and_no_request_data(
    tmp_path, monkeypatch
) -> None:
    _allow_mock_hosts(monkeypatch)
    query_canary = "RATE-QUERY-CANARY-22865"
    header_canary = "RATE-HEADER-CANARY-22865"
    body_canary = "RATE-BODY-CANARY-22865"
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "19"},
            text=body_canary,
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Rate-limited feed",
            "url": f"https://source.example/feed?token={query_canary}",
            "source_type": "rss",
            "custom_headers": {"X-Feed-Token": header_canary},
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert seen_requests, "anti-vacuity: the MockTransport must receive the request"
    request = seen_requests[0]
    assert query_canary in str(request.url)
    assert request.headers["X-Feed-Token"] == header_canary
    assert request.headers["User-Agent"] == PRODUCT_USER_AGENT
    assert completed["failure_category"] == "rate_limited"
    assert completed["http_status"] == 429
    assert completed["retry_after_seconds"] == 19
    public = json.dumps(completed, default=str)
    for canary in (query_canary, header_canary, body_canary):
        assert canary not in public


@pytest.mark.asyncio
async def test_mock_transport_401_retains_status_without_request_or_response_data(
    tmp_path, monkeypatch
) -> None:
    _allow_mock_hosts(monkeypatch)
    query_canary = "AUTH-QUERY-CANARY-22865"
    request_header_canary = "AUTH-REQUEST-HEADER-CANARY-22865"
    response_header_canary = "AUTH-RESPONSE-HEADER-CANARY-22865"
    body_canary = "AUTH-BODY-CANARY-22865"
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            401,
            headers={"X-Response-Secret": response_header_canary},
            text=body_canary,
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Authenticated feed",
            "url": f"https://source.example/feed?token={query_canary}",
            "source_type": "rss",
            "custom_headers": {"X-Feed-Token": request_header_canary},
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert seen_requests, "anti-vacuity: the MockTransport must receive the request"
    request = seen_requests[0]
    assert query_canary in str(request.url)
    assert request.headers["X-Feed-Token"] == request_header_canary
    assert request.headers["User-Agent"] == PRODUCT_USER_AGENT
    assert completed["failure_category"] == "authentication_required"
    assert completed["http_status"] == 401
    assert completed["retryable"] is False
    assert completed["retry_after_seconds"] is None
    public = json.dumps(completed, default=str)
    for canary in (
        query_canary,
        request_header_canary,
        response_header_canary,
        body_canary,
    ):
        assert canary not in public


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_type", "content_type", "body"),
    [
        ("rss", "application/xml", "<rss><channel>"),
        ("rss", "application/xml", "<html><body>not a feed</body></html>"),
        ("json_feed", "application/json", "{"),
        ("json_feed", "application/json", "{}"),
    ],
)
async def test_malformed_and_nonfeed_payloads_are_invalid_feed_failures(
    tmp_path, monkeypatch, source_type: str, content_type: str, body: str
) -> None:
    _allow_mock_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": content_type},
            text=body,
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Invalid feed fixture",
            "url": "https://source.example/feed",
            "source_type": source_type,
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "failed"
    assert completed["failure_category"] == "invalid_feed"
    assert completed["retryable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"items": [None]},
        {"items": ["NESTED-ITEM-CANARY-22865"]},
        {"items": [{"author": "NESTED-AUTHOR-CANARY-22865"}]},
        {"items": [{"authors": "NESTED-AUTHORS-LIST-CANARY-22865"}]},
        {"items": [{"authors": ["NESTED-AUTHORS-ENTRY-CANARY-22865"]}]},
        {"items": [{"attachments": "NESTED-ATTACHMENTS-LIST-CANARY-22865"}]},
        {
            "items": [
                {"attachments": ["NESTED-ATTACHMENTS-ENTRY-CANARY-22865"]}
            ]
        },
    ],
)
async def test_malformed_nested_json_feed_shapes_are_safe_invalid_feed_failures(
    tmp_path, monkeypatch, payload: object
) -> None:
    _allow_mock_hosts(monkeypatch)
    body = json.dumps(payload)
    served_bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served_bodies.append(body)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            text=body,
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Nested invalid JSON feed fixture",
            "url": "https://source.example/feed",
            "source_type": "json_feed",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert served_bodies == [body], "anti-vacuity: the malformed payload was fetched"
    assert completed["status"] == "failed"
    assert completed["failure_category"] == "invalid_feed"
    assert completed["retryable"] is False
    public = json.dumps(completed, default=str)
    assert "NESTED-" not in public


@pytest.mark.asyncio
async def test_parser_logs_do_not_include_raw_exception_text(monkeypatch) -> None:
    from tldw_chatbook.Subscriptions import monitoring_engine as me

    _allow_mock_hosts(monkeypatch)
    body_canary = "PARSER-BODY-CANARY-22865"
    exception_canary = "PARSER-EXCEPTION-CANARY-22865"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/xml"},
            text=f"<rss>{body_canary}</rss>",
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)

    def fail_parse(content: str):
        assert body_canary in content
        raise me.ET.ParseError(exception_canary)

    monkeypatch.setattr(me.ET, "fromstring", fail_parse)
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)))
    try:
        with pytest.raises(me.ET.ParseError) as exc_info:
            await me.FeedMonitor().check_feed(
                {
                    "id": 1,
                    "source": "https://source.example/feed",
                    "type": "rss",
                    "auth_config": None,
                }
            )
    finally:
        logger.remove(sink_id)

    assert exception_canary in str(exc_info.value), (
        "anti-vacuity: the raw exception text must reach the logging boundary"
    )
    assert exception_canary not in "".join(captured)


@pytest.mark.parametrize("parser_name", ["_parse_rss_item", "_parse_atom_entry"])
def test_item_parser_logs_do_not_include_raw_exception_text(
    monkeypatch, parser_name: str
) -> None:
    from tldw_chatbook.Subscriptions import monitoring_engine as me

    exception_canary = f"ITEM-PARSER-CANARY-22865-{parser_name}"
    monitor = me.FeedMonitor()

    def fail_text(*_args, **_kwargs):
        raise RuntimeError(exception_canary)

    monkeypatch.setattr(monitor, "_get_text", fail_text)
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)))
    try:
        result = getattr(monitor, parser_name)(me.ET.fromstring("<item />"))
    finally:
        logger.remove(sink_id)

    assert result is None
    assert exception_canary not in "".join(captured)


@pytest.mark.asyncio
async def test_failure_bookkeeping_log_does_not_include_resolution_exception(
    tmp_path, monkeypatch
) -> None:
    canary = "RUN-RESOLUTION-LOG-CANARY-22865 /private/watchlists.db"
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Resolution logging fixture",
            "url": "https://source.example/feed",
            "source_type": "rss",
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    async def fail_get_run(_run_id):
        raise RuntimeError(canary)

    monkeypatch.setattr(service, "get_run", fail_get_run)
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)))
    try:
        with pytest.raises(RuntimeError, match=canary):
            await service.record_run_failure(
                launched["run_id"], error=ConnectionError("safe test failure")
            )
    finally:
        logger.remove(sink_id)

    assert canary not in "".join(captured)


@pytest.mark.asyncio
async def test_all_fetch_paths_send_the_exact_product_user_agent(monkeypatch, tmp_path) -> None:
    from tldw_chatbook.Subscriptions import monitoring_engine as me

    _allow_mock_hosts(monkeypatch)
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.headers.get("User-Agent", "")
        if request.headers.get("User-Agent") != PRODUCT_USER_AGENT:
            return httpx.Response(403, request=request)
        if request.url.path == "/feed":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/xml"},
                text="<rss><channel /></rss>",
                request=request,
            )
        if request.url.path == "/page":
            return httpx.Response(200, text="plain page", request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    "<urlset><url><loc>https://source.example/page</loc>"
                    "</url></urlset>"
                ),
                request=request,
            )
        return httpx.Response(200, json={"items": []}, request=request)

    real_async_client, transport = _install_mock_transport(monkeypatch, handler)
    async with real_async_client(transport=transport) as control_client:
        control = await control_client.get("https://source.example/control")
    assert control.status_code == 403
    assert seen["/control"].startswith("python-httpx/")

    await me.FeedMonitor().check_feed(
        {
            "id": 1,
            "source": "https://source.example/feed",
            "type": "rss",
            "auth_config": None,
        }
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    await me.URLMonitor(db)._fetch_url_content(
        {
            "source": "https://source.example/page",
            "extraction_method": "raw",
        }
    )
    await LocalWatchlistsService._items_for_api_source(
        {"source": "https://source.example/api"}
    )
    await LocalWatchlistsService._urls_for_sitemap(
        {"source": "https://source.example/sitemap.xml"}
    )

    assert seen == {
        "/control": seen["/control"],
        "/feed": PRODUCT_USER_AGENT,
        "/page": PRODUCT_USER_AGENT,
        "/api": PRODUCT_USER_AGENT,
        "/sitemap.xml": PRODUCT_USER_AGENT,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["rss", "url", "url_list", "sitemap", "api"])
async def test_guarded_fetch_container_errors_are_safe_connection_failures(
    tmp_path, monkeypatch, source_type: str
) -> None:
    _allow_mock_hosts(monkeypatch)
    query_canary = f"{source_type.upper()}-FETCH-CANARY-22865"
    source_url = f"https://source.example/{source_type}?token={query_canary}"
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(302, request=request)

    _install_mock_transport(monkeypatch, handler)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    payload = {
        "name": f"Guarded fetch {source_type}",
        "url": source_url,
        "source_type": source_type,
    }
    if source_type == "url_list":
        payload["extraction_rules"] = {"urls": [source_url]}
    source = await service.create_source(payload)
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert seen_requests, "anti-vacuity: the production builder must issue a request"
    assert any(query_canary in str(request.url) for request in seen_requests)
    assert completed["status"] == "failed"
    assert completed["failure_category"] == "connection_failure"
    assert completed["retryable"] is True
    assert completed["http_status"] is None
    assert query_canary not in json.dumps(completed, default=str)


@pytest.mark.asyncio
async def test_feed_user_agent_override_survives_cross_origin_redirect(
    monkeypatch,
) -> None:
    from tldw_chatbook.Subscriptions import monitoring_engine as me

    _allow_mock_hosts(monkeypatch)
    custom_user_agent = "custom-feed-client/2.0"
    final_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "source.example":
            return httpx.Response(
                302,
                headers={"Location": "https://mirror.example/feed"},
                request=request,
            )
        final_requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/xml"},
            text="<rss><channel /></rss>",
            request=request,
        )

    _install_mock_transport(monkeypatch, handler)
    await me.FeedMonitor().check_feed(
        {
            "id": 1,
            "source": "https://source.example/feed",
            "type": "rss",
            "auth_config": None,
            "custom_headers": json.dumps({"User-Agent": custom_user_agent}),
        }
    )

    assert final_requests
    assert final_requests[0].headers["User-Agent"] == custom_user_agent


@pytest.mark.asyncio
async def test_all_error_url_list_run_keeps_a_safe_classified_recovery(
    tmp_path, monkeypatch
) -> None:
    canary = "URL-LIST-ERROR-CANARY-22865"

    class FakeURLMonitor:
        def __init__(self, _db):
            pass

        async def check_url(self, _subscription):
            raise _http_error(503, message=canary)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "All-error URL list",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": [
                    "https://source.example/a",
                    "https://source.example/b",
                ]
            },
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "failed"
    assert completed["stats"]["dispositions"]["error"] == 2
    assert completed["failure_category"] == "temporary_server_error"
    assert completed["retryable"] is True
    assert completed["http_status"] == 503
    assert completed["next_action"] == "Retry later."
    stored_source = db.get_subscription(source["source_id"])
    assert stored_source["last_error"] == "The source is temporarily unavailable."
    assert canary not in json.dumps(completed, default=str)
    assert canary not in str(stored_source["last_error"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statuses", "retry_delays", "expected_status", "expected_retry"),
    [
        ((500, 503), (None, None), None, None),
        ((429, 429), ("11", "29"), 429, None),
        ((429, 429), ("17", "17"), 429, 17),
    ],
)
async def test_uniform_url_list_category_carries_only_unanimous_per_url_metadata(
    tmp_path,
    monkeypatch,
    statuses,
    retry_delays,
    expected_status,
    expected_retry,
) -> None:
    class FakeURLMonitor:
        def __init__(self, _db):
            pass

        async def check_url(self, subscription):
            path = httpx.URL(subscription["source"]).path
            index = 0 if path == "/a" else 1
            raise _http_error(
                statuses[index],
                message=f"UNIFORM-{path}-CANARY-22865",
                retry_after=retry_delays[index],
            )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def run(urls: list[str]) -> dict:
        source = await service.create_source(
            {
                "name": f"Uniform URL list {urls[0][-1]}",
                "source_type": "url_list",
                "extraction_rules": {"urls": urls},
            }
        )
        launched = await service.launch_run(source_id=source["source_id"])
        return await service.execute_run(launched["run_id"])

    forward = await run(
        ["https://source.example/a", "https://source.example/b"]
    )
    reverse = await run(
        ["https://source.example/b", "https://source.example/a"]
    )

    for completed in (forward, reverse):
        expected_category = (
            "rate_limited" if statuses[0] == 429 else "temporary_server_error"
        )
        assert completed["failure_category"] == expected_category
        assert completed["retryable"] is True
        assert completed["http_status"] == expected_status
        assert completed["retry_after_seconds"] == expected_retry
        assert completed["stats"]["dispositions"]["error"] == 2
        assert "UNIFORM-" not in json.dumps(completed, default=str)


@pytest.mark.asyncio
async def test_url_list_recovery_aggregates_failures_without_treating_skips_as_errors(
    tmp_path, monkeypatch
) -> None:
    class FakeURLMonitor:
        def __init__(self, _db):
            pass

        async def check_url(self, subscription):
            if httpx.URL(subscription["source"]).path == "/failed":
                raise _http_error(503, message="ERROR-WITH-SKIP-CANARY-22865")
            return None, {
                "kind": "skipped_in_flight",
                "reason": None,
                "withheld_percentage": None,
            }

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "Error plus skip",
            "source_type": "url_list",
            "extraction_rules": {
                "urls": [
                    "https://source.example/failed",
                    "https://source.example/skipped",
                ]
            },
        }
    )
    launched = await service.launch_run(source_id=source["source_id"])

    completed = await service.execute_run(launched["run_id"])

    assert completed["status"] == "failed"
    assert completed["failure_category"] == "temporary_server_error"
    assert completed["retryable"] is True
    assert completed["http_status"] == 503
    assert completed["stats"]["dispositions"]["error"] == 1
    assert completed["stats"]["dispositions"]["skipped"] == 1
    assert "ERROR-WITH-SKIP-CANARY" not in json.dumps(completed, default=str)


@pytest.mark.asyncio
async def test_mixed_category_url_list_failure_is_generic_and_order_independent(
    tmp_path, monkeypatch
) -> None:
    canaries = {
        "/a": "MIXED-SERVER-CANARY-22865",
        "/b": "MIXED-ACCESS-CANARY-22865",
    }

    class FakeURLMonitor:
        def __init__(self, _db):
            pass

        async def check_url(self, subscription):
            path = httpx.URL(subscription["source"]).path
            status = 503 if path == "/a" else 403
            raise _http_error(status, message=canaries[path])

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.URLMonitor",
        FakeURLMonitor,
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)

    async def run(urls: list[str]) -> dict:
        source = await service.create_source(
            {
                "name": f"Mixed URL list {len(urls)} {urls[0][-1]}",
                "source_type": "url_list",
                "extraction_rules": {"urls": urls},
            }
        )
        launched = await service.launch_run(source_id=source["source_id"])
        return await service.execute_run(launched["run_id"])

    forward = await run(
        ["https://source.example/a", "https://source.example/b"]
    )
    reverse = await run(
        ["https://source.example/b", "https://source.example/a"]
    )

    for completed in (forward, reverse):
        assert completed["status"] == "failed"
        assert completed["stats"]["dispositions"]["error"] == 2
        assert completed["failure_category"] is None
        assert completed["retryable"] is False
        assert completed["error_msg"] == "Watchlists source check failed."
        assert completed["next_action"] == (
            "Review the source configuration before trying again."
        )
        public = json.dumps(completed, default=str)
        assert all(canary not in public for canary in canaries.values())
