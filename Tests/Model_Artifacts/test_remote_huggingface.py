"""Tests for bounded Hugging Face metadata search (TASK-596.1)."""

from __future__ import annotations

from collections.abc import Callable
import traceback

import httpx
import pytest

from tldw_chatbook.Model_Artifacts.remote_huggingface import (
    HuggingFaceRemoteAdapter,
    RemoteDiscoveryError,
    RemoteModelSummary,
    is_exact_repository,
)


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[[], httpx.AsyncClient]:
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_search_trims_query_and_uses_fixed_bounded_request() -> None:
    """Catches a wrong endpoint, untrimmed query, or unbounded search."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    assert await adapter.search("  whisper  ", token="secret") == ()
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url).split("?")[0] == "https://huggingface.co/api/models"
    assert requests[0].url.params["search"] == "whisper"
    assert requests[0].url.params["limit"] == "50"
    assert requests[0].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_search_disables_redirects() -> None:
    """Catches a metadata request that could forward credentials via redirects."""
    recorded: list[bool] = []

    class TrackingClient(httpx.AsyncClient):
        def stream(self, method: str, url: str | httpx.URL, **kwargs: object):
            recorded.append(bool(kwargs["follow_redirects"]))
            return super().stream(method, url, **kwargs)

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: TrackingClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
        )
    )

    assert await adapter.search("whisper", token="secret") == ()
    assert recorded == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "code", "retryable"),
    [
        (401, "authentication_required", False),
        (403, "access_forbidden", False),
        (404, "repository_not_found", False),
        (429, "rate_limited", True),
    ],
)
async def test_search_sanitizes_expected_http_errors(
    status_code: int, code: str, retryable: bool
) -> None:
    """Catches raw upstream error propagation or a wrong retry policy."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(status_code, text="upstream secret detail")
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert raised.value.details == ()
    assert "upstream secret detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_search_sanitizes_timeout() -> None:
    """Catches timeout leakage instead of a recoverable discovery failure."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream secret", request=request)

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "network_error",
        True,
        (),
    )
    assert "upstream secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_search_timeout_traceback_has_no_upstream_secret_or_cause() -> None:
    """Catches chained HTTPX errors that expose credentials or upstream text."""
    secret = "upstream-secret-and-token"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper", token=secret)

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"metadata":"json-secret-marker",}',
        b'{"metadata":"utf8-secret-marker"}\xff',
    ],
    ids=["malformed-json", "invalid-utf8"],
)
async def test_search_parser_failures_have_no_cause_or_upstream_marker(
    body: bytes,
) -> None:
    """Catches parser exception chains exposing an upstream response body."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, content=body))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "secret-marker" not in rendered


@pytest.mark.asyncio
async def test_search_rejects_redirect_status_as_remote_error() -> None:
    """Catches a redirect response being parsed as trusted search metadata."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(302, json=[]))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "remote_error",
        False,
        (),
    )


@pytest.mark.parametrize(
    ("code", "details"),
    [
        ("unexpected", ()),
        ("invalid_response", ("x" * 513,)),
        ("invalid_response", ("line\nbreak",)),
        ("invalid_response", tuple("warning" for _ in range(21))),
        ("invalid_response", ["warning"]),
    ],
)
def test_remote_discovery_error_rejects_unbounded_or_unsanitized_values(
    code: str, details: object
) -> None:
    """Catches public error values that could retain arbitrary upstream content."""
    with pytest.raises(ValueError):
        RemoteDiscoveryError(code, details=details)  # type: ignore[arg-type]


def test_remote_discovery_error_retains_bounded_display_safe_warnings() -> None:
    """Catches rejection of the bounded warning capacity needed by Task 2."""
    details = tuple("model.gguf missing 00001" for _ in range(20))

    error = RemoteDiscoveryError("no_eligible_gguf", details=details)

    assert error.details == details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [b"not json", b"[" + (b" " * (2 * 1024 * 1024)) + b"]"],
)
async def test_search_rejects_malformed_or_oversized_response(body: bytes) -> None:
    """Catches decoding before response-size enforcement or raw parse errors."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, content=body))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert raised.value.code in {"invalid_response", "response_too_large"}
    assert raised.value.retryable is False
    assert raised.value.details == ()


@pytest.mark.asyncio
async def test_search_parses_valid_bounded_metadata() -> None:
    """Catches loss of valid private/gated metadata or malformed optional fields."""
    models = [
        {
            "modelId": "acme/private",
            "private": True,
            "gated": False,
            "downloads": 42,
            "likes": 3,
            "lastModified": "2026-08-01T00:00:00Z",
        },
        {
            "modelId": "acme/auto",
            "private": False,
            "gated": "auto",
            "downloads": -1,
            "likes": "3",
            "lastModified": "x" * 65,
        },
        {
            "modelId": "acme/manual",
            "private": False,
            "gated": "manual",
            "downloads": 2**63 - 1,
            "likes": 0,
            "lastModified": "updated",
        },
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, json=models))
    )

    assert await adapter.search("models") == (
        RemoteModelSummary(
            repository="acme/private",
            private=True,
            gated="none",
            downloads=42,
            likes=3,
            last_modified="2026-08-01T00:00:00Z",
        ),
        RemoteModelSummary(
            repository="acme/auto", private=False, gated="auto"
        ),
        RemoteModelSummary(
            repository="acme/manual",
            private=False,
            gated="manual",
            downloads=2**63 - 1,
            likes=0,
            last_modified="updated",
        ),
    )


@pytest.mark.asyncio
async def test_search_caps_results_at_fifty() -> None:
    """Catches a server response exceeding the declared result limit."""
    models = [
        {"modelId": f"owner/model-{index}", "private": False, "gated": False}
        for index in range(51)
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, json=models))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("models")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "invalid_response",
        False,
        (),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("owner/repository", True),
        ("a/b", True),
        ("owner/repository/extra", False),
        ("owner", False),
        (" owner/repository", False),
        ("owner/repository ", False),
        ("owner/" + ("a" * 91), False),
        ("owner/repo?query", False),
        ("owner/repo--name", False),
        ("owner/repo..name", False),
        ("owner/repo-", False),
    ],
)
def test_is_exact_repository_requires_one_bounded_portable_pair(
    value: str, expected: bool
) -> None:
    """Catches unsafe, ambiguous, or oversized exact repository identifiers."""
    assert is_exact_repository(value) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "x" * 257])
async def test_search_rejects_empty_or_oversized_trimmed_query(query: str) -> None:
    """Catches unbounded or blank user input reaching the remote endpoint."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: pytest.fail("request was sent"))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search(query)

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "invalid_query",
        False,
        (),
    )
