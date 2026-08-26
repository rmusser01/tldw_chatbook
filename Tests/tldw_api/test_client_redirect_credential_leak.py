"""task-19557: ``X-API-KEY`` must never survive a cross-origin redirect.

``TLDWAPIClient`` (``tldw_api/client.py``) authenticates via a client-level
``X-API-KEY`` header (api-key is the DEFAULT auth mode; an optional bearer
``Authorization`` may also be present). httpx's built-in redirect-follower
strips only ``Authorization``/``Cookie`` on a cross-host hop -- it has no
notion of ``X-API-KEY``, so a redirecting or compromised server (or a MITM on
an ``http://`` base URL) could otherwise capture the real API key verbatim.

Fixed by constructing the shared client with ``follow_redirects=False`` and
refusing (``APIConnectionError``) any 3xx response before it is processed --
see ``TLDWAPIClient._raise_if_redirected``. This mirrors the already-shipped
``x-goog-api-key`` fix in ``LLM_Calls/LLM_API_Calls.py`` (``chat_with_google``,
task-686): refuse to follow rather than partially forward credentials.

Born-red: reverting ``follow_redirects=False`` back to ``True`` (and/or
removing the ``_raise_if_redirected`` calls) makes
``test_x_api_key_absent_on_cross_origin_redirect_hop`` fail by showing the
sentinel key delivered to the cross-origin host.

Also pins three Qodo-round fixes to ``_raise_if_redirected`` itself:
unclosed redirect responses (a connection leak, worse on the streaming
paths), 304 Not Modified mis-treated as a redirect (breaks conditional
GETs like ``get_user_profile_catalog(if_none_match=...)``), and the raw
``Location`` header being reflected into the exception message (server-
and, on a hostile endpoint, attacker-controlled data).
"""

from __future__ import annotations

import httpx
import pytest

import tldw_chatbook.tldw_api.client as client_module
from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIConnectionError

# Synthetic sentinel -- never a real credential.
_SENTINEL_KEY = "sentinel-test-x-api-key-must-never-leak"


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Patch ``httpx.AsyncClient`` construction to inject a ``MockTransport``.

    Deliberately does NOT construct the client's ``httpx.AsyncClient``
    itself -- that would bypass the very kwargs (``follow_redirects``) this
    test exists to pin. Instead it wraps the real constructor so
    ``_get_client()``'s own kwargs reach a fake, in-process transport
    instead of a real socket. No test in this module makes a real network
    call.
    """
    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _patched)


@pytest.mark.asyncio
async def test_x_api_key_absent_on_cross_origin_redirect_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers_by_host: dict[str, dict[str, str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen_headers_by_host[host] = {
            k.lower(): v for k, v in request.headers.items()
        }
        if host == "good.example":
            # The configured server "redirects" to a different, untrusted
            # host -- the exact shape a hostile or compromised endpoint (or
            # a MITM on an http:// base_url) would exploit.
            return httpx.Response(
                302, headers={"Location": "https://evil.example/steal"}
            )
        if host == "evil.example":
            return httpx.Response(200, json={"stolen": "no"})
        raise AssertionError(f"unexpected host in test transport: {host}")

    _install_mock_transport(monkeypatch, handler)

    client = TLDWAPIClient("https://good.example", token=_SENTINEL_KEY)
    raised: Exception | None = None
    try:
        await client._request("GET", "/api/v1/notes")
    except APIConnectionError as exc:
        raised = exc
    finally:
        await client.close()

    # Sanity: the first hop really did carry the credential (otherwise the
    # test would prove nothing).
    assert "good.example" in seen_headers_by_host
    assert seen_headers_by_host["good.example"].get("x-api-key") == _SENTINEL_KEY

    # The load-bearing assertion, checked independently of whether the call
    # raised, so a regression shows up as "the credential leaked" rather
    # than merely "no exception was raised". Framed as "absent from any
    # request that reached evil.example" (not "evil.example was never
    # called") so this stays correct under either a refuse-outright fix
    # (today's implementation -- evil.example is never called at all) or a
    # future strip-and-follow implementation.
    evil_headers = seen_headers_by_host.get("evil.example", {})
    assert "x-api-key" not in evil_headers, (
        f"X-API-KEY leaked to the cross-origin redirect target: {evil_headers}"
    )
    assert "authorization" not in evil_headers

    # And the redirect must actually be refused, not merely have its
    # credential incidentally dropped by some other mechanism.
    assert raised is not None, "expected APIConnectionError on redirect refusal"

    # Qodo round: the raw Location must never be reflected into the raised
    # message -- it is server- (and on a hostile/compromised endpoint,
    # attacker-) controlled data.
    assert "evil.example" not in str(raised), (
        f"redirect Location leaked into the exception message: {raised}"
    )


@pytest.mark.asyncio
async def test_304_not_modified_is_not_treated_as_redirect() -> None:
    """304 is cache-validation, not a redirect -- must not be refused.

    Real caller this protects: ``get_user_profile_catalog(if_none_match=
    ...)`` sends a conditional GET and expects a legitimate 304 back from
    the server, not an ``APIConnectionError``. 304 has no ``Location``
    and httpx itself never treats it as `is_redirect`.
    """
    response = httpx.Response(304)
    try:
        # Must return None without raising.
        await TLDWAPIClient._raise_if_redirected(
            response, "/api/v1/users/profile/catalog"
        )
    finally:
        await response.aclose()


def _handler_with_counting_close(
    captured: dict[str, httpx.Response], close_calls: list[int]
):
    """Build a MockTransport handler whose 302 response's ``aclose`` is spied.

    ``response.is_closed`` alone can't discriminate "``_raise_if_redirected``
    explicitly closed it" from "httpx's own ``stream()`` context manager
    closed it during unwind" -- both leave ``is_closed`` True, and in fact
    the latter closes it regardless of whether the explicit call exists
    (verified: removing the explicit ``await response.aclose()`` from
    ``_raise_if_redirected`` still left ``is_closed`` True, because the
    ``async with client.stream(...)`` block's own ``finally: aclose()``
    covers it -- so an ``is_closed``-only assertion doesn't pin this fix).

    Instead this wraps the instance's ``aclose`` in a counting spy *before*
    the response is returned from the handler, so both the explicit call in
    ``_raise_if_redirected`` and the automatic one on ``async with`` exit are
    independently recorded. Two calls means both fired (the explicit one
    included); one call means only the automatic one did.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            302, headers={"Location": "https://evil.example/steal"}
        )
        original_aclose = response.aclose

        async def _counting_aclose() -> None:
            close_calls.append(1)
            await original_aclose()

        response.aclose = _counting_aclose
        captured["response"] = response
        return response

    return handler


@pytest.mark.asyncio
async def test_streaming_redirect_response_is_closed_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect reached via ``_stream_request`` must not leak the connection.

    ``_stream_request``/``_sse_request`` obtain their response via
    ``client.stream(...)`` (not the eagerly-read ``client.request(...)``),
    so an unclosed response here is a real leaked streaming connection --
    worse than the non-streaming methods, where httpx already reads (and
    thereby releases) the body before ``_raise_if_redirected`` ever runs.
    """
    captured: dict[str, httpx.Response] = {}
    close_calls: list[int] = []
    _install_mock_transport(
        monkeypatch, _handler_with_counting_close(captured, close_calls)
    )

    client = TLDWAPIClient("https://good.example", token=_SENTINEL_KEY)
    try:
        with pytest.raises(APIConnectionError):
            async for _ in client._stream_request("POST", "/api/v1/media/ingest"):
                pass
    finally:
        await client.close()

    assert "response" in captured
    assert captured["response"].is_closed
    # The load-bearing count: 1 means only httpx's own stream()-exit close
    # ran; 2 means _raise_if_redirected's explicit close ran too.
    assert len(close_calls) == 2, (
        f"expected _raise_if_redirected to explicitly close the redirect "
        f"response before raising (in addition to httpx's own stream-exit "
        f"close); observed {len(close_calls)} aclose() call(s)"
    )


@pytest.mark.asyncio
async def test_sse_redirect_response_is_closed_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as above, for the SSE path (``_sse_request``)."""
    captured: dict[str, httpx.Response] = {}
    close_calls: list[int] = []
    _install_mock_transport(
        monkeypatch, _handler_with_counting_close(captured, close_calls)
    )

    client = TLDWAPIClient("https://good.example", token=_SENTINEL_KEY)
    try:
        with pytest.raises(APIConnectionError):
            async for _ in client._sse_request("GET", "/api/v1/events"):
                pass
    finally:
        await client.close()

    assert "response" in captured
    assert captured["response"].is_closed
    assert len(close_calls) == 2, (
        f"expected _raise_if_redirected to explicitly close the redirect "
        f"response before raising (in addition to httpx's own stream-exit "
        f"close); observed {len(close_calls)} aclose() call(s)"
    )
