"""All five `tldw_api` request primitives raise the same typed error.

Tier-2 review S06, P2 [D4]: `client.py` carried five hand-rolled copies of
the HTTP-error handler. Only `_request` had the structured
``{"detail": {"code", "message", ...}}`` branch that TASK "schedules task 6
round 2, D9" added; the other four had a `detail`-as-string branch only, so
a tldw_server refusal reached the user as the raw httpx text.

Worse on the two STREAMING primitives, because `raise_for_status()` fires
*inside* `async with client.stream(...)`: by the time the `except` body
runs, `__aexit__` has closed the response.

* `_stream_request`'s `await e.response.aread()` therefore always raised
  `StreamClosed`, and its `except Exception` swallowed it -- the server's
  explanation was dropped and `response_data` was `{"raw_text": ""}`.
* `_sse_request`'s `e.response.json()` raised `httpx.ResponseNotRead`,
  which is **not** a `ValueError` (MRO: `ResponseNotRead -> StreamError ->
  RuntimeError`), so it escaped the `except ValueError` guard and left the
  package's exception family entirely.

The mock below returns a genuinely *unread* streaming response (an async
generator body). A pre-buffered `httpx.Response(json=...)` gives a false
negative: httpx precomputes its content, so `is_stream_consumed` is already
True before `raise_for_status()` runs and both defects vanish.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import (
    APIRequestError,
    APIResponseError,
    AuthenticationError,
)

_STRUCTURED_REFUSAL = {
    "detail": {
        "code": "session_token_expired",
        "message": "Your session token expired.",
        "retryable": False,
    }
}


def _client(handler) -> TLDWAPIClient:
    """A client whose transport is `handler`, with no real network."""
    client = TLDWAPIClient("http://api.test", "secret")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


def _streaming(status: int, body: dict):
    """A handler returning an UNREAD streaming response (the real shape)."""

    def handler(request: httpx.Request) -> httpx.Response:
        async def chunks():
            yield json.dumps(body).encode()

        return httpx.Response(status, content=chunks())

    return handler


def _buffered(status: int, body: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


@pytest.mark.asyncio
async def test_sse_request_401_raises_authentication_error_not_response_not_read():
    client = _client(_streaming(401, _STRUCTURED_REFUSAL))

    with pytest.raises(AuthenticationError) as exc:
        async for _ in client._sse_request("GET", "/api/v1/notifications/stream"):
            pass

    assert "Your session token expired." in str(exc.value)
    assert exc.value.response_data == _STRUCTURED_REFUSAL


@pytest.mark.asyncio
async def test_stream_request_401_surfaces_the_servers_own_explanation():
    client = _client(_streaming(401, _STRUCTURED_REFUSAL))

    with pytest.raises(AuthenticationError) as exc:
        async for _ in client._stream_request("POST", "/api/v1/media/process-video"):
            pass

    assert "Your session token expired." in str(exc.value)
    # Never the raw httpx text, which is what the user would otherwise see.
    assert "developer.mozilla.org" not in str(exc.value)
    assert exc.value.response_data == _STRUCTURED_REFUSAL


@pytest.mark.asyncio
async def test_sse_request_409_dict_detail_surfaces_the_server_message():
    client = _client(_streaming(409, _STRUCTURED_REFUSAL))

    with pytest.raises(APIResponseError) as exc:
        async for _ in client._sse_request("GET", "/api/v1/mcp/hub/events/stream"):
            pass

    assert exc.value.status_code == 409
    assert "Your session token expired." in str(exc.value)


@pytest.mark.asyncio
async def test_binary_request_dict_detail_surfaces_the_server_message():
    client = _client(_buffered(409, _STRUCTURED_REFUSAL))

    with pytest.raises(APIResponseError) as exc:
        await client._binary_request("GET", "/api/v1/reading/export")

    assert "Your session token expired." in str(exc.value)
    assert "developer.mozilla.org" not in str(exc.value)


@pytest.mark.asyncio
async def test_headers_request_dict_detail_surfaces_the_server_message():
    client = _client(_buffered(409, _STRUCTURED_REFUSAL))

    with pytest.raises(APIResponseError) as exc:
        await client._headers_request("HEAD", "/api/v1/media/1")

    assert "Your session token expired." in str(exc.value)
    assert "developer.mozilla.org" not in str(exc.value)


@pytest.mark.asyncio
async def test_streaming_primitives_classify_422_like_request_does():
    """422 was an `APIRequestError` on `_request` and an `APIResponseError`
    on both streaming primitives -- the same server refusal, two types."""
    client = _client(_streaming(422, {"detail": [{"msg": "bad", "loc": ["body", "q"]}]}))

    with pytest.raises(APIRequestError):
        async for _ in client._sse_request("GET", "/api/v1/x/stream"):
            pass
