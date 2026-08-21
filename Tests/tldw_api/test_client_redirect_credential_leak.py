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
