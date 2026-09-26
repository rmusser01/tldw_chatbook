"""A value interpolated into a request path cannot escape its segment.

Tier-2 review S06, P2 [D1]: `client.py` builds 705 request paths with
f-strings. 355 interpolate an `int`-annotated parameter (safe) and 33 go
through `quote(..., safe="")` -- the remaining 334 interpolate a
`str`-annotated parameter with neither. Several of those are values the
user pastes: a public share token, a shared-conversation token, a consent
purpose, a provider name.

httpx resolves a relative path through `base_url.join()`, which treats `?`
and `#` as delimiters and normalises `../`. So a pasted token containing
`?` appends attacker-chosen query parameters to an authenticated request,
and one containing `../` walks out of the API namespace with the
`X-API-KEY` header still attached.

The guard is at the single choke point every request primitive passes
through, not at 334 call sites, and it fails CLOSED: no endpoint in the
package carries a literal `?` or `#`.
"""

from __future__ import annotations

import httpx
import pytest

from tldw_chatbook.Utils.input_validation import validate_request_endpoint_path
from tldw_chatbook.tldw_api import client as client_module
from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIRequestError


def _client() -> tuple[TLDWAPIClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client = TLDWAPIClient("http://api.test", "secret")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client, seen


@pytest.mark.asyncio
async def test_a_pasted_share_token_cannot_append_query_parameters():
    client, seen = _client()

    with pytest.raises(APIRequestError) as exc:
        await client.preview_public_share("tok?api_key=leaked")

    assert "escaped its path segment" in str(exc.value)
    assert seen == [], "the request must not reach the transport at all"


@pytest.mark.asyncio
async def test_a_pasted_token_cannot_walk_out_of_the_api_namespace():
    client, seen = _client()

    with pytest.raises(APIRequestError):
        await client.preview_public_share("../../admin/users")

    assert seen == []


@pytest.mark.asyncio
async def test_a_fragment_delimiter_is_refused_too():
    client, seen = _client()

    with pytest.raises(APIRequestError):
        await client.preview_public_share("tok#frag")

    assert seen == []


@pytest.mark.asyncio
async def test_an_ordinary_token_still_reaches_the_server_unchanged():
    """The guard fails closed, so prove it does not reject the real shape.

    `_request` directly rather than `preview_public_share`, because the
    response model is not what is under test here.
    """
    client, seen = _client()

    await client._request("GET", "/api/v1/sharing/public/aXb-cYd_eZf012345678901")

    assert len(seen) == 1
    assert seen[0].url.path.endswith("/aXb-cYd_eZf012345678901")
    assert seen[0].url.query == b""


# ---------------------------------------------------------------------------
# Qodo review on PR #2806: the policy itself belongs to the shared input
# boundary, not to a one-off copy inside the API client -- five request
# primitives depend on it, so a later change made in only one of the two
# places would diverge silently.
# ---------------------------------------------------------------------------


def test_the_shared_validator_owns_the_policy():
    """`Utils.input_validation` refuses, and the client reuses that function."""
    for bad in ("tok?a=1", "tok#f", "a/../b"):
        with pytest.raises(ValueError):
            validate_request_endpoint_path(bad)

    assert validate_request_endpoint_path("/api/v1/x/y") == "/api/v1/x/y"
    assert (
        client_module.validate_request_endpoint_path
        is validate_request_endpoint_path
    )


@pytest.mark.asyncio
async def test_the_request_primitive_calls_the_shared_validator(monkeypatch):
    """Identity is not use: prove the endpoint actually goes through it."""
    seen: list[str] = []

    def spy(endpoint: str) -> str:
        seen.append(endpoint)
        return endpoint

    monkeypatch.setattr(client_module, "validate_request_endpoint_path", spy)
    client, _ = _client()

    await client._request("GET", "/api/v1/sharing/public/tok")

    assert seen == ["/api/v1/sharing/public/tok"]
