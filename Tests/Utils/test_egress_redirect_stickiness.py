# test_egress_redirect_stickiness.py
# Description: Regression coverage for task-19863 (the cross-origin
#              credential strip re-armed when a redirect chain returned home)
"""
``Utils/egress.py``'s guarded-fetch loops used to decide credential
forwarding per hop by comparing each hop's URL against the ORIGINAL request
URL. The strip was therefore not sticky: a chain
``feed.example -> evil.example -> feed.example`` stripped the credential on
hop 2 and RE-ATTACHED it on hop 3, because hop 3 is same-origin with the
start. The Fetch specification removes a credential permanently once it has
been stripped, precisely so a chain cannot launder its way back into an
authenticated state; every loop now shares ``_StickyOriginPolicy``, which
implements exactly that.

The attacker in the A->B->A shape never receives the credential directly --
they choose the path and query string of the request that carries it. Every
credential value below is a synthetic sentinel, and every transport is
in-process (``httpx.MockTransport`` / a fake adapter / a fake aiohttp
session); nothing opens a socket.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import (
    _StickyOriginPolicy,
    guarded_fetch_httpx,
    guarded_fetch_httpx_async,
    guarded_fetch_requests,
)

# The guarded fetches read the egress policy through the config-participant
# admission chain, which fails closed with RecoveryRequired under the
# per-test config redirect (same admission signature as the egress suites on
# Tests/conftest.py's keep-bootstrap list; the marker is the self-contained
# form of that list's entries).
pytestmark = pytest.mark.bootstrap_profile

SENTINEL = "sentinel-not-a-real-key-19863"
CUSTOM_HEADER = "X-Feed-Token"

#: feed.example -> evil.example -> feed.example/return -> 200
ABA_ROUTES = {
    "https://feed.example/start": (302, {"location": "https://evil.example/x"}, b""),
    "https://evil.example/x": (
        302,
        {"location": "https://feed.example/return"},
        b"",
    ),
    "https://feed.example/return": (200, {"content-type": "text/plain"}, b"done"),
}

#: Three hops that never leave home.
SAME_ORIGIN_ROUTES = {
    "https://feed.example/start": (302, {"location": "/second"}, b""),
    "https://feed.example/second": (302, {"location": "/third"}, b""),
    "https://feed.example/third": (200, {"content-type": "text/plain"}, b"done"),
}


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Keep these transport tests independent of DNS/egress policy."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _resolve_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _resolve_async)


def _transport(routes, seen):
    """MockTransport recording every request it is handed."""

    def handler(request):
        seen.append(request)
        for prefix, (status, headers, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, headers=headers, content=body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# The unit rule itself
# ---------------------------------------------------------------------------


def test_the_policy_is_sticky_across_a_return_home():
    """A->B keeps credentials only until B; the return to A does not re-arm."""
    policy = _StickyOriginPolicy("https://feed.example/start")
    assert policy.hop_carries_credentials("https://feed.example/start") is True
    assert policy.hop_carries_credentials("https://evil.example/x") is False
    # The return home stays credential-less: the strip is permanent.
    assert policy.hop_carries_credentials("https://feed.example/return") is False


def test_the_policy_never_strips_a_chain_that_stays_home():
    policy = _StickyOriginPolicy("https://feed.example/start")
    assert policy.hop_carries_credentials("https://feed.example/start") is True
    assert policy.hop_carries_credentials("https://feed.example/second") is True
    assert policy.hop_carries_credentials("https://feed.example/third") is True


def test_scheme_port_and_host_all_count_as_origins():
    """``same_origin``'s (scheme, host, port) rule is inherited unchanged."""
    policy = _StickyOriginPolicy("https://feed.example/start")
    assert policy.hop_carries_credentials("http://feed.example/x") is False
    assert policy.hop_carries_credentials("https://feed.example:8443/x") is False
    policy2 = _StickyOriginPolicy("https://feed.example/start")
    assert policy2.hop_carries_credentials("https://feed.example/x?a=b") is True


# ---------------------------------------------------------------------------
# The A -> B -> A chain, per transport
# ---------------------------------------------------------------------------


def test_httpx_sync_aba_chain_does_not_re_arm_the_credential():
    seen = []
    with httpx.Client(transport=_transport(ABA_ROUTES, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    assert len(seen) == 3, [str(r.url) for r in seen]
    first, second, third = seen
    # Hop 1 (home): the credential rides.
    assert first.headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    # Hop 2 (attacker): stripped.
    assert CUSTOM_HEADER.lower() not in second.headers
    # Hop 3 (home again): STILL stripped -- this is the fix.
    assert "feed.example" in str(third.url)
    assert CUSTOM_HEADER.lower() not in third.headers, dict(third.headers)
    assert SENTINEL not in "".join(third.headers.values())


@pytest.mark.asyncio
async def test_httpx_async_aba_chain_does_not_re_arm_the_credential():
    seen = []
    transport = _transport(ABA_ROUTES, seen)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    assert len(seen) == 3, [str(r.url) for r in seen]
    first, second, third = seen
    assert first.headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert CUSTOM_HEADER.lower() not in second.headers
    assert CUSTOM_HEADER.lower() not in third.headers, dict(third.headers)


@pytest.mark.asyncio
async def test_httpx_async_aba_chain_does_not_re_arm_the_auth_argument():
    """The helper's ``auth=`` parameter follows the same sticky rule: after
    the chain leaves home, an explicit auth object is not re-applied."""
    seen = []
    auth_header = "X-Auth-Flow"

    class _StampingAuth(httpx.Auth):
        """Marks each request it is applied to, like a real auth flow would."""

        def auth_flow(self, request):
            request.headers[auth_header] = SENTINEL
            yield request

    transport = _transport(ABA_ROUTES, seen)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            auth=_StampingAuth(),
        )
    assert resp.content == b"done"
    assert len(seen) == 3, [str(r.url) for r in seen]
    first, second, third = seen
    # Hop 1 (never left home): the auth flow ran.
    assert first.headers.get(auth_header) == SENTINEL
    # Hop 2 (attacker) and hop 3 (home again): the helper passed
    # ``auth=None`` -- the chain had left home, so the explicit auth
    # argument is not re-applied on the return hop.
    assert auth_header not in second.headers
    assert auth_header not in third.headers, dict(third.headers)


def test_requests_aba_chain_does_not_re_arm_the_credential():
    requests = pytest.importorskip("requests")

    seen = []

    class _Raw:
        def __init__(self, body):
            self._body = body

        def stream(self, chunk_size, decode_content=True):
            yield self._body

        def read(self, *a, **k):
            return self._body

        def release_conn(self):
            pass

        def close(self):
            pass

    class _Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            seen.append(request)
            resp = requests.Response()
            resp.request = request
            resp.url = request.url
            if request.url.startswith("https://feed.example/start"):
                resp.status_code = 302
                resp.headers["location"] = "https://evil.example/x"
                resp.raw = _Raw(b"")
            elif request.url.startswith("https://evil.example/x"):
                resp.status_code = 302
                resp.headers["location"] = "https://feed.example/return"
                resp.raw = _Raw(b"")
            else:
                resp.status_code = 200
                resp.raw = _Raw(b"done")
            return resp

        def close(self):
            pass

    session = requests.Session()
    session.mount("https://", _Adapter())
    try:
        guarded_fetch_requests(
            "https://feed.example/start",
            session=session,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    finally:
        session.close()

    assert len(seen) == 3, [r.url for r in seen]
    assert seen[0].headers.get(CUSTOM_HEADER) == SENTINEL
    assert CUSTOM_HEADER not in seen[1].headers
    assert seen[2].url.startswith("https://feed.example/return")
    assert CUSTOM_HEADER not in seen[2].headers, dict(seen[2].headers)


@pytest.mark.asyncio
async def test_aiohttp_aba_chain_strips_the_caller_credential_permanently():
    """The caller-header layer of the aiohttp helper follows the same rule."""
    from tldw_chatbook.Utils.egress import guarded_fetch_aiohttp

    class _FakeBody:
        def __init__(self, body):
            self._body = body

        async def iter_chunked(self, size):
            for i in range(0, len(self._body), size):
                yield self._body[i : i + size]

    class _FakeResp:
        def __init__(self, status, headers, body, url):
            from multidict import CIMultiDict

            self.status = status
            self.headers = CIMultiDict(headers)
            self.content = _FakeBody(body)
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        """Bare session (no session-level credential to refuse)."""

        def __init__(self, routes):
            self.routes = routes
            self.seen = []
            self.auth = None
            self.headers = {}

        def get(self, url, **kwargs):
            from multidict import CIMultiDict

            self.seen.append((url, dict(kwargs.get("headers") or {})))
            for prefix, (status, resp_headers, body) in self.routes.items():
                if url.startswith(prefix):
                    return _FakeResp(status, resp_headers, body, url)
            return _FakeResp(404, {}, b"", url)

    session = _FakeSession(
        {
            "https://feed.example/start": (302, {"Location": "https://evil.example/x"}, b""),
            "https://evil.example/x": (
                302,
                {"Location": "https://feed.example/return"},
                b"",
            ),
            "https://feed.example/return": (200, {}, b"done"),
        }
    )
    resp = await guarded_fetch_aiohttp(
        "https://feed.example/start",
        session=session,
        max_bytes=1024,
        headers={CUSTOM_HEADER: SENTINEL},
    )
    assert resp.content == b"done"
    assert len(session.seen) == 3, session.seen
    _first_url, first_headers = session.seen[0]
    _second_url, second_headers = session.seen[1]
    third_url, third_headers = session.seen[2]
    assert first_headers.get(CUSTOM_HEADER) == SENTINEL
    assert CUSTOM_HEADER not in second_headers
    assert third_url == "https://feed.example/return"
    assert CUSTOM_HEADER not in third_headers, third_headers


# ---------------------------------------------------------------------------
# No regression for chains that never leave home
# ---------------------------------------------------------------------------


def test_same_origin_chain_keeps_credentials_on_every_hop():
    """AC#3: the ordinary authenticated feed that redirects within its own
    origin keeps working exactly as before -- three hops, three carries."""
    seen = []
    with httpx.Client(transport=_transport(SAME_ORIGIN_ROUTES, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    assert len(seen) == 3, [str(r.url) for r in seen]
    for request in seen:
        assert request.headers.get(CUSTOM_HEADER.lower()) == SENTINEL, dict(
            request.headers
        )


# ---------------------------------------------------------------------------
# The shared-rule lock
# ---------------------------------------------------------------------------


def test_all_four_loops_share_the_one_sticky_rule():
    """AC#4: every ``guarded_fetch_*`` loop decides per-hop origin through
    ``_StickyOriginPolicy`` -- none calls ``same_origin(url, current)``
    directly, so a loop added tomorrow inherits the rule by copying its
    siblings (and this test fails if any of the four regresses to the
    per-hop comparison)."""
    source = Path(inspect.getfile(egress)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    loops = {
        "guarded_fetch_httpx",
        "guarded_fetch_httpx_async",
        "guarded_fetch_requests",
        "guarded_fetch_aiohttp",
    }
    found_loops = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in loops:
            continue
        found_loops.add(node.name)
        segment = ast.get_source_segment(source, node) or ""
        assert "_StickyOriginPolicy(url)" in segment, node.name
        assert "origins.hop_carries_credentials(current)" in segment, node.name
        # The per-hop comparison this task removed must not come back.
        assert "same_origin(url, current)" not in segment, node.name
    assert found_loops == loops, found_loops
    # And the rule itself is defined exactly once.
    assert source.count("class _StickyOriginPolicy") == 1
