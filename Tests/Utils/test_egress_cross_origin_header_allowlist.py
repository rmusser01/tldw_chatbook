"""Cross-origin redirect hops carry an allowlist, not a denylist (task-19733).

The credential header NAME in this app is user-supplied: a subscription's
``auth_config`` picks it (``monitoring_engine._fetch_and_parse_feed``, which
only *defaults* to ``X-API-Key``), and so does a ``SiteConfig``
(``site_config_manager.SiteConfig.get_headers``, same default). A denylist of
literal header names therefore cannot be correct -- it closes whichever names
someone thought of and forwards every other one verbatim to whatever host the
feed decides to redirect to.

These tests pin the inverted rule: on a hop that leaves the original origin,
a caller-supplied header is dropped unless it is explicitly known to be safe
to forward. They are deliberately written so the DEFAULT name and a CUSTOM
name are separate cases -- appending ``"x-api-key"`` to the old denylist
passes the default case and still leaks the custom one.

All transports are in-process (``httpx.MockTransport`` / a fake aiohttp
session); nothing here opens a socket. Every credential value is a synthetic
sentinel.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import (
    guarded_fetch_httpx,
    guarded_fetch_httpx_async,
    guarded_fetch_requests,
)

# Synthetic sentinels -- never a real credential.
SENTINEL = "sentinel-not-a-real-key-19733"
CUSTOM_HEADER = "X-Feed-Token"


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


CROSS_ORIGIN_ROUTES = {
    "https://feed.example/": (302, {"location": "https://evil.example/x"}, b""),
    "https://evil.example/": (200, {"content-type": "text/plain"}, b"done"),
}

SAME_ORIGIN_ROUTES = {
    "https://feed.example/start": (302, {"location": "/moved"}, b""),
    "https://feed.example/moved": (200, {"content-type": "text/plain"}, b"done"),
}


# ---------------------------------------------------------------------------
# The custom header name -- the case a longer denylist cannot reach
# ---------------------------------------------------------------------------


def test_sync_custom_named_credential_header_dropped_cross_origin():
    """A user-chosen credential header name must not reach a second origin.

    ``X-Feed-Token`` is on no denylist anywhere in the codebase. If the rule
    is name-based, this leaks.
    """
    seen = []
    with httpx.Client(transport=_transport(CROSS_ORIGIN_ROUTES, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    first, second = seen[0], seen[1]
    assert first.headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert "evil.example" in str(second.url)
    assert CUSTOM_HEADER.lower() not in second.headers
    assert SENTINEL not in "".join(second.headers.values())


@pytest.mark.asyncio
async def test_async_custom_named_credential_header_dropped_cross_origin():
    """Async guarded fetch, same rule (the subscriptions path uses this one)."""
    seen = []
    transport = _transport(CROSS_ORIGIN_ROUTES, seen)
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    first, second = seen[0], seen[1]
    assert first.headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert CUSTOM_HEADER.lower() not in second.headers
    assert SENTINEL not in "".join(second.headers.values())


def test_sync_custom_client_default_header_dropped_cross_origin():
    """The same name set as an httpx client DEFAULT header also must not leak.

    ``_hop_headers`` never sees client-default headers -- httpx merges them
    onto the built request -- so this is a separate escape route.
    """
    seen = []
    with httpx.Client(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen),
        headers={CUSTOM_HEADER: SENTINEL},
    ) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start", client=client, max_bytes=1024
        )
    assert resp.content == b"done"
    assert seen[0].headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert CUSTOM_HEADER.lower() not in seen[1].headers


@pytest.mark.asyncio
async def test_async_custom_client_default_header_dropped_cross_origin():
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen),
        headers={CUSTOM_HEADER: SENTINEL},
    ) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start", client=client, max_bytes=1024
        )
    assert resp.content == b"done"
    assert seen[0].headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert CUSTOM_HEADER.lower() not in seen[1].headers


# ---------------------------------------------------------------------------
# The default header name -- the case the filing named
# ---------------------------------------------------------------------------


def test_sync_default_x_api_key_dropped_cross_origin():
    seen = []
    with httpx.Client(transport=_transport(CROSS_ORIGIN_ROUTES, seen)) as client:
        guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={"X-API-Key": SENTINEL},
        )
    assert seen[0].headers.get("x-api-key") == SENTINEL
    assert "x-api-key" not in seen[1].headers


@pytest.mark.asyncio
async def test_async_default_x_api_key_dropped_cross_origin():
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen)
    ) as client:
        await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={"X-API-Key": SENTINEL},
        )
    assert seen[0].headers.get("x-api-key") == SENTINEL
    assert "x-api-key" not in seen[1].headers


def test_requests_path_drops_custom_credential_header_cross_origin():
    """``guarded_fetch_requests`` shares the rule (article/audio/Confluence)."""
    requests = pytest.importorskip("requests")

    seen = []

    class _Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            seen.append(request)
            resp = requests.Response()
            resp.request = request
            resp.url = request.url
            if request.url.startswith("https://feed.example/"):
                resp.status_code = 302
                resp.headers["location"] = "https://evil.example/x"
                resp.raw = _Raw(b"")
            else:
                resp.status_code = 200
                resp.raw = _Raw(b"done")
            return resp

        def close(self):
            pass

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

    assert len(seen) == 2
    assert seen[0].headers.get(CUSTOM_HEADER) == SENTINEL
    assert CUSTOM_HEADER not in seen[1].headers


# ---------------------------------------------------------------------------
# The rule must not be "drop everything"
# ---------------------------------------------------------------------------

FORWARDABLE = {
    "User-Agent": "tldw-chatbook/1.0",
    "Accept": "application/rss+xml, application/xml",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "If-None-Match": '"etag-abc"',
    "If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT",
    "Range": "bytes=100-",
}


def test_sync_forwardable_headers_still_cross_origin():
    """Content negotiation / conditional / range headers keep working.

    A CDN redirect (the normal case for feeds and artifact downloads) must
    still receive these, or resume and conditional GET break.
    """
    seen = []
    with httpx.Client(transport=_transport(CROSS_ORIGIN_ROUTES, seen)) as client:
        guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers=dict(FORWARDABLE),
        )
    second = seen[1]
    for name, value in FORWARDABLE.items():
        assert second.headers.get(name.lower()) == value, name


@pytest.mark.asyncio
async def test_async_forwardable_headers_still_cross_origin():
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen)
    ) as client:
        await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers=dict(FORWARDABLE),
        )
    second = seen[1]
    for name, value in FORWARDABLE.items():
        assert second.headers.get(name.lower()) == value, name


# ---------------------------------------------------------------------------
# Same-origin redirects keep authenticating
# ---------------------------------------------------------------------------


def test_sync_same_origin_redirect_keeps_custom_credential_header():
    seen = []
    with httpx.Client(transport=_transport(SAME_ORIGIN_ROUTES, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL, "X-API-Key": SENTINEL},
        )
    assert resp.content == b"done"
    assert len(seen) == 2
    for request in seen:
        assert request.headers.get(CUSTOM_HEADER.lower()) == SENTINEL
        assert request.headers.get("x-api-key") == SENTINEL


@pytest.mark.asyncio
async def test_async_same_origin_redirect_keeps_custom_credential_header():
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(SAME_ORIGIN_ROUTES, seen)
    ) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers={CUSTOM_HEADER: SENTINEL},
        )
    assert resp.content == b"done"
    assert len(seen) == 2
    assert all(r.headers.get(CUSTOM_HEADER.lower()) == SENTINEL for r in seen)


# ---------------------------------------------------------------------------
# The allowlist itself
# ---------------------------------------------------------------------------


def test_allowlist_never_admits_a_credential_shaped_name():
    """Guard on future edits to CROSS_ORIGIN_SAFE_HEADERS.

    The allowlist is the whole safety property now, so an addition to it is a
    security decision. Everything the old denylist named, plus anything whose
    name reads like a secret, must stay off it. The ``_STRIP_HEADERS`` half is
    enforced by construction (both exemption sets subtract it) -- this asserts
    the enforcement is actually wired up, not just intended.
    """
    for name in egress._STRIP_HEADERS:
        assert name not in egress.CROSS_ORIGIN_SAFE_HEADERS, name
    for name in egress.CROSS_ORIGIN_SAFE_HEADERS:
        assert name == name.lower(), f"{name} must be lowercase to match lookups"
        for marker in ("key", "token", "secret", "auth", "cookie", "password", "sig"):
            assert marker not in name, f"{name} looks credential-bearing"


def test_transport_headers_are_disjoint_from_the_allowlist():
    """The two exemption sets must not overlap -- one reason per header."""
    assert not (egress.CROSS_ORIGIN_SAFE_HEADERS & egress._TRANSPORT_HEADERS)
    for name in egress._STRIP_HEADERS:
        assert name not in egress._TRANSPORT_HEADERS, name


# ---------------------------------------------------------------------------
# Producer path: a real subscription auth_config, not _hop_headers in isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_api_key_auth_does_not_survive_cross_origin_redirect(
    monkeypatch,
):
    """``FeedMonitor`` with a custom-named ``api_key`` auth config.

    This is the reachable production path: feed auth is user-configurable and
    the header name comes straight from ``auth_config["header"]``.
    """
    from tldw_chatbook.Subscriptions import monitoring_engine

    seen = []
    routes = {
        "https://feed.example/": (302, {"location": "https://evil.example/x"}, b""),
        "https://evil.example/": (
            200,
            {"content-type": "application/xml"},
            b"<rss><channel></channel></rss>",
        ),
    }
    transport = _transport(routes, seen)

    real_async_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(monitoring_engine.httpx, "AsyncClient", _client_factory)

    monitor = monitoring_engine.FeedMonitor()
    subscription = {
        "id": 1,
        "source": "https://feed.example/start",
        "type": "rss",
        "auth_config": json.dumps(
            {"type": "api_key", "header": CUSTOM_HEADER, "key": SENTINEL}
        ),
    }

    await monitor._fetch_and_parse_feed(subscription)

    assert len(seen) == 2, [str(r.url) for r in seen]
    assert seen[0].headers.get(CUSTOM_HEADER.lower()) == SENTINEL
    assert "evil.example" in str(seen[1].url)
    assert CUSTOM_HEADER.lower() not in seen[1].headers
    assert SENTINEL not in "".join(seen[1].headers.values())


@pytest.mark.asyncio
async def test_site_config_scraper_headers_do_not_survive_cross_origin_redirect():
    """``SiteConfig.get_headers()`` is the other producer of the same shape."""
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfig

    config = SiteConfig(
        "feed.example",
        {
            "auth_type": "api_key",
            "auth_credentials": {"key_name": CUSTOM_HEADER, "key_value": SENTINEL},
        },
    )
    headers = config.get_headers({"User-Agent": "tldw-chatbook/1.0"})
    assert headers[CUSTOM_HEADER] == SENTINEL  # the producer really emits it

    seen = []
    async with httpx.AsyncClient(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen)
    ) as client:
        await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            headers=headers,
        )
    assert CUSTOM_HEADER.lower() not in seen[1].headers
    assert seen[1].headers.get("user-agent") == "tldw-chatbook/1.0"


# ---------------------------------------------------------------------------
# The other route a credential reaches the wire: a CLIENT-level auth=
# ---------------------------------------------------------------------------


def test_sync_client_level_auth_not_applied_cross_origin():
    """``httpx.Client(auth=...)`` must not re-attach on a cross-origin hop.

    Header stripping cannot reach this one: httpx applies a client-level
    ``auth`` inside ``send()``, AFTER ``build_request`` produced the request
    the guard filtered. Found by independent review of task-19733 -- the
    allowlist closed the ``headers=``/client-default-header routes and this
    third route still put ``Authorization: Basic ...`` on the wire to the
    second origin.
    """
    seen = []
    with httpx.Client(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen),
        auth=("alice", SENTINEL),
    ) as client:
        resp = guarded_fetch_httpx(
            "https://feed.example/start", client=client, max_bytes=1024
        )
    assert resp.content == b"done"
    assert len(seen) == 2
    assert "authorization" in seen[0].headers  # same-origin hop still authenticates
    assert "authorization" not in seen[1].headers
    assert SENTINEL not in "".join(seen[1].headers.values())


@pytest.mark.asyncio
async def test_async_client_level_auth_not_applied_cross_origin():
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(CROSS_ORIGIN_ROUTES, seen),
        auth=("alice", SENTINEL),
    ) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start", client=client, max_bytes=1024
        )
    assert resp.content == b"done"
    assert len(seen) == 2
    assert "authorization" in seen[0].headers
    assert "authorization" not in seen[1].headers


@pytest.mark.asyncio
async def test_async_same_origin_redirect_still_applies_client_level_auth():
    """The suppression is per-hop, not permanent: same-origin still authenticates."""
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(SAME_ORIGIN_ROUTES, seen),
        auth=("alice", SENTINEL),
    ) as client:
        resp = await guarded_fetch_httpx_async(
            "https://feed.example/start", client=client, max_bytes=1024
        )
    assert resp.content == b"done"
    assert len(seen) == 2
    assert all("authorization" in r.headers for r in seen)


@pytest.mark.asyncio
async def test_async_explicit_auth_argument_still_applies_same_origin():
    """The function's own ``auth=`` parameter keeps working on same-origin hops."""
    seen = []
    async with httpx.AsyncClient(
        transport=_transport(SAME_ORIGIN_ROUTES, seen)
    ) as client:
        await guarded_fetch_httpx_async(
            "https://feed.example/start",
            client=client,
            max_bytes=1024,
            auth=httpx.BasicAuth("alice", SENTINEL),
        )
    assert len(seen) == 2
    assert all("authorization" in r.headers for r in seen)
