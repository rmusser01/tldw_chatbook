"""Policy tests for tldw_chatbook.Utils.egress (no real DNS, no network)."""

import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    check_url_or_raise,
    evaluate_url_policy,
    evaluate_url_policy_async,
)


@pytest.fixture(autouse=True)
def _no_real_dns_or_config(monkeypatch):
    """Default: everything resolves public; config enabled with no allowlist."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _fake_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _resolve_to(monkeypatch, ips):
    monkeypatch.setattr(egress, "_resolve", lambda host: list(ips))

    async def _fake_async(host):
        return list(ips)

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)


def test_public_url_allowed():
    d = evaluate_url_policy("https://example.com/page")
    assert d.allowed and d.reason == "ok"


def test_non_http_schemes_blocked():
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x", "data:text/html,hi"):
        d = evaluate_url_policy(url)
        assert not d.allowed and d.reason == "scheme", url


def test_missing_host_blocked():
    assert not evaluate_url_policy("https:///nohost").allowed


def test_loopback_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["127.0.0.1"])
    d = evaluate_url_policy("http://myhost.example/")
    assert not d.allowed and d.reason == "private"


def test_rfc1918_blocked(monkeypatch):
    for ip in ("10.0.0.5", "172.16.3.4", "192.168.1.1"):
        _resolve_to(monkeypatch, [ip])
        assert not evaluate_url_policy("http://h.example/").allowed


def test_ipv6_ula_and_mapped_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["fd12::1"])
    assert not evaluate_url_policy("http://h.example/").allowed
    _resolve_to(monkeypatch, ["::ffff:192.168.0.1"])
    assert evaluate_url_policy("http://h.example/").reason == "private"


def test_cgnat_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["100.64.0.7"])
    assert not evaluate_url_policy("http://h.example/").allowed


def test_metadata_ip_blocked_even_when_trusted(monkeypatch):
    _resolve_to(monkeypatch, ["169.254.169.254"])
    d = evaluate_url_policy(
        "http://h.example/", trusted_origins=frozenset({"h.example"})
    )
    assert not d.allowed and d.reason == "metadata"


def test_metadata_hostname_blocked_pre_resolution(monkeypatch):
    def _boom(host):  # pragma: no cover - must not be called
        raise AssertionError("resolved a metadata hostname")

    monkeypatch.setattr(egress, "_resolve", _boom)
    d = evaluate_url_policy("http://metadata.google.internal/computeMetadata/")
    assert not d.allowed and d.reason == "metadata"


def test_ip_literal_hosts_classified_directly():
    d4 = evaluate_url_policy("http://169.254.169.254/latest/meta-data/")
    assert not d4.allowed and d4.reason == "metadata"
    d6 = evaluate_url_policy("http://[::1]:8080/")
    assert not d6.allowed and d6.reason == "private"


def test_any_bad_record_blocks(monkeypatch):
    _resolve_to(monkeypatch, ["93.184.216.34", "192.168.0.9"])
    assert not evaluate_url_policy("http://h.example/").allowed


def test_dns_failure_fail_closed(monkeypatch):
    def _fail(host):
        raise OSError("nxdomain")

    monkeypatch.setattr(egress, "_resolve", _fail)
    d = evaluate_url_policy("http://nope.invalid/")
    assert not d.allowed and d.reason == "dns_failure"


def test_trusted_origin_allows_private(monkeypatch):
    _resolve_to(monkeypatch, ["192.168.1.50"])
    d = evaluate_url_policy(
        "http://wiki.corp.example/page",
        trusted_origins=frozenset({"wiki.corp.example"}),
    )
    assert d.allowed


def test_trusted_match_is_hostname_only_case_insensitive(monkeypatch):
    _resolve_to(monkeypatch, ["10.1.2.3"])
    d = evaluate_url_policy(
        "https://Wiki.CORP.example:8443/x",
        trusted_origins=frozenset({"wiki.corp.example"}),
    )
    assert d.allowed


def test_allowlist_overrides_metadata(monkeypatch):
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda s, k=None, d=None: ["metadata.google.internal"]
        if k == "allowed_hosts"
        else d,
    )
    d = evaluate_url_policy("http://metadata.google.internal/")
    assert d.allowed


def test_kill_switch_short_circuits_before_dns(monkeypatch):
    def _boom(host):  # pragma: no cover
        raise AssertionError("resolved while disabled")

    monkeypatch.setattr(egress, "_resolve", _boom)
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda s, k=None, d=None: False if k == "enabled" else d,
    )
    d = evaluate_url_policy("http://192.168.0.1/")
    assert d.allowed and d.reason == "disabled"


def test_check_url_or_raise_raises_with_remedy(monkeypatch):
    _resolve_to(monkeypatch, ["127.0.0.1"])
    with pytest.raises(EgressBlockedError) as exc:
        check_url_or_raise("http://internal.example/")
    assert "internal.example" in str(exc.value)
    assert "allowed_hosts" in str(exc.value)
    assert exc.value.reason == "private"


@pytest.mark.asyncio
async def test_async_variant_same_policy(monkeypatch):
    async def _fake(host):
        return ["169.254.169.254"]

    monkeypatch.setattr(egress, "_resolve_async", _fake)
    d = await evaluate_url_policy_async("http://h.example/")
    assert not d.allowed and d.reason == "metadata"


# ---------------------------------------------------------------------------
# Guarded httpx helpers
# ---------------------------------------------------------------------------
import httpx

from tldw_chatbook.Utils.egress import (
    EgressFetchError,
    GuardedResponse,
    guarded_fetch_httpx,
    guarded_fetch_httpx_async,
)


def _transport(routes, seen):
    """MockTransport: routes = {url_prefix: (status, headers, body)}."""

    def handler(request):
        seen.append(request)
        for prefix, (status, headers, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, headers=headers, content=body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_httpx_basic_fetch_returns_guarded_response():
    seen = []
    routes = {"https://example.com/": (200, {"content-type": "text/html; charset=utf-8"}, b"<html>ok</html>")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://example.com/page", client=client, max_bytes=1024
        )
    assert isinstance(resp, GuardedResponse)
    assert resp.status_code == 200
    assert resp.content == b"<html>ok</html>"
    assert resp.text == "<html>ok</html>"
    assert resp.final_url == "https://example.com/page"


def test_httpx_redirect_to_internal_blocked(monkeypatch):
    def fake_resolve(host):
        return ["192.168.1.1"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    seen = []
    routes = {
        "https://example.com/": (302, {"location": "http://internal.example/"}, b""),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressBlockedError) as exc:
            guarded_fetch_httpx("https://example.com/r", client=client, max_bytes=1024)
    assert exc.value.reason == "private"
    # the internal host was never actually requested
    assert all("internal.example" not in str(r.url) for r in seen)


def test_httpx_same_origin_redirect_allowed_and_followed():
    seen = []
    routes = {
        "https://example.com/old": (301, {"location": "/new"}, b""),
        "https://example.com/new": (200, {"content-type": "text/html"}, b"moved"),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://example.com/old", client=client, max_bytes=1024
        )
    assert resp.status_code == 200 and resp.content == b"moved"
    assert resp.final_url == "https://example.com/new"


def test_httpx_cross_origin_hop_strips_credentials(monkeypatch):
    _resolve_to(monkeypatch, ["93.184.216.34"])
    seen = []
    routes = {
        "https://a.example/": (302, {"location": "https://b.example/x"}, b""),
        "https://b.example/": (200, {}, b"done"),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://a.example/start",
            client=client,
            max_bytes=1024,
            headers={"Authorization": "Bearer secret", "Cookie": "sid=1", "X-Keep": "y"},
        )
    assert resp.content == b"done"
    first, second = seen[0], seen[1]
    assert first.headers.get("authorization") == "Bearer secret"
    assert "authorization" not in second.headers
    assert "cookie" not in second.headers
    assert second.headers.get("x-keep") == "y"


def test_httpx_hop_cap():
    seen = []
    routes = {"https://example.com/": (302, {"location": "/loop"}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="redirect"):
            guarded_fetch_httpx("https://example.com/loop", client=client, max_bytes=64)


def test_httpx_redirect_without_location():
    seen = []
    routes = {"https://example.com/": (302, {}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="[Ll]ocation"):
            guarded_fetch_httpx("https://example.com/x", client=client, max_bytes=64)


def test_httpx_byte_cap_aborts():
    seen = []
    routes = {"https://example.com/": (200, {}, b"x" * 2048)}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="exceeds"):
            guarded_fetch_httpx("https://example.com/big", client=client, max_bytes=1024)


def test_httpx_304_passes_through_without_raise():
    seen = []
    routes = {"https://example.com/": (304, {"etag": "abc"}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx("https://example.com/feed", client=client, max_bytes=64)
    assert resp.status_code == 304


def test_guarded_response_raise_for_status_delegates_httpx():
    seen = []
    routes = {"https://example.com/": (404, {}, b"nope")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx("https://example.com/x", client=client, max_bytes=64)
    with pytest.raises(httpx.HTTPStatusError):
        resp.raise_for_status()


@pytest.mark.asyncio
async def test_httpx_async_variant_and_auth_suppression(monkeypatch):
    seen = []
    routes = {
        "https://a.example/": (302, {"location": "https://b.example/t"}, b""),
        "https://b.example/": (200, {}, b"ok"),
    }
    async with httpx.AsyncClient(transport=_transport(routes, seen)) as client:
        resp = await guarded_fetch_httpx_async(
            "https://a.example/s",
            client=client,
            max_bytes=64,
            auth=("user", "pw"),
        )
    assert resp.content == b"ok"
    assert "authorization" in seen[0].headers
    assert "authorization" not in seen[1].headers


# ---------------------------------------------------------------------------
# Guarded requests helper
# ---------------------------------------------------------------------------
import io

import requests as requests_lib
from requests.adapters import BaseAdapter
from requests.models import Response as RequestsResponse

from tldw_chatbook.Utils.egress import guarded_fetch_requests


class _FakeHTTPAdapter(BaseAdapter):
    """Serves canned responses; records every prepared request."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes  # {url_prefix: (status, headers, body)}
        self.seen = []

    def send(self, request, **kwargs):
        self.seen.append(request)
        for prefix, (status, headers, body) in self.routes.items():
            if request.url.startswith(prefix):
                resp = RequestsResponse()
                resp.status_code = status
                resp.headers.update(headers)
                resp.raw = io.BytesIO(body)
                resp.url = request.url
                resp.request = request
                return resp
        resp = RequestsResponse()
        resp.status_code = 404
        resp.raw = io.BytesIO(b"")
        resp.url = request.url
        return resp

    def close(self):
        pass


def _session_with(routes):
    sess = requests_lib.Session()
    adapter = _FakeHTTPAdapter(routes)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess, adapter


def test_requests_basic_fetch_preloads_content():
    sess, adapter = _session_with(
        {"https://example.com/": (200, {"content-type": "text/html"}, b"<p>hi</p>")}
    )
    resp = guarded_fetch_requests(
        "https://example.com/p", session=sess, max_bytes=1024
    )
    assert resp.status_code == 200
    assert resp.content == b"<p>hi</p>"
    assert resp.text == "<p>hi</p>"


def test_requests_redirect_to_internal_blocked(monkeypatch):
    def fake_resolve(host):
        return ["10.0.0.5"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    sess, adapter = _session_with(
        {"https://example.com/": (302, {"location": "http://internal.example/"}, b"")}
    )
    with pytest.raises(EgressBlockedError):
        guarded_fetch_requests("https://example.com/r", session=sess, max_bytes=64)
    assert all("internal.example" not in r.url for r in adapter.seen)


def test_requests_session_auth_suppressed_cross_origin():
    sess, adapter = _session_with(
        {
            "https://a.example/": (302, {"location": "https://b.example/n"}, b""),
            "https://b.example/": (200, {}, b"fin"),
        }
    )
    sess.auth = ("user", "pw")
    resp = guarded_fetch_requests("https://a.example/s", session=sess, max_bytes=64)
    assert resp.content == b"fin"
    assert "Authorization" in adapter.seen[0].headers
    assert "Authorization" not in adapter.seen[1].headers


def test_requests_byte_cap():
    sess, _ = _session_with({"https://example.com/": (200, {}, b"y" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests("https://example.com/big", session=sess, max_bytes=1024)


def test_requests_sink_streams_without_buffering():
    body = b"z" * 3000
    sess, _ = _session_with({"https://example.com/": (200, {}, body)})
    sink = io.BytesIO()
    resp = guarded_fetch_requests(
        "https://example.com/file", session=sess, max_bytes=4096, sink=sink
    )
    assert sink.getvalue() == body
    assert resp.content == b""
    assert resp.status_code == 200


def test_requests_sink_cap_still_enforced():
    sess, _ = _session_with({"https://example.com/": (200, {}, b"w" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests(
            "https://example.com/file", session=sess, max_bytes=1024, sink=io.BytesIO()
        )


# ---------------------------------------------------------------------------
# Guarded aiohttp helper + Playwright chain validation
# ---------------------------------------------------------------------------
from tldw_chatbook.Utils.egress import (
    collect_navigation_chain,
    guarded_fetch_aiohttp,
    validate_navigation_chain,
    validate_navigation_chain_async,
)


class _FakeAiohttpContent:
    def __init__(self, body):
        self._body = body

    async def iter_chunked(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]


class _FakeAiohttpResponse:
    def __init__(self, status, headers, body, url):
        self.status = status
        self.headers = headers
        self.content = _FakeAiohttpContent(body)
        self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeAiohttpSession:
    def __init__(self, routes):
        self.routes = routes
        self.seen = []

    def get(self, url, **kwargs):
        self.seen.append((url, kwargs))
        for prefix, (status, headers, body) in self.routes.items():
            if url.startswith(prefix):
                return _FakeAiohttpResponse(status, headers, body, url)
        return _FakeAiohttpResponse(404, {}, b"", url)


@pytest.mark.asyncio
async def test_aiohttp_fetch_and_redirect_revalidation(monkeypatch):
    def fake_resolve_ok(host):
        return ["10.9.9.9"] if host == "intra.example" else ["93.184.216.34"]

    async def fake_async(host):
        return fake_resolve_ok(host)

    monkeypatch.setattr(egress, "_resolve_async", fake_async)
    session = _FakeAiohttpSession(
        {"https://example.com/": (302, {"Location": "http://intra.example/"}, b"")}
    )
    with pytest.raises(EgressBlockedError):
        await guarded_fetch_aiohttp(
            "https://example.com/r", session=session, max_bytes=64
        )
    assert all("intra.example" not in u for u, _ in session.seen)


@pytest.mark.asyncio
async def test_aiohttp_basic_fetch_capped():
    session = _FakeAiohttpSession(
        {"https://example.com/": (200, {"Content-Type": "text/html"}, b"page")}
    )
    resp = await guarded_fetch_aiohttp(
        "https://example.com/x", session=session, max_bytes=64
    )
    assert resp.status_code == 200 and resp.content == b"page"
    # allow_redirects must be forced off (manual loop owns redirects)
    assert session.seen[0][1].get("allow_redirects") is False

    big = _FakeAiohttpSession({"https://example.com/": (200, {}, b"q" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        await guarded_fetch_aiohttp("https://example.com/b", session=big, max_bytes=1024)


class _FakePlaywrightRequest:
    def __init__(self, url, redirected_from=None):
        self.url = url
        self.redirected_from = redirected_from


class _FakePlaywrightResponse:
    def __init__(self, request):
        self.request = request


def test_collect_navigation_chain_walks_redirects():
    first = _FakePlaywrightRequest("https://a.example/start")
    second = _FakePlaywrightRequest("https://a.example/hop", redirected_from=first)
    final = _FakePlaywrightRequest("https://b.example/end", redirected_from=second)
    chain = collect_navigation_chain(_FakePlaywrightResponse(final))
    assert chain == [
        "https://a.example/start",
        "https://a.example/hop",
        "https://b.example/end",
    ]


def test_validate_navigation_chain_blocks_internal_hop(monkeypatch):
    def fake_resolve(host):
        return ["169.254.169.254"] if host == "meta.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    with pytest.raises(EgressBlockedError):
        validate_navigation_chain(
            ["https://ok.example/", "http://meta.example/x"]
        )
    validate_navigation_chain(["https://ok.example/"])  # no raise


@pytest.mark.asyncio
async def test_validate_navigation_chain_async(monkeypatch):
    async def fake_async(host):
        return ["192.168.7.7"]

    monkeypatch.setattr(egress, "_resolve_async", fake_async)
    with pytest.raises(EgressBlockedError):
        await validate_navigation_chain_async(["http://internal.example/"])
    # trusted origin passes
    await validate_navigation_chain_async(
        ["http://internal.example/"],
        trusted_origins=frozenset({"internal.example"}),
    )
