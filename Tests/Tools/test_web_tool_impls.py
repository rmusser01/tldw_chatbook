import socket
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import (
    FETCH_MAX_BYTES,
    FETCH_MAX_REDIRECTS,
    LocalToolError,
    validate_outbound_url,
    web_fetch,
)


def test_accepts_public_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    assert validate_outbound_url("https://example.com/page") == "https://example.com/page"


def test_rejects_bad_schemes():
    for url in ("file:///etc/passwd", "ftp://x/y", "gopher://x", "javascript:alert(1)", "data:text/html,hi"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)


def test_rejects_loopback_and_private_literals():
    for url in ("http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "http://172.16.0.1/",
                "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data", "http://[::1]/",
                "http://0.0.0.0/"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)


def test_rejects_private_dns_answer(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 80))])
    with pytest.raises(LocalToolError, match="private|internal|not allowed"):
        validate_outbound_url("http://evil.internal.example.com/")


def test_rejects_unresolvable_host(monkeypatch):
    import socket

    def boom(*a, **k):
        raise socket.gaierror("no")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(LocalToolError):
        validate_outbound_url("http://does-not-exist.invalid/")


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

_PUBLIC_IP = "93.184.216.34"  # example.com's historical public IP

_ARTICLE_HTML = (
    "<html><head><title>Test Page</title>"
    "<script>var marker = 'SCRIPT_MARKER_XYZZY';</script></head>"
    "<body><article><h1>Hello</h1>"
    "<p>This is the main article body sentence that the extractor should keep. "
    "It carries enough plain prose words to look like real page content.</p>"
    "</article></body></html>"
)


class _FakeClock:
    """Instant fake time: sleep() advances the clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fetch_env(monkeypatch):
    """Mocked network (MockTransport), fake DNS (public IP), fake clock.

    validate_outbound_url does REAL DNS on every hop, so getaddrinfo is
    monkeypatched to resolve everything to a public IP; literal-IP URLs in
    tests skip DNS entirely.
    """
    routes: dict[str, object] = {}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        item = routes[url]
        if isinstance(item, Exception):
            raise item
        return item

    # DNS: every hostname resolves to a public IP.
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (_PUBLIC_IP, 80))]
    )
    monkeypatch.setattr(
        web_tool_impls, "_transport", httpx.MockTransport(handler)
    )
    clock = _FakeClock()
    monkeypatch.setattr(web_tool_impls, "time", clock)
    web_tool_impls._reset_state_for_tests()
    yield SimpleNamespace(routes=routes, calls=calls, clock=clock)
    web_tool_impls._reset_state_for_tests()


def _text_page(body: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "text/plain"})


def _html_page(html: str = _ARTICLE_HTML) -> httpx.Response:
    return httpx.Response(200, content=html.encode(), headers={"content-type": "text/html"})


def test_fetch_extracts_text(fetch_env):
    fetch_env.routes["http://example.com/page"] = _html_page()
    result = web_fetch("http://example.com/page")
    assert "main article body sentence" in result
    assert "SCRIPT_MARKER_XYZZY" not in result  # script stripped
    assert "<" not in result.split("main article")[0]  # tags gone
    assert fetch_env.calls == ["http://example.com/page"]


def test_fetch_validates_each_redirect_hop(fetch_env):
    fetch_env.routes["http://example.com/start"] = httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data"}
    )
    with pytest.raises(LocalToolError, match="ssrf"):
        web_fetch("http://example.com/start")
    # The redirect target was never requested (blocked before the hop).
    assert fetch_env.calls == ["http://example.com/start"]


def test_fetch_redirect_cap(fetch_env):
    # FETCH_MAX_REDIRECTS + 1 chained redirects -> every hop is a redirect.
    for i in range(FETCH_MAX_REDIRECTS + 1):
        fetch_env.routes[f"http://example.com/r{i}"] = httpx.Response(
            302, headers={"location": f"http://example.com/r{i + 1}"}
        )
    with pytest.raises(LocalToolError, match="redirect-limit"):
        web_fetch("http://example.com/r0")
    assert len(fetch_env.calls) == FETCH_MAX_REDIRECTS + 1


def test_fetch_byte_cap_sets_truncated(fetch_env):
    body = b"x" * (FETCH_MAX_BYTES + 5000)
    fetch_env.routes["http://example.com/big"] = _text_page(body)
    result = web_fetch("http://example.com/big")
    assert "truncated" in result
    assert len(result) <= FETCH_MAX_BYTES + 200  # body capped + short marker


def test_fetch_rate_limits_per_domain(fetch_env):
    fetch_env.routes["http://example.com/a"] = _text_page(b"page a")
    fetch_env.routes["http://example.com/b"] = _text_page(b"page b")
    web_fetch("http://example.com/a")
    assert fetch_env.clock.sleeps == []  # first request: no wait
    fetch_env.clock.now = 0.2
    web_fetch("http://example.com/b")  # same domain, different URL (no cache hit)
    assert fetch_env.clock.sleeps == [pytest.approx(0.8)]


def test_fetch_rate_limited_raises_when_clock_frozen(fetch_env):
    fetch_env.routes["http://example.com/a"] = _text_page(b"page a")
    fetch_env.routes["http://example.com/b"] = _text_page(b"page b")
    web_fetch("http://example.com/a")
    # Freeze the clock: a sleep that never advances monotonic time must raise
    # instead of spinning forever.
    fetch_env.clock.sleep = lambda s: fetch_env.clock.sleeps.append(s)
    with pytest.raises(LocalToolError, match="rate-limited"):
        web_fetch("http://example.com/b")


def test_fetch_caches_within_ttl(fetch_env):
    fetch_env.routes["http://example.com/page"] = _text_page(b"cached page")
    first = web_fetch("http://example.com/page")
    second = web_fetch("http://example.com/page")
    assert first == second == "cached page"
    assert fetch_env.calls == ["http://example.com/page"]  # transport hit once


def test_fetch_http_error_status(fetch_env):
    fetch_env.routes["http://example.com/missing"] = _text_page(b"nope", status=404)
    with pytest.raises(LocalToolError, match="http-404"):
        web_fetch("http://example.com/missing")


def test_fetch_timeout(fetch_env):
    fetch_env.routes["http://example.com/slow"] = httpx.ReadTimeout("too slow")
    with pytest.raises(LocalToolError, match="timeout"):
        web_fetch("http://example.com/slow")


def test_fetch_invalid_url(fetch_env):
    with pytest.raises(LocalToolError, match="invalid-url"):
        web_fetch("ftp://example.com/x")
    assert fetch_env.calls == []  # never touched the network


# ---------------------------------------------------------------------------
# SSRF bypass-class regressions + hardening
# ---------------------------------------------------------------------------

def test_rejects_cgnat_and_shared_space_literals():
    # 100.64.0.0/10 (RFC 6598 CGNAT/shared space — Tailscale tailnets, carrier
    # gear) is NOT covered by ipaddress.is_private on Python 3.12; the guard
    # must block it explicitly. 192.0.0.0/24 is IETF protocol assignments.
    for url in ("http://100.64.0.1/", "http://100.100.100.100/", "http://192.0.0.1/"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)


def test_rejects_decimal_and_hex_ip_forms(monkeypatch):
    # libc getaddrinfo translates these odd literal forms to the canonical
    # IPv4 address (2130706433 == 0x7f000001 == 127.0.0.1); the guard must
    # see and refuse the translated answer.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 80))])
    for url in ("http://2130706433/", "http://0x7f000001/"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)


def test_rejects_ipv4_mapped_ipv6_loopback():
    with pytest.raises(LocalToolError):
        validate_outbound_url("http://[::ffff:127.0.0.1]/")


def test_rejects_userinfo_host_trick():
    # The real host is 127.0.0.1; "example.com" is userinfo. No DNS needed.
    with pytest.raises(LocalToolError):
        validate_outbound_url("http://example.com@127.0.0.1/")


def test_rejects_port_out_of_range():
    # parts.port raises "Port out of range" ValueError; must be a LocalToolError.
    with pytest.raises(LocalToolError):
        validate_outbound_url("http://example.com:99999/")


def test_fetch_rejects_bad_port(fetch_env):
    with pytest.raises(LocalToolError, match="invalid-url"):
        web_fetch("http://example.com:99999/")
    assert fetch_env.calls == []  # never touched the network


def test_fetch_rejects_garbage_max_bytes(fetch_env):
    fetch_env.routes["http://example.com/page"] = _text_page(b"ok")
    with pytest.raises(LocalToolError):
        web_fetch("http://example.com/page", max_bytes="10MB")


def test_fetch_client_disables_trust_env(fetch_env, monkeypatch):
    # With HTTP(S)_PROXY set, trust_env=True would let the proxy do its own
    # DNS and connect anywhere, bypassing the guard. The client must opt out.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    captured: dict = {}
    real_client = httpx.Client

    def recording_client(*args, **kwargs):
        captured.update(kwargs)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(web_tool_impls.httpx, "Client", recording_client)
    fetch_env.routes["http://example.com/page"] = _text_page(b"ok")
    assert web_fetch("http://example.com/page") == "ok"
    assert captured.get("trust_env") is False


def test_fetch_httpx_invalid_url_becomes_local_tool_error(fetch_env):
    # \x7f in the path passes urlsplit and the guard, but httpx's stricter
    # parser rejects it with httpx.InvalidURL — NOT an HTTPError subclass,
    # so it must be caught explicitly.
    with pytest.raises(LocalToolError, match="invalid-url"):
        web_fetch("http://example.com/\x7f")


def test_cache_keyed_by_max_bytes(fetch_env):
    """A small-cap fetch must not poison a later full-cap fetch (spec §1)."""
    body = b"y" * 600
    fetch_env.routes["http://example.com/sized"] = _text_page(body)
    small = web_fetch("http://example.com/sized", max_bytes=100)
    assert "truncated" in small
    fetch_env.clock.now += 2.0  # clear the rate-limit interval
    full = web_fetch("http://example.com/sized")
    assert "truncated" not in full
    # Two distinct requests: different caps are different cache entries.
    assert fetch_env.calls.count("http://example.com/sized") == 2


def test_cache_entry_cap_evicts_earliest_expiry(fetch_env):
    from tldw_chatbook.Tools.web_tool_impls import FETCH_CACHE_MAX_ENTRIES

    for i in range(FETCH_CACHE_MAX_ENTRIES):
        url = f"http://example.com/p{i}"
        fetch_env.routes[url] = _text_page(f"page {i}".encode())
        web_fetch(url)
        fetch_env.clock.now += 2.0
    assert len(web_tool_impls._fetch_cache) == FETCH_CACHE_MAX_ENTRIES
    # One more insert evicts the earliest-expiry entry (p0).
    fetch_env.routes["http://example.com/extra"] = _text_page(b"extra")
    web_fetch("http://example.com/extra")
    assert len(web_tool_impls._fetch_cache) == FETCH_CACHE_MAX_ENTRIES
    assert ("http://example.com/p0", FETCH_MAX_BYTES) not in web_tool_impls._fetch_cache
    assert ("http://example.com/extra", FETCH_MAX_BYTES) in web_tool_impls._fetch_cache
