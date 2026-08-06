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


# ---------------------------------------------------------------------------
# PDF fetch (spec 2026-08-06 §1)
# ---------------------------------------------------------------------------

# NOT a module-level pytest.importorskip: that raises Skipped at import time,
# which would skip the WHOLE FILE (including the 27 pre-existing v1 tests and
# task 1's cache tests) in any environment without the optional `pdf` extra.
# Only the PDF-dependent tests below should skip; a plain `pip install
# -e ".[dev]"` install must still exercise SSRF/redirect/rate-limit/cache.
try:
    import pymupdf
except ImportError:
    pymupdf = None

requires_pymupdf = pytest.mark.skipif(
    pymupdf is None, reason="pymupdf not installed (pdf extra)"
)


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_response(body: bytes, content_type: str = "application/pdf") -> httpx.Response:
    headers = {"content-type": content_type} if content_type else {}
    return httpx.Response(200, content=body, headers=headers)


@requires_pymupdf
def test_fetch_extracts_pdf_text(fetch_env):
    body = _make_pdf(["alpha page one text", "beta page two text"])
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/doc.pdf")
    assert "alpha page one text" in result
    assert "beta page two text" in result


@requires_pymupdf
def test_fetch_pdf_sniff_beats_mislabeled_content_type(fetch_env):
    body = _make_pdf(["sniffed content"])
    for ctype in ("application/octet-stream", "text/html", ""):
        fetch_env.routes["http://example.com/mislabeled"] = _pdf_response(body, ctype)
        web_tool_impls._reset_state_for_tests()
        result = web_fetch("http://example.com/mislabeled")
        assert "sniffed content" in result, f"failed for content-type {ctype!r}"


@requires_pymupdf
def test_fetch_pdf_reads_past_html_cap(fetch_env):
    """Mid-stream cap raise: a >max_bytes PDF must be read in full (spec §1)."""
    filler = "lorem ipsum dolor sit amet " * 40
    body = _make_pdf([f"page {i} {filler}" for i in range(400)])
    assert len(body) > 64 * 1024
    fetch_env.routes["http://example.com/big.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/big.pdf", max_bytes=64 * 1024)
    assert "page 0" in result  # parsed => the full byte stream was read


@requires_pymupdf
def test_fetch_pdf_over_ceiling_refused(fetch_env, monkeypatch):
    monkeypatch.setattr(web_tool_impls, "PDF_MAX_BYTES", 1024)
    body = _make_pdf(["x" * 500] * 20)
    assert len(body) > 1024
    fetch_env.routes["http://example.com/huge.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"too-large.*media ingestion"):
        web_fetch("http://example.com/huge.pdf")


@requires_pymupdf
def test_fetch_pdf_extracted_text_truncated_with_page_count(fetch_env):
    body = _make_pdf([f"page {i} " + "words " * 200 for i in range(30)])
    fetch_env.routes["http://example.com/long.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/long.pdf", max_bytes=2048)
    assert "truncated: extracted text exceeded max_bytes=2048" in result
    assert "of 30 pages" in result
    # early stop: not every page was processed
    assert "processed 30 of 30" not in result


@requires_pymupdf
def test_fetch_pdf_encrypted_refused(fetch_env):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    body = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="hunter2")
    doc.close()
    fetch_env.routes["http://example.com/locked.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"pdf-error.*encrypted"):
        web_fetch("http://example.com/locked.pdf")


@requires_pymupdf
def test_fetch_pdf_textless_points_at_ocr(fetch_env):
    doc = pymupdf.open()
    doc.new_page()  # one blank page, no text layer
    body = doc.tobytes()
    doc.close()
    fetch_env.routes["http://example.com/scan.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"empty-content.*OCR"):
        web_fetch("http://example.com/scan.pdf")


@requires_pymupdf
def test_fetch_pdf_damaged_bytes_error(fetch_env):
    # pymupdf is sometimes lenient with garbage after a valid header: it may
    # raise at open (-> pdf-error) or yield a zero-page doc (-> empty-content).
    # Either way the caller gets a structured refusal, never a crash.
    fetch_env.routes["http://example.com/junk.pdf"] = _pdf_response(b"%PDF-1.7 garbage not a real pdf")
    with pytest.raises(LocalToolError, match=r"pdf-error|empty-content"):
        web_fetch("http://example.com/junk.pdf")


@requires_pymupdf
def test_fetch_pdf_missing_dep_message(fetch_env, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_pymupdf(name, *a, **k):
        if name == "pymupdf":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pymupdf)
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(_make_pdf(["hi"]))
    with pytest.raises(LocalToolError, match=r"missing-dep.*tldw_chatbook\[pdf\]"):
        web_fetch("http://example.com/doc.pdf")


@requires_pymupdf
def test_fetch_pdf_result_is_cached(fetch_env):
    body = _make_pdf(["cache me"])
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(body)
    first = web_fetch("http://example.com/doc.pdf")
    second = web_fetch("http://example.com/doc.pdf")
    assert first == second
    assert fetch_env.calls.count("http://example.com/doc.pdf") == 1


def test_fetch_html_types_unaffected(fetch_env):
    """Regression guard: ordinary HTML still goes through trafilatura/tag-strip."""
    fetch_env.routes["http://example.com/page"] = _html_page()
    assert "main article body sentence" in web_fetch("http://example.com/page")


# ---------------------------------------------------------------------------
# Multi-chunk streaming (coverage gap: every fixture above yields one chunk,
# so _fetch_once's mid-stream %PDF- sniff and short-body fallback were never
# actually exercised across multiple response.iter_bytes() iterations)
# ---------------------------------------------------------------------------

@requires_pymupdf
def test_fetch_pdf_dribbled_one_byte_at_a_time_still_sniffed(fetch_env):
    """Spec §1: 'The sniff buffers until at least 5 body bytes have
    arrived before deciding — a server dribbling one byte per chunk must
    not defeat it.' Mislabel content-type so detection depends entirely on
    the sniff (not the `declared == "application/pdf"` shortcut), and feed
    the body as one httpx chunk per byte."""
    body = _make_pdf(["dribbled content"])
    fetch_env.routes["http://example.com/dribble.pdf"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "application/octet-stream"},
    )
    result = web_fetch("http://example.com/dribble.pdf")
    assert "dribbled content" in result


def test_fetch_short_body_under_pdf_magic_length_extracts_as_text(fetch_env):
    """A body shorter than len(_PDF_MAGIC) == 5 bytes must fall through
    _fetch_once's post-loop fallback sniff (`is_pdf is None` never gets
    resolved inside the loop) without error, and extract as plain text —
    not be misdetected as a PDF. Delivered one byte per chunk."""
    body = b"abc"  # 3 bytes: shorter than the %PDF- magic prefix
    fetch_env.routes["http://example.com/short"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "text/plain"},
    )
    result = web_fetch("http://example.com/short")
    assert result == "abc"
