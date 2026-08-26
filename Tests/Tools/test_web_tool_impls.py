import socket
import sys
import threading
import zipfile
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import (
    FETCH_MAX_BYTES,
    FETCH_MAX_REDIRECTS,
    LocalToolError,
    robots_allows_for_scrape,
    validate_outbound_url,
    web_fetch,
)


# ---------------------------------------------------------------------------
# Module ephemerality: spec §5 guards against persistence layer coupling
# ---------------------------------------------------------------------------


def test_module_never_imports_persistence():
    """Verify web_tool_impls never imports database or media-storage modules.

    Spec §5 requirement: web_tool_impls is a pure-helper module with no
    coupling to the application's persistence layer (Client_Media_DB_v2,
    ChaChaNotes_DB, Local_Ingestion, RAG_Indexing, sqlite3, etc.).
    """
    import inspect
    import re
    src = inspect.getsource(web_tool_impls)
    # tempfile/mkstemp/mkdtemp joined the pattern with task-1359's binary
    # support: its amended AC ("zero on-disk persistence") rests on THIS
    # test, so the guard must actually cover temp-file writes.
    assert re.search(r"Client_Media_DB|ChaChaNotes|Local_Ingestion|RAG_Indexing|sqlite3|tempfile|mkstemp|mkdtemp", src) is None


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
    # robots.txt enforcement (task-2833) defaults to ON in shipped config,
    # but every existing test in this module was written against a world
    # with no robots machinery at all: with the real default, a robots
    # pre-fetch would add transport-call entries and an extra
    # _enforce_rate_limit hit that break exact-list/count/sleep assertions
    # across this file (design doc, Critical 1). This is a TEST-FIXTURE
    # default only -- the shipped config default remains true. Robots
    # tests opt back in explicitly via their own monkeypatch.
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": False}
    )
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


def test_fetch_pdf_too_large_message_reflects_configured_ceiling(fetch_env, monkeypatch):
    """The [too-large] message must render the number FROM PDF_MAX_BYTES,
    not a hardcoded '20 MB' string — a caller that monkeypatches the
    constant to a non-default value must see THAT value quoted back, not
    the module's original default. Uses raw sniffable bytes (not a real
    pymupdf PDF): the [too-large] refusal fires on size alone, before any
    parse. Forces _pymupdf_available() True regardless of the real
    environment: this test is about MESSAGE RENDERING in the [too-large]
    branch, not extraction, and sub-item (f) made that branch reachable
    only when pymupdf is (believed) present — without this patch, a plain
    `.[dev]` install (no pdf extra) would take the [missing-dep] branch
    instead and this test would fail, not skip."""
    monkeypatch.setattr(web_tool_impls, "_pymupdf_available", lambda: True)
    monkeypatch.setattr(web_tool_impls, "PDF_MAX_BYTES", 3 * 1024 * 1024)
    body = b"%PDF-1.4\n" + b"x" * (3 * 1024 * 1024 + 100)
    fetch_env.routes["http://example.com/huge3.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"too-large.*3 MB.*media ingestion"):
        web_fetch("http://example.com/huge3.pdf")


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


def test_pymupdf_available_spec_less_stub_returns_false_not_valueerror(fetch_env, monkeypatch):
    """_pymupdf_available() must be TOTAL: importlib.util.find_spec raises a
    raw ValueError (not caught anywhere else in this module) when
    sys.modules holds a stub entry with __spec__ = None — e.g. a test or
    a bad partial-import elsewhere in the process leaves such a stub
    behind. That must not escape web_fetch as an uncaught ValueError; the
    module's failure contract is all-LocalToolError."""
    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(__spec__=None))
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(b"%PDF-1.4 stub")
    with pytest.raises(LocalToolError, match=r"missing-dep"):
        web_fetch("http://example.com/doc.pdf")


def test_fetch_pdf_missing_dep_skips_20mb_download(fetch_env, monkeypatch):
    """When pymupdf is absent, web_fetch must decide that BEFORE opening the
    20 MB PDF read ceiling: pass pdf_max_bytes=None so the stream aborts at
    the caller's ordinary max_bytes cap, then raise [missing-dep] — never
    download up to 20 MB of a PDF it cannot parse anyway. Guard iterator
    proves the byte cap actually in effect: it raises if pulled past the
    default FETCH_MAX_BYTES cap, which only happens if the 20 MB ceiling
    were (wrongly) still in play."""
    monkeypatch.setattr(web_tool_impls, "_pymupdf_available", lambda: False)

    def guarded_chunks():
        # One oversized chunk, already past FETCH_MAX_BYTES: _fetch_once
        # must break out of the read loop without pulling a second chunk.
        yield b"%PDF-" + b"a" * (FETCH_MAX_BYTES + 5000)
        raise AssertionError(
            "guarded iterator pulled past the caller's byte cap — the 20 MB "
            "PDF ceiling was used despite pymupdf being unavailable"
        )

    fetch_env.routes["http://example.com/huge.pdf"] = httpx.Response(
        200, content=guarded_chunks(), headers={"content-type": "application/pdf"}
    )
    with pytest.raises(LocalToolError, match=r"missing-dep"):
        web_fetch("http://example.com/huge.pdf")


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
    """A body shorter than the 12-byte sniff window must fall through
    _fetch_once's post-loop fallback sniff (`kind` never resolves inside
    the loop) without error, and extract as plain text — not be
    misdetected as any binary kind. Delivered one byte per chunk."""
    body = b"abc"  # 3 bytes: shorter than the %PDF- magic prefix
    fetch_env.routes["http://example.com/short"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "text/plain"},
    )
    result = web_fetch("http://example.com/short")
    assert result == "abc"


# ---------------------------------------------------------------------------
# robots.txt enforcement (task-2833)
# ---------------------------------------------------------------------------
#
# fetch_env's own fixture default sets respect_robots_txt=False (existing-
# suite compatibility, design doc Critical 1) -- every test below opts back
# in explicitly via _enable_robots().

def _enable_robots(monkeypatch, respect: bool = True) -> None:
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": respect}
    )


def test_fetch_robots_disallowed_path_refused(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /private\n")
    fetch_env.routes["http://example.com/private/page"] = _text_page(b"secret")
    with pytest.raises(LocalToolError) as exc_info:
        web_fetch("http://example.com/private/page")
    assert str(exc_info.value).startswith("[robots-disallowed] http://example.com/private/page")
    assert "http://example.com/private/page" not in fetch_env.calls  # blocked before the hop


def test_fetch_robots_allowed_path_proceeds(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /private\n")
    fetch_env.routes["http://example.com/public"] = _text_page(b"hello public")
    assert web_fetch("http://example.com/public") == "hello public"


def test_fetch_robots_specific_ua_beats_wildcard(fetch_env, monkeypatch):
    """Fixture-authoring caveat (design doc Minor 6): stdlib RobotFileParser
    is first-match-wins in FILE order, not longest-path -- irrelevant here
    since each group has exactly one rule, but the file deliberately puts
    the wildcard group's Disallow SECOND to prove it is our own specific
    group (declared first) that actually governs us, not file order."""
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: tldw-chatbook-web-fetch\n"
        b"Disallow: /secret\n"
        b"\n"
        b"User-agent: *\n"
        b"Disallow: /forbidden\n"
    )
    fetch_env.routes["http://example.com/secret"] = _text_page(b"nope")
    fetch_env.routes["http://example.com/forbidden"] = _text_page(b"actually ours")
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch("http://example.com/secret")
    fetch_env.clock.now += 2.0  # clear the per-domain rate-limit interval
    # Our specific-UA group applies and does NOT disallow /forbidden -- the
    # wildcard-only rule is never even consulted for our own user agent.
    assert web_fetch("http://example.com/forbidden") == "actually ours"


def test_fetch_robots_missing_route_fails_open(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    # No robots.txt route registered at all -> fetch fails -> fail open.
    assert web_fetch("http://example.com/page") == "hello"


def test_fetch_robots_500_fails_open(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = httpx.Response(500)
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"


def test_fetch_robots_garbage_body_fails_open(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"\x00\x01 not remotely robots.txt syntax \xff\xfe"
    )
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"


def test_fetch_robots_truncated_body_fails_open(fetch_env, monkeypatch):
    """A body truncated at ROBOTS_MAX_BYTES must not be trusted (design doc
    Minor 7): a half-file could silently drop trailing Disallow lines. This
    robots.txt, if read in FULL, would disallow everything -- truncation
    must still fail the fetch open."""
    _enable_robots(monkeypatch)
    huge = b"User-agent: *\nDisallow: /\n" + b"# padding\n" * web_tool_impls.ROBOTS_MAX_BYTES
    assert len(huge) > web_tool_impls.ROBOTS_MAX_BYTES
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(huge)
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"


def test_fetch_robots_cache_not_refetched_on_second_request(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nAllow: /\n")
    fetch_env.routes["http://example.com/a"] = _text_page(b"page a")
    fetch_env.routes["http://example.com/b"] = _text_page(b"page b")
    web_fetch("http://example.com/a")
    assert fetch_env.calls.count("http://example.com/robots.txt") == 1
    fetch_env.clock.now += 2.0  # clear the per-domain rate-limit interval
    web_fetch("http://example.com/b")
    assert fetch_env.calls.count("http://example.com/robots.txt") == 1  # cached, no re-fetch


def test_fetch_robots_cache_ttl_expiry_refetches(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nAllow: /\n")
    fetch_env.routes["http://example.com/a"] = _text_page(b"page a")
    fetch_env.routes["http://example.com/b"] = _text_page(b"page b")
    web_fetch("http://example.com/a")
    assert fetch_env.calls.count("http://example.com/robots.txt") == 1
    fetch_env.clock.now += web_tool_impls.ROBOTS_CACHE_TTL_SECONDS + 1
    web_fetch("http://example.com/b")
    assert fetch_env.calls.count("http://example.com/robots.txt") == 2  # TTL expired, re-fetched


def test_fetch_robots_negative_cache_holds_for_ttl(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    # No robots.txt route registered: the fetch fails (KeyError inside the
    # mock handler), caught broadly, cached as None (fail open).
    fetch_env.routes["http://example.com/a"] = _text_page(b"page a")
    fetch_env.routes["http://example.com/b"] = _text_page(b"page b")
    web_fetch("http://example.com/a")
    first_attempts = fetch_env.calls.count("http://example.com/robots.txt")
    assert first_attempts == 1
    fetch_env.clock.now += 2.0  # well under ROBOTS_CACHE_TTL_SECONDS
    web_fetch("http://example.com/b")
    assert fetch_env.calls.count("http://example.com/robots.txt") == first_attempts  # negative cache held


def test_fetch_robots_redirect_into_disallowed_path_refused_mid_chain(fetch_env, monkeypatch):
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /private\n")
    fetch_env.routes["http://example.com/start"] = httpx.Response(
        302, headers={"location": "http://example.com/private/page"}
    )
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch("http://example.com/start")
    assert "http://example.com/private/page" not in fetch_env.calls  # blocked before the hop


def test_fetch_robots_toggle_off_makes_no_robots_fetch(fetch_env):
    # fetch_env's own fixture default is respect_robots_txt=False; this
    # test proves a PRESENT, disallowing robots.txt is never even fetched
    # while the toggle is off.
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /\n")
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"
    assert "http://example.com/robots.txt" not in fetch_env.calls


def test_fetch_robots_cache_hit_rechecks_and_refuses(fetch_env, monkeypatch):
    """Ruling 3: a cache-hit web_fetch re-checks robots the same way it
    re-checks SSRF policy -- a cached body plus a newly-disallowing
    robots.txt must refuse, not silently hand back the cached text."""
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nAllow: /\n")
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"

    # Robots policy changes; force the next call to see it by clearing only
    # the robots cache (the fetch/body cache stays warm and TTL-valid).
    web_tool_impls._robots_cache.clear()
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /\n")
    fetch_env.clock.now += 2.0  # clear the per-domain rate-limit interval
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch("http://example.com/page")
    # The page body itself was fetched only once -- the refusal came from
    # the CACHE-HIT path's robots re-check, not a second real fetch.
    assert fetch_env.calls.count("http://example.com/page") == 1


def test_fetch_robots_txt_fetch_is_itself_rate_limited(fetch_env, monkeypatch):
    """Ruling 5: the robots.txt fetch goes through the same per-domain rate
    limiter as any other request -- two back-to-back rate-limited requests
    to a brand-new host (robots.txt, then the page) cost one sleep."""
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nAllow: /\n")
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    web_fetch("http://example.com/page")
    assert fetch_env.clock.sleeps == [pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# robots.txt enforcement -- fix round 1 (review findings)
# ---------------------------------------------------------------------------

def test_fetch_robots_txt_redirect_is_followed_and_enforced(fetch_env, monkeypatch):
    """Important 1: a redirecting robots.txt (e.g. HTTP canonicalization)
    must be FOLLOWED, not treated as an unreachable fetch -- the latter
    negative-caches the host (fail open) for ROBOTS_CACHE_TTL_SECONDS,
    silently disabling enforcement for any host whose robots.txt 3xxs."""
    _enable_robots(monkeypatch)
    fetch_env.routes["http://example.com/robots.txt"] = httpx.Response(
        301, headers={"location": "http://example.com/static/robots.txt"}
    )
    fetch_env.routes["http://example.com/static/robots.txt"] = _text_page(
        b"User-agent: *\nDisallow: /\n"
    )
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch("http://example.com/page")
    # The redirect target must have actually been followed and consulted,
    # not just the initial 301 response.
    assert "http://example.com/robots.txt" in fetch_env.calls
    assert "http://example.com/static/robots.txt" in fetch_env.calls


def test_fetch_robots_txt_redirect_loop_exhausts_cap_and_fails_open(fetch_env, monkeypatch):
    """A robots.txt redirect chain that never resolves must still fail open
    once the bounded hop cap is exhausted, not raise or hang."""
    _enable_robots(monkeypatch)
    for i in range(FETCH_MAX_REDIRECTS + 2):
        fetch_env.routes[f"http://example.com/robots{i}.txt"] = httpx.Response(
            301, headers={"location": f"http://example.com/robots{i + 1}.txt"}
        )
    # The FIRST hop must be at the well-known /robots.txt location.
    fetch_env.routes["http://example.com/robots.txt"] = httpx.Response(
        301, headers={"location": "http://example.com/robots0.txt"}
    )
    fetch_env.routes["http://example.com/page"] = _text_page(b"hello")
    assert web_fetch("http://example.com/page") == "hello"


def test_webfetch_settings_defaults_to_true_with_no_config_override():
    """Important 2a: _webfetch_settings() itself is otherwise exercised by
    ZERO tests -- every fixture/helper in this file replaces the whole
    seam, so a section-name typo or a flipped default in the real function
    would ship robots-off with every other test green. This calls the REAL
    function against the per-test sandboxed config (autouse
    isolate_test_environment in Tests/conftest.py provides a fresh
    per-test config dir with no [webfetch] respect_robots_txt override
    present, i.e. the shipped default)."""
    assert web_tool_impls._webfetch_settings() == {"respect_robots_txt": True}


def test_webfetch_settings_raw_string_false_disables(monkeypatch):
    """Important 2b."""
    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key, default):
        if (section, key) == ("webfetch", "respect_robots_txt"):
            return "false"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    assert web_tool_impls._webfetch_settings() == {"respect_robots_txt": False}


def test_webfetch_settings_raw_string_true_uppercase_stays_enabled(monkeypatch):
    """Important 2c: an uppercase "TRUE" must not be misread as disabled --
    the same lesson _deep_search_settings recorded, applied to a flag whose
    default is already True."""
    import tldw_chatbook.config as config_module

    def fake_get_cli_setting(section, key, default):
        if (section, key) == ("webfetch", "respect_robots_txt"):
            return "TRUE"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    assert web_tool_impls._webfetch_settings() == {"respect_robots_txt": True}


@pytest.fixture
def fetch_env_default_settings(monkeypatch):
    """Like fetch_env, but deliberately does NOT monkeypatch
    _webfetch_settings -- used for the one composition test (Important 2d)
    that proves the REAL, unpatched settings seam defaults robots
    enforcement ON end-to-end against the per-test sandboxed config."""
    routes: dict[str, object] = {}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        item = routes[url]
        if isinstance(item, Exception):
            raise item
        return item

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


def test_webfetch_default_on_end_to_end_refuses_without_patching_seam(fetch_env_default_settings):
    """Important 2d: composition test -- sandboxed default config (no
    monkeypatch of _webfetch_settings anywhere in this test) plus a
    disallowing robots route means web_fetch must refuse, proving the
    real default really is on end-to-end, not just at the unit level."""
    env = fetch_env_default_settings
    env.routes["http://example.com/robots.txt"] = _text_page(b"User-agent: *\nDisallow: /\n")
    env.routes["http://example.com/page"] = _text_page(b"hello")
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch("http://example.com/page")


# ---------------------------------------------------------------------------
# robots.txt enforcement -- fix round 2 (Qodo PR review finding)
# ---------------------------------------------------------------------------

# 2606:4700::1111 (Cloudflare anycast), not a 2001:db8::/32 documentation
# address: Python's ipaddress module classifies the documentation range as
# is_private=True, so it would never even reach the robots check -- the
# ordinary SSRF guard (_validate_hop, shared by every hop) would refuse it
# first. This is a literal-IP URL either way: validate_outbound_url's
# literal-IP branch handles it directly (ipaddress.ip_address(host)
# succeeds), so no DNS resolution happens at all -- fetch_env's IPv4-only
# fake getaddrinfo is irrelevant to these two tests.
_IPV6_LITERAL = "2606:4700::1111"


def test_fetch_robots_ipv6_literal_host_disallowed_refused(fetch_env, monkeypatch):
    """Qodo finding: urlsplit(...).hostname strips the [...] brackets an
    IPv6 literal needs in a URL. Without re-bracketing it for the
    constructed robots.txt URL, the assembled string is malformed,
    _validate_hop rejects it inside _fetch_robots_parser, and the broad
    fail-open catch there silently disables robots enforcement for EVERY
    IPv6-literal host -- an input class validate_outbound_url explicitly
    supports."""
    _enable_robots(monkeypatch)
    fetch_env.routes[f"http://[{_IPV6_LITERAL}]/robots.txt"] = _text_page(
        b"User-agent: *\nDisallow: /\n"
    )
    fetch_env.routes[f"http://[{_IPV6_LITERAL}]/page"] = _text_page(b"hello")
    with pytest.raises(LocalToolError, match=r"\[robots-disallowed\]"):
        web_fetch(f"http://[{_IPV6_LITERAL}]/page")
    # The robots.txt URL was actually well-formed and fetched -- not
    # silently skipped via the fail-open path.
    assert f"http://[{_IPV6_LITERAL}]/robots.txt" in fetch_env.calls


def test_fetch_robots_ipv6_literal_host_allowed_proceeds(fetch_env, monkeypatch):
    """Control for the test above: an ALLOWING robots.txt on the same
    IPv6-literal host must let the fetch proceed normally."""
    _enable_robots(monkeypatch)
    fetch_env.routes[f"http://[{_IPV6_LITERAL}]/robots.txt"] = _text_page(
        b"User-agent: *\nAllow: /\n"
    )
    fetch_env.routes[f"http://[{_IPV6_LITERAL}]/page"] = _text_page(b"hello")
    assert web_fetch(f"http://[{_IPV6_LITERAL}]/page") == "hello"


# ---------------------------------------------------------------------------
# Binary-file fetch: images, ZIP archives, audio
# (task-1359, Docs/superpowers/specs/2026-08-07-web-fetch-binary-design.md)
# ---------------------------------------------------------------------------

try:
    _webp_probe = BytesIO()
    Image.new("RGB", (2, 2)).save(_webp_probe, format="WEBP")
    _webp_supported = True
except Exception:
    _webp_supported = False

requires_webp = pytest.mark.skipif(
    not _webp_supported, reason="Pillow webp write support not available"
)


def _png_bytes(size: tuple[int, int] = (16, 32)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=(200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (40, 20)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=(10, 200, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def _webp_bytes(size: tuple[int, int] = (6, 6)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=(5, 5, 200)).save(buf, format="WEBP")
    return buf.getvalue()


def _zip_bytes(members: dict) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _zip_bytes_with_encrypted_flag(name: str, data: bytes) -> bytes:
    """A ZIP whose one member is FLAGGED encrypted (flag_bits bit 0 set)
    without actually being encrypted -- infolist() doesn't care (encryption
    only blocks a member's own .read()), so this is sufficient to exercise
    the "(encrypted)" annotation without needing genuine ZipCrypto/AES
    encryption, which the stdlib zipfile writer doesn't support anyway.

    Setting ``ZipInfo.flag_bits`` before ``writestr()`` does NOT survive:
    the writer recomputes the general-purpose bit flag itself (verified
    interactively), clobbering whatever was set beforehand. So this
    patches the flag field directly in the CENTRAL DIRECTORY record's raw
    bytes after writing -- the same field a genuinely encrypted zip would
    carry, and the only one ``infolist()`` actually reads.
    """
    raw = bytearray(_zip_bytes({name: data}))
    idx = raw.find(b"PK\x01\x02")  # central directory file header signature
    assert idx != -1, "central directory record not found"
    flag_offset = idx + 8  # general purpose bit flag field
    flags = int.from_bytes(raw[flag_offset:flag_offset + 2], "little")
    flags |= 0x1
    raw[flag_offset:flag_offset + 2] = flags.to_bytes(2, "little")
    return bytes(raw)


def _binary_response(body: bytes, content_type: str) -> httpx.Response:
    headers = {"content-type": content_type} if content_type else {}
    return httpx.Response(200, content=body, headers=headers)


# --- images -----------------------------------------------------------------


def test_fetch_png_returns_image_metadata(fetch_env):
    fetch_env.routes["http://example.com/pic.png"] = _binary_response(
        _png_bytes((16, 32)), "image/png"
    )
    result = web_fetch("http://example.com/pic.png")
    assert result.startswith("[image] PNG 16×32,")


def test_fetch_jpeg_returns_image_metadata(fetch_env):
    fetch_env.routes["http://example.com/pic.jpg"] = _binary_response(
        _jpeg_bytes((40, 20)), "image/jpeg"
    )
    result = web_fetch("http://example.com/pic.jpg")
    assert result.startswith("[image] JPEG 40×20,")


def test_fetch_corrupt_image_bytes_refused(fetch_env):
    body = b"\x89PNG\r\n\x1a\n" + b"not a real png payload, just garbage bytes"
    fetch_env.routes["http://example.com/bad.png"] = _binary_response(body, "image/png")
    with pytest.raises(LocalToolError, match=r"\[image-error\]"):
        web_fetch("http://example.com/bad.png")


def test_fetch_png_sniff_beats_mislabeled_html_content_type(fetch_env):
    """PNG served with a wrong declared type: the sniff wins over the
    declared type (design doc ruling 3), same shape as the PDF precedent."""
    fetch_env.routes["http://example.com/mislabeled"] = _binary_response(
        _png_bytes((5, 5)), "text/html"
    )
    result = web_fetch("http://example.com/mislabeled")
    assert result.startswith("[image] PNG 5×5,")


def test_fetch_image_over_binary_ceiling_refused(fetch_env, monkeypatch):
    monkeypatch.setattr(web_tool_impls, "BINARY_MAX_BYTES", 100)
    body = b"\x89PNG\r\n\x1a\n" + b"x" * 500
    fetch_env.routes["http://example.com/huge.png"] = _binary_response(body, "image/png")
    with pytest.raises(LocalToolError, match=r"too-large.*image.*media ingestion"):
        web_fetch("http://example.com/huge.png")


@requires_webp
def test_fetch_webp_dribbled_one_byte_at_a_time_still_sniffed(fetch_env):
    """The 12-byte WEBP two-anchor magic (RIFF....WEBP, size field at [4:8]
    ignored) must still resolve when delivered one byte per chunk --
    generalizes the PDF dribble test to the wider sniff window (design
    doc, Minor 12)."""
    body = _webp_bytes((6, 6))
    fetch_env.routes["http://example.com/pic.webp"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "application/octet-stream"},
    )
    result = web_fetch("http://example.com/pic.webp")
    assert result.startswith("[image] WEBP 6×6,")


# --- sniff robustness ---------------------------------------------------


def test_fetch_body_shorter_than_sniff_window_still_extracts_as_text(fetch_env):
    """A body shorter than the 12-byte sniff prefix but LONGER than the old
    5-byte %PDF- magic must not crash the WEBP two-anchor index (body[8:12])
    and must still fall through to plain-text extraction."""
    body = b"hello world"  # 11 bytes: > 5, < _SNIFF_PREFIX_LEN (12)
    fetch_env.routes["http://example.com/short11"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "text/plain"},
    )
    result = web_fetch("http://example.com/short11")
    assert result == "hello world"


# --- ZIP archives -------------------------------------------------------


def test_fetch_zip_lists_members_with_sizes(fetch_env):
    body = _zip_bytes({"readme.txt": b"hello", "data/nested.bin": b"12345678"})
    fetch_env.routes["http://example.com/archive.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/archive.zip")
    assert result.startswith("[archive] ZIP,")
    assert "2 members" in result
    assert "readme.txt —" in result
    assert "data/nested.bin —" in result


def test_fetch_zip_over_list_max_shows_more_marker(fetch_env):
    members = {f"file{i}.txt": b"x" for i in range(25)}
    body = _zip_bytes(members)
    fetch_env.routes["http://example.com/many.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/many.zip")
    assert "25 members" in result
    assert "… and 5 more" in result
    # Exactly ARCHIVE_LIST_MAX member lines are actually listed.
    assert result.count("file") == web_tool_impls.ARCHIVE_LIST_MAX


def test_fetch_zip_hostile_member_names_flagged_not_verbatim(fetch_env):
    """Traversal screen (design doc ruling 2, mirrors
    chatbook_importer._validated_archive_parts): every shape of hostile
    name is flagged and repr-escaped, never printed as a raw path."""
    body = _zip_bytes({
        "../../etc/passwd": b"x",
        "/etc/shadow": b"x",
        "evil\\name": b"x",
        "C:/win/cmd.exe": b"x",
    })
    fetch_env.routes["http://example.com/hostile.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/hostile.zip")
    assert result.count("[suspicious name]") == 4
    # repr()'d, not printed as a bare path -- quoted form present for each.
    assert "'../../etc/passwd'" in result
    assert "'/etc/shadow'" in result
    assert "'evil\\\\name'" in result  # repr() escapes the backslash itself
    assert "'C:/win/cmd.exe'" in result


def test_member_display_name_flags_all_hostile_shapes():
    """Direct unit test of the traversal screen predicate -- NOT
    round-tripped through a real ZIP for the NUL case: CPython's
    zipfile.ZipInfo constructor silently truncates a filename at the first
    NUL byte on BOTH write and read (verified interactively:
    ``zipfile.ZipInfo("a\\x00b").filename == "a"``), so a genuine zip
    fixture can never carry a NUL byte through to ``infolist()`` -- the
    NUL branch in ``_member_display_name`` is only reachable via a direct
    call like this one."""
    from tldw_chatbook.Tools.web_tool_impls import _member_display_name

    assert _member_display_name("readme.txt") == "readme.txt"
    assert _member_display_name("dir/nested.bin") == "dir/nested.bin"
    for hostile in (
        "../../etc/passwd", "/etc/shadow", "evil\\name", "nul\x00byte", "C:/win/cmd.exe",
    ):
        result = _member_display_name(hostile)
        assert result == f"[suspicious name] {hostile!r}", f"not flagged correctly: {hostile!r}"


def test_fetch_zip_encrypted_member_annotated_not_refused(fetch_env):
    """Encryption blocks only a member's own .read() -- infolist() works
    fine, so an encrypted-but-well-formed ZIP lists successfully with an
    "(encrypted)" annotation, not an [archive-error] refusal (design doc
    ruling 2)."""
    body = _zip_bytes_with_encrypted_flag("secret.txt", b"shh")
    fetch_env.routes["http://example.com/locked.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/locked.zip")
    assert "[archive] ZIP" in result
    assert "secret.txt" in result
    assert "(encrypted)" in result


def test_fetch_zip_corrupt_bytes_refused(fetch_env):
    body = b"PK\x03\x04" + b"not a real zip structure at all, just garbage padding"
    fetch_env.routes["http://example.com/broken.zip"] = _binary_response(body, "application/zip")
    with pytest.raises(LocalToolError, match=r"\[archive-error\]"):
        web_fetch("http://example.com/broken.zip")


def test_fetch_zip_sniff_beats_octet_stream_content_type(fetch_env):
    body = _zip_bytes({"a.txt": b"hi"})
    fetch_env.routes["http://example.com/download"] = _binary_response(
        body, "application/octet-stream"
    )
    result = web_fetch("http://example.com/download")
    assert result.startswith("[archive] ZIP,")
    assert "a.txt —" in result


# --- audio ---------------------------------------------------------------


def test_fetch_audio_returns_metadata_line(fetch_env):
    fetch_env.routes["http://example.com/song.mp3"] = _binary_response(
        b"ID3 fake mp3 payload bytes", "audio/mpeg"
    )
    result = web_fetch("http://example.com/song.mp3")
    assert result.startswith("[audio] audio/mpeg,")


def test_fetch_audio_accepts_nonstandard_subtype_variants(fetch_env):
    """Audio has no sniff to rescue mislabels -- declared TOP-LEVEL type
    only (design doc ruling 2), so real-world variants like audio/mp3 and
    audio/x-wav must all resolve, not just the canonical audio/mpeg."""
    for ctype in ("audio/mp3", "audio/x-wav", "audio/mpeg"):
        fetch_env.routes["http://example.com/clip"] = _binary_response(b"binarydata", ctype)
        web_tool_impls._reset_state_for_tests()
        result = web_fetch("http://example.com/clip")
        assert result.startswith(f"[audio] {ctype},"), f"failed for content-type {ctype!r}"


# --- regression pin: unsupported binary types unchanged -------------------


def test_fetch_unsupported_binary_type_still_refused(fetch_env):
    """Regression pin (design doc, closing the recon gap): a binary type
    NOT on the allowlist keeps the pre-existing refusal, unchanged."""
    body = b"MZ\x90\x00" + b"x" * 50
    fetch_env.routes["http://example.com/app.exe"] = _binary_response(
        body, "application/x-msdownload"
    )
    with pytest.raises(LocalToolError, match=r"\[empty-content\] unsupported content type"):
        web_fetch("http://example.com/app.exe")


# --- review fix round 1 (control chars, GIF sniff, cap-ordering, sizes) ---


def test_fetch_zip_control_char_member_names_flagged_not_verbatim(fetch_env):
    """Review Important 1: a member name is attacker-controlled text
    embedded in a structured listing. A newline can FORGE a listing row,
    ESC/BEL can smuggle terminal control sequences, U+202E reverses the
    rendered name. All must be repr-escaped [suspicious name] entries;
    none may reach the output raw."""
    forged = "ok.txt\n[archive] ZIP, 1 B, 0 members"
    body = _zip_bytes({
        forged: b"x",
        "\x1b]0;evil\x07innocent.txt": b"x",
        "photo‮gnp.exe": b"x",
    })
    fetch_env.routes["http://example.com/sneaky.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/sneaky.zip")
    assert result.count("[suspicious name]") == 3
    assert "\x1b" not in result
    assert "‮" not in result
    # The forged header must never exist as its OWN line — repr-escaping
    # keeps it embedded (quoted) inside the [suspicious name] row, where
    # the substring is harmless.
    header_lines = [ln for ln in result.split("\n") if ln.startswith("[archive] ZIP,")]
    assert len(header_lines) == 1
    assert "\nok.txt\n" not in result


def test_fetch_zip_printable_unicode_member_names_list_plainly(fetch_env):
    """isprintable() must not overreach: ordinary non-ASCII names are
    legitimate and list verbatim."""
    body = _zip_bytes({"naïve.txt": b"x", "日本語.txt": b"y"})
    fetch_env.routes["http://example.com/unicode.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/unicode.zip")
    assert "[suspicious name]" not in result
    assert "naïve.txt —" in result
    assert "日本語.txt —" in result


def test_member_display_name_flags_control_and_invisible_chars():
    from tldw_chatbook.Tools.web_tool_impls import _member_display_name

    for hostile in ("evil\nname", "\x1b]0;evil\x07x.txt", "photo‮gnp.exe"):
        assert _member_display_name(hostile) == f"[suspicious name] {hostile!r}"
    for benign in ("naïve.txt", "日本語.txt"):
        assert _member_display_name(benign) == benign


def test_fetch_gif_sniff_beats_wrong_content_type(fetch_env):
    buf = BytesIO()
    Image.new("P", (7, 3)).save(buf, format="GIF")
    fetch_env.routes["http://example.com/anim"] = _binary_response(
        buf.getvalue(), "application/octet-stream"
    )
    result = web_fetch("http://example.com/anim")
    assert result.startswith("[image] GIF 7×3,")


def test_fetch_zip_over_binary_ceiling_refused_as_too_large(fetch_env, monkeypatch):
    """Review Minor 8: the ZIP case is the one that could LIE about what
    happened — a truncated central directory would BadZipFile into
    [archive-error] if the size check didn't come first. Pin the ordering:
    over-ceiling is [too-large], never [archive-error]."""
    monkeypatch.setattr(web_tool_impls, "BINARY_MAX_BYTES", 64)
    body = _zip_bytes({f"file{i}.txt": b"payload" for i in range(10)})
    assert len(body) > 64
    fetch_env.routes["http://example.com/big.zip"] = _binary_response(body, "application/zip")
    with pytest.raises(LocalToolError, match=r"\[too-large\]"):
        web_fetch("http://example.com/big.zip")


def test_fetch_zip_normal_members_carry_no_encrypted_annotation_and_sizes(fetch_env):
    """Review Minor 8: pin the ABSENCE of "(encrypted)" on ordinary
    members, and pin one _format_size suffix so the size column is not
    entirely unasserted."""
    body = _zip_bytes({"readme.txt": b"hello"})
    fetch_env.routes["http://example.com/plain.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/plain.zip")
    assert "(encrypted)" not in result
    assert "readme.txt — 5 B" in result


def test_fetch_zip_nonbadzipfile_parse_errors_normalized(fetch_env, monkeypatch):
    """Qodo PR #1442 (2): only LocalToolError may escape web_fetch — a
    hostile central directory raising beyond BadZipFile (struct/Overflow/
    Value errors) must still surface as [archive-error], with a FIXED
    message (never an attacker-influenced exception string)."""
    body = _zip_bytes({"a.txt": b"x"})  # build BEFORE the patch: zipfile is one shared module

    def explode(*args, **kwargs):
        raise ValueError("weird central directory field")
    monkeypatch.setattr(web_tool_impls.zipfile, "ZipFile", explode)
    fetch_env.routes["http://example.com/weird.zip"] = _binary_response(body, "application/zip")
    with pytest.raises(LocalToolError, match=r"\[archive-error\] could not read ZIP \(malformed metadata\)"):
        web_fetch("http://example.com/weird.zip")


def test_fetch_zip_long_member_names_display_capped(fetch_env):
    """Qodo PR #1442 (3): member names are attacker-controlled (the zip
    format allows 64 KiB per name); the listing must bound each line
    itself, not rely on the runtime's head-first truncation (which would
    eat the '… and N more' marker off the END)."""
    long_name = "a" * 5000 + ".txt"
    body = _zip_bytes({long_name: b"x", "short.txt": b"y"})
    fetch_env.routes["http://example.com/long.zip"] = _binary_response(body, "application/zip")
    result = web_fetch("http://example.com/long.zip")
    for line in result.split("\n"):
        assert len(line) <= web_tool_impls.ARCHIVE_MEMBER_NAME_MAX + 40, f"unbounded line: {len(line)} chars"
    assert "… [name truncated]" in result
    assert "short.txt — 1 B" in result


@requires_pymupdf
def test_fetch_small_max_bytes_still_fills_sniff_window_for_mislabeled_pdf(fetch_env):
    """Review Minor 3: a caller max_bytes below the 12-byte sniff window
    must not break the read loop before the sniff can resolve — the old
    code read the full PDF here, and so must the new code (cap-raise
    applies once the kind resolves). Dribbled one byte per chunk to force
    the mid-stream path."""
    body = _make_pdf(["tiny cap content"])
    fetch_env.routes["http://example.com/tinycap.pdf"] = httpx.Response(
        200,
        content=iter(bytes([b]) for b in body),
        headers={"content-type": "application/octet-stream"},
    )
    result = web_fetch("http://example.com/tinycap.pdf", max_bytes=4)
    # The fix guarantees the READ side: the full document reaches
    # extraction (no 5-byte fragment, no [pdf-error]) and the trailer
    # proves every page was processed. The EXTRACTED TEXT is still
    # bounded by the caller's max_bytes — pre-existing PDF-path behavior,
    # unchanged here.
    assert "tiny" in result
    assert "processed 1 of 1 pages" in result
    assert "[pdf-error]" not in result


# ---------------------------------------------------------------------------
# web_search result cache (task-2832) + the _fetch_cache TTL-expiry gap the
# spec review found (design doc, "What already exists")
# ---------------------------------------------------------------------------


def _search_payload(n=2):
    return {
        "results": [
            {"title": f"R{i}", "url": f"https://example.com/{i}", "content": f"body {i}"}
            for i in range(1, n + 1)
        ]
    }


def _patch_search(monkeypatch, fn):
    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch", fn
    )


def test_search_cache_hit_skips_backend(fetch_env, monkeypatch):
    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), _search_payload())[1])
    first = web_tool_impls.web_search("python asyncio")
    second = web_tool_impls.web_search("python asyncio")
    assert len(calls) == 1
    assert first == second
    assert "R1" in first


def test_search_cache_key_normalizes_whitespace_and_case(fetch_env, monkeypatch):
    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), _search_payload())[1])
    web_tool_impls.web_search("  Foo   BAR ")
    web_tool_impls.web_search("foo bar")
    assert len(calls) == 1
    # Different engine or count = different entry.
    web_tool_impls.web_search("foo bar", search_engine="bing")
    web_tool_impls.web_search("foo bar", result_count=3)
    assert len(calls) == 3


def test_search_cache_ttl_expiry_reinvokes_backend(fetch_env, monkeypatch):
    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), _search_payload())[1])
    web_tool_impls.web_search("stale query")
    fetch_env.clock.now += web_tool_impls.SEARCH_CACHE_TTL_SECONDS + 1
    web_tool_impls.web_search("stale query")
    assert len(calls) == 2


def test_search_backend_exception_not_cached(fetch_env, monkeypatch):
    calls = []

    def boom(**kw):
        calls.append(kw)
        raise RuntimeError("provider down")

    _patch_search(monkeypatch, boom)
    out = web_tool_impls.web_search("flaky")
    assert out.startswith("[search-failed]")
    web_tool_impls.web_search("flaky")
    assert len(calls) == 2  # second call re-invoked the backend


def test_search_error_envelope_and_malformed_not_cached(fetch_env, monkeypatch):
    """Design doc ruling 1 shapes (ii) and (iii): the unmarked
    malformed-response string and the [search-failed] envelope string are
    both transient-failure shapes — neither may pin for the TTL."""
    payloads = iter([
        {"error": "quota exceeded"},        # (iii) envelope error
        "not a dict at all",                # (ii) non-dict
        _search_payload(),                  # recovery
    ])
    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), next(payloads))[1])
    assert "[search-failed]" in web_tool_impls.web_search("recovering")
    assert "unexpected response format" in web_tool_impls.web_search("recovering")
    assert "R1" in web_tool_impls.web_search("recovering")
    assert len(calls) == 3  # nothing was cached until the genuine success


def test_search_confirmed_empty_not_cached(fetch_env, monkeypatch):
    """Design doc ruling 1 shape (v): a zero-result response is as often a
    broken parser as a true empty (this repo's tavily/searx history) — it
    must not pin for the TTL."""
    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), {"results": []})[1])
    out = web_tool_impls.web_search("nothing here")
    assert out.startswith("No results found for")
    web_tool_impls.web_search("nothing here")
    assert len(calls) == 2


def test_search_cache_eviction_bounds_size(fetch_env, monkeypatch):
    _patch_search(monkeypatch, lambda **kw: _search_payload())
    for i in range(web_tool_impls.SEARCH_CACHE_MAX_ENTRIES + 5):
        # Distinct expiry per entry (review Minor 4): with a frozen clock
        # every entry ties on expires_at and "earliest-expiry" is
        # untestable; advancing the clock makes the POLICY assertable.
        fetch_env.clock.now += 1
        web_tool_impls.web_search(f"query number {i}")
    assert len(web_tool_impls._search_cache) == web_tool_impls.SEARCH_CACHE_MAX_ENTRIES
    # Earliest-expiry policy: the very first entries are the evicted ones.
    remaining = {k[1] for k in web_tool_impls._search_cache}
    assert "query number 0" not in remaining
    assert f"query number {web_tool_impls.SEARCH_CACHE_MAX_ENTRIES + 4}" in remaining


def test_search_cache_logs_never_carry_query_text(fetch_env, monkeypatch, capsys):
    """Design doc ruling 6 (scoped to the wrapper layer): across a
    miss/hit/failure cycle, the query text appears in no captured log
    output. loguru writes to stderr in tests."""
    secret = "EXTREMELYUNIQUEQUERYTOKEN"
    payloads = iter([_search_payload(), RuntimeError("down")])

    def backend(**kw):
        item = next(payloads)
        if isinstance(item, Exception):
            raise item
        return item

    _patch_search(monkeypatch, backend)
    # Review Important 1: capsys does NOT observe loguru here (its handler
    # binds pytest's global capture at import) -- a sink is the only real
    # observer. The house pattern (~15 files) is a list-appending sink.
    from loguru import logger as _logger
    records: list[str] = []
    sink_id = _logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        web_tool_impls.web_search(secret)          # miss + store
        web_tool_impls.web_search(secret)          # hit
        web_tool_impls.web_search(secret + " v2")  # failure path (logs engine only)
    finally:
        _logger.remove(sink_id)
    assert not any(secret in r for r in records), records
    captured = capsys.readouterr()  # still catches stray print()s
    assert secret not in captured.err
    assert secret not in captured.out


def test_deep_search_phase1_bypasses_search_cache(fetch_env, monkeypatch):
    """Design doc ruling 4: web_deep_search's phase-1 fan-out must keep
    hitting the live pipeline even when web_search has a warm entry for
    the same normalized query. NEW fixture shape (spec review, Important
    2): mock perform_websearch at source and let the REAL
    generate_and_search run with sub-queries off."""
    from tldw_chatbook.Web_Scraping import WebSearch_APIs

    calls = []
    _patch_search(monkeypatch, lambda **kw: (calls.append(kw), _search_payload())[1])
    # Warm the web_search cache for this exact normalized query AND count
    # (review Important 2: the deep call below forwards result_count=2, so
    # the warm entry must use count=2 or the keys never coincide and the
    # test proves nothing).
    web_tool_impls.web_search("shared question", result_count=2)
    assert len(calls) == 1
    # Real generate_and_search, sub-queries off -> exactly one search call.
    out = WebSearch_APIs.generate_and_search(
        "shared question",
        {
            "engine": "duckduckgo",
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 2,
            "subquery_generation": False,
        },
    )
    assert len(calls) == 2, "deep-search phase 1 must not consult the web_search cache"
    assert out["web_search_results_dict"]
    # And phase 1 must not WRITE the shared cache either: still exactly the
    # one entry web_search itself stored.
    assert len(web_tool_impls._search_cache) == 1


def test_fetch_cache_ttl_expiry_refetches(fetch_env):
    """The _fetch_cache expiry gap the task-2832 spec review found: hit and
    size-eviction were tested, but nothing ever advanced the clock past
    FETCH_CACHE_TTL_SECONDS."""
    fetch_env.routes["http://example.com/page"] = _html_page()
    web_fetch("http://example.com/page")
    web_fetch("http://example.com/page")
    assert fetch_env.calls.count("http://example.com/page") == 1  # warm hit
    fetch_env.clock.now += web_tool_impls.FETCH_CACHE_TTL_SECONDS + 1
    web_fetch("http://example.com/page")
    assert fetch_env.calls.count("http://example.com/page") == 2  # stale -> refetch


# ---------------------------------------------------------------------------
# task-3770: _fetch_cache / _robots_cache locking (barrier-based, deterministic)
# ---------------------------------------------------------------------------
#
# A genuine race REPRODUCTION (concurrent threads actually corrupting a dict
# mid-iteration) is inherently flaky -- out of scope per the design doc. This
# instead proves each lock is genuinely HELD across its cache op: a worker
# thread is parked *inside* the critical section (via a fake clock whose
# .monotonic() blocks on an Event, standing in for the real work done under
# the lock), and the test asserts a non-blocking acquire from this thread
# fails while the worker is parked there, then succeeds again once released.


class _BlockingOnceClock:
    """`time` stand-in: the FIRST `.monotonic()` call signals `entered` and
    blocks on `release` before returning -- holds whatever critical section
    called it open for exactly as long as the test needs. Subsequent calls
    (a caller that reads the clock more than once) return immediately."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self._entered = entered
        self._release = release
        self._first = True

    def monotonic(self) -> float:
        if self._first:
            self._first = False
            self._entered.set()
            # Hard-fail rather than silently proceeding (Qodo PR #1451): a
            # silent expiry lets the worker leave the critical section
            # before the contention assertion runs on a loaded runner --
            # turning a deterministic test into a flaky one.
            assert self._release.wait(timeout=30), "lock-test release never set"
        return 0.0


@pytest.mark.parametrize(
    "lock_attr, put_fn_name, put_args",
    [
        ("_fetch_cache_lock", "_cache_put", (("http://lock-test.example/", 100), "cached text")),
        ("_robots_cache_lock", "_robots_cache_put", ("http://lock-test.example", None)),
    ],
    ids=["fetch_cache", "robots_cache"],
)
def test_cache_put_holds_its_lock_across_the_critical_section(
    monkeypatch, lock_attr, put_fn_name, put_args
):
    web_tool_impls._reset_state_for_tests()
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(web_tool_impls, "time", _BlockingOnceClock(entered, release))

    put_fn = getattr(web_tool_impls, put_fn_name)
    lock = getattr(web_tool_impls, lock_attr)

    worker = threading.Thread(target=lambda: put_fn(*put_args))
    worker.start()
    try:
        assert entered.wait(timeout=5), "worker never entered the critical section"
        # The lock must be HELD while the worker is parked inside the
        # critical section -- a non-blocking acquire from this thread must
        # fail. This is the actual discriminator: an unlocked _cache_put
        # (the pre-task-3770 shape) would let this acquire succeed.
        assert lock.acquire(blocking=False) is False, (
            f"{lock_attr} was not held while {put_fn_name} was inside its critical section"
        )
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive(), f"{put_fn_name} worker did not finish"
    # Lock is free again once the critical section exits.
    assert lock.acquire(blocking=False) is True
    lock.release()


# ---------------------------------------------------------------------------
# task-3260: robots_allows_for_scrape (web_deep_search's scrape-path helper)
# ---------------------------------------------------------------------------


def test_robots_allows_for_scrape_disallowed_path_refused(fetch_env):
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: *\nDisallow: /private\n"
    )
    assert robots_allows_for_scrape("http://example.com/private/page") is False


def test_robots_allows_for_scrape_allowed_path_proceeds(fetch_env):
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: *\nDisallow: /private\n"
    )
    assert robots_allows_for_scrape("http://example.com/public/page") is True


def test_robots_allows_for_scrape_client_disables_trust_env(fetch_env, monkeypatch):
    # Same ratchet as test_fetch_client_disables_trust_env above: with
    # HTTP(S)_PROXY set, trust_env=True would let the proxy do its own DNS
    # and connect anywhere, silently defeating validate_outbound_url's SSRF
    # check on the robots.txt URL. robots_allows_for_scrape (task-3260)
    # builds its OWN httpx.Client, a separate construction site from
    # _new_web_fetch_client -- this is its own, independent ratchet.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    captured: dict = {}
    real_client = httpx.Client

    def recording_client(*args, **kwargs):
        captured.update(kwargs)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(web_tool_impls.httpx, "Client", recording_client)
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: *\nAllow: /\n"
    )
    assert robots_allows_for_scrape("http://example.com/page") is True
    assert captured.get("trust_env") is False


def test_robots_allows_for_scrape_uses_own_truthful_user_agent(fetch_env):
    """Ruling 3: robots.txt is checked with _DEEP_SEARCH_ROBOTS_UA, distinct
    from _USER_AGENT (web_fetch) and _CRAWL_USER_AGENT (web_crawl) -- a
    group scoped to this tool's own UA product token must be honored
    specifically, not only the wildcard group. (RobotFileParser's own
    matching strips the *caller's* "/version" suffix before comparing --
    the file's own token is conventionally written without one, same as
    real-world "User-agent: Googlebot" groups.)"""
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: tldw-chatbook-deep-search\nDisallow: /\n"
        b"User-agent: *\nAllow: /\n"
    )
    assert robots_allows_for_scrape("http://example.com/page") is False


def test_robots_allows_for_scrape_unreachable_fails_open(fetch_env):
    # No robots.txt route registered at all -> fetch fails -> fail open.
    assert robots_allows_for_scrape("http://example.com/page") is True


def test_robots_allows_for_scrape_shares_cache_with_web_fetch(fetch_env, monkeypatch):
    """Ruling 3: robots_allows_for_scrape shares the module robots cache --
    a host already warmed by web_fetch's own robots consult must not cost
    a second robots.txt fetch here."""
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": True}
    )
    fetch_env.routes["http://example.com/robots.txt"] = _text_page(
        b"User-agent: *\nAllow: /\n"
    )
    fetch_env.routes["http://example.com/page"] = _html_page()
    web_fetch("http://example.com/page")
    assert fetch_env.calls.count("http://example.com/robots.txt") == 1

    assert robots_allows_for_scrape("http://example.com/other-page") is True
    assert fetch_env.calls.count("http://example.com/robots.txt") == 1  # cached, no re-fetch
