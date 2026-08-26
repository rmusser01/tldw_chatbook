"""web_crawl: pure-helper unit tests (no transport) + crawl behavior tests."""

import socket
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import (
    CRAWL_TITLE_MAX_CHARS,
    LocalToolError,
    _CrawlLinkParser,
    _coerce_budget,
    _crawl_host,
    _format_crawl_result,
    _normalize_crawl_url,
    _parse_sitemap,
)


def _parse(html: str) -> _CrawlLinkParser:
    parser = _CrawlLinkParser()
    parser.feed(html)
    parser.close()
    return parser


def test_parser_collects_links_title_and_base():
    p = _parse(
        "<html><head><title>My &amp; Page</title><base href='/de/'></head>"
        "<body><a href='a.html'>A</a><a href='/b'>B</a>"
        "<a>no href</a><a href='c.html'>C</a></body></html>"
    )
    assert p.title == "My & Page"
    assert p.base_href == "/de/"
    assert p.links == ["a.html", "/b", "c.html"]


def test_parser_survives_malformed_html():
    p = _parse("<a href='x'><b><title>t</<><a href='y'>")
    assert "x" in p.links


def test_title_accumulation_bounded():
    """An unclosed <title> followed by a huge stream of text data must not
    grow the accumulator unboundedly — handle_data is called once per text
    chunk, so an unclosed tag can otherwise concatenate arbitrarily much."""
    parser = _CrawlLinkParser()
    parser.feed("<title>" + ("x" * 100_000))
    parser.close()
    assert len(parser.title) <= CRAWL_TITLE_MAX_CHARS


def test_normalize_folds_www_case_and_fragment():
    assert (
        _normalize_crawl_url("HTTP://WWW.Example.COM/Path?q=1#frag")
        == "http://example.com/Path?q=1"
    )
    assert _normalize_crawl_url("http://example.com") == "http://example.com/"


def test_normalize_survives_malformed_urls():
    # Bad port should not raise; return the input unchanged for stable visited-set identity
    assert _normalize_crawl_url("http://example.com:abc/") == "http://example.com:abc/"
    # Malformed IPv6 should not raise; return the input unchanged
    assert _normalize_crawl_url("http://[::1") == "http://[::1"


def test_crawl_host_folds_www():
    assert _crawl_host("https://www.Example.com/x") == "example.com"
    assert _crawl_host("https://example.com/x") == "example.com"
    assert _crawl_host("not a url") == ""


def test_coerce_budget_clamps_and_defaults():
    assert _coerce_budget(3, 20, 40) == 3
    assert _coerce_budget(999, 20, 40) == 40
    assert _coerce_budget(0, 20, 40) == 1
    assert _coerce_budget(-5, 20, 40) == 1
    assert _coerce_budget("garbage", 20, 40) == 20
    assert _coerce_budget(None, 20, 40) == 20
    assert _coerce_budget("7", 20, 40) == 7


_URLSET = b"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a</loc></url>
  <url><loc> https://example.com/b </loc></url>
</urlset>"""

_INDEX = b"""<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/s1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/s2.xml</loc></sitemap>
</sitemapindex>"""


def test_parse_sitemap_urlset_and_index():
    pages, children = _parse_sitemap(_URLSET)
    assert pages == ["https://example.com/a", "https://example.com/b"]
    assert children == []
    pages, children = _parse_sitemap(_INDEX)
    assert pages == []
    assert children == ["https://example.com/s1.xml", "https://example.com/s2.xml"]


def test_parse_sitemap_garbage_raises():
    with pytest.raises(LocalToolError, match="crawl-failed"):
        _parse_sitemap(b"this is not xml at all")


def _page(url, title="T", excerpt="ex", marker=None):
    return {"url": url, "title": title, "excerpt": excerpt, "marker": marker}


def test_format_blocks_footer_and_marker():
    out = _format_crawl_result(
        [_page("http://e.com/1", "One", "first words"),
         _page("http://e.com/m.pdf", "", "", marker="[application/pdf]")],
        failed=2, blocked=1, stop_reason="page budget reached",
    )
    assert "1. One\n   URL: http://e.com/1\n   first words" in out
    assert "2. [application/pdf]\n   URL: http://e.com/m.pdf" in out
    assert out.endswith("Crawled 2 pages (2 failed, 1 blocked). Stopped: page budget reached.")


def test_format_total_cap_omits_pages():
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_RESULT_MAX_BYTES

    pages = [_page(f"http://e.com/{i}", f"T{i}", "x" * 190) for i in range(200)]
    out = _format_crawl_result(pages, failed=0, blocked=0, stop_reason="page budget reached")
    assert "further pages omitted" in out
    assert len(out.encode("utf-8")) < CRAWL_RESULT_MAX_BYTES + 2048  # footer + omission slack
    assert "Crawled 200 pages" in out  # footer reports the crawl, not the capped list


def test_format_empty_crawl_is_just_footer():
    out = _format_crawl_result([], failed=1, blocked=0, stop_reason="no more links within depth")
    assert out == "Crawled 0 pages (1 failed, 0 blocked). Stopped: no more links within depth."


def test_format_duplicate_redirects_skipped_clause():
    """New optional param (item 3): a nonzero count renders a third footer
    clause, following the existing `children_skipped` idiom exactly."""
    out = _format_crawl_result(
        [], failed=0, blocked=0, stop_reason="page budget reached",
        duplicates_skipped=1,
    )
    assert out == "Crawled 0 pages (0 failed, 0 blocked; 1 duplicate redirects skipped). Stopped: page budget reached."


def test_format_duplicate_redirects_skipped_clause_absent_when_zero():
    out = _format_crawl_result([], failed=0, blocked=0, stop_reason="page budget reached")
    assert "duplicate redirects skipped" not in out


# ---------------------------------------------------------------------------
# web_crawl BFS (transport-level tests, v1 fetch_env conventions)
# ---------------------------------------------------------------------------

from tldw_chatbook.Tools.web_tool_impls import web_crawl

_PUBLIC_IP = "93.184.216.34"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def crawl_env(monkeypatch):
    """MockTransport + fake DNS + fake clock, mirroring test_web_tool_impls."""
    routes: dict[str, object] = {}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        item = routes.get(url)
        if item is None:
            return httpx.Response(404)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (_PUBLIC_IP, 80))]
    )
    monkeypatch.setattr(web_tool_impls, "_transport", httpx.MockTransport(handler))
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


def _html(body_text: str, links: list[str] = (), title: str = "Page") -> httpx.Response:
    anchors = "".join(f"<a href='{href}'>l</a>" for href in links)
    html = (
        f"<html><head><title>{title}</title></head>"
        f"<body><p>{body_text}</p>{anchors}</body></html>"
    )
    return httpx.Response(200, content=html.encode(), headers={"content-type": "text/html"})


def _site(env, spec: dict) -> None:
    """spec: url -> (body_text, links) tuples or ready Responses."""
    for url, item in spec.items():
        if isinstance(item, tuple):
            body_text, links = item
            env.routes[url] = _html(body_text, links, title=f"Title {url.rsplit('/', 1)[-1]}")
        else:
            env.routes[url] = item


def test_crawl_lists_pages_with_titles_and_excerpts(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("home page words", ["/a", "/b"]),
        "http://example.com/a": ("alpha page words", []),
        "http://example.com/b": ("beta page words", []),
    })
    out = web_crawl("http://example.com/")
    assert "Title a" in out and "alpha page words" in out
    assert "Title b" in out and "beta page words" in out
    assert "Crawled 3 pages (0 failed, 0 blocked)" in out
    assert out.endswith("Stopped: no more links within depth.")


def test_crawl_respects_max_pages_budget(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", [f"/p{i}" for i in range(10)]),
        **{f"http://example.com/p{i}": (f"page {i}", []) for i in range(10)},
    })
    out = web_crawl("http://example.com/", max_pages=4)
    assert len(crawl_env.calls) == 4
    assert "Stopped: page budget reached." in out


def test_crawl_respects_max_depth(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("d0", ["/d1"]),
        "http://example.com/d1": ("d1", ["/d2"]),
        "http://example.com/d2": ("d2", ["/d3"]),
        "http://example.com/d3": ("d3", []),
    })
    web_crawl("http://example.com/", max_depth=1)
    assert "http://example.com/d2" not in crawl_env.calls
    assert "http://example.com/d1" in crawl_env.calls


def test_crawl_stays_on_host_and_folds_www(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", [
            "http://other.com/x",
            "http://www.example.com/a",     # same host after www fold
            "http://example.com/a",         # duplicate of the above
            "http://example.com/a#section", # fragment variant: same identity
        ]),
        "http://www.example.com/a": ("alpha", []),
    })
    web_crawl("http://example.com/")
    assert "http://other.com/x" not in crawl_env.calls
    # www/apex/fragment variants collapse to one fetch.
    assert crawl_env.calls.count("http://www.example.com/a") == 1
    assert not any(c.startswith("http://example.com/a") for c in crawl_env.calls)


def test_crawl_honors_base_href(crawl_env):
    crawl_env.routes["http://example.com/dir/"] = httpx.Response(
        200,
        content=(
            b"<html><head><title>B</title><base href='/other/'></head>"
            b"<body><a href='rel.html'>r</a></body></html>"
        ),
        headers={"content-type": "text/html"},
    )
    _site(crawl_env, {"http://example.com/other/rel.html": ("resolved", [])})
    web_crawl("http://example.com/dir/")
    assert "http://example.com/other/rel.html" in crawl_env.calls


def test_crawl_blocked_link_counted_not_fatal(crawl_env, monkeypatch):
    """DNS flips private mid-crawl (rebinding): the later same-host URL is
    guard-blocked, counted, and the crawl continues.

    Same-host crawls share one hostname, so per-URL DNS is impossible — the
    deterministic construction flips the answer inside an earlier page's
    handler. BFS order: / -> /ok (flips DNS) -> /blocked-later (resolves
    private, refused before any request)."""
    state = {"private": False}

    def dns(host, *a, **k):
        ip = "10.0.0.5" if state["private"] else _PUBLIC_IP
        return [(2, 1, 6, "", (ip, 80))]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    _site(crawl_env, {
        "http://example.com/": ("root", ["/ok", "/blocked-later"]),
        "http://example.com/blocked-later": ("never seen", []),
    })

    def ok_then_flip(request):
        state["private"] = True
        return _html("fine words", [])

    crawl_env.routes["http://example.com/ok"] = ok_then_flip
    out = web_crawl("http://example.com/")
    assert "1 blocked" in out
    assert "fine words" in out
    assert "http://example.com/blocked-later" not in crawl_env.calls


def test_crawl_failed_page_counted_not_fatal(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/dead", "/ok"]),
        "http://example.com/ok": ("fine", []),
    })
    crawl_env.routes["http://example.com/dead"] = httpx.Response(500)
    out = web_crawl("http://example.com/")
    assert "1 failed" in out
    assert "fine" in out


def test_crawl_ssrf_substring_in_url_not_misclassified_as_blocked(crawl_env):
    """A same-host link whose URL text happens to CONTAIN the literal
    substring "[ssrf]" and simply 404s must be counted as failed, not
    blocked. The classifier must key off the message's PREFIX (which only
    _validate_hop controls), not an unspoofable-in-name-only substring
    search — an attacker-served URL could otherwise forge a "blocked"
    classification for what is really an ordinary failure."""
    _site(crawl_env, {
        "http://example.com/": ("root", ["/x[ssrf]y"]),
    })
    crawl_env.routes["http://example.com/x[ssrf]y"] = httpx.Response(404)
    out = web_crawl("http://example.com/")
    assert "1 failed, 0 blocked" in out


def test_crawl_start_url_failure_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/"] = httpx.Response(500)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/")


def test_crawl_nonhtml_listed_with_marker_not_expanded(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/doc.pdf"]),
    })
    crawl_env.routes["http://example.com/doc.pdf"] = httpx.Response(
        200, content=b"%PDF-1.7 pretend", headers={"content-type": "application/pdf"}
    )
    out = web_crawl("http://example.com/")
    assert "[application/pdf]" in out
    assert "http://example.com/doc.pdf" in out


def test_crawl_offhost_redirect_listed_not_expanded(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/moved"]),
        "http://elsewhere.com/final": ("away content", ["/next-on-elsewhere"]),
    })
    crawl_env.routes["http://example.com/moved"] = httpx.Response(
        302, headers={"location": "http://elsewhere.com/final"}
    )
    out = web_crawl("http://example.com/")
    assert "http://elsewhere.com/final" in out       # listed at final URL
    assert "http://elsewhere.com/next-on-elsewhere" not in crawl_env.calls


def test_crawl_redirect_duplicate_targets_listed_once(crawl_env):
    """Two different same-host pages that both redirect to the SAME final
    URL must list that final URL once, not once per redirecting source —
    each redirect still spends its own attempt slot (the budget contract),
    but the duplicate row must not appear in the output or the page count.

    The spent-but-invisible attempt must still be surfaced somewhere: the
    footer's "N duplicate redirects skipped" clause (item 3) — root
    contributes 1 listed page (via /one's redirect), /two's redirect to the
    same target is the one deduped skip."""
    _site(crawl_env, {
        "http://example.com/": ("root", ["/one", "/two"]),
    })
    crawl_env.routes["http://example.com/one"] = httpx.Response(
        302, headers={"location": "http://example.com/target"}
    )
    crawl_env.routes["http://example.com/two"] = httpx.Response(
        302, headers={"location": "http://example.com/target"}
    )
    _site(crawl_env, {"http://example.com/target": ("target page words", [])})
    out = web_crawl("http://example.com/")
    assert out.count("URL: http://example.com/target") == 1
    assert "Crawled 2 pages" in out
    assert "1 duplicate redirects skipped" in out


def test_crawl_no_duplicates_omits_duplicate_redirects_clause(crawl_env):
    """A crawl with no deduped redirects must not mention the clause at
    all — it's additive, not a permanent zero-value fixture in the footer."""
    _site(crawl_env, {
        "http://example.com/": ("home page words", ["/a", "/b"]),
        "http://example.com/a": ("alpha page words", []),
        "http://example.com/b": ("beta page words", []),
    })
    out = web_crawl("http://example.com/")
    assert "duplicate redirects skipped" not in out


def test_crawl_redirect_into_private_space_blocked(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/trap"]),
    })
    crawl_env.routes["http://example.com/trap"] = httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data"}
    )
    out = web_crawl("http://example.com/")
    assert "1 blocked" in out
    assert "http://169.254.169.254/latest/meta-data" not in crawl_env.calls


def test_crawl_deadline_stops(crawl_env):
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_DEADLINE_SECONDS

    def slow_then_page(request):
        crawl_env.clock.now += CRAWL_DEADLINE_SECONDS + 1
        return _html("slow", [])

    _site(crawl_env, {"http://example.com/": ("root", ["/slow", "/never"])})
    crawl_env.routes["http://example.com/slow"] = slow_then_page
    _site(crawl_env, {"http://example.com/never": ("never", [])})
    out = web_crawl("http://example.com/")
    assert "Stopped: deadline reached." in out
    assert "http://example.com/never" not in crawl_env.calls


def test_crawl_deadline_stops_during_redirect_hop(crawl_env):
    """Between-hops deadline coverage: a redirect whose handler advances the
    clock past CRAWL_DEADLINE_SECONDS must cause the crawl to stop, and the
    redirect target must never be fetched (exercises _CrawlDeadline raise/catch)."""
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_DEADLINE_SECONDS

    def redirect_with_deadline_advance(request):
        crawl_env.clock.now += CRAWL_DEADLINE_SECONDS + 1
        return httpx.Response(302, headers={"location": "http://example.com/after-deadline"})

    _site(crawl_env, {"http://example.com/": ("root", ["/redirect"])})
    crawl_env.routes["http://example.com/redirect"] = redirect_with_deadline_advance
    _site(crawl_env, {"http://example.com/after-deadline": ("should not fetch", [])})
    out = web_crawl("http://example.com/")
    assert "Stopped: deadline reached." in out
    assert "http://example.com/after-deadline" not in crawl_env.calls


def test_crawl_rate_limits_between_pages(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/a"]),
        "http://example.com/a": ("alpha", []),
    })
    web_crawl("http://example.com/")
    assert crawl_env.clock.sleeps  # second same-domain fetch waited


def test_crawl_uses_crawl_user_agent(crawl_env):
    seen_agents: list[str] = []

    def capture(request):
        seen_agents.append(request.headers.get("user-agent", ""))
        return _html("root", [])

    crawl_env.routes["http://example.com/"] = capture
    web_crawl("http://example.com/")
    assert seen_agents == ["tldw-chatbook-web-crawl/1.0"]


def test_crawl_warm_writes_fetch_cache(crawl_env):
    from tldw_chatbook.Tools.web_tool_impls import web_fetch

    _site(crawl_env, {"http://example.com/": ("cache me please words", [])})
    web_crawl("http://example.com/")
    n_calls = len(crawl_env.calls)
    result = web_fetch("http://example.com/")
    assert "cache me please words" in result
    assert len(crawl_env.calls) == n_calls  # served from cache, no new request


def test_crawl_invalid_args(crawl_env):
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_crawl("")
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_crawl("   ")


# ---------------------------------------------------------------------------
# Fix-round: unguarded urljoin() ValueError + unmarked truncated cache write
# ---------------------------------------------------------------------------


def test_crawl_survives_malformed_href_in_link_loop(crawl_env):
    """A malformed href (urljoin raises ValueError: 'Invalid IPv6 URL') must
    be skipped, not crash the whole crawl."""
    _site(crawl_env, {
        "http://example.com/": ("root", ["http://[", "/good"]),
        "http://example.com/good": ("good page words", []),
    })
    out = web_crawl("http://example.com/")
    assert "good page words" in out
    assert "Crawled 2 pages (0 failed, 0 blocked)" in out


def test_crawl_survives_malformed_base_href(crawl_env):
    """A malformed <base href> must fall back to final_url as the resolution
    base, not crash the whole crawl."""
    crawl_env.routes["http://example.com/"] = httpx.Response(
        200,
        content=(
            b"<html><head><title>T</title><base href='http://['></head>"
            b"<body><a href='/good'>g</a></body></html>"
        ),
        headers={"content-type": "text/html"},
    )
    _site(crawl_env, {"http://example.com/good": ("good page words", [])})
    out = web_crawl("http://example.com/")
    assert "good page words" in out
    assert "http://example.com/good" in crawl_env.calls


def test_crawl_malformed_redirect_location_counted_as_failed(crawl_env):
    """A redirect Location that urljoin cannot parse becomes a per-page
    [invalid-url] failure, not an uncaught ValueError."""
    _site(crawl_env, {
        "http://example.com/": ("root", ["/trap", "/ok"]),
        "http://example.com/ok": ("fine words", []),
    })
    crawl_env.routes["http://example.com/trap"] = httpx.Response(
        302, headers={"location": "http://["}
    )
    out = web_crawl("http://example.com/")
    assert "1 failed" in out
    assert "fine words" in out


def test_crawl_warm_write_includes_truncation_marker(crawl_env):
    """A truncated HTML page must warm-write the SAME truncation marker
    web_fetch would produce, so a later default web_fetch() cache hit does
    not silently hand back a cut page as if it were complete."""
    from tldw_chatbook.Tools.web_tool_impls import FETCH_MAX_BYTES, web_fetch

    big_html = "<html><body><p>" + ("y " * ((FETCH_MAX_BYTES + 5000) // 2)) + "</p></body></html>"
    crawl_env.routes["http://example.com/"] = httpx.Response(
        200, content=big_html.encode(), headers={"content-type": "text/html"}
    )
    web_crawl("http://example.com/")
    n_calls = len(crawl_env.calls)
    result = web_fetch("http://example.com/")
    assert result.endswith(f"[... truncated: response exceeded max_bytes={FETCH_MAX_BYTES} ...]")
    assert len(crawl_env.calls) == n_calls  # served from cache, no new request


# ---------------------------------------------------------------------------
# sitemap mode (spec §2)
# ---------------------------------------------------------------------------

def _sitemap_response(xml: bytes) -> httpx.Response:
    return httpx.Response(200, content=xml, headers={"content-type": "application/xml"})


# NOT a module-level pytest.importorskip: see test_web_tool_impls.py's
# requires_pymupdf comment — that would skip the whole file. Only the
# defusedxml-specific refusal tests below need to skip when it's absent
# (the stdlib fallback parser parses internal entities without complaint).
try:
    import defusedxml  # noqa: F401
except ImportError:
    defusedxml = None

requires_defusedxml = pytest.mark.skipif(
    defusedxml is None, reason="defusedxml not installed (websearch/ebook/subscriptions extra)"
)

_ENTITY_SITEMAP = (
    b'<?xml version="1.0"?><!DOCTYPE urlset [<!ENTITY x "y">]>'
    b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<url><loc>http://example.com/&x;</loc></url></urlset>"
)


@requires_defusedxml
def test_sitemap_entity_declaration_root_is_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(_ENTITY_SITEMAP)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")


@requires_defusedxml
def test_sitemap_entity_declaration_child_is_skipped(crawl_env):
    index = (b'<?xml version="1.0"?>'
             b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             b"<sitemap><loc>http://example.com/bad.xml</loc></sitemap>"
             b"<sitemap><loc>http://example.com/good.xml</loc></sitemap>"
             b"</sitemapindex>")
    good = (b'<?xml version="1.0"?>'
            b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            b"<url><loc>http://example.com/page</loc></url></urlset>")
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/bad.xml"] = _sitemap_response(_ENTITY_SITEMAP)
    crawl_env.routes["http://example.com/good.xml"] = _sitemap_response(good)
    _site(crawl_env, {"http://example.com/page": ("still works", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "still works" in out


def test_sitemap_mode_seeds_pages_and_skips_expansion(crawl_env):
    xml = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/a</loc></url>"
        b"<url><loc>http://example.com/b</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {
        "http://example.com/a": ("alpha words", ["/should-not-follow"]),
        "http://example.com/b": ("beta words", []),
        "http://example.com/should-not-follow": ("nope", []),
    })
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "alpha words" in out and "beta words" in out
    # sitemap IS the discovery: links on seeded pages are not expanded
    assert "http://example.com/should-not-follow" not in crawl_env.calls
    assert "Stopped: sitemap exhausted." in out


def test_sitemap_index_one_level(crawl_env):
    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/s1.xml</loc></sitemap>"
        b"<sitemap><loc>http://cdn-other.com/s2.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    child = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/from-child</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/s1.xml"] = _sitemap_response(child)
    _site(crawl_env, {"http://example.com/from-child": ("child page words", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "child page words" in out
    # off-host child sitemap (different host than sitemap_url) never fetched
    assert "http://cdn-other.com/s2.xml" not in crawl_env.calls


def test_sitemap_offhost_page_urls_filtered(crawl_env):
    xml = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/ok</loc></url>"
        b"<url><loc>http://other.com/evil</loc></url>"
        b"<url><loc>http://10.0.0.5/internal</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {"http://example.com/ok": ("fine", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "fine" in out
    assert "http://other.com/evil" not in crawl_env.calls
    assert "http://10.0.0.5/internal" not in crawl_env.calls  # off-host AND private


def test_sitemap_respects_max_pages(crawl_env):
    urls = "".join(
        f"<url><loc>http://example.com/p{i}</loc></url>".encode().decode()
        for i in range(10)
    )
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(10)})
    web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3)
    page_calls = [c for c in crawl_env.calls if "/p" in c]
    assert len(page_calls) == 3


def test_sitemap_unfetchable_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = httpx.Response(500)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")


def test_sitemap_garbage_xml_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(b"not xml")
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")


def test_sitemap_namespace_free_urlset_seeds_pages(crawl_env):
    """A sitemap without the sitemaps.org XML namespace (some generators
    emit these) must still seed pages — pre-fix, the namespaced-only
    `.//{ns}loc` findall returns zero matches, so the crawl seeds nothing
    and reports "sitemap exhausted" instead of the page's content."""
    xml = b'<?xml version="1.0"?><urlset><url><loc>http://example.com/a</loc></url></urlset>'
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {"http://example.com/a": ("alpha words", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "alpha words" in out
    assert "Stopped: sitemap exhausted." in out


def test_parse_sitemap_namespace_free_urlset():
    xml = b'<?xml version="1.0"?><urlset><url><loc>http://example.com/a</loc></url></urlset>'
    pages, children = _parse_sitemap(xml)
    assert pages == ["http://example.com/a"]
    assert children == []


def test_sitemap_deadline_during_child_fetch_reports_deadline_reached(crawl_env):
    """The child-sitemap loop's bound check is a plain `break`, not an
    exception: if the clock crosses the deadline while fetching a child
    sitemap, the seed can come back short/empty for a reason that has
    nothing to do with the sitemap being exhausted. The stop reason must
    say so, not claim "sitemap exhausted" when the crawl actually ran out
    of time."""
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_DEADLINE_SECONDS

    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/s1.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)

    def slow_child(request):
        crawl_env.clock.now += CRAWL_DEADLINE_SECONDS + 1
        return _sitemap_response(
            b'<?xml version="1.0"?>'
            b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
        )

    crawl_env.routes["http://example.com/s1.xml"] = slow_child
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert out.endswith("Stopped: deadline reached.")


# ---------------------------------------------------------------------------
# Fix wave (whole-branch review): sniffed-PDF cache poisoning, unbounded
# frontier, sitemap child-fetch amplification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctype", ["application/pdf", "text/html", ""])
def test_crawl_pdf_sniff_not_poisoned_into_html_branch(crawl_env, ctype):
    """CRITICAL: a PDF served with a misleading (or absent) content-type
    must be classified via `_fetch_once`'s is_pdf sniff, not the declared
    content-type alone. Pre-fix, `_crawl_fetch_page` dropped the sniff
    result entirely, so a PDF labeled `text/html` (or unlabeled) fell
    through to the HTML branch: it was listed as a normal page (raw PDF
    bytes decoded into its 'excerpt' for the unlabeled case) and its
    garbage 'extracted text' warm-wrote the shared web_fetch cache under
    (url, FETCH_MAX_BYTES) — the exact key a later web_fetch(url) reads.

    Marker contract (clarified post-review): the sniff wins over the
    declared type for classification AND for the marker text — when
    is_pdf is true the marker is always "[application/pdf]" regardless of
    what the server claimed, matching spec §1's "the sniff wins over the
    declared type." Only a genuinely non-PDF, non-HTML response is
    labeled with its own declared content-type."""
    pdf_body = b"%PDF-1.7 pretend pdf body padding " + b"z" * 20
    headers = {"content-type": ctype} if ctype else {}
    _site(crawl_env, {"http://example.com/": ("root", ["/doc"])})
    crawl_env.routes["http://example.com/doc"] = httpx.Response(200, content=pdf_body, headers=headers)
    out = web_crawl("http://example.com/")
    # All three parametrized content types wrap a %PDF- sniffed body, so
    # is_pdf is true in every case: the marker is always [application/pdf].
    assert "[application/pdf]" in out
    assert "%PDF-" not in out
    assert ("http://example.com/doc", web_tool_impls.FETCH_MAX_BYTES) not in web_tool_impls._fetch_cache


def test_crawl_html_only_aborts_sniffed_pdf_despite_declared_html(crawl_env):
    """A response DECLARED text/html but whose body sniffs as a PDF must
    still abort the read early under html_only (crawl) mode. Pre-fix, the
    early-break only looked at the declared content-type — since
    "text/html" IS an HTML type, the break never fired and the crawl drained
    up to the 1 MiB page cap instead of aborting once the type is known
    from the sniff."""
    def guarded_chunks():
        yield b"%PDF-"  # 5 bytes: under the 12-byte sniff window, resolves NEXT chunk
        for i in range(3):
            yield f"filler chunk {i}".encode()
        raise AssertionError(
            "crawl drained the full sniffed-PDF body under html_only=True — "
            "the abort-after-sniff did not fire"
        )

    _site(crawl_env, {"http://example.com/": ("root", ["/doc"])})
    crawl_env.routes["http://example.com/doc"] = httpx.Response(
        200, content=guarded_chunks(), headers={"content-type": "text/html"}
    )
    out = web_crawl("http://example.com/")
    assert "[application/pdf]" in out


def test_crawl_mislabeled_binary_declared_html_reads_full_body(crawl_env):
    """Review fix round 1 (Important 2): a binary served AS text/html
    during a crawl keeps today's full-read behavior (the binary design
    doc's stated non-goal) — the html_only early-abort must key on
    kind == "pdf" plus the DECLARED type, never a sniffed image/zip/audio
    kind, which would cut the body at the 12-byte sniff window while
    misreporting truncated=False. The ASCII tail marker survives the
    UTF-8-replace decode, so its presence in the excerpt proves the body
    was drained past the sniff window. CHUNKED delivery is load-bearing
    (re-review): a single-chunk body is fully captured before the abort
    check runs, so only a multi-chunk response lets the buggy predicate
    actually cut the tail — verified red against the pre-fix code."""
    def chunked():
        yield b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # 16 bytes: sniff resolves kind=image here
        yield b"TAILMARKER after the sniff window"
    _site(crawl_env, {"http://example.com/": ("root", ["/masq"])})
    crawl_env.routes["http://example.com/masq"] = httpx.Response(
        200, content=chunked(), headers={"content-type": "text/html"}
    )
    out = web_crawl("http://example.com/")
    assert "TAILMARKER" in out


def test_crawl_warm_cache_does_not_mask_binary_metadata_for_later_fetch(crawl_env):
    """task-3280 / Qodo PR #1442 (1): crawl's mojibake decode of a
    MISLABELED binary must not warm the (url, FETCH_MAX_BYTES) cache key
    web_fetch reads — otherwise the same URL returns different result
    shapes depending on which tool touched it first, for the whole cache
    TTL. Crawl-then-fetch must yield the binary metadata shape."""
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20 + b"tail bytes beyond the sniff window"
    _site(crawl_env, {"http://example.com/": ("root", ["/pic"])})
    crawl_env.routes["http://example.com/pic"] = httpx.Response(
        200, content=png, headers={"content-type": "text/html"}
    )
    web_crawl("http://example.com/")
    # The body is magic-only (not a valid PNG), so the metadata path
    # refuses with [image-error] — the point is the SHAPE: web_fetch must
    # take the binary path and raise, never return crawl's cached
    # mojibake text (which would come back as a plain string).
    with pytest.raises(LocalToolError, match=r"\[image-error\]"):
        web_tool_impls.web_fetch("http://example.com/pic")


def test_crawl_nonpdf_nonhtml_marker_uses_declared_type(crawl_env):
    """The is_pdf branch must not swallow every non-HTML response into a
    hardcoded '[application/pdf]' marker: a genuinely non-PDF, non-HTML
    response (body does not sniff as %PDF-) is still labeled with its
    actual declared content-type."""
    _site(crawl_env, {"http://example.com/": ("root", ["/image.png"])})
    crawl_env.routes["http://example.com/image.png"] = httpx.Response(
        200, content=b"\x89PNG\r\n\x1a\n" + b"binarydata", headers={"content-type": "image/png"}
    )
    out = web_crawl("http://example.com/")
    assert "[image/png]" in out


def test_crawl_caps_links_enqueued_per_page(crawl_env, monkeypatch):
    """IMPORTANT: an unbounded frontier lets one page's link spam blow up
    visited/queue memory (52,975 entries measured from a single page in the
    review). A page must stop contributing links to the queue once it hits
    CRAWL_MAX_LINKS_PER_PAGE."""
    monkeypatch.setattr(web_tool_impls, "CRAWL_MAX_LINKS_PER_PAGE", 5)
    links = [f"/p{i}" for i in range(20)]
    _site(crawl_env, {
        "http://example.com/": ("root", links),
        **{f"http://example.com/p{i}": (f"page {i}", []) for i in range(20)},
    })
    web_crawl("http://example.com/", max_pages=100)
    # root fetch + exactly the capped number of links enqueued from it.
    # Expected: root (1) + at most 5 links from CRAWL_MAX_LINKS_PER_PAGE = 6 total.
    assert len(crawl_env.calls) == 6


def test_sitemap_index_caps_children_fetched(crawl_env, monkeypatch):
    """IMPORTANT: the sitemapindex child loop is bounded only by deadline +
    rate limit today — off-host children never trip max_pages's early
    exit, so a same-host index with ~119 children can pull ~600 MB in one
    call (5 MiB SITEMAP_MAX_BYTES each). Cap the number of child sitemaps
    actually fetched at SITEMAP_MAX_CHILDREN."""
    monkeypatch.setattr(web_tool_impls, "SITEMAP_MAX_CHILDREN", 3)
    children_locs = "".join(
        f"<sitemap><loc>http://example.com/s{i}.xml</loc></sitemap>" for i in range(10)
    )
    index = (
        '<?xml version="1.0"?>'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{children_locs}</sitemapindex>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    empty_child = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    )
    for i in range(10):
        crawl_env.routes[f"http://example.com/s{i}.xml"] = _sitemap_response(empty_child)
    web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=100)
    child_calls = [c for c in crawl_env.calls if c != "http://example.com/sitemap.xml"]
    assert len(child_calls) == 3


def test_sitemap_child_budget_reached_stop_reason(crawl_env, monkeypatch):
    """When the SITEMAP_MAX_CHILDREN break fires (children left unfetched),
    the footer must say so honestly instead of claiming "sitemap exhausted"
    — which implies every child sitemap was consulted when it was not."""
    monkeypatch.setattr(web_tool_impls, "SITEMAP_MAX_CHILDREN", 1)
    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/s1.xml</loc></sitemap>"
        b"<sitemap><loc>http://example.com/s2.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    empty_child = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/s1.xml"] = _sitemap_response(empty_child)
    crawl_env.routes["http://example.com/s2.xml"] = _sitemap_response(empty_child)
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert out.endswith("Stopped: sitemap child budget reached.")


# ---------------------------------------------------------------------------
# Final-review fix wave (task-2620): redirect-dedup regression, child
# skip-and-count, budget-truncated sitemap seeding honesty
# ---------------------------------------------------------------------------


def test_crawl_redirect_dedup_lists_content_fetched_via_redirect(crawl_env):
    """REGRESSION vs dev: the redirect-dedup guard checked the final URL
    against `visited` — but `visited` holds ENQUEUED urls, not LISTED ones.
    Root links to both /x and /y; /x 302s onto /y. /y's body (fetched via
    /x's attempt) must be listed even though /y was separately enqueued as
    its own link and, with the budget spent on root+/-x, never gets its own
    turn to be popped from the queue. Pre-fix this silently discarded the
    fetched page and reported "Crawled 1 pages"."""
    _site(crawl_env, {"http://example.com/": ("root", ["/x", "/y"])})
    crawl_env.routes["http://example.com/x"] = httpx.Response(
        302, headers={"location": "http://example.com/y"}
    )
    _site(crawl_env, {"http://example.com/y": ("y content words", [])})
    out = web_crawl("http://example.com/", max_pages=2)
    assert "y content words" in out
    assert "Crawled 2 pages" in out


@requires_defusedxml
def test_sitemap_children_skipped_counted_in_footer(crawl_env):
    """AC#1's skip-and-COUNT half (previously unimplemented): child fetch
    failures and parse refusals were swallowed by a bare `continue` in the
    child loop, so a sitemapindex whose children ALL fail reported zero
    signal ("Crawled 0 pages (0 failed, 0 blocked). Stopped: sitemap
    exhausted."). One child refuses to parse (defusedxml entity-declaration
    refusal) and one 500s — the footer must count both as skipped."""
    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/bad.xml</loc></sitemap>"
        b"<sitemap><loc>http://example.com/broken.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/bad.xml"] = _sitemap_response(_ENTITY_SITEMAP)
    crawl_env.routes["http://example.com/broken.xml"] = httpx.Response(500)
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "2 child sitemaps skipped" in out
    assert out.endswith("Stopped: sitemap exhausted.")


def test_sitemap_budget_truncated_reports_page_budget_reached(crawl_env):
    """A plain urlset with MORE same-host URLs than max_pages — the default
    path for nearly every real sitemap at the default max_pages=20 — must
    not claim "sitemap exhausted"; `take()`'s cap left candidates behind."""
    urls_xml = "".join(f"<url><loc>http://example.com/p{i}</loc></url>" for i in range(10))
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls_xml}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(10)})
    out = web_crawl(
        "http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3
    )
    assert out.endswith("Stopped: page budget reached.")


def test_sitemap_exactly_consumed_still_reports_exhausted(crawl_env):
    """Boundary: a sitemap with EXACTLY max_pages same-host URLs (nothing
    left over) must still report "sitemap exhausted" — take() only flips the
    truncation flag when a candidate was actually left unconsidered."""
    urls_xml = "".join(f"<url><loc>http://example.com/p{i}</loc></url>" for i in range(3))
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls_xml}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(3)})
    out = web_crawl(
        "http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3
    )
    assert out.endswith("Stopped: sitemap exhausted.")


def test_sitemap_trailing_offhost_loc_does_not_flip_budget_truncated(crawl_env):
    """False-positive guard: take()'s max_pages check must run AFTER the
    scope_host filter. 3 same-host locs exactly fill max_pages=3; a 4th,
    TRAILING off-host loc would be discarded by the host filter regardless
    — it must not flip budget_truncated and make the footer claim "page
    budget reached" when every same-host candidate was actually considered."""
    urls_xml = "".join(f"<url><loc>http://example.com/p{i}</loc></url>" for i in range(3))
    urls_xml += "<url><loc>http://other.com/off-host</loc></url>"
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls_xml}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(3)})
    out = web_crawl(
        "http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3
    )
    assert "http://other.com/off-host" not in crawl_env.calls
    assert out.endswith("Stopped: sitemap exhausted.")


def test_sitemap_trailing_duplicate_loc_does_not_flip_budget_truncated(crawl_env):
    """Same false-positive guard, duplicate-filter side: a 4th, TRAILING loc
    that duplicates the first same-host URL would be discarded by the
    `seen` filter regardless of max_pages — it must not flip
    budget_truncated either."""
    urls_xml = "".join(f"<url><loc>http://example.com/p{i}</loc></url>" for i in range(3))
    urls_xml += "<url><loc>http://example.com/p0</loc></url>"  # duplicate of the first
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls_xml}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(3)})
    out = web_crawl(
        "http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3
    )
    assert out.endswith("Stopped: sitemap exhausted.")


def test_sitemap_seed_deadline_and_budget_both_hit_deadline_wins(crawl_env):
    """Coverage add (final-review observation): when a sitemap child fetch
    simultaneously exhausts the wall-clock deadline AND fills max_pages via
    take()'s cap, the seeding path's documented priority order — "deadline
    (wall-clock, non-negotiable) > children_capped > budget_truncated >
    exhausted" (see the comment above web_crawl's `if time.monotonic() >=
    deadline: ... elif seed.budget_truncated: ...` chain) — must report the
    deadline, not "page budget reached". The child handler advances the
    clock past the deadline while returning 5 same-host page URLs against
    max_pages=3, so take() sets budget_truncated=True in the same call that
    pushes the clock past the deadline."""
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_DEADLINE_SECONDS

    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/s1.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)

    def slow_child_over_budget(request):
        crawl_env.clock.now += CRAWL_DEADLINE_SECONDS + 1
        urls_xml = "".join(f"<url><loc>http://example.com/p{i}</loc></url>" for i in range(5))
        xml = (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"{urls_xml}</urlset>"
        ).encode()
        return _sitemap_response(xml)

    crawl_env.routes["http://example.com/s1.xml"] = slow_child_over_budget
    out = web_crawl(
        "http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3
    )
    assert out.endswith("Stopped: deadline reached.")
    page_calls = [c for c in crawl_env.calls if "/p" in c]
    assert page_calls == []  # deadline hit before the seeded pages get their own attempt


# ---------------------------------------------------------------------------
# robots.txt enforcement (task-2833)
# ---------------------------------------------------------------------------
#
# crawl_env's own fixture default sets respect_robots_txt=False (existing-
# suite compatibility, design doc Critical 1) -- every test below opts back
# in explicitly via _enable_robots().

def _enable_robots(monkeypatch, respect: bool = True) -> None:
    monkeypatch.setattr(
        web_tool_impls, "_webfetch_settings", lambda: {"respect_robots_txt": respect}
    )


def _robots_txt(body: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "text/plain"})


def test_crawl_disallowed_page_skipped_and_counted(crawl_env, monkeypatch):
    _enable_robots(monkeypatch)
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(b"User-agent: *\nDisallow: /private\n")
    _site(crawl_env, {
        "http://example.com/": ("root", ["/private", "/ok"]),
        "http://example.com/ok": ("fine words", []),
    })
    out = web_crawl("http://example.com/")
    assert "1 robots-disallowed" in out
    assert "fine words" in out
    assert "http://example.com/private" not in crawl_env.calls  # blocked before the hop


def test_crawl_disallowed_child_sitemap_skipped(crawl_env, monkeypatch):
    _enable_robots(monkeypatch)
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(b"User-agent: *\nDisallow: /bad.xml\n")
    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/bad.xml</loc></sitemap>"
        b"<sitemap><loc>http://example.com/good.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    good = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/page</loc></url></urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/bad.xml"] = _sitemap_response(good)  # would succeed IF fetched
    crawl_env.routes["http://example.com/good.xml"] = _sitemap_response(good)
    _site(crawl_env, {"http://example.com/page": ("still works", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "still works" in out
    assert "1 child sitemaps skipped" in out
    assert "http://example.com/bad.xml" not in crawl_env.calls  # blocked before the hop


def test_crawl_disallowed_start_url_returns_structured_refusal(crawl_env, monkeypatch):
    _enable_robots(monkeypatch)
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(b"User-agent: *\nDisallow: /\n")
    crawl_env.routes["http://example.com/"] = _html("root", [])
    with pytest.raises(LocalToolError) as exc_info:
        web_crawl("http://example.com/")
    msg = str(exc_info.value)
    # Double-wrapped (design doc): the disallowed seed flows through the
    # existing unconditional start-URL wrap, same as any other seed failure.
    assert msg.startswith("[crawl-failed] start URL could not be fetched: ")
    assert "[robots-disallowed]" in msg
    assert "http://example.com/" not in crawl_env.calls  # blocked before the hop


def test_crawl_disallowed_root_sitemap_returns_structured_refusal(crawl_env, monkeypatch):
    _enable_robots(monkeypatch)
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(b"User-agent: *\nDisallow: /sitemap.xml\n")
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(
        b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    )
    with pytest.raises(LocalToolError) as exc_info:
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    msg = str(exc_info.value)
    assert msg.startswith("[crawl-failed] sitemap could not be fetched: ")
    assert "[robots-disallowed]" in msg
    assert "http://example.com/sitemap.xml" not in crawl_env.calls  # blocked before the hop


def test_crawl_robots_uses_crawl_user_agent(crawl_env, monkeypatch):
    """Disallow the web_fetch UA but allow the crawl UA -- proves web_crawl
    checks robots.txt against _CRAWL_USER_AGENT, not _USER_AGENT."""
    _enable_robots(monkeypatch)
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(
        b"User-agent: tldw-chatbook-web-fetch\nDisallow: /\n\n"
        b"User-agent: tldw-chatbook-web-crawl\nAllow: /\n"
    )
    _site(crawl_env, {"http://example.com/": ("root words", [])})
    out = web_crawl("http://example.com/")
    assert "root words" in out


def test_crawl_toggle_off_makes_no_robots_fetch(crawl_env):
    # crawl_env's own fixture default is respect_robots_txt=False; this
    # test proves a PRESENT, fully-disallowing robots.txt is never even
    # fetched while the toggle is off.
    crawl_env.routes["http://example.com/robots.txt"] = _robots_txt(b"User-agent: *\nDisallow: /\n")
    _site(crawl_env, {"http://example.com/": ("root words", [])})
    out = web_crawl("http://example.com/")
    assert "root words" in out
    assert "http://example.com/robots.txt" not in crawl_env.calls
