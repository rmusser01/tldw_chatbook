"""web_crawl: pure-helper unit tests (no transport) + crawl behavior tests."""

import socket
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import (
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
