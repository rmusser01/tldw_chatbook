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
