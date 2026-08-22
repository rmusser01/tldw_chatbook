"""TASK-19556 (c): sitemap/crawl seams must not trust their own input URL.

`Utils/egress.py`'s module contract is explicit:

    Shared pipeline code must NEVER auto-trust its own input URL -- trust is
    seeded only at boundaries where user intent is known and threaded down.

and `config.py`'s `[web_security]` block says the same thing from the
config side, naming the exact inputs:

    content-derived URLs (redirects, **sitemap/crawl discoveries**, feed
    items) must resolve to public IPs; URLs you explicitly configure (feed
    sources, Confluence base_url, ingest URLs) may be private.

At this branch's base four seams contradicted both:

* `Article_Extractor_Lib.scrape_from_sitemap:1032` computed
  `origins = origin_set(sitemap_url)` and used it for its OWN fetch --
  and then handed that same trust to every URL it found *inside* the
  fetched XML (`scrape_article(url.text, trusted_origins=origins)`).
* `Article_Scraper/crawler.get_urls_from_sitemap:352` self-trusted the
  same way.
* `Article_Extractor_Lib.collect_internal_links:1089` and
  `Article_Scraper/crawler.crawl_site:178` seeded from `base_url` and then
  carried that trust onto every link discovered mid-crawl.

The fix makes `trusted_origins` an explicit, fail-closed keyword on each
(matching `scrape_article`, `get_page_title` and `ScraperConfig`, which
already default to `frozenset()`), and applies it to the caller-named
entry URL only -- never to a discovery.

REACHABILITY, re-checked at this base and stated honestly: none of these
four functions has an in-app caller. `scrape_from_sitemap` is reached only
via `scrape_and_convert_with_filter` (no callers), `get_urls_from_sitemap`
only from its own docstring example, `collect_internal_links` from
`scrape_entire_site`/`create_filtered_sitemap` (no callers), `crawl_site`
from its docstring example. The Watchlists `sitemap` source type -- the
one shipped path that fetches a sitemap -- goes through
`Subscriptions/local_watchlists_service._urls_for_sitemap`, whose
`origin_set(source)` IS provenance-correct (`source` is the subscription
URL the user configured). So this is a correctness defect with LATENT
reach, fixed as such, and the change is behaviour-neutral for the shipped
app.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL
from tldw_chatbook.Web_Scraping.Article_Scraper import crawler as CR

SITEMAP_URL = "http://sitemap.internal/map.xml"
SITEMAP_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>http://sitemap.internal/secret-page</loc></url>
</urlset>
"""

BASE_URL = "http://crawl.internal/"
BASE_HTML = '<html><body><a href="/deep/page">deep</a></body></html>'


class _FakeResponse:
    def __init__(self, content: bytes = b"", text: str = "", status: int = 200):
        self.content = content
        self.text = text or content.decode("utf-8", "replace")
        self.status_code = status
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def raise_for_status(self) -> None:
        return None


class _Recorder:
    """Records every guarded fetch's URL and `trusted_origins`."""

    def __init__(self, responder):
        self.calls: List[Dict[str, Any]] = []
        self._responder = responder

    def __call__(self, url: str, **kwargs: Any):
        self.calls.append({"url": url, **kwargs})
        return self._responder(url)

    @property
    def trust_by_url(self) -> Dict[str, frozenset]:
        return {c["url"]: c.get("trusted_origins", frozenset()) for c in self.calls}


# ---------------------------------------------------------------------------
# scrape_from_sitemap
# ---------------------------------------------------------------------------


def test_scrape_from_sitemap_does_not_trust_its_own_input_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(lambda _u: _FakeResponse(content=SITEMAP_XML))
    monkeypatch.setattr(AEL, "guarded_fetch_requests", recorder)
    monkeypatch.setattr(AEL, "scrape_article", lambda *a, **k: None)

    AEL.scrape_from_sitemap(SITEMAP_URL)

    assert recorder.calls, "the sitemap was never fetched"
    trusted = recorder.calls[0].get("trusted_origins", frozenset())
    assert trusted == frozenset(), (
        f"scrape_from_sitemap self-trusted its own input URL: {trusted}"
    )


def test_sitemap_discovered_urls_do_not_inherit_the_sitemap_hosts_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact input the policy exists to catch."""
    recorder = _Recorder(lambda _u: _FakeResponse(content=SITEMAP_XML))
    monkeypatch.setattr(AEL, "guarded_fetch_requests", recorder)
    seen: List[Dict[str, Any]] = []

    def _fake_scrape_article(url: str, *args: Any, **kwargs: Any):
        seen.append({"url": url, **kwargs})
        return None

    monkeypatch.setattr(AEL, "scrape_article", _fake_scrape_article)

    # Even a caller that legitimately trusts the sitemap host must not have
    # that trust forwarded to URLs the sitemap CONTENT names.
    AEL.scrape_from_sitemap(SITEMAP_URL, trusted_origins=frozenset({"sitemap.internal"}))

    assert seen, "no sitemap-discovered URL was scraped"
    trusted = seen[0].get("trusted_origins", frozenset())
    assert trusted == frozenset(), (
        f"a sitemap-discovered URL was handed trust: {trusted}"
    )


def test_scrape_from_sitemap_threads_caller_supplied_trust_to_the_entry_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trust is still seedable -- from the caller, not from the URL itself."""
    recorder = _Recorder(lambda _u: _FakeResponse(content=SITEMAP_XML))
    monkeypatch.setattr(AEL, "guarded_fetch_requests", recorder)
    monkeypatch.setattr(AEL, "scrape_article", lambda *a, **k: None)

    AEL.scrape_from_sitemap(SITEMAP_URL, trusted_origins=frozenset({"sitemap.internal"}))

    assert recorder.calls[0]["trusted_origins"] == frozenset({"sitemap.internal"})


# ---------------------------------------------------------------------------
# get_urls_from_sitemap (async / aiohttp arm)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_urls_from_sitemap_does_not_trust_its_own_input_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    class _Guarded:
        status_code = 200
        headers = {"Content-Type": "application/xml"}
        text = SITEMAP_XML.decode()

        def raise_for_status(self) -> None:
            return None

    async def _fake_fetch(url: str, **kwargs: Any):
        calls.append({"url": url, **kwargs})
        return _Guarded()

    monkeypatch.setattr(CR, "guarded_fetch_aiohttp", _fake_fetch)

    await CR.get_urls_from_sitemap(SITEMAP_URL)

    assert calls, "the sitemap was never fetched"
    trusted = calls[0].get("trusted_origins", frozenset())
    assert trusted == frozenset(), (
        f"get_urls_from_sitemap self-trusted its own input URL: {trusted}"
    )


# ---------------------------------------------------------------------------
# crawl discoveries
# ---------------------------------------------------------------------------


def test_collect_internal_links_does_not_trust_discovered_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(lambda _u: _FakeResponse(text=BASE_HTML))
    monkeypatch.setattr(AEL, "guarded_fetch_requests", recorder)

    AEL.collect_internal_links(BASE_URL, trusted_origins=frozenset({"crawl.internal"}))

    trust = recorder.trust_by_url
    assert BASE_URL in trust, f"the crawl root was never fetched: {list(trust)}"
    assert trust[BASE_URL] == frozenset({"crawl.internal"})
    discovered = {u: t for u, t in trust.items() if u != BASE_URL}
    assert discovered, "no link was discovered, so the assertion below is vacuous"
    assert all(t == frozenset() for t in discovered.values()), (
        f"a crawl discovery inherited the root's trust: {discovered}"
    )


@pytest.mark.asyncio
async def test_crawl_site_does_not_trust_discovered_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    class _Guarded:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        text = BASE_HTML

    async def _fake_fetch(url: str, **kwargs: Any):
        calls.append({"url": url, **kwargs})
        return _Guarded()

    monkeypatch.setattr(CR, "guarded_fetch_aiohttp", _fake_fetch)

    await CR.crawl_site(
        BASE_URL, max_pages=5, max_depth=2, trusted_origins=frozenset({"crawl.internal"})
    )

    trust = {c["url"]: c.get("trusted_origins", frozenset()) for c in calls}
    assert trust.get(BASE_URL) == frozenset({"crawl.internal"})
    discovered = {u: t for u, t in trust.items() if u != BASE_URL}
    assert discovered, "no link was discovered, so the assertion below is vacuous"
    assert all(t == frozenset() for t in discovered.values()), (
        f"a crawl discovery inherited the root's trust: {discovered}"
    )


# ---------------------------------------------------------------------------
# Fail-closed signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [
        AEL.scrape_from_sitemap,
        AEL.collect_internal_links,
        CR.get_urls_from_sitemap,
        CR.crawl_site,
    ],
    ids=["scrape_from_sitemap", "collect_internal_links", "get_urls_from_sitemap", "crawl_site"],
)
def test_sitemap_and_crawl_entry_points_default_to_no_trust(func) -> None:
    parameter = inspect.signature(func).parameters.get("trusted_origins")
    assert parameter is not None, (
        f"{func.__name__} has no explicit trusted_origins seam, so a caller "
        "cannot thread user intent down and the function must be inventing it"
    )
    assert parameter.default == frozenset()
