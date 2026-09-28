"""Wiring tests: Article_Extractor_Lib routes fetches through the egress guard."""

from unittest.mock import patch

import pytest

from tldw_chatbook.Utils.egress import EgressBlockedError
from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL


def test_get_page_title_uses_guarded_fetch_and_contains_block():
    with patch.object(
        AEL, "guarded_fetch_requests", side_effect=EgressBlockedError("u", "private")
    ) as mocked:
        title = AEL.get_page_title("http://internal.example/x")
    assert mocked.called
    kwargs = mocked.call_args.kwargs
    assert kwargs["max_bytes"] == 10 * 1024 * 1024
    assert kwargs["timeout"] == 10
    assert title == "Untitled (Blocked URL)"


def test_scrape_from_sitemap_blocked_returns_empty():
    with patch.object(
        AEL, "guarded_fetch_requests", side_effect=EgressBlockedError("u", "private")
    ) as mocked:
        result = AEL.scrape_from_sitemap("http://sitemap.internal/map.xml")
    assert result == []
    assert mocked.call_args.kwargs["max_bytes"] == 50 * 1024 * 1024
    # (TASK-19556 (c)) This used to assert `frozenset({"sitemap.internal"})`
    # -- i.e. it pinned the defect: the function trusted the origin of the
    # very URL it was about to fetch, so the guard was a no-op for exactly
    # the content-derived input it exists to catch. Trust is now the
    # caller's to seed and defaults to none; see
    # Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py.
    assert mocked.call_args.kwargs["trusted_origins"] == frozenset()


def test_scrape_article_signature_defaults_fail_closed():
    import inspect

    sig = inspect.signature(AEL.scrape_article)
    assert sig.parameters["trusted_origins"].default == frozenset()
    sig_async = inspect.signature(AEL.scrape_article_async)
    assert sig_async.parameters["trusted_origins"].default == frozenset()


def test_scraper_config_has_fail_closed_trusted_origins_default():
    from tldw_chatbook.Web_Scraping.Article_Scraper.config import ScraperConfig

    assert ScraperConfig().trusted_origins == frozenset()


def _confluence_auth_with_recorders(monkeypatch):
    """Real `ConfluenceAuth` with both outbound seams recorded, none live."""
    from tldw_chatbook.Web_Scraping.Confluence import confluence_auth as ca

    guarded = {}
    raw = {}

    def fake_guarded(url, **kwargs):
        guarded["url"] = url
        guarded.update(kwargs)

        class R:
            status_code = 200

        return R()

    def fake_raw(self, method, url, **kwargs):  # requests.Session.request
        raw["url"] = url
        raw["method"] = method
        raw.update(kwargs)

        class R:
            status_code = 200

        return R()

    checked = []

    monkeypatch.setattr(ca, "guarded_fetch_requests", fake_guarded)
    monkeypatch.setattr(ca.requests.Session, "request", fake_raw)
    # Stubbed so the raw branch records rather than tripping ADR-126's
    # recovery gate on the config read inside the real egress check.
    monkeypatch.setattr(
        ca, "check_url_or_raise", lambda url, **kwargs: checked.append(url)
    )
    auth = ca.ConfluenceAuth("https://wiki.corp.example")
    auth._auth_configured = True
    return auth, guarded, raw, checked


def test_confluence_make_request_gets_timeout_and_guard(monkeypatch):
    auth, guarded, raw, checked = _confluence_auth_with_recorders(monkeypatch)
    auth.make_request("GET", "/rest/api/content/123")
    assert raw == {}, "a plain GET must not reach the raw session"
    assert guarded["url"] == "https://wiki.corp.example/rest/api/content/123"
    assert guarded["timeout"] == 30
    assert guarded["trusted_origins"] == frozenset({"wiki.corp.example"})


def test_confluence_make_request_with_params_is_still_guarded(monkeypatch):
    """TASK-32894: the shape all SEVEN production callers actually use.

    `confluence_scraper.py:78,150,214,376` and `confluence_crawler.py:239,
    271,292` every one pass `params={...}`. The old guard predicate was
    `set(kwargs) <= {"headers", "timeout"}`, so every production call fell
    through to the raw `session.request` else-branch: no size cap, and
    redirects followed with `session.auth` still attached. The old version
    of the test above called `make_request` with no `params` -- the one
    shape production never uses -- so it was green on a branch nothing
    reached.
    """
    auth, guarded, raw, checked = _confluence_auth_with_recorders(monkeypatch)
    auth.make_request(
        "GET",
        "/rest/api/content",
        params={"spaceKey": "DOCS", "limit": 25},
    )
    assert raw == {}, (
        "a params-carrying GET bypassed the egress-guarded fetch: "
        f"{raw.get('url')}"
    )
    assert guarded["url"] == (
        "https://wiki.corp.example/rest/api/content?spaceKey=DOCS&limit=25"
    )
    assert guarded["max_bytes"] > 0
    assert guarded["trusted_origins"] == frozenset({"wiki.corp.example"})


def test_confluence_non_get_still_pre_checks_the_url(monkeypatch):
    """Non-GET keeps the raw path; the egress pre-check must still run."""
    auth, guarded, raw, checked = _confluence_auth_with_recorders(monkeypatch)
    auth.make_request("POST", "/rest/api/content", json={"title": "x"})
    assert guarded == {}
    assert raw["method"] == "POST"
    assert checked == ["https://wiki.corp.example/rest/api/content"]
