"""TASK-38603: the scraper's timeout retry path must behave like the
generic error path — back off between attempts and raise
``MaxRetriesExceededError`` after the final one — and a ``retries`` value
below 1 must be rejected loudly instead of silently returning ``""``.

Everything playwright/egress-shaped is faked; no network, no browser.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import tldw_chatbook.Web_Scraping.Article_Extractor_Lib as AEL


FAKE_HTML = "<html><body><p>Hello article body</p></body></html>"
URL = "https://example.com/article"


class _FakePage:
    def __init__(self, goto_outcomes):
        # goto_outcomes: list of either Exception instances to raise or
        # values to return; the last entry repeats once the list is spent.
        self._outcomes = list(goto_outcomes)
        self.goto_calls = 0

    async def goto(self, url, **kwargs):
        self.goto_calls += 1
        if len(self._outcomes) > 1:
            outcome = self._outcomes.pop(0)
        else:
            outcome = self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def wait_for_load_state(self, *args, **kwargs):
        return None

    async def content(self):
        return FAKE_HTML


def _install_fake_browser(monkeypatch, page):
    class _FakeContext:
        async def new_page(self):
            return page

        async def add_cookies(self, cookies):
            return None

    context = _FakeContext()

    class _FakeBrowser:
        async def new_context(self, **kwargs):
            return context

        async def close(self):
            return None

    browser = _FakeBrowser()

    class _FakeChromium:
        async def launch(self, **kwargs):
            return browser

    class _FakePlaywrightCM:
        async def __aenter__(self):
            return SimpleNamespace(chromium=_FakeChromium())

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(AEL, "async_playwright", lambda: _FakePlaywrightCM())
    return browser


def _maybe(value):
    async def _coro(*args, **kwargs):
        return value

    return _coro


class _FakeTrafilatura:
    seen_html = None

    @staticmethod
    def extract(html, **kwargs):
        _FakeTrafilatura.seen_html = html
        return "Hello article body"

    @staticmethod
    def extract_metadata(html, **kwargs):
        return None


def _install_harness(monkeypatch, goto_outcomes, retry_count):
    page = _FakePage(goto_outcomes)
    _install_fake_browser(monkeypatch, page)

    sleeps: list[float] = []
    monkeypatch.setattr(AEL, "asyncio", SimpleNamespace(sleep=_record(sleeps)))

    monkeypatch.setattr(
        AEL,
        "load_and_log_configs",
        lambda: {
            "web_scraper": {
                "web_scraper_retry_count": retry_count,
                "web_scraper_retry_timeout": 5,
                "web_scraper_stealth_playwright": False,
            }
        },
    )
    monkeypatch.setattr(AEL, "check_url_or_raise_async", _maybe(None))
    monkeypatch.setattr(AEL, "validate_navigation_chain_async", _maybe(None))
    monkeypatch.setattr(AEL, "collect_navigation_chain", lambda nav: [])
    monkeypatch.setattr(AEL, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(AEL, "_ensure_playwright", lambda: True)
    monkeypatch.setattr(AEL, "TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(AEL, "_ensure_trafilatura", lambda: True)
    monkeypatch.setattr(AEL, "trafilatura", _FakeTrafilatura)
    _FakeTrafilatura.seen_html = None
    return page, sleeps


def _record(sinks):
    async def _sleep(delay, *args, **kwargs):
        sinks.append(delay)

    return _sleep


async def test_timeout_exhaustion_backs_off_and_raises(monkeypatch):
    """Every attempt times out -> sleeps happen between attempts and the
    final one raises MaxRetriesExceededError (surfaced by scrape_article
    as a scraping failure, not empty content)."""
    page, sleeps = _install_harness(
        monkeypatch, [TimeoutError("page load timed out")], retry_count=3
    )

    result = await AEL.scrape_article(URL)

    assert page.goto_calls == 3
    assert sleeps == [2, 4]  # backoff between attempts, none after the last
    assert result["extraction_successful"] is False
    assert "Scraping Failed" in result["title"] or "3" in result["content"]
    assert "timeout" in str(result.get("error", "")).lower()


async def test_timeout_then_success_still_returns_html(monkeypatch):
    """A timeout followed by a successful load returns the page HTML."""
    page, sleeps = _install_harness(
        monkeypatch, [TimeoutError("page load timed out"), object()], retry_count=3
    )

    result = await AEL.scrape_article(URL)

    assert page.goto_calls == 2
    assert sleeps == [2]
    assert result["extraction_successful"] is True
    assert _FakeTrafilatura.seen_html == FAKE_HTML
    assert "Hello article body" in result["content"]


async def test_retries_below_one_is_rejected_loudly(monkeypatch):
    """retries = 0 used to make range(0) return "" immediately; it must
    now raise a ValueError naming the config key."""
    _install_harness(monkeypatch, [object()], retry_count=0)

    with pytest.raises(ValueError, match="web_scraper_retry_count"):
        await AEL.scrape_article(URL)


async def test_retries_non_integer_is_rejected(monkeypatch):
    _install_harness(monkeypatch, [object()], retry_count="3")

    with pytest.raises(ValueError, match="web_scraper_retry_count"):
        await AEL.scrape_article(URL)
