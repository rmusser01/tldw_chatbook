"""Confluence HTTP must not run on the event loop thread (TASK-585).

`ConfluenceAuth.make_request` is a synchronous `requests` call and
`ConfluenceScraper._extract_page_id_from_url` falls back to a blocking fetch.
Both were invoked inline from `async def` methods, so a slow or hanging
Confluence server stalled the whole loop for up to the full request timeout
(30s since task-328 bounded it; unbounded before that), starving every other
concurrent operation in the app.

These tests assert the property directly -- the blocking callable must
execute on a DIFFERENT thread than the running loop -- rather than asserting
that some particular wrapper was used, so they keep holding if the code later
moves to httpx async instead of a worker thread.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.Web_Scraping.Confluence.confluence_crawler import ConfluenceCrawler
from tldw_chatbook.Web_Scraping.Confluence.confluence_scraper import ConfluenceScraper


class ThreadRecordingAuth:
    """Stands in for ConfluenceAuth, recording where make_request ran."""

    def __init__(self) -> None:
        self.threads: list[int] = []
        self.base_url = "https://example.invalid/wiki"

    def make_request(self, method: str, endpoint: str, **kwargs):
        self.threads.append(threading.get_ident())
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "id": "123",
            "title": "T",
            "body": {"storage": {"value": "<p>x</p>"}},
            "version": {"number": 1},
            "space": {"key": "S", "name": "S"},
            "results": [],
            "size": 0,
        }
        response.content = b""
        return response


@pytest.mark.asyncio
async def test_scrape_page_by_id_does_not_block_the_loop_thread():
    auth = ThreadRecordingAuth()
    scraper = ConfluenceScraper(auth)

    loop_thread = threading.get_ident()
    await scraper.scrape_page_by_id("123")

    assert auth.threads, "make_request was never called; the test proved nothing"
    assert loop_thread not in auth.threads, (
        "ConfluenceAuth.make_request ran on the event loop thread: a slow "
        "Confluence server would stall every other concurrent operation."
    )


@pytest.mark.asyncio
async def test_crawler_child_page_lookup_does_not_block_the_loop_thread():
    auth = ThreadRecordingAuth()
    crawler = ConfluenceCrawler(auth)

    loop_thread = threading.get_ident()
    await crawler._get_child_pages("123")

    assert auth.threads, "make_request was never called; the test proved nothing"
    assert loop_thread not in auth.threads


@pytest.mark.asyncio
async def test_page_id_extraction_does_not_block_the_loop_thread(monkeypatch):
    """The URL helper falls back to fetching the page, so it blocks too."""
    auth = ThreadRecordingAuth()
    scraper = ConfluenceScraper(auth)

    seen: list[int] = []

    def recording_extract(url: str):
        seen.append(threading.get_ident())
        return "123"

    monkeypatch.setattr(scraper, "_extract_page_id_from_url", recording_extract)

    loop_thread = threading.get_ident()
    await scraper.scrape_page_by_url("https://example.invalid/wiki/x")

    assert seen, "_extract_page_id_from_url was never called"
    assert loop_thread not in seen, (
        "_extract_page_id_from_url ran on the event loop thread; its fallback "
        "path performs a blocking HTTP fetch."
    )


@pytest.mark.asyncio
async def test_the_loop_stays_responsive_while_confluence_is_slow():
    """End-to-end property: a slow request must not freeze other tasks.

    This is the symptom users would actually feel, and it fails loudly if
    someone reverts the offload without touching the assertions above.
    """
    auth = ThreadRecordingAuth()
    slow = threading.Event()

    def slow_request(method: str, endpoint: str, **kwargs):
        slow.wait(0.30)  # stand-in for a slow/hanging server
        return ThreadRecordingAuth.make_request(auth, method, endpoint, **kwargs)

    auth.make_request = slow_request  # type: ignore[assignment]
    scraper = ConfluenceScraper(auth)

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat reach its first await
    await scraper.scrape_page_by_id("123")
    # Sample BEFORE draining the heartbeat: awaiting it first would let it
    # finish regardless and the assertion could never fail.
    ticks_during_request = ticks
    slow.set()
    await beat

    assert ticks_during_request >= 5, (
        f"the event loop only advanced {ticks_during_request} times while a "
        "Confluence request was in flight; it was blocked rather than awaiting"
    )


@pytest.mark.asyncio
async def test_concurrent_scrapes_do_not_use_the_session_simultaneously(monkeypatch):
    """Offloading made `scrape_many`'s concurrency real -- and unsafe.

    `requests.Session` is not thread-safe (mutable cookies, connection pool,
    headers). That was harmless while every call ran inline on the event loop:
    `scrape_many`'s `asyncio.gather` looked concurrent, but each blocking
    `make_request` serialized the others anyway. Moving those calls to
    `asyncio.to_thread` made N worker threads able to touch one session at
    once -- a hazard this change introduced, found in review.

    Drives the REAL `ConfluenceAuth.make_request` (only the session's
    `request` is stubbed) so the lock under test is the production one. An
    earlier version of this test wrapped a FAKE auth's `make_request`, which
    bypassed that lock entirely and measured only the stub.
    """
    import asyncio

    from tldw_chatbook.Web_Scraping.Confluence import confluence_auth as auth_module
    from tldw_chatbook.Web_Scraping.Confluence.confluence_auth import ConfluenceAuth

    # The egress policy rejects the synthetic host before the session is ever
    # reached; this test is about session concurrency, not egress.
    monkeypatch.setattr(auth_module, "check_url_or_raise", lambda *a, **k: None)

    auth = ConfluenceAuth("https://example.invalid/wiki")
    auth._auth_configured = True

    overlap = {"max": 0, "cur": 0}
    guard = threading.Lock()

    def recording_request(method, url, **kwargs):
        with guard:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
        try:
            threading.Event().wait(0.02)  # widen the window
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "id": "1", "title": "T",
                "body": {"storage": {"value": "<p>x</p>"}},
                "version": {"number": 1},
                "space": {"key": "S", "name": "S"},
                "results": [], "size": 0,
            }
            response.content = b""
            return response
        finally:
            with guard:
                overlap["cur"] -= 1

    auth.session.request = recording_request  # type: ignore[assignment]
    scraper = ConfluenceScraper(auth)

    await asyncio.gather(*(scraper.scrape_page_by_id(str(i)) for i in range(6)))

    assert overlap["max"] >= 1, "no request reached the session; test proved nothing"
    assert overlap["max"] == 1, (
        f"{overlap['max']} concurrent entries into the shared requests.Session; "
        "it is not thread-safe and must be serialized"
    )
