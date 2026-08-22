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


class _ThreadRecordingAuth:
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
    auth = _ThreadRecordingAuth()
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
    auth = _ThreadRecordingAuth()
    crawler = ConfluenceCrawler(auth)

    loop_thread = threading.get_ident()
    await crawler._get_child_pages("123")

    assert auth.threads, "make_request was never called; the test proved nothing"
    assert loop_thread not in auth.threads


@pytest.mark.asyncio
async def test_page_id_extraction_does_not_block_the_loop_thread(monkeypatch):
    """The URL helper falls back to fetching the page, so it blocks too."""
    auth = _ThreadRecordingAuth()
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
    auth = _ThreadRecordingAuth()
    slow = threading.Event()

    def slow_request(method: str, endpoint: str, **kwargs):
        slow.wait(0.30)  # stand-in for a slow/hanging server
        return _ThreadRecordingAuth.make_request(auth, method, endpoint, **kwargs)

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
