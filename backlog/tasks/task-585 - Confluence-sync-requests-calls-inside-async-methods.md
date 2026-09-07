---
id: TASK-585
title: Confluence replace sync requests calls inside async methods
status: Done
assignee: []
created_date: '2026-07-23 12:00'
labels: [subscriptions, performance]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConfluenceAuth.make_request` and `ConfluenceScraper._extract_page_id_from_url` make synchronous `requests` library calls from inside `async def` methods, blocking the event loop for up to the full request timeout (30s after task-328's fix; previously unbounded). This is a performance hazard that can starve concurrent operations. Port these to httpx async or run via a thread executor to avoid blocking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] No synchronous HTTP call remains on the event loop in `Web_Scraping/Confluence/` (ConfluenceAuth and ConfluenceScraper)
- [x] Async methods use either httpx async or run_in_executor for blocking sync calls
- [x] Tests verify no blocking calls are made from async contexts
<!-- AC:END -->

## Implementation Notes

All seven `auth.make_request` call sites across `confluence_scraper.py` and
`confluence_crawler.py`, plus `_extract_page_id_from_url` (whose fallback
path performs its own blocking fetch), now run via `asyncio.to_thread`.
Chose the executor route over an httpx port: it satisfies the AC at a
fraction of the risk, leaving session/auth/retry handling untouched.

Tests assert the property -- the blocking callable runs on a different thread
than the loop -- rather than asserting a specific wrapper, so they survive a
later move to httpx async. A fourth test pins the user-visible symptom: the
loop keeps ticking while a slow request is in flight.

One test-quality note worth recording: that fourth test initially passed
against the UNFIXED code because it sampled the heartbeat counter after
awaiting the heartbeat, which always completes. It now samples before
draining and fails with the fix reverted, like the other three.

Note for anyone running these locally: this package imports `playwright` and
`trafilatura` transitively, so its tests (including the two pre-existing
ones) do not collect without those optional deps installed.

Files: `tldw_chatbook/Web_Scraping/Confluence/confluence_scraper.py`,
`confluence_crawler.py`,
`Tests/Web_Scraping/Confluence/test_confluence_no_blocking_io_on_loop.py`.
