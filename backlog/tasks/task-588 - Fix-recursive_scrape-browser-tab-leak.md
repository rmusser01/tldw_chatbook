---
id: TASK-588
title: Fix recursive_scrape browser-tab leak on page.goto block
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-23 12:00'
labels: [web, performance]
dependencies: [task-328]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`recursive_scrape` inner `page.goto` block skips `page.close()` when a link is blocked, causing browser-tab leaks per blocked link. Wrap the block in `try/finally` to ensure cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] `recursive_scrape` always closes browser pages in the inner loop, even when goto or extraction fails
- [x] Browser-tab leaks are eliminated for blocked-link scenarios
<!-- AC:END -->

## Implementation Plan

1. Write a RED regression test at the `Article_Extractor_Lib.async_playwright` module seam: a stub playwright chain whose `check_url_or_raise_async` raises (blocked link), asserting `page.close()` was called.
2. Wrap the link-discovery block (page creation through link enqueue) in `try/finally: await page.close()`.
3. Run the new tests plus the full `Tests/Web_Scraping/` suite; ruff-check the touched files against the pre-change baseline.

ADR required: no
ADR path: N/A
Reason: Localized resource-cleanup fix inside one function; no boundary or contract change.

## Implementation Notes

Wrapped the link-discovery block in `recursive_scrape` (`tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py`) in `try/finally: await page.close()`. Previously `page.close()` was the last statement of the block, so any raise between `context.new_page()` and it -- most realistically `check_url_or_raise_async` rejecting an egress-blocked URL -- leaked the tab; the outer `except Exception` logged and the crawl continued, so one tab leaked per failed discovery for the remaining crawl lifetime (the browser only closes after the while loop).

TDD evidence: `Tests/Web_Scraping/test_recursive_scrape_tab_lifecycle.py` (new) stubs the playwright chain at the documented patchable `async_playwright` module global. RED: `test_blocked_link_discovery_page_is_closed` failed with `close_calls` 0 == 1 on unmodified dev; GREEN after the fix. `test_successful_link_discovery_page_is_closed_once` pins the happy path at exactly one close. Suite: `Tests/Web_Scraping/` 315 passed, 3 skipped (playwright installed via the `websearch` extra; the 3 prior collection errors were this venv lacking optional deps, unrelated to the change). Ruff: new test file clean; `Article_Extractor_Lib.py` shows the same 78 pre-existing violations as at HEAD -- zero delta.
