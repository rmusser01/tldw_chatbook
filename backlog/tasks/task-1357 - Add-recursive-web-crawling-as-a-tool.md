---
id: TASK-1357
title: Add recursive web crawling as a tool
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 06:04'
updated_date: '2026-08-06'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Article_Extractor_Lib already implements sitemap and recursive crawlers (scrape_from_sitemap, sync_recursive_scrape, scrape_by_url_level). Wrap them in a budgeted web_crawl tool behind the egress guard — page list returned to the model, no permanent ingestion.

**Owner ruling (2026-08-06 brainstorm), superseding the "wrap them" premise above:** the crawler was built as a new lightweight httpx BFS on the web-tools v1 core (`web_tool_impls.py`) plus a sitemap mode — NOT a wrapper around Article_Extractor_Lib. The premise predated v1's merged design: those crawlers launch Playwright Chromium per call, write a `scrape_progress.json` resume file into the cwd, use the older egress guard, and their link collector has no page budget. Ruling and rationale recorded in `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 web_crawl with max_pages/max_depth budgets + egress guard,Per-domain rate limiting honored,Ask default; results ephemeral; tests with local fixture site
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md` §2; plan: `Docs/superpowers/plans/2026-08-06-web-crawl-pdf-fetch.md` (tasks 3–6). Same-host BFS reusing v1's SSRF guard/rate limiter/transport seam; sitemap seeding mode; LocalToolSpec registration (MCP exposure automatic via the generic `_register_local_agent_tools` path).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on `feat/web-crawl-pdf-fetch` as a new sync core `web_crawl` in `tldw_chatbook/Tools/web_tool_impls.py` plus one `LocalToolSpec` in `Agents/local_tool_provider.py` (tags=() → Ask default from the permission store's global default; exposed over MCP automatically, fail-closed for external callers).

- Budgets: `max_pages` (default 20, ceiling 40) counts fetch ATTEMPTS incl. failures and guard-blocks; `max_depth` (2/5); 120 s wall-clock deadline checked per dequeue and between redirect hops; 1 s/domain rate limit shared with web_fetch; `CRAWL_MAX_LINKS_PER_PAGE = 500` bounds the frontier (link spam measured at ~53k links/page otherwise).
- Every URL (BFS-discovered, sitemap-discovered, redirect hop) passes the v1 egress guard individually; sitemap XML parsed via defusedxml-with-fallback; `SITEMAP_MAX_CHILDREN = 20` bounds index amplification.
- Output: numbered page list (URL, title, ~200-char excerpt), per-block/total byte caps, honest footer with exact stop reason. Ephemeral by construction — no DB imports in the module.
- Crawl warm-writes the fetch cache (truncation-marker parity with web_fetch) so follow-up `web_fetch` of a listed page is instant; never reads it.
- Review trail (7 review rounds total incl. a whole-branch opus review): notable catches were an unguarded `urljoin` ValueError that let one malformed href abort a whole crawl, sniffed-PDF cache poisoning at the task-2/task-4 seam, and two resource-amplification bounds now pinned as constants. Deferred minors + parked findings: TASK-2620.
- Tests: `Tests/Tools/test_web_crawl.py` (fixture site over `httpx.MockTransport`, hermetic DNS incl. a mid-crawl DNS-rebinding case, fake clock), 396 passing across the branch-relevant suites at completion.
<!-- SECTION:NOTES:END -->
