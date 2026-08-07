---
id: TASK-2910
title: Web-tools residual hardening
status: To Do
assignee: []
created_date: '2026-08-06'
labels:
  - web-tools
  - tech-debt
dependencies:
  - TASK-2620
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2620's final review round (2026-08-06) re-triaged the web-tools v2 (`web_crawl`/`web_fetch` PDF support) module's known residuals. Two of the four originally-listed residuals turned out to be real bugs and were fixed in that same round (the redirect-dedup regression and the sitemap budget-truncation honesty gap); the remaining items below are won't-fix-for-now calls, each with the reasoning recorded in task-2620's Implementation Notes, that should be re-examined or fixed properly later:

1. `_pymupdf_available()` raises an uncaught `ValueError` on a `sys.modules` stub with no `__spec__` — a non-total probe (only reachable through test monkeypatching of `sys.modules`), but it contradicts the module's all-`LocalToolError` failure contract in principle. Should probably guard with `except (ImportError, ValueError)`.
2. `_pymupdf_available()` (`importlib.util.find_spec`) is called on every redirect hop inside `web_fetch`'s loop rather than once per call — ruled cheap (`find_spec` benefits from import-system caching) but never actually measured.
3. A redirect deduped by `web_crawl`'s listed-URL guard still spends its own attempt slot but produces no row in the footer's page count — accepted attempt-slot-contract behavior, documented in `web_crawl`'s docstring as of task-2620, but the footer itself gives the model no visibility into how many attempts were spent vs. listed.
4. Foreign-namespace (neither sitemaps.org-namespaced nor bare/namespace-less) sitemaps still parse to zero locs — collapses into the already-accepted parse-zero-locs case; the original spec asked for namespace-less support specifically, not arbitrary foreign namespaces.
5. The `[too-large]` PDF-ceiling message's "N MB" copy (`PDF_MAX_BYTES // (1024 * 1024)`) can render "0 MB" under integer-division truncation — only reachable via a test that monkeypatches `PDF_MAX_BYTES` to well under 1 MB, not with the real module constant, but the message would be misleading if that constant were ever lowered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each residual fixed or re-ruled in this task's notes
<!-- AC:END -->

## Additional residuals (final fix-wave re-review, 2026-08-06)

6. `budget_truncated` false-positive: the seed's max_pages check runs BEFORE the host/dup filters, so trailing candidates that take() would discard anyway flip the flag — footer says "page budget reached" when the truth is "sitemap exhausted". Conservative direction (under-claims completeness; invites a pointless higher-max_pages retry). Verified fix: reorder the cap check after the host+seen filters (re-reviewer confirmed against all shipped tests); same shape in the child loop's break when remaining children are all off-host.
7. `children_skipped` counts deadline-expired child fetches (`_CrawlDeadline`) but the _SitemapSeed docstring and spec enumerate only fetch-error/oversized/parse-refusal — complete the enumerations.
8. web_crawl's attempt-invariant docstring sentence describes only the redirect-lands-on-listed case; the mirror (a plain fetch of a URL an earlier redirect already listed) is the commoner row-less attempt.
