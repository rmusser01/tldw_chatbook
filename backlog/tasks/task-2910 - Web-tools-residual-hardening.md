---
id: TASK-2910
title: Web-tools residual hardening
status: In Progress
assignee:
  - '@claude'
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

1. ~~`_pymupdf_available()` raises an uncaught `ValueError` on a `sys.modules` stub with no `__spec__`~~ — **RESOLVED on PR #1393**: `_pymupdf_available()` now wraps the `find_spec` call in `except (ImportError, ValueError): return False`, so it is total and stays inside the module's all-`LocalToolError` contract. RED test: `test_pymupdf_available_spec_less_stub_returns_false_not_valueerror`.
2. ~~`_pymupdf_available()` (`importlib.util.find_spec`) is called on every redirect hop inside `web_fetch`'s loop rather than once per call~~ — **RESOLVED on PR #1393**: `web_fetch` now probes once (`pymupdf_ok = _pymupdf_available()`) before the redirect loop and reuses that value for both the `pdf_max_bytes` selection and the post-fetch `[missing-dep]`-vs-`[too-large]` branch, closing the "never actually measured" gap by removing the repeated call outright and eliminating the theoretical mid-call `sys.modules`-mutation disagreement between the two call sites.
3. A redirect deduped by `web_crawl`'s listed-URL guard still spends its own attempt slot but produces no row in the footer's page count — accepted attempt-slot-contract behavior, documented in `web_crawl`'s docstring as of task-2620, but the footer itself gives the model no visibility into how many attempts were spent vs. listed.
4. Foreign-namespace (neither sitemaps.org-namespaced nor bare/namespace-less) sitemaps still parse to zero locs — collapses into the already-accepted parse-zero-locs case; the original spec asked for namespace-less support specifically, not arbitrary foreign namespaces.
5. The `[too-large]` PDF-ceiling message's "N MB" copy (`PDF_MAX_BYTES // (1024 * 1024)`) can render "0 MB" under integer-division truncation — only reachable via a test that monkeypatches `PDF_MAX_BYTES` to well under 1 MB, not with the real module constant, but the message would be misleading if that constant were ever lowered.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Fix 3 (duplicate-redirect footer clause `; N duplicate redirects skipped`, spec-pinned, RED-first), 7+8 (prose enumerations/docstring), add deadline-vs-budget precedence test (final-review observation); re-rule 4+5 with reasoning. Close Done.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each residual fixed or re-ruled in this task's notes
<!-- AC:END -->

## Additional residuals (final fix-wave re-review, 2026-08-06)

6. ~~`budget_truncated` false-positive: the seed's max_pages check runs BEFORE the host/dup filters, so trailing candidates that take() would discard anyway flip the flag — footer says "page budget reached" when the truth is "sitemap exhausted".~~ — **RESOLVED on PR #1393**: `take()`'s `len(urls) >= max_pages` check now runs after the `scope_host` and `seen` filters, and the child loop's off-host check now runs before its own budget check, matching the verified fix shape. RED tests: `test_sitemap_trailing_offhost_loc_does_not_flip_budget_truncated`, `test_sitemap_trailing_duplicate_loc_does_not_flip_budget_truncated`; pre-existing `test_sitemap_budget_truncated_reports_page_budget_reached` (true-positive) and `test_sitemap_exactly_consumed_still_reports_exhausted` (exact-consumption) stayed green throughout.
7. `children_skipped` counts deadline-expired child fetches (`_CrawlDeadline`) but the _SitemapSeed docstring and spec enumerate only fetch-error/oversized/parse-refusal — complete the enumerations.
8. web_crawl's attempt-invariant docstring sentence describes only the redirect-lands-on-listed case; the mirror (a plain fetch of a URL an earlier redirect already listed) is the commoner row-less attempt.
