---
id: TASK-2910
title: Web-tools residual hardening
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06'
updated_date: '2026-08-07 03:06'
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
3. ~~A redirect deduped by `web_crawl`'s listed-URL guard still spends its own attempt slot but produces no row in the footer's page count — accepted attempt-slot-contract behavior, documented in `web_crawl`'s docstring as of task-2620, but the footer itself gives the model no visibility into how many attempts were spent vs. listed.~~ — **RESOLVED**: `_format_crawl_result` gained an optional `duplicates_skipped` param; `web_crawl` increments a `duplicates_skipped` counter at the exact `final_norm in listed` continue in the shared fetch loop (BFS and sitemap mode both go through it) and renders `; N duplicate redirects skipped` in the footer parenthetical when nonzero, e.g. `Crawled 2 pages (0 failed, 0 blocked; 1 duplicate redirects skipped). Stopped: no more links within depth.`. RED tests: `test_format_duplicate_redirects_skipped_clause`, `test_crawl_redirect_duplicate_targets_listed_once` (the `/one`,`/two`→`/target` scenario, asserts the clause with N=1); absence pinned by `test_format_duplicate_redirects_skipped_clause_absent_when_zero` and `test_crawl_no_duplicates_omits_duplicate_redirects_clause`.
4. ~~Foreign-namespace (neither sitemaps.org-namespaced nor bare/namespace-less) sitemaps still parse to zero locs — collapses into the already-accepted parse-zero-locs case; the original spec asked for namespace-less support specifically, not arbitrary foreign namespaces.~~ — **WON'T-FIX**: the spec's scope was namespace-less sitemaps specifically, not arbitrary foreign namespaces. A foreign-namespaced sitemap is a malformed-for-our-purposes document and collapses honestly into the already-accepted parse-zero-locs outcome (`_parse_sitemap` returns `([], [])`; the crawl reports "sitemap exhausted" / 0 pages) — supporting arbitrary XML namespaces would mean guessing at generators never verified against, not closing a defined gap.
5. ~~The `[too-large]` PDF-ceiling message's "N MB" copy (`PDF_MAX_BYTES // (1024 * 1024)`) can render "0 MB" under integer-division truncation — only reachable via a test that monkeypatches `PDF_MAX_BYTES` to well under 1 MB, not with the real module constant, but the message would be misleading if that constant were ever lowered.~~ — **WON'T-FIX**: only reachable by monkeypatching the module constant to well under 1 MiB; `PDF_MAX_BYTES` is code (a maintainer-set module constant), not a runtime/config value, and the derived copy (`PDF_MAX_BYTES // (1024 * 1024)`) already stays in lockstep with the constant at any realistic value — at the real `20 * 1024 * 1024` there is no drift to guard against.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each residual fixed or re-ruled in this task's notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Fix 3 (duplicate-redirect footer clause `; N duplicate redirects skipped`, spec-pinned, RED-first), 7+8 (prose enumerations/docstring), add deadline-vs-budget precedence test (final-review observation); re-rule 4+5 with reasoning. Close Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed the tail of the web-tools v2 residuals list (items 3-8; 1/2/6 were already resolved on PR #1393).

Fixed (item 3, the one behavior change): duplicate-redirect visibility in
web_crawl's footer. `_format_crawl_result` gained an optional
`duplicates_skipped: int = 0` param; `web_crawl` counts every
`final_norm in listed` continue in the shared BFS+sitemap fetch loop into a
new `duplicates_skipped` local and threads it through, rendering
"; N duplicate redirects skipped" in the footer parenthetical when nonzero
(idiom matches the existing "; N child sitemaps skipped" clause). RED-first:
test_format_duplicate_redirects_skipped_clause and the extended
test_crawl_redirect_duplicate_targets_listed_once (/one,/two -> /target)
failed before the fix, passed after; absence pinned by
test_format_duplicate_redirects_skipped_clause_absent_when_zero and
test_crawl_no_duplicates_omits_duplicate_redirects_clause.

Prose completed (items 7, 8): _SitemapSeed's children_skipped docstring and
spec Sec.2's child-sitemap-skip sentence now list "a deadline expiry
mid-fetch" alongside fetch/redirect error, oversized body, and parse
refusal (matches the `except (LocalToolError, _CrawlDeadline)` catch already
in _seed_from_sitemap's child loop). web_crawl's attempt-invariant
docstring and spec Sec.2's Output section now describe both row-less-attempt
cases -- redirect-lands-on-listed, and a plain
non-redirecting fetch of a URL an earlier redirect already listed -- and
both note the pair is now surfaced via item 3's footer clause.

Coverage add (final-review observation): pinned deadline-vs-budget
precedence in the sitemap seeding path with
test_sitemap_seed_deadline_and_budget_both_hit_deadline_wins. Read the code
first: web_crawl's post-seed `if time.monotonic() >= deadline: ... elif
seed.budget_truncated: ...` chain already checks the deadline before
budget_truncated, so deadline wins where both are true simultaneously --
this is a coverage-only test (passed on first run, no code change), not a
RED/GREEN fix.

Re-ruled, no code change (items 4, 5): item 4 (foreign-namespace sitemaps)
WON'T-FIX -- spec scope was namespace-less specifically; a foreign namespace
collapses honestly into the already-accepted parse-zero-locs path. Item 5
("0 MB" PDF-ceiling copy) WON'T-FIX -- only reachable by monkeypatching
PDF_MAX_BYTES below 1 MiB in a test; the constant is code, and the derived
copy stays in lockstep with it at any realistic value.

Files touched:
- tldw_chatbook/Tools/web_tool_impls.py (_format_crawl_result signature +
  footer clause, web_crawl duplicates_skipped counter + docstring,
  _SitemapSeed docstring)
- Tests/Tools/test_web_crawl.py (5 new tests, 1 extended existing test)
- Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md (footer
  sentence + example, children-skipped enumeration)
- backlog/tasks/task-2910 (this file: items 3/4/5/7/8 struck through with
  rulings, matching 1/2/6's style)

Test evidence: `pytest Tests/Tools/test_web_crawl.py
Tests/Tools/test_web_tool_impls.py -v -p no:randomly` -> 107 passed. Full
Tests/Tools/ --collect-only sweep: 325 collected, no import errors.

Follow-up (coordinator review): dropped the unverified '(the more common case in practice)' frequency ranking from web_crawl's docstring and spec Sec.2 -- which row-less-attempt shape is commoner depends on BFS discovery order and was never measured; replaced with neutral both-ways wording in both places.
<!-- SECTION:NOTES:END -->

## Additional residuals (final fix-wave re-review, 2026-08-06)

6. ~~`budget_truncated` false-positive: the seed's max_pages check runs BEFORE the host/dup filters, so trailing candidates that take() would discard anyway flip the flag — footer says "page budget reached" when the truth is "sitemap exhausted".~~ — **RESOLVED on PR #1393**: `take()`'s `len(urls) >= max_pages` check now runs after the `scope_host` and `seen` filters, and the child loop's off-host check now runs before its own budget check, matching the verified fix shape. RED tests: `test_sitemap_trailing_offhost_loc_does_not_flip_budget_truncated`, `test_sitemap_trailing_duplicate_loc_does_not_flip_budget_truncated`; pre-existing `test_sitemap_budget_truncated_reports_page_budget_reached` (true-positive) and `test_sitemap_exactly_consumed_still_reports_exhausted` (exact-consumption) stayed green throughout.
7. ~~`children_skipped` counts deadline-expired child fetches (`_CrawlDeadline`) but the _SitemapSeed docstring and spec enumerate only fetch-error/oversized/parse-refusal — complete the enumerations.~~ — **RESOLVED**: `_SitemapSeed`'s `children_skipped` docstring and spec §2's child-sitemap-skip sentence both now list "a deadline expiry mid-fetch" alongside fetch/redirect error, oversized body, and parse refusal, matching the `except (LocalToolError, _CrawlDeadline): children_skipped += 1` catch in `_seed_from_sitemap`'s child loop.
8. ~~web_crawl's attempt-invariant docstring sentence describes only the redirect-lands-on-listed case; the mirror (a plain fetch of a URL an earlier redirect already listed) is the commoner row-less attempt.~~ — **RESOLVED**: `web_crawl`'s docstring and spec §2's Output section now describe both row-less-attempt cases — a redirect landing on an already-listed final URL, and a plain, non-redirecting fetch of a URL an earlier page's redirect already listed (no frequency ranking: which occurs first depends on discovery order) — and both note the pair is surfaced via item 3's footer clause.
