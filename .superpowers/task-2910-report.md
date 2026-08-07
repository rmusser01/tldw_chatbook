# task-2910 — Web-tools residual hardening — report

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/web-tools-residuals-2910`
Items in scope: 3, 4, 5, 7, 8 (+ coverage-add). Items 1/2/6 were already resolved on PR #1393 before this task started.

## Item 3 — FIX: duplicate-redirect visibility in the footer

**Change:** `_format_crawl_result()` gained an optional `duplicates_skipped: int = 0`
param. `web_crawl()` now increments a `duplicates_skipped` local at the exact
`final_norm in listed: continue` site in the shared BFS+sitemap fetch loop
(`tldw_chatbook/Tools/web_tool_impls.py`), and threads it into the final
`_format_crawl_result(...)` call. When nonzero, the footer parenthetical grows
a third clause, following the existing `; N child sitemaps skipped` idiom:

```
Crawled 2 pages (0 failed, 0 blocked; 1 duplicate redirects skipped). Stopped: no more links within depth.
```

Unit-level (`duplicates_skipped=1` passed directly to `_format_crawl_result`):
```
Crawled 0 pages (0 failed, 0 blocked; 1 duplicate redirects skipped). Stopped: page budget reached.
```

**RED (before fix):**
```
Tests/Tools/test_web_crawl.py::test_format_duplicate_redirects_skipped_clause FAILED
  TypeError: _format_crawl_result() got an unexpected keyword argument 'duplicates_skipped'
Tests/Tools/test_web_crawl.py::test_crawl_redirect_duplicate_targets_listed_once FAILED
  AssertionError: assert '1 duplicate redirects skipped' in
  '...Crawled 2 pages (0 failed, 0 blocked). Stopped: no more links within depth.'
```

**GREEN (after fix):** both above pass; plus two absence-guard tests
(`test_format_duplicate_redirects_skipped_clause_absent_when_zero`,
`test_crawl_no_duplicates_omits_duplicate_redirects_clause`) confirm the clause
does not appear when the count is zero.

Spec (`Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md`) updated
with the same footer sentence + example, right after the existing
child-sitemaps-skipped paragraph.

## Item 7 — prose: complete the skip-reason enumerations

`_SitemapSeed`'s `children_skipped` docstring and the spec's child-sitemap-skip
sentence enumerated fetch/redirect error, oversized body, and parse refusal but
missed the `_CrawlDeadline` case already counted by
`except (LocalToolError, _CrawlDeadline): children_skipped += 1` in
`_seed_from_sitemap`'s child loop. Both now read "...a fetch/redirect error, a
deadline expiry mid-fetch, an oversized body, or a parse refusal...". No
behavior change; no new test needed (existing
`test_sitemap_deadline_during_child_fetch_reports_deadline_reached` already
exercises this path and stayed green).

## Item 8 — prose: complete the attempt-invariant sentence

`web_crawl`'s docstring described only the redirect-lands-on-already-listed
case. Added the mirror — a plain, non-redirecting fetch of a URL an earlier
page's redirect already listed — and noted (per item 3) that both cases are
now surfaced via the footer's duplicate-redirects-skipped clause. Same
addition made to spec §2's Output section. Prose-only; no test change beyond
what item 3 already covers (the existing
`test_crawl_redirect_dedup_lists_content_fetched_via_redirect` test already
exercises the redirect-lands-on-listed path).

## Coverage add — deadline-vs-budget precedence

Read the code first, as instructed: `web_crawl`'s post-seed chain is
```python
if time.monotonic() >= deadline:
    stop_reason = "deadline reached"
elif seed.children_capped:
    stop_reason = "sitemap child budget reached"
elif seed.budget_truncated:
    stop_reason = "page budget reached"
else:
    stop_reason = "sitemap exhausted"
```
— the deadline check already runs first, matching the docstring comment above
it ("deadline (wall-clock, non-negotiable) > children_capped > budget_truncated
> exhausted"). Added
`test_sitemap_seed_deadline_and_budget_both_hit_deadline_wins`: a sitemapindex
child fetch handler advances the clock past `CRAWL_DEADLINE_SECONDS` while
returning 5 same-host page URLs against `max_pages=3`, so `take()` sets
`budget_truncated=True` in the same call that pushes the clock past the
deadline. Result: footer ends `Stopped: deadline reached.` — this test passed
on first run (no code change); it is a pin, not a RED/GREEN fix. **Budget does
not win anywhere in the seeding path; deadline strictly outranks it, as
documented.**

## Items 4, 5 — re-ruled, no code change

Both re-ruled in the task file with strikethrough + ruling text, matching the
style of items 1/2/6:

- **Item 4** (foreign-namespace sitemaps): **WON'T-FIX**. Spec scope was
  namespace-less sitemaps specifically; a foreign-namespaced sitemap is
  malformed-for-our-purposes and collapses honestly into the already-accepted
  parse-zero-locs path (`_parse_sitemap` returns `([], [])`; footer reports
  "sitemap exhausted" / 0 pages). Supporting arbitrary namespaces would mean
  guessing at generators never verified against.
- **Item 5** ("0 MB" PDF-ceiling copy): **WON'T-FIX**. Only reachable by
  monkeypatching `PDF_MAX_BYTES` below 1 MiB in a test; the constant is code
  (maintainer-set), not runtime config, and the derived copy
  (`PDF_MAX_BYTES // (1024 * 1024)`) stays in lockstep with the constant at
  any realistic value — no drift exists at the real `20 * 1024 * 1024`.

## Test evidence (final run)

```
$ /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly
...
107 passed, 5 warnings in 1.24s
```

Full-module collect-only sweep (per the "targeted tests, not full suites" rule):
```
$ /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Tools/ --collect-only -q
325 tests collected in 1.20s
```
No import errors, no collection failures.

## Files touched

- `tldw_chatbook/Tools/web_tool_impls.py` — `_format_crawl_result` signature +
  footer clause; `web_crawl` `duplicates_skipped` counter, increment site,
  threaded call, docstring; `_SitemapSeed.children_skipped` docstring.
- `Tests/Tools/test_web_crawl.py` — 5 new tests
  (`test_format_duplicate_redirects_skipped_clause`,
  `test_format_duplicate_redirects_skipped_clause_absent_when_zero`,
  `test_crawl_no_duplicates_omits_duplicate_redirects_clause`,
  `test_sitemap_seed_deadline_and_budget_both_hit_deadline_wins`), 1 extended
  existing test (`test_crawl_redirect_duplicate_targets_listed_once`).
- `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md` — footer
  sentence + example, children-skipped enumeration.
- `backlog/tasks/task-2910 - Web-tools-residual-hardening.md` — items 3/4/5/7/8
  struck through with resolutions/rulings; AC #1 checked; status Done;
  Implementation Notes added.

## Concerns / notes for reviewer

- Item 8's docstring phrase "the more common case" is a design-rationale claim
  from the task brief, not something independently measured in this session —
  it describes typical link-graph shapes (pages linking directly to an
  already-redirected-to target being more common than two separate redirects
  converging), not a runtime-verified frequency.
- The duplicate-redirect footer clause and the child-sitemaps-skipped clause
  can both appear together (e.g. a sitemap crawl with both child failures and
  redirect dedup); no test exercises that combination specifically, but the
  string-concatenation logic in `_format_crawl_result` is straightforward
  (two independent `if count > 0: counts += "; ..."` blocks) and the unit test
  `test_format_duplicate_redirects_skipped_clause` exercises the
  `duplicates_skipped`-only path in isolation.
