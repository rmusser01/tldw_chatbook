---
id: TASK-2620
title: Web-crawl/PDF-fetch deferred review findings
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-06'
labels:
  - web-tools
  - tech-debt
dependencies:
  - TASK-1357
  - TASK-1358
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The web-tools v2 branch (`feat/web-crawl-pdf-fetch`, tasks 1357/1358) went through seven review rounds; everything Critical/Important was fixed before merge. This task collects the findings every reviewer agreed were real but safely deferrable, so they aren't lost. The two "parked" items were found by the final fix-wave re-review and have rulings recorded; the rest are polish/coverage.

**Parked (real, small):**
1. defusedxml's refusals (`EntitiesForbidden` — a ValueError, NOT a ParseError) escape `_parse_sitemap`'s `except xET.ParseError`: a root sitemap declaring entities reaches the model as a raw exception string instead of `[crawl-failed]`, and one hostile CHILD sitemap aborts the whole crawl instead of being skipped. Ruling: no security exposure (the parse is refused either way); fix by widening the except to `(xET.ParseError, ValueError)`.
2. A crawl truncated by `SITEMAP_MAX_CHILDREN` reports `Stopped: sitemap exhausted.` — a new stop-reason honesty gap in the same family task-5's deadline fix closed.

**Deferred minors worth batching:**
3. `web_fetch`/provider docstrings lag behavior: no PDF mention, no `(url, max_bytes)` cache key or 256-entry bound; `LocalToolProvider`'s module docstring tool list still ends at `web_search`.
4. `[ssrf]` substring classification for blocked-vs-failed footer accounting is spoofable by a URL path containing `[ssrf]` — should ride an exception attribute.
5. `_fetch_once`'s `html_only` early-abort keys on the DECLARED type only, so a PDF mislabeled `text/html` streams up to 1 MiB during a crawl before the (correct) marker branch discards it — bandwidth only.
6. Two same-host URLs redirecting to one target are fetched and listed twice (visited marked only post-fetch; intermediate hops never marked).
7. No static ephemerality regression guard (spec §5 called for a "no media-DB import in web_tool_impls" assertion) — true by inspection, unpinned.
8. Namespace-less (non-conformant) sitemaps silently parse to zero URLs → "sitemap exhausted" instead of a parse complaint.
9. Optional-dep imports (`defusedxml`, `pymupdf`) use the module's local try/except pattern (v1's trafilatura precedent) rather than the central `optional_deps.py` helper — Qodo rule 497159 flagged it on PR #1376; centralize when touching these imports. `[too-large]` copy hardcodes "20 MB" independent of `PDF_MAX_BYTES`; a pymupdf-absent pre-check (`importlib.util.find_spec`) before raising the 20 MB read ceiling would avoid pointless downloads; `_CrawlLinkParser` title accumulation unbounded on unclosed `<title>` (transient memory only); frontier cap test asserts `<= 6` where `== 6` would pin both directions; between-hops `_CrawlDeadline` path uncovered.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
`Docs/superpowers/plans/2026-08-06-web-tools-polish-2620.md` — Task 1 behavioral fixes (sitemap refusal contract, child-cap stop reason, unspoofable blocked/failed, sniff-abort in html_only, derived [too-large] copy, pymupdf pre-check, title bound, namespace-less sitemaps, redirect-dup dedup); Task 2 coverage + docstring truth + closure with won't-fix rulings.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sitemap parse failures of ANY kind (incl. defusedxml refusals) surface as [crawl-failed] for the root and skip-and-count for children
- [ ] #2 A child-cap-truncated crawl's stop reason does not claim the sitemap was exhausted
- [ ] #3 The remaining minors are each either fixed or explicitly re-ruled as won't-fix in this task's notes
<!-- AC:END -->
