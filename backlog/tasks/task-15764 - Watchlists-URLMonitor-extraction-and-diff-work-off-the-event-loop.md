---
id: TASK-15764
title: 'Watchlists: URLMonitor extraction and diff work off the event loop'
status: Done
assignee: []
created_date: '2026-08-13 12:31'
updated_date: '2026-08-15 00:09'
labels:
  - perf
  - watchlists
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up named explicitly in task-15463's "Scope" section (input-latency
burn-down, `Docs/Design/2026-08-11-input-latency-audit.md`): task-15463 moved
the watchlist scheduler's sqlite work and feed-body parsing off the event
loop, but deliberately left `URLMonitor`'s own CPU work on the loop as "a
separate, larger change." For `url`/`url_list`/`sitemap` sources this covers:

- `ContentExtractor.extract_text_from_html` (BeautifulSoup over a page up to
  `MAX_FETCH_BYTES_PAGE`, reached from `monitoring_engine._fetch_url_content`)
- the difflib work in `check_url` (`_segment_for_diff` called twice,
  `build_change_diff`, `added_and_removed_text`, `classify_change_type`)

Both are pure CPU with no sqlite involvement, so `asyncio.to_thread` applies
without task-15463's in-memory-connection hazard (no `:memory:` guard is
needed here — this is not a DB call). A `url_list` source multiplies both
costs by its URL count, so a source that watches several URLs pays this
synchronously on the loop once per check, per URL.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `ContentExtractor.extract_text_from_html` runs off the event loop for
      `url`/`url_list`/`sitemap` checks (evidence: thread-identity test)
- [x] #2 `check_url`'s difflib work (`_segment_for_diff` ×2, `build_change_diff`,
      `added_and_removed_text`, `classify_change_type`) runs off the event loop
- [x] #3 A `url_list` source with multiple URLs shows the same off-loop behavior
      for each URL, not just the first
- [x] #4 Existing Watchlists/Subscriptions suites (including task-15463's
      `test_watchlists_db_instance_and_off_loop.py`) stay green; due-detection,
      run records, and change-classification semantics are unchanged
- [x] #5 `ContentExtractor.calculate_change_percentage` runs off the event loop
      while the existing below-threshold short circuit remains authoritative
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add thread-identity RED coverage for HTML extraction and offload the existing
   extractor with `asyncio.to_thread`.
2. Add RED coverage for percentage, significant-diff, threshold, and
   cancellation semantics; introduce one private grouped comparison helper and
   offload it.
3. Prove a real multi-URL source uses the worker path for every URL while
   retaining sequential order and accounting.
4. Run only impacted Watchlists tests plus scoped Ruff/format/diff checks.
5. Self-review, record ADR=no and verification evidence, complete acceptance
   criteria, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented standard-library asyncio.to_thread offloads for ContentExtractor.extract_text_from_html, calculate_change_percentage, and one private grouped significant-change helper; preserved sequential url_list processing, thresholds, persistence, circuit-breaker behavior, and public item/disposition shapes. Changed files: tldw_chatbook/Subscriptions/monitoring_engine.py, Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py, and Tests/Subscriptions/test_local_watchlists_service.py. Verification used the user-mandated impacted-only scope; no full suite was run. Exact three-module pytest command: 72 passed, 1 warning in 22.48s. Scoped Ruff check: all checks passed. git diff --check d63e3dbf29c516dc8bb4fea5fcdc73786a9e6cfd..HEAD: clean. Ruff format --check truthfully reports all three legacy files; base-vs-HEAD formatter-delta hashes are identical per file (monitoring_engine ce293878499d3e1b90ee8b11f614a825bb595e725792e187db35ffd759f1a467; DB/off-loop test ba7ec063a6b46ed3adc13fe048fa6812b99296377f040226a0b6f1dfb91a9c23; local service test ebd027dc98279634c68dcfc58de308f67c382c80e5bce655e9a49c7386f47e0a), proving pre-existing whole-file formatter debt outside introduced hunks; no mass-format was applied. ADR required: no; ADR path: N/A. Reason: direct residual performance fix with no storage, runtime, service, public-contract, dependency, or long-lived-boundary change. Plan deviation: cancellation coverage gates the later grouped significant-change worker rather than initial extraction because this directly pins that cancellation cannot resume into snapshot persistence or breaker success; extraction retains separate thread-identity coverage. Self-review: no custom executor, fan-out, schema, public-result, ordering, or threshold change; no DB call in the new plain helper; each text side is segmented exactly once.
<!-- SECTION:NOTES:END -->
