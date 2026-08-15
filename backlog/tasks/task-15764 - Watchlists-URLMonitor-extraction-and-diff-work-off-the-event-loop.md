---
id: TASK-15764
title: 'Watchlists: URLMonitor extraction and diff work off the event loop'
status: Done
assignee: []
created_date: '2026-08-13 12:31'
updated_date: '2026-08-15 00:20'
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
- [x] #4 The impacted Watchlists/Subscriptions modules
      `test_watchlists_db_instance_and_off_loop.py`,
      `test_watchlist_content_kind_producer.py`, and
      `test_local_watchlists_service.py` stay green in the approved impacted-only
      verification scope; due-detection, run records, and change-classification
      semantics are unchanged
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
Implemented standard-library `asyncio.to_thread` offloads for
`ContentExtractor.extract_text_from_html`, `calculate_change_percentage`, and
one private grouped significant-change helper. Sequential `url_list` processing,
thresholds, persistence, circuit-breaker behavior, and public item/disposition
shapes remain unchanged.

- Implementation/test files: `tldw_chatbook/Subscriptions/monitoring_engine.py`,
  `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`, and
  `Tests/Subscriptions/test_local_watchlists_service.py`.
- Closeout record:
  `backlog/tasks/task-15764 - Watchlists-URLMonitor-extraction-and-diff-work-off-the-event-loop.md`.
- Verification used the user-mandated impacted-only scope; no full suite was run.
  Exact command:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py \
    Tests/Subscriptions/test_watchlist_content_kind_producer.py \
    Tests/Subscriptions/test_local_watchlists_service.py
  ```

  Result: `72 passed, 1 warning in 22.48s`; the warning was the existing
  `RequestsDependencyWarning` about the installed
  urllib3/chardet/charset_normalizer versions.
- Scoped Ruff lint passed.
  `git diff --check d63e3dbf29c516dc8bb4fea5fcdc73786a9e6cfd..HEAD`
  passed.
- Scoped `ruff format --check` truthfully reported all three legacy files. To
  classify it, the three files were archived from base
  `d63e3dbf29c516dc8bb4fea5fcdc73786a9e6cfd` and from `HEAD`, each copy was
  formatted with the same Ruff command, and each raw-to-formatted patch was
  compared with `diff -U0` after removing only the path and `@@` line-number
  headers. Base and HEAD had identical remaining formatter-diff bodies (one
  hunk in `monitoring_engine.py`, two in the DB/off-loop test, and seventeen in
  the local-service test); only their line offsets changed. The remaining Ruff
  deltas therefore predate this task and are outside the introduced hunks. No
  mass-format was applied.
- ADR required: no. ADR path: N/A. This is a direct residual performance fix
  with no storage, runtime, service, public-contract, dependency, or long-lived
  boundary change.
- Plan deviation: cancellation coverage gates the later grouped
  significant-change worker rather than initial extraction because this
  directly pins that cancellation cannot resume into snapshot persistence or
  breaker success; extraction retains separate thread-identity coverage.
- Self-review found no custom executor, fan-out, schema, public-result,
  ordering, or threshold change; no DB call exists in the new plain helper;
  each text side is segmented exactly once.
- Lessons hygiene: no reusable lesson was added because the work surfaced no
  new generalizable repository trap.
<!-- SECTION:NOTES:END -->
