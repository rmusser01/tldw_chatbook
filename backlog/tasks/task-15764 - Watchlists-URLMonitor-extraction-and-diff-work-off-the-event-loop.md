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

## Reconciliation (2026-08-15, duplicate-implementation event -- task-596 playbook)

Two sessions implemented this task independently: the merged-first
implementation above (Codex, commits `8f638f815` + `8fbd1426d`, PR #1650,
merged 2026-08-14 17:38 -0700) STANDS; a second session on burn-down base
`bb91fef73` (task file still `To Do` at that base) produced a structurally
convergent implementation (`d07cfc50f` on `task/15764-burn`, independently
reviewed with verdict MERGE), whose review-hardened delta is ported on top.
Incident recorded in `lessons-backlog-hygiene.md` (4th "status is not a
lock" instance).

**Incumbent audit (by the second session, against base `bb91fef73`):**

- Segment-once sharing: PRESERVED -- `_build_significant_change_details`
  segments each side exactly once and hands the segments to both consumers.
- Semantic identity: VERIFIED byte-identical across 11 full baseline+change
  cycles (whole `change_info` minus `published_date`, both dispositions,
  all persisted snapshot rows) covering the truncation path, unicode,
  markup-only/unchanged, additions/removals-only, empty-page both
  directions, and the below-threshold withheld path. No defect found; no
  production change needed.
- `response.text` decode placement: identical to the reviewed version
  (decoded once on the loop, ~0.5 ms at the 10 MB cap per the review's
  measurement); the incumbent's comments make no claims, so nothing
  overclaims.

**Ported delta:**

- `Tests/Subscriptions/test_url_monitor_off_loop.py` -- whole-check
  extraction thread-identity through `check_url` (the incumbent probes
  `_fetch_url_content` directly); the url_list coverage AC #3 was ticked
  without: two URLs x two runs through the real `launch_run`/`execute_run`
  path, every URL's extraction AND diff off-loop; the `_segment_for_diff`
  called-exactly-twice pin. Born-red evidence against the incumbent
  implementation via per-hop runtime re-inlining (a shim replacing only
  `monitoring_engine.asyncio.to_thread` for one target at a time):
  extract -> extraction+url_list tests fail; percentage -> diff+url_list
  fail; details -> diff+url_list fail; clean run 3/3 green.
- `lessons-testing-evidence.md`: the review-CORRECTED cost-profile lesson
  (the original 16.2 s / "99.8%" figures on 160 KB Latin text did not
  reproduce -- autojunk fast path, 20-40 ms; the quadratic regime is
  repertoire-dependent, ~7 min at the 10 MB cap on CJK/unicode-heavy text,
  so the off-loop move is MORE justified than the original numbers implied).

**Review follow-ups queued for filing:** per-(subscription,url) in-flight
guard (pre-existing scheduled-vs-Check-Now double-report window, present at
`bb91fef73`); bound `calculate_change_percentage` by input size (a worker
thread can hold the GIL for minutes at the fetch cap on large-repertoire
text); autojunk makes the reported change percentage meaningless for large
Latin-text pages (pre-existing correctness oddity).
