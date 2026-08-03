---
id: TASK-1394
title: One failing URL fails a whole url_list run and discards collected items
status: In Progress
assignee: []
created_date: '2026-07-30 05:20'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`local_watchlists_service.py:894-921`: the `url_list`/`sitemap` arms loop URLs with no per-URL
`try/except`, so one failing URL (timeout, SSRF block, HTTP error) raises out of the loop, fails
the whole run via `record_run_failure`, and discards the items already collected from the URLs
that succeeded.

Pre-existing, found during TASK-1362's whole-branch review; newly load-bearing because per-URL
baselines and dispositions (TASK-1361/1362) make large multi-URL sources more useful, and a
50-URL source with one dead link currently yields nothing at all.

Design note: isolating failures per URL wants an error disposition (the current vocabulary is
changed/unchanged/withheld/baseline/rebaselined) so a partially-failed run says so in the Runs
pane rather than reporting clean counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A url_list run with one failing URL persists the items and dispositions from the URLs that succeeded
- [ ] #2 The failure is visible per run (an error count or disposition in the Runs detail), not silently absorbed
- [ ] #3 A test with one poisoned URL among several fails under the old all-or-nothing behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `_default_run_executor` (`local_watchlists_service.py`): wrap each `monitor.check_url(...)` in the
   `url_list` AND `sitemap` loops in try/except. On exception, append a NEW `error` disposition
   (type-only, no URL/content in the recorded value) and continue — never add to `items`, never
   raise out of the loop. Items/dispositions from the URLs that succeeded persist (AC#1).
2. Add a 6th `error` counter to `_ALL_DISPOSITION_COUNTERS` (:68) + `_disposition_counts` mapping so
   a partially/fully-failed run reports its error count in `run_stats["dispositions"]` (AC#2).
3. Render the error count in `runs_pane.py`'s Checks line (~:162) alongside changed/unchanged/etc.
4. Tests: one poisoned URL among several → succeeded items persist + error counter = 1 (AC#1/#2);
   the same test REDs without the try/except (poisoned URL raises, whole run fails — AC#3).
<!-- SECTION:PLAN:END -->
