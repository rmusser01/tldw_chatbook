---
id: TASK-1394
title: One failing URL fails a whole url_list run and discards collected items
status: To Do
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
`local_watchlists_service.py:838-862`: the `url_list`/`sitemap` arms loop URLs with no per-URL
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
