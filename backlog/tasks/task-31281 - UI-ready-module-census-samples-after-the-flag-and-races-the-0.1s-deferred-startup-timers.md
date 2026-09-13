---
id: TASK-31281
title: >-
  UI-ready module census samples after the flag and races the 0.1s
  deferred-startup timers
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-04 14:54'
updated_date: '2026-09-04 14:55'
labels:
  - testing
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The perf-guard census (Tests/Performance/test_ui_ready_module_census.py) polls _ui_ready every 5ms and then copies sys.modules. TldwCli sets the flag and keeps running the rest of its mount path, which arms 0.1s timers (collections-capture wiring, audio service, workspace provisioning) that are deferred past _ui_ready by design. On a slow runner the loop is starved long enough after the flag that those timers fire before the census wakes, and their imports are counted. Observed on PR #2373 (2026-09-04): three runs of the same tree measured 973, 973 and 977 -- the 977 run carried +5 Library.collections_capture_* modules that neither sibling run nor dev's own run at the merge commit had. Dev sits exactly at the 972 ratchet, so the documented +/-1 wobble headroom is gone and this race flips the check red.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census records sys.modules at the instant _ui_ready is set, not on the next poll tick
- [x] #2 Three consecutive local census runs are green and report the same count
- [x] #3 The evidence (the three PR #2373 run ids and their +/- module lists) is in the task notes
- [x] #4 A lesson entry with the incident is added to backlog/docs/lessons-testing-evidence.md
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Intercept the flag with a class-level property in _CENSUS_SCRIPT whose setter copies sys.modules synchronously.
2. Use that copy for the census; keep the poll only to wait for readiness.
3. Run the census test three times locally; confirm the count is no higher than before.
4. Lesson entry; PR; Qodo; merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the census polled _ui_ready every 5ms and copied sys.modules when it woke, while TldwCli sets the flag mid-mount and then arms 0.1s deferred-startup timers (_schedule_deferred_startup_work: collections-capture wiring, audio service, workspace provisioning) later in the same path. On a starved runner those timers won the race. Fix: _CENSUS_SCRIPT installs a class-level property for _ui_ready whose setter copies sys.modules synchronously inside the assignment; the poll only waits for readiness and the census iterates that copy. Evidence, PR #2373 (same tree after the 2nd run): run 33844095943 = 973 (+18/-13, incl. the PR's own module), run 33882174928 = 973 (+19/-14, incl. that module), run 33884089897 = 977 (+23/-14, that module ABSENT, +5 Library.collections_capture_{models,repository,service}, collections_legacy_recovery, collections_offline_store); dev's own perf-guard run 33885569458 at the merge commit 50c9918935 = success. Local after the fix: three consecutive runs 966/972 (was 969 on this platform per the constant's comment), drift +16/-18 identical each time. Files: Tests/Performance/test_ui_ready_module_census.py, backlog/docs/lessons-testing-evidence.md (new entry).
<!-- SECTION:NOTES:END -->
