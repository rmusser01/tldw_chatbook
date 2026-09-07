---
id: TASK-1468
title: >-
  Restore per-test gc for app-mounting test dirs — periodic-only gc exposed Textual app-cycle interference between adjacent tests
status: Done
assignee: []
created_date: '2026-07-30 12:55'
labels:
  - testing
  - bug
priority: high
dependencies: [task-1454]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1454's periodic gc (every 25 tests) surfaced a latent coupling the old per-test double-collect had been masking: a Textual `App` is a reference cycle that only `gc.collect()` reclaims, and an uncollected app from the previous test interferes with the next app-mounting test. Post-merge evidence: a 10-test batch (TestChatApiCall + one Skills UI test + two Library git UI tests) failed a **different** UI test on consecutive runs — the victim rotates with heap state — passed 10/10 with `TLDW_TEST_GC_EVERY=1`, and passed 10/10 on pre-task-1454 dev. Interval tuning cannot fix adjacency (any interval >1 leaves neighboring app tests unprotected), so per-test collection is restored for exactly the directories the audit's `run_test()` census shows mount apps; everything else keeps periodic collection, which is where task-1454's measured win (Tests/Notes 131.9s→111.7s) lives.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] The rotating-victim batch passes repeatedly (3×) under default settings
- [x] Tests in app-mounting dirs (UI, Widgets, Watchlists, Skills, Library, Event_Handlers, integration, Chat) collect per test; other dirs remain periodic
- [x] Non-app dirs keep the task-1454 speedup (no behavior change for them)

## Implementation Plan

1. Confirm the mechanism: batch A/B with `TLDW_TEST_GC_EVERY=1` vs default, and against pre-task-1454 dev
2. Add an app-dir path predicate to `cleanup_file_descriptors` alongside the marker and counter
3. Verify the batch 3× and record the lesson

## Implementation Notes

`_APP_MOUNTING_DIR_PARTS` (from the audit's run_test call-site census) added as a
third trigger in `cleanup_file_descriptors`. The old behavior was TWO collects per
test everywhere; the new steady state is ONE collect per test in ~8 app dirs and
one per 25 tests elsewhere. Lesson recorded in lessons-testing-evidence.md.
Modified: `Tests/conftest.py`.
