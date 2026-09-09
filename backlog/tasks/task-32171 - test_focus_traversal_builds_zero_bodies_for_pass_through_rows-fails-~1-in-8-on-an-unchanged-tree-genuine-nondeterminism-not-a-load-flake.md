---
id: TASK-32171
title: >-
  test_focus_traversal_builds_zero_bodies_for_pass_through_rows fails ~1-in-8 on
  an unchanged tree (genuine nondeterminism, not a load flake)
status: To Do
assignee: []
created_date: '2026-09-09 10:03'
labels:
  - library
  - tests
  - flaky-test
  - phase-c
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_media_reader_traversal_t22207.py::test_focus_traversal_builds_zero_bodies_for_pass_through_rows fails about 1 run in 8 in isolation on an unchanged tree. Confirmed production-revert-controlled during phase-C task 3: 'git checkout <base> -- tldw_chatbook/' then running the test alone STILL failed, which no production-caused regression can survive; eight runs per arm gave 7 passed / 1 failed on BOTH the branch and the base commit -- an identical rate. Two prior phase-C reports labelled it a 'load flake, passes in isolation', which was too generous: it fails alone, at a stable ~1-in-8 rate. It keeps costing attribution time in every paired sweep of the media battery, because a genuine intermittent is indistinguishable from a branch-caused regression until the revert control is run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The source of the nondeterminism in the zero-body pass-through traversal path is identified (systematic-debugging: find the varying input, likely a settle/ordering/timing-dependent assertion)
- [ ] #2 The test is made deterministic -- passes N-in-N or fails 0-in-N reliably over at least eight isolated runs -- OR the production nondeterminism it legitimately catches is found and fixed
- [ ] #3 The fix is verified with n>=8 isolated runs on the current tree showing a stable rate, recorded in the task's implementation notes
<!-- AC:END -->
