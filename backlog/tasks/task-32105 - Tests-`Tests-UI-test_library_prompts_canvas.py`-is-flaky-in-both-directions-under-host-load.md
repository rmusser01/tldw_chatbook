---
id: TASK-32105
title: >-
  Tests: `Tests/UI/test_library_prompts_canvas.py` is flaky in both directions
  under host load
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - tests
  - library
  - prompts
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the critique-8 wave two full runs of this file on the same pair of commits disagreed in both directions (five names failed on base and passed on branch, two the other way); name-set diffs against it require a quiet machine, and it took 85 minutes under load. Found while verifying PR #2528. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The flaky names are identified and either stabilised or marked with the reason
- [ ] #2 A full run of the file on a loaded host gives the same failing-name set twice
<!-- AC:END -->
