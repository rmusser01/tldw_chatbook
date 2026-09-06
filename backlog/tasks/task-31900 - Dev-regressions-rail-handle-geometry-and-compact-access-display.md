---
id: TASK-31900
title: >-
  Dev regressions: vertical rail-handle geometry and compact-access rail display
status: To Do
assignee: []
created_date: '2026-09-06 09:00'
labels: [console, tests, regression]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while baselining PR #2453 against origin/dev @ 5894f4755e (paired arms in a
throwaway worktree): five failures reproduce IDENTICALLY on the untouched dev tree.
Two are the known wall-clock flakes, but two look like genuine dev regressions:
test_console_rail_handle vertical-handle geometry fails on content_region.width
3 == 1, and test_console_inspector_compact_access x2 fail on rail.display is False
at widths where the rail should show. Neither is caused by the burn-down branch
(verified base-vs-branch byte-identical failure sets across three commits).
Bisect against dev history; the 472-commit window between 7e904737c7 and
5894f4755e contains the culprit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Both failures bisected to their introducing commit(s) on dev and either fixed or the tests re-pinned with a documented behavior ruling
- [ ] #2 The known wall-clock flakes in the same files are annotated or stabilized so they stop polluting baseline sweeps
<!-- AC:END -->
