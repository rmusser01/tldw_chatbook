---
id: TASK-1078
title: Remove stale deferred initial-tab test after startup refactor
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 22:01'
updated_date: '2026-07-27 22:50'
labels:
  - ui
  - tests
  - baseline
dependencies: []
references:
  - Tests/UI/test_screen_navigation.py
  - >-
    backlog/tasks/task-288 -
    Canonicalize-current_tab-through-route-aliases-at-startup.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The scoped TASK-944 baseline run found `test_deferred_initial_tab_uses_first_run_home_route` still calling `TldwCli._set_initial_tab`, which was removed by commit 1df0c4cb4. Repair or remove the stale test while preserving current first-run Home routing coverage; do not restore the deleted production method or broaden into TASK-288 alias canonicalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The navigation suite no longer calls the removed `_set_initial_tab` method
- [x] #2 Current first-run Home routing remains covered through the supported startup or resolver path
- [x] #3 The focused navigation test file passes without production runtime changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the stale `_set_initial_tab` failure with its exact test node.
2. Compare adjacent first-run routing coverage and remove the obsolete test if the supported resolver path already pins the same outcome; otherwise rewrite only that test against the current startup path.
3. Run only `Tests/UI/test_screen_navigation.py`, then Ruff on that test file and `git diff --check`.
4. Review the minimal diff, document verification, complete the acceptance criteria, and mark TASK-1078 Done.

ADR required: no
ADR path: N/A
Reason: Test-only cleanup after an existing startup refactor; no application behavior or architecture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the single obsolete `test_deferred_initial_tab_uses_first_run_home_route` test. It duplicated the adjacent resolver-level first-run Home assertion while calling `_set_initial_tab`, a production method removed by commit 1df0c4cb4. No replacement test or production compatibility shim was added because `test_first_run_initial_route_defaults_to_home` already covers the supported routing contract.

TDD evidence: the exact stale node first failed with `AttributeError: TldwCli has no attribute _set_initial_tab`; after deletion, the focused navigation file passed 55 tests. Ruff check and format-check passed for the one touched test file, no `_set_initial_tab` reference remains there, and `git diff --check` passed. The run emitted only existing RequestsDependencyWarning and train-journey SyntaxWarnings.

ADR required: no. ADR path: N/A. Test-only cleanup with no production or architectural change.
<!-- SECTION:NOTES:END -->
