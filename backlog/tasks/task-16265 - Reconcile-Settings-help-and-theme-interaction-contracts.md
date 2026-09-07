---
id: TASK-16265
title: Reconcile Settings help and theme interaction contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 18:34'
updated_date: '2026-08-14 18:54'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Settings help and theme-editor evidence after help dispatch became synchronous and interaction timing changed under the combined Settings cohort.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Obsolete generic Settings help tests are removed in favor of the dedicated category-scoped help contract.
- [x] #2 Theme preset keyboard activation passes in the affected ordered Settings cohort without speculative production changes.
- [x] #3 Affected modules and checkpoint 62 pass with static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve and classify the four checkpoint failures.
2. Remove stale generic-binding help assertions that duplicate the current category-scoped help suite.
3. Reproduce the theme interaction under its real predecessor order and change it only if the failure is deterministic.
4. Run affected modules, checkpoint 62, and static checks.

ADR required: no
ADR path: N/A
Reason: test-harness reconciliation only; no production architecture or behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed three stale RAG help tests that still expected Settings help to flatten
the screen-wide `BINDINGS` superset. Settings now intentionally builds F1 help
from the active category's footer shortcuts, and the dedicated footer/help suite
already verifies that contract. The theme swatch failure passed in the affected
two-module slice and again in the exact 25-file checkpoint order, so no timing or
production workaround was added. Verification: 139 affected tests and 502
checkpoint tests passed; the current help contract test, Ruff lint, and diff
checks passed. Ruff formatting remains red identically on the recorded base file.
<!-- SECTION:NOTES:END -->
