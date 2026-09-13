---
id: TASK-20011
title: Repair File Notes collection decorator drift
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 18:12'
updated_date: '2026-08-23 18:25'
labels:
  - testing
  - ui
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore collection of the File Notes workspace UI suite after parametrization decorators attached to the wrong intervening test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 File Notes workspace test module collects without errors
- [x] #2 Wide-files test remains a single case
- [x] #3 Path-transition test collects the intended four parameter combinations
- [x] #4 No production module changes
- [x] #5 Focused tests and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the failing collect-only command as RED evidence.
2. Move only the misplaced save/push parametrizers to their consuming path-transition test.
3. Verify exact collection cardinality, focused behavior, full module, and static checks.
4. Record ADR required: no; this is a test-only correction with no architectural decision.
5. Complete review and task hygiene.

ADR required: no
ADR path: N/A
Reason: This test-only decorator correction changes no architecture, production boundary, storage, security, or long-lived application behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the existing save-state and push-state parametrization decorators from the intervening wide Files-return test to the path-transition authority test that consumes those arguments. The correction is test-only: no production code or architectural boundary changed, so no ADR is required. Collect-only verification now reports one wide-return case and the intended four path-transition combinations; the five focused cases and all 124 tests in the File Notes workspace module pass. Ruff lint, Ruff range-format verification for the changed region, and diff checks pass. No new lesson was added because the existing task and regression evidence fully capture this localized decorator-placement error.
<!-- SECTION:NOTES:END -->
