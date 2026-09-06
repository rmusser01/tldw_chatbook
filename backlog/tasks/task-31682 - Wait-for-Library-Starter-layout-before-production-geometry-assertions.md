---
id: TASK-31682
title: Wait for Library Starter layout before production geometry assertions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:21'
updated_date: '2026-09-05 18:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the integrated Starter geometry failures by distinguishing settled evidence state from the rendered production rail and landing, retaining every geometry and keyboard-order assertion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both production geometry sizes retain all geometry and focus-order assertions and pass reliably
- [x] #2 Readiness follows observable mounted and painted state without fixed delays or increased timeouts
- [x] #3 Focused Starter tests and static checks pass with no production DOM or budget changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both production-CSS geometry sizes without edits and inspect Starter recompose versus layout readiness.
2. Wait at the local harness boundary for mounted and painted Starter widgets, retaining every original visibility, geometry, and keyboard-order assertion.
3. Run both sizes repeatedly and related Starter/graduation coverage, then scoped static checks and review.
ADR required: no
ADR path: N/A
Reason: Test-only readiness repair preserving existing production lifecycle, DOM, CSS, and architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Waits for all existing required Starter selectors to resolve to current compositor widgets before geometry assertions. Both baseline cases failed with zero-sized collapse regions; settled evidence had preceded replacement layout. All original geometry, visibility, keyboard-order and painted-copy assertions remain unchanged.
Verification: Starter/graduation selection 21 passed, 803 deselected in 36.16s; independent two-size geometry rerun 2 passed in 7.49s. Ruff check, changed-function range-format and git diff --check pass. Whole-file format debt elsewhere was preserved. Parent review found no blocking issue.
Documented the specific settlement/layout incident in library-decomposition-recipe section 23. ADR required: no; test-only readiness, no production behavior or architectural boundary change.
<!-- SECTION:NOTES:END -->
