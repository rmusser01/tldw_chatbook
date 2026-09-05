---
id: TASK-31689
title: Repair Library prompt browse and history settlement regressions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:30'
updated_date: '2026-09-05 18:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Diagnose and repair the five prompt-canvas failures from the integrated UI sweep, preserving latest user focus, stale-request exclusion, retry behavior and collapsed history ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Latest surviving focus and filter caret remain authoritative during prompt browse settlement
- [x] #2 History retries, collapse during restore, and no-change selection preserve their existing behavioral assertions
- [x] #3 Causal reproduction distinguishes runtime bugs from harness readiness and focused related tests pass without weakened assertions or higher budgets
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the five integrated failures together and trace request, recomposition, focus and history state ownership; record non-reproduced races separately.
2. Characterize causal failure points with existing real callback and worker seams before proposing runtime repairs; obtain root review for behavior changes.
3. Repair only proven harness readiness or state-ownership defects, retaining assertions and timeout ceilings; split distinct runtime policies into separate tasks when necessary.
4. Run the reported five together plus prompt browse/history controller and related canvas coverage, scoped static checks and review.
ADR required: no new ADR anticipated
ADR path: N/A
Reason: Bounded regression repair intended to preserve existing latest-user-focus and history lifecycle contracts. Reassess if diagnosis requires a changed cross-module contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only readiness repair. The stale-caret baseline issued the next search with screen.focused=None during loading recomposition; now a deliberately gated loading recompose settles its current filter/caret before the next action. Surviving-focus likewise waits for loading admission. Final caret and stale-result assertions are unchanged.
A deterministic real remove/mount probe proved _wait_for_selector returned its pre-Pilot-pause owner. The helper re-queries after pause and requires exact current attached identity. Textual is_mounted remains true after removal, so is_attached is the relevant ownership check. History retry/collapse/no-change use current rendered controls, and stale-conflict completion includes detail adoption/notification rather than only earlier SQLite v4.
Baseline reported five selection:1 failed4 passed; expanded related selection before readiness fixes:2 failed75 passed (retry-page and stale-conflict notification). After:78 passed289 deselected in107.75s; independent complete browse/history controller files25 passed in2.29s. New stale-owner regression red before and green after; delayed loading caret also green. Ruff check, touched-range/new-file format and git diff --check pass; unrelated whole-file format debt preserved. Parent review found no actionable issue.
ADR required:no; no runtime behavior or architecture change. Specific incidents recorded in library-decomposition-recipe section24. Broader Notes responsive-scroll work remains separate.
<!-- SECTION:NOTES:END -->
