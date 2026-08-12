---
id: TASK-399.4.4
title: Preserve File Notes focus across Library breakpoints
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 00:54'
updated_date: '2026-08-12 01:00'
labels:
  - notes
  - library
  - ux
  - accessibility
dependencies:
  - TASK-399.4.3
parent_task_id: TASK-399.4
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep keyboard focus inside the retained File Notes canvas when the Library shell crosses compact and wide breakpoints, including when optional Library backends are unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused File Notes editor remains in the visible Library canvas across 40-column and wide breakpoint transitions
- [x] #2 Breakpoint recomposition never leaves the Library screen with no focused widget while File Notes remains mounted
- [x] #3 Focused production-shell and static checks pass without weakening optional-backend error handling
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a localized focus-retention regression within the existing Library shell structure.

1. Reproduce and identify the breakpoint that clears focus.
2. Restore focus only when retained content loses it during shell layout changes.
3. Run production-shell, compact-layout, and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Extended the Library Notes breakpoint identity mechanism to serialize a focused retained File Notes control by stable widget ID and resolve it inside the retained workspace.
- Added the responsive focus memory as the lowest-priority resize fallback, covering the case where Textual clears focus before the return transition is captured while preserving explicit user-focus overrides.
- Strengthened the production-shell test to require the same editor instance, not merely any visible focus, across compact/wide round trips.
- Evidence: reproduced loss at the first 40x20 transition and then at the 120x40 return transition; corrected production-shell test 1 passed; four Database Notes breakpoint/user-veto regressions passed; compileall and `git diff --check` passed. Ruff passed with `E721` excluded; the full-file Ruff invocation still reports seven pre-existing `E721` violations at unrelated legacy lines.
- No documentation or ADR change was required.
<!-- SECTION:NOTES:END -->
