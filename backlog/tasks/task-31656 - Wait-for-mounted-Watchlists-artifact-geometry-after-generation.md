---
id: TASK-31656
title: Wait for mounted Watchlists artifact geometry after generation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:24'
updated_date: '2026-09-05 17:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent generation test helpers from accepting populated replacement tables before Textual has mounted and laid them out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generation readiness waits for the current visible mounted table geometry as well as database row agreement.
- [x] #2 A controlled delayed-layout regression retains the original list, button, body placement and painted-label assertions.
- [x] #3 The complete affected Watchlists test file and scoped static checks pass.
- [x] #4 Cast readiness likewise tolerates replacement-table absence and waits for the current mounted geometry while preserving script row and content assertions.
- [x] #5 Synthesize readiness waits for its current mounted script detail before audio-content assertions, including transient replacement absence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the observed zero-region failure with controlled delayed-layout regressions.
2. Extend the bounded Generate and Cast readiness polls with current-table presence, mounted/displayed nonzero geometry, and explicit timeout failures.
3. Apply the same current-widget lifecycle predicate to the Synthesize detail after full-file evidence catches its final database-only readiness gap.
4. Run regressions, the full affected file, and scoped static checks; document evidence and commit task-owned files only.
ADR required: no
ADR path: N/A
Reason: test-only readiness correction; production layout and runtime ownership remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned Generate, Cast, and Synthesize test helper completion with the current mounted, displayed, nonzero-geometry widget as well as their original database/toast conditions. All polls retain their existing 20-second deadline and now explicitly fail on timeout; missing replacements remain within the bounded poll. Production code, layout, row/content assertions, and terminal placement checks are unchanged.

The controlled delayed-layout regression reproduced the exact zero-region failure before the Generate fix and now covers both briefing and script tables. Full-file evidence also caught replacement absence in Generate, seven Cast cases, and one Synthesize detail case; the task acceptance criteria/plan were extended before those same-contract sibling repairs. Focused delayed-layout and both original geometry cases passed (4 tests). Final complete file: 146 passed in 204.60s in the isolated installed Python 3.12 environment with corrected GID20 pytest temp root. Full-file Ruff lint, all changed-region formatter checks, and git diff --check passed; unrelated pre-existing whole-file formatting drift was preserved. Root reviewed the Generate predicate and regression with no actionable findings; self-review covered the final sibling changes. Added the observed lifecycle trap to lessons-testing-evidence.md.

ADR required: no; test-only mounted-readiness correction does not change product boundaries. Files: Watchlists artifacts tests, this task, and testing-evidence lessons.
<!-- SECTION:NOTES:END -->
