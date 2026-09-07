---
id: TASK-15104
title: Repair hanging Library source-transition test
status: Done
assignee: []
created_date: '2026-08-11 00:58'
updated_date: '2026-08-11 01:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct a stale test harness note-flush stub so the Library Files-to-Collections transition test exercises the current typed flush contract instead of timing out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The exact hanging node completes using a typed permitted note-flush result
- [x] #2 Adjacent source-transition tests pass
- [x] #3 No production behavior changes
- [x] #4 The mandatory full-suite gate can proceed past this node
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-harness contract correction only; it does not change production behavior or architecture.

1. Preserve the authoritative branch and clean-dev 300-second timeout results as RED evidence.
2. Update only Tests/UI/test_screen_navigation.py so the stale note-flush stub returns the current typed permitted outcome.
3. Run the exact node and adjacent transition tests GREEN; perform a safe mutation/revert proof of the stale return value.
4. Run focused Ruff and committed/working-tree diff-quality checks.
5. Record evidence in Implementation Notes, check every acceptance criterion, mark the task Done, and commit the task record with the test correction atomically.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected the stale Files-to-Collections navigation-test stub to return the current typed permitted note-flush outcome. This keeps production behavior unchanged while allowing the test to reach its intended transition/recompose assertions.

Evidence: the unchanged stub timed out at 300 seconds on both the TASK-2512 branch and clean origin/dev 8d764c03, and a safe local mutation reproduced the timeout in 3 seconds. After restoration, the exact node passed in 1.08 seconds; the eight-node adjacent transition and typed-flush group passed in 2.18 seconds. Ruff check passed, the changed range is Ruff-formatted, and git diff --check passed.

Modified files: Tests/UI/test_screen_navigation.py and this task record.

Integration note: this later claimant moved from TASK-14913 to TASK-15104 after
the exact add-commit audit showed that the Console context-limit task had already
claimed TASK-14913 on dev.
<!-- SECTION:NOTES:END -->
