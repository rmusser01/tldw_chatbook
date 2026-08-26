---
id: TASK-16245
title: Wire image controller in live-work restore fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 11:26'
updated_date: '2026-08-14 11:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the live-work state restoration integration fixture aligned with the controller initialization contract so screen recreation remains covered without a mounted Textual app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The bare restore fixture attaches the shared fail-loud image controller stub.
- [x] #2 The staged live-work handoff restoration regression passes.
- [x] #3 The affected module and exact checkpoint chunk pass.
- [x] #4 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-fixture correction that preserves the existing controller boundary.

1. Preserve the failing focused test as RED evidence and trace the restore dependency.
2. Attach the existing fail-loud image-controller stub in the bare restore fixture.
3. Run the focused regression, affected test module, and exact checkpoint chunk.
4. Run scoped static checks, self-review the diff, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated the bare live-work restore fixture to attach the repository's existing fail-loud `ConsoleImageController` stub alongside its message-controller stub; production code was unchanged.
- Verified the original failing node (1 passed), the full handoff module (65 passed), and checkpoint chunk 49 (865 passed).
- `git diff --check` passed. Scoped Ruff check/format remain red only for the exact pre-existing HEAD baseline in the legacy test file (one unused `App` import and existing formatting drift); the new hunk introduces no new static violation.
- ADR check: no ADR was required because this correction preserves the existing controller boundary.
<!-- SECTION:NOTES:END -->
