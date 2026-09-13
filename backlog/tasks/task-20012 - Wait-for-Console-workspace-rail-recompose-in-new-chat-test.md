---
id: TASK-20012
title: Wait for Console workspace rail recompose in new-chat test
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 18:12'
updated_date: '2026-08-23 18:51'
labels:
  - testing
  - console
  - ui
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove a load-sensitive false failure by asserting the eventual mounted Console workspace conversation rail after Textual completes its scheduled recompose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New-chat rail test waits through the existing bounded paint-settle seam.
- [x] #2 The test verifies Chat 1 remains and Chat 2 becomes selected.
- [x] #3 No fixed sleep or production module change is introduced.
- [x] #4 The required four-suite aggregate gate passes.
- [x] #5 Focused tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the aggregate-suite failure and direct-DOM timing diagnosis as RED evidence.
2. Reuse the existing bounded workspace-conversation paint helper for both Chat 1 and selected Chat 2 assertions.
3. Verify focused test, full Console native-flow file, required four-suite aggregate, and static checks.
4. Record the ADR determination for this test-only synchronization correction.
5. Complete review and task hygiene.

ADR required: no
ADR path: N/A
Reason: This is a test-only synchronization correction with no runtime behavior or architectural boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the new-chat test's immediate workspace-rail DOM reads with the
  existing bounded `_wait_for_workspace_conversation_text` paint-settle helper.
  The test now waits for Chat 1 before and after the new-chat action and for
  Chat 2 with `selected=True`; no fixed sleep or production change was added.
- RED evidence preserved from diagnosis: the exact four-suite aggregate failed
  twice with `601 passed, 1 failed`, both times at the owned immediate rail
  assertion. A third unchanged pre-edit run passed all 602, confirming the
  scheduling defect was intermittent rather than a deterministic product
  failure.
- Verification: victim test `1 passed`; neighboring new-chat/new-conversation
  selection `7 passed`; full native Console flow `320 passed`; exact four-suite
  aggregate `602 passed`; Ruff lint, Python compilation, touched-function Ruff
  formatting, and `git diff --check` passed.
- Whole-file Ruff formatting continues to identify two unrelated pre-existing
  blocks at lines 2542 and 2582. The exact touched function formats cleanly, and
  those baseline lines were left unchanged to keep this task atomic.
- Self-review moved the post-click selected Chat 2 wait ahead of the retained
  Chat 1 assertion, so Chat 1 cannot be matched against the old pre-recompose
  DOM. The final full-file and aggregate gates were rerun after that correction.
ADR required: no
ADR path: N/A
Reason: This is a test-only synchronization correction with no runtime behavior or architectural boundary change.
<!-- SECTION:NOTES:END -->
