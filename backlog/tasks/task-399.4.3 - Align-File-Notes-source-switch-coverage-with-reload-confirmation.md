---
id: TASK-399.4.3
title: Align File Notes source-switch coverage with reload confirmation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 00:51'
updated_date: '2026-08-12 00:53'
labels:
  - notes
  - library
  - ux
  - tests
dependencies:
  - TASK-399.4.2
parent_task_id: TASK-399.4
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the end-to-end Database/Files source-switch workflow aligned with the intentional destructive-reload confirmation so retained drafts are never discarded implicitly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source-switch integration test verifies the conflict draft remains unchanged before confirmation
- [x] #2 The test explicitly confirms reload before expecting the disk version and source-switch veto to clear
- [x] #3 Focused reload and source-switch tests plus static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is test coverage alignment with the already-shipped destructive reload confirmation, not a contract change.

1. Reproduce the integration failure.
2. Update the workflow assertion to exercise confirmation and retained-draft safety.
3. Run focused reload/source-switch and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated the Database/Files source-switch integration trace to wait for the destructive reload confirmation, assert the conflict draft remains intact, and explicitly confirm before expecting the disk version.
- Kept production behavior unchanged because the confirmation is the intended safety contract and is already covered in dedicated keyboard and stale-state tests.
- Evidence: reproduced the stale integration failure; corrected source-switch test 1 passed; complete reload matrix 9 passed; Ruff, compileall, and `git diff --check` passed.
- No documentation or ADR change was required.
<!-- SECTION:NOTES:END -->
