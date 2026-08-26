---
id: TASK-16270
title: Align Collections empty-state test with current copy contract
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:47'
updated_date: '2026-08-14 20:50'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Collections deduplication test aligned with the current TASK-4023 empty-state contract instead of retired TASK-2855 copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test proves the current empty-state copy renders exactly once.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the stale-copy failure and trace the current TASK-4023 contract.
2. Assert the canonical state copy through its single owned widget and retain retired-widget absence checks.
3. Run focused, module, checkpoint, and static verification.

ADR required: no
ADR path: N/A
Reason: test-only reconciliation with an already-shipped copy contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the assertion for TASK-2855's retired sentence with an exact assertion on the current TASK-4023 `state.empty_copy` rendered by its single owned widget; retired duplicate widget IDs remain absent.
- Preserved RED evidence: the isolated test and original checkpoint both reported zero matches for the deleted sentence. Restoring that assertion reproduces the failure.
- Verified the full Collections panel module (11 passed) and original 25-file checkpoint (441 passed). Ruff lint and `git diff --check` pass. Ruff format is already red on `HEAD` and remains unchanged; no unrelated formatting churn was included.
- ADR required: no. This is test-only reconciliation with existing product copy.
<!-- SECTION:NOTES:END -->
