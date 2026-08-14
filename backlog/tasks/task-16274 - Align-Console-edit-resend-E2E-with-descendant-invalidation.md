---
id: TASK-16274
title: Align Console edit-resend E2E with descendant invalidation
status: Done
assignee: []
created_date: '2026-08-14 21:14'
updated_date: '2026-08-14 21:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Console edit-and-resend integration test aligned with the durable rule that an in-place ancestor edit invalidates stale assistant descendants without creating a sibling branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The E2E proves in-place Save creates no sibling branch.
- [x] #2 The E2E proves stale descendants are removed from the active transcript and persistence.
- [x] #3 Focused Console integration coverage passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the current persisted ancestor-edit contract and reproduce the stale assertion.
2. Update the E2E expectation to prove descendant invalidation and no sibling creation.
3. Run focused and adjacent integration verification.

ADR required: no
ADR path: N/A
Reason: test reconciliation for an existing durable edit contract; no architecture or ownership change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the edit-and-resend E2E to match the durable ancestor-edit contract introduced by the continuation lifecycle hardening: a plain Save still edits the selected user row without creating a sibling, while its now-stale assistant descendant is removed from both the active transcript and the database. The original assertion failed with the stale reply still expected; focused verification passed, and the exact 25-file regression slice passed with 292 tests and 6 optional-dependency skips. ADR required: no; this is test reconciliation for existing behavior.
<!-- SECTION:NOTES:END -->
