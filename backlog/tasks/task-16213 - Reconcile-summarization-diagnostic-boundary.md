---
id: TASK-16213
title: Reconcile summarization diagnostic boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:25'
updated_date: '2026-08-14 00:29'
labels:
  - test-health
  - security
  - diagnostics
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the summarization privacy fixture after the repository-wide diagnostic inventory refresh removed the historical checked-versus-generated drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fixture pins the independently computed checked and generated normalized inventories at their current values.
- [x] #2 The boundary continues to reject unrelated inventory drift and summarization-owner mismatches.
- [x] #3 The complete privacy module, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this refreshes test evidence after an existing governed inventory reconciliation; it changes no diagnostic policy or runtime boundary.

1. Preserve the three stale-boundary failures and independently calculate both normalized inventory hashes.
2. Refresh the fixture and replace the obsolete historical-inequality assertion with a reconciled-source assertion.
3. Prove unrelated and owner-specific mutations remain rejected, then run the full module, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed both independently evaluated normalized inventory hashes after the governed global diagnostic reconciliation made the checked manifest and live source inventory equal. Replaced the obsolete inequality assertion with equality while preserving all unrelated-drift and summarization-owner mismatch checks. A one-character hash mutation failed the boundary as intended. The full privacy module passed 257 tests and the containing chunk passed 1,242 tests.
<!-- SECTION:NOTES:END -->
