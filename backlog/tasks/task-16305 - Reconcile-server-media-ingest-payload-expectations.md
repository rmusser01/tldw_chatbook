---
id: TASK-16305
title: Reconcile server media ingest payload expectations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 01:08'
updated_date: '2026-08-14 01:10'
labels:
  - test-health
  - media
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile server media ingest tests with the live-verified endpoint contract that omits unsupported forced embedding regeneration from submissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ingest submission expectations exclude `force_regenerate_embeddings` when false.
- [x] #2 Reprocess expectations retain the supported force-regeneration field.
- [x] #3 The focused files, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this updates stale tests to the existing TASK-3309 live-verified server API contract.

1. Preserve the two exact payload mismatches and verify the endpoint decision history.
2. Remove only the unsupported submission-field expectations.
3. Run focused files, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the obsolete forced-regeneration field from two exact ingest-submission payload expectations, matching TASK-3309's live-verified server endpoint contract. The reprocess expectation still requires the supported `force_regenerate_embeddings=False` field. Both focused tests and all 324 chunk-26 tests passed; Ruff lint/format and diff checks passed.
<!-- SECTION:NOTES:END -->
