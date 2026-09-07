---
id: TASK-16217
title: Reconcile local transcription test doubles
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:52'
updated_date: '2026-08-14 00:56'
labels:
  - test-health
  - transcription
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile local transcription test doubles with the current provenance, timestamp, and native backend contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio model-directory routing verifies all current transcription kwargs without dropping provenance defaults.
- [x] #2 The fake transcribe-cpp model accepts and verifies the current backend selection argument.
- [x] #3 The focused files, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: these are stale test doubles for existing transcription contracts; no runtime boundary changes.

1. Preserve the exact kwargs mismatch and sanitized artifact-incompatible failure.
2. Extend only the two expected test interfaces.
3. Run focused files, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the model-directory routing assertion with the current provenance identifiers, retry provenance, model path, and timestamp defaults. Updated the fake transcribe-cpp model to accept and verify the required `backend="auto"` load argument; this prevents the test double itself from being sanitized as an incompatible artifact. The focused three-module gate passed 11 tests and final chunk 23 passed 342 tests. Ruff lint/format and diff checks passed.
<!-- SECTION:NOTES:END -->
