---
id: TASK-16271
title: Pass explicit speech precision through wizard allowlist test
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:48'
updated_date: '2026-08-14 20:50'
labels:
  - testing
  - speech
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the wizard section-allowlist test aligned with the exact speech transcription commit contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The allowlist test supplies and validates the required transcription precision.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the missing-precision TypeError and confirm the builder contract.
2. Add the canonical INT8 value to the allowlist fixture without changing production defaults.
3. Run focused, wizard, checkpoint, and static verification.

ADR required: no
ADR path: N/A
Reason: test-only reconciliation with the existing explicit speech configuration contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the canonical `int8` precision to the speech transcription commit fixture, matching the builder's explicit provider/model/language/precision contract without adding a production default.
- Preserved RED evidence: the isolated test and original checkpoint raised the same missing keyword-only `precision` TypeError. Removing the argument reproduces it.
- Verified both first-run wizard files (105 passed) and the original 25-file checkpoint (441 passed). `git diff --check` passes. Ruff reports the same three pre-existing E402 diagnostics on `HEAD` and the changed file; Ruff format is likewise already red on `HEAD`.
- ADR required: no. This is test-only reconciliation with the existing speech persistence contract.
<!-- SECTION:NOTES:END -->
