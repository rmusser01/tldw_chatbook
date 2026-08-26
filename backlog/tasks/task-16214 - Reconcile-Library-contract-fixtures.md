---
id: TASK-16214
title: Reconcile Library contract fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:35'
updated_date: '2026-08-14 00:46'
labels:
  - test-health
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile three Library unit fixtures with the current ingest capability, export reconciliation, and host-path contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The generic ingest capability assertion covers the complete current field schema.
- [x] #2 The memory-backed export double supplies and verifies current request ownership context.
- [x] #3 The path-normalization test exercises host-valid dot segments while independently characterizing case policy.
- [x] #4 The focused files, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: these are stale test fixtures for existing Library contracts; no runtime or architectural decision changes.

1. Preserve the three exact chunk failures and compare each test to current production ownership.
2. Update only the obsolete expected tuple, export double, and host-path input.
3. Run focused files, mutations/characterizations, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the generic capability tuple for the existing ten-field schema, brought the memory-backed export double forward to the generation/route/request reconciliation contract, and corrected the path fixture to use host-valid separators while still mocking case policy. No production behavior changed. The four focused Library files passed 352 tests with one Windows-only skip; final chunk 21 passed 962 tests with one Windows-only skip. Ruff lint and formatting passed on the final touched files.
<!-- SECTION:NOTES:END -->
