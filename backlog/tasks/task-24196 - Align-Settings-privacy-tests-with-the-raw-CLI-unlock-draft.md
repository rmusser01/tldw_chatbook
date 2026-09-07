---
id: TASK-24196
title: Align Settings privacy tests with the raw CLI unlock draft
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 07:01'
updated_date: '2026-08-29 12:24'
labels:
  - settings
  - test-regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair legacy Settings hub tests that still classify Privacy & Security as read-only after the raw CLI host-access unlock became its bounded guided draft.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Privacy & Security tests require Save and Revert only for the raw CLI unlock while preserving read-only privacy posture and the Check Privacy action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test maintenance for an already-implemented Settings boundary; no production policy changes. Update the stale mounted and footer contracts, run the exact tests and the full Settings hub file, then document evidence and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated the mounted Privacy & Security, unsupported-mutation, redaction, privacy-check, and footer contracts to distinguish the bounded raw CLI unlock draft from the otherwise read-only privacy and credential posture.
- Save and Revert remain disabled until that one raw CLI field changes; Check Privacy remains visible and keyboard reachable. Read-only Overview and Diagnostics categories still omit draft actions.
- Verification: the complete Settings configuration hub passed **388/388**; the canonical joined Notes/Settings/Library matrix passed **1,879/1,879**. Scoped Ruff and compileall passed. The touched Settings files were already Ruff-format-red at HEAD, so their inherited whole-file formatting was preserved rather than creating an unrelated rewrite.
- ADR required: no. ADR path: N/A. This aligns tests to an existing Settings boundary and changes no privacy, credential, storage, or service policy.
<!-- SECTION:NOTES:END -->
