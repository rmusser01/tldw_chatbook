---
id: TASK-16267
title: Reconcile repository credential ignore coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:17'
updated_date: '2026-08-14 20:19'
labels:
  - testing
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep repository credential-ignore tests aligned with the intentional any-depth API-key scratch-file guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root credential scratch filenames remain ignored.
- [x] #2 Nested API-key scratch filenames remain ignored.
- [x] #3 Unrelated nested files remain trackable and fatal Git errors fail closed.
- [x] #4 The focused checkpoint and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the stale nested-path assertion as RED and trace the governing ignore rule.
2. Update only the repository ignore contract to match the intentional any-depth guard while retaining a non-credential negative control.
3. Run the focused module, original 25-file checkpoint, and static checks.

ADR required: no
ADR path: N/A
Reason: test-only reconciliation with an existing repository security rule; no architecture or policy boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled the stale root-only test with the intentional `*-api-key.txt` any-depth ignore rule introduced by TASK-1355; `.gitignore` itself was already correct and unchanged.
- Retained an unrelated nested filename as a negative control and the fatal `git check-ignore` fail-closed characterization.
- RED: the prior nested credential assertion failed because both files were ignored. GREEN: focused module 3 passed; original 25-file Utils/Video checkpoint 595 passed and 2 Windows-only tests skipped; Ruff lint/format and diff hygiene passed.
<!-- SECTION:NOTES:END -->
