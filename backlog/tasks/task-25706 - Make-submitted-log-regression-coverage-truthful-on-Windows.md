---
id: TASK-25706
title: Make submitted-log regression coverage truthful on Windows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 17:52'
updated_date: '2026-08-30 17:52'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct Windows-only failures exposed by the native submitted-log validation so the matrix distinguishes portable behavior from intentionally POSIX-only security contracts and reports actionable evidence without weakening ADR-029.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The submitted-log regression matrix completes on native Windows without pytest internal errors.
- [ ] #2 POSIX-only profile-migration security tests are capability-gated and Windows retains a tested fail-closed contract.
- [ ] #3 Windows-illegal filename fixtures do not mask SQLite URI coverage.
- [ ] #4 The optional-datasets notice test ignores unrelated expected platform diagnostics.
- [ ] #5 The compact selection-menu case is independently diagnosed and either fixed or proven stable on Windows.
- [ ] #6 Durable CI coverage exercises the corrected Windows contract.
- [ ] #7 Generated diagnostic inventories have the same deterministic path order on Windows and POSIX hosts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the native Windows run as RED evidence and add focused platform-contract tests for timeout selection, filenames, optional-datasets logging, and profile-migration fail-closed behavior.
2. Apply narrow test capability gates without changing ADR-029 or weakening production privacy checks.
3. Run the compact selection-menu case independently on Windows with actionable geometry diagnostics and fix only a reproduced product defect.
4. Make diagnostic inventory ordering platform-independent and keep the floating menu contained after late layout measurement.
5. Add durable native-Windows submitted-log coverage with uploaded structured results.
6. Run targeted local tests, the native Windows matrix, static checks, and self-review; document results and close the task.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already defines Windows privacy as unverified pending separately approved native ACL work; this task corrects test and CI portability while preserving that runtime boundary.
<!-- SECTION:PLAN:END -->
