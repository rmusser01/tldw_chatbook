---
id: TASK-31780
title: Qualify the quiescent SQLite backup wrapper in the inventory
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:53'
updated_date: '2026-09-06 00:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The direct backup census counts the existing quiescence wrapper as a new destination-owning operation. Track exact qualified delegation without hiding new direct backup calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inventory accepts only the existing seam and exact qualified super backup delegation with multiplicity enforced
- [x] #2 Negative controls reject extra, moved or receiver-changed backup calls
- [x] #3 Real backups block quiescence while active and release their reservation on success or failure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced backup inventory failure and inspect the wrapper plus all backup callers. 2. Add real backup/quiescence tests and negative scanner controls; use existing qualified AST visitor for an exact call-site census, without changing production backup logic. 3. Run full inventory, quiescence, core-owner and compaction files, static checks and independent review. ADR required: no. ADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: test-only reconciliation of existing backup delegation and maintenance exclusion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced module-only backup count with the existing qualified AST visitor and an exact Counter of module, symbol, receiver and multiplicity for the seam and super() wrapper. Five negative controls reject added/duplicate/moved/receiver-changed/other-module calls; no blanket allowlist or production backup change. Real SQLite backup tests prove exclusion while callbacks execute and reservation release after both success and callback cancellation. Process-local mutations bypassing reservation or omitting release each failed both variants as intended. Final seven complete files:434 passed,2 Windows-only skips,2 dependency warnings in88.89s, XML:/private/tmp/tldw-sqlite-recovery-backup-reviewed.xml. Whole changed-file Ruff and formatting plus diff-check pass. Independent scoped review clear. Inventory documents delegation, not an additional destination or backup authority. ADR required:no, existing ADR-029 and ADR-097 contracts unchanged.

Third rebase onto dev2b4973971e preserved this repair byte-for-byte. Fresh eight-file SQLite/recovery/compaction/regeneration qualification:440 passed,2 Windows-only skips in104.16s, XML:/private/tmp/tldw-third-rebase-sqlite-regeneration.xml. Scoped Ruff/format and diff checks pass; broader review remains open.
<!-- SECTION:NOTES:END -->
