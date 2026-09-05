---
id: TASK-31778
title: Register the physical trace maintenance SQLite owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:34'
updated_date: '2026-09-05 23:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The physical trace compactor opens its maintenance connection using the ChaChaNotes module owner ID, so the module-owned SQLite boundary guard fails. Register its actual owner without bypassing private-path checks or widening maintenance authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trace maintenance uses a literal owner registered to its actual module and only allows private file targets
- [x] #2 Existing maintenance, integrity, cancellation and private SQLite behavior remain passing
- [x] #3 The classified inventory includes the owner and unchanged guards reject module-owner mismatches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced literal-owner violation in the complete SQLite inventory baseline and inspect all maintenance callers.
2. Add one file-only maintenance policy, switch only the owner literal, and add the corresponding explicit inventory row. No new opener, backup permission, bypass or path behavior.
3. Verify the complete private SQLite owner matrix, compaction/admission files and inventory file. Record the two independent raw-recovery/backup-wrapper failures separately if they persist. Run scoped static checks and independent review.
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Register an existing connection through the established module-owned seam; same-file compaction and private storage policies are unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Registered chat.trace_maintenance for PhysicalTraceCompactor through the existing private SQLite seam and documented C55. Only the owner literal changed at the connection site: private-file only, must_exist, connection options and PRAGMAs preserved; no backup authority granted. Baseline inventory 21 passed / 3 failed; the actual module-owner mismatch now passes. Final five complete SQLite/private-owner/compaction/admission files: 383 passed / 2 unrelated inventory failures / 2 Windows-only skips in 57.10s, XML /private/tmp/tldw-trace-owner-final.xml. The open failures are LegacyCollectionsRecovery._read_transaction raw connection and DB/base_db super().backup wrapper; neither was excluded or claimed resolved. Scoped Ruff and edited-range formatting pass; unrelated pre-existing whole-file compactor formatting drift preserved. Self-review and independent bounded review clear; new registry owner exercises real SQLite via the existing owner/target matrix. ADR required: no; existing ADR-029 private-storage and ADR-097 same-file maintenance contracts unchanged. Files: DB/private_sqlite.py, Chat/console_trace_maintenance.py, Tests/DB/test_private_sqlite_inventory.py, sqlite-private-owner-inventory.md. No performance, privacy, security or license boundary widened.
<!-- SECTION:NOTES:END -->
