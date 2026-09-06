---
id: TASK-31779
title: Route legacy Collections recovery through the read-only SQLite owner
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:52'
updated_date: '2026-09-06 00:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The SQLite inventory detects a raw recovery connection. Keep legacy inspection non-mutating and schema-independent while enforcing the existing registered read-only boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recovery uses its module-owned read-only connection and preserves existing source modes and schema
- [x] #2 Unsafe or disappearing source paths fail with a stable content-free error and connections close on failure
- [x] #3 Complete recovery and SQLite boundary files pass their relevant checks without raw-site exemptions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced raw-connection failure and inspect the recovery and existing read-only seam contracts. 2. Add behavioral guards for read-only/source-mode and rejected source replacement, then register one read-only owner and route the path-only branch through it. 3. Run complete recovery, private SQLite and inventory files, lint and independent review. ADR required: no. ADR path: backlog/decisions/029-local-private-data-boundary.md. Reason: adopt the established read-only source-preserving seam, not a new authority or migration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Routed path-only recovery through library.legacy_recovery, a read-only/source-mode-preserving owner with no backup authority, and added C56 without raw-site exclusions. Setup/rollback cleanup now always closes the handle. Initial regressions reproduced symlink replacement, unsafe parent and setup-leak failures (3 failed / 2 passed); source bytes/modes and enforced read-only access remain checked. Independent review caught pre-seam Path.resolve erasing initial aliases; added leaf/parent entry tests, watched both fail, and retained the lexical absolute path. Disappearance fixture now injects at the retained is_file boundary. Final seven complete recovery/SQLite/quiescence/compaction files:434 passed,2 Windows-only skips,2 dependency warnings in88.89s. XML:/private/tmp/tldw-sqlite-recovery-backup-reviewed.xml. Whole changed-file Ruff check/format and diff-check pass; independent re-review clear. Lessons entry records the constructor canonicalization incident. ADR required:no, existing ADR-029 read-only boundary adopted; no new schema, authority or backup policy.
<!-- SECTION:NOTES:END -->
