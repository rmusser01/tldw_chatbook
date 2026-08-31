---
id: TASK-26069
title: Add durable bounded Tool Pack receipts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 19:46'
updated_date: '2026-08-31 21:51'
labels:
  - tool-packs
  - receipts
  - security
  - durability
dependencies:
  - TASK-26068
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist privacy-safe, non-authoritative import and tombstone receipts with capacity reservation, private crash-safe writes, authenticated reads, bounded reconciliation, and compaction for Tool Pack lifecycle recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capacity is reserved under one process lock before commit, enforces per-receipt/total limits, and releases idempotently without truncating existing receipts.
- [x] #2 Receipt directories/files and private sibling temporaries use restrictive modes, durable atomic replacement, authenticated random names, cleanup on failure, and exact-digest reads.
- [x] #3 Strict import and compact-tombstone receipt variants contain only approved safe metadata and reject unknown, sensitive, malformed, or mismatched fields.
- [x] #4 Reconciliation removes only authenticated old unreferenced regular receipts after the 24-hour grace period while preserving linked, live, recent, corrupt-referenced, unknown, and symlink entries.
- [x] #5 Tombstone compaction preserves required lineage while reducing receipt size, and focused receipt/static tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for capacity reservation, privacy modes, authenticated reads/names, crash residue, and idempotent release.
2. Implement strict privacy-safe receipt dataclasses, reservation accounting, and private durable atomic commit/read paths.
3. Add red tests for 24-hour reconciliation, protected entry classes, corrupt referenced receipts, and tombstone compaction.
4. Implement bounded reconciliation and compact tombstone receipts without granting authority.
5. Run focused receipt tests, scoped static checks, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Accepted ADR-107 already defines non-authoritative receipt ownership, privacy fields, durability, capacity, reconciliation, and compaction boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a standard-library-only receipt store with strict immutable import
and compact-tombstone unions, shared per-root capacity reservations, private
atomic writes, digest-authenticated reads, 24-hour orphan reconciliation, and
lineage-preserving compaction. Independent review drove two durability fixes:
the strictest root cap is now shared across store instances, and receipt-root
plus parent-directory fsync is mandatory, with post-replace failures reported
as uncertain. Focused verification: 41 receipt tests passed; scoped Ruff and
`git diff --check` passed. ADR-107 remains the governing decision; no new ADR,
dependency, schema migration, or Windows-specific implementation was added.
<!-- SECTION:NOTES:END -->
