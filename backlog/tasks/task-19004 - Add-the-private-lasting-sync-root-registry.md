---
id: TASK-19004
title: Add the private lasting-sync root registry
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:41'
updated_date: '2026-08-21 02:18'
labels:
  - notes
  - sync
  - database
dependencies:
  - TASK-19003
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing device-private notes.sync_state owner with local root, binding, representation, status, and receipt records while preserving the shipped import ledger contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing `notes.sync_state` private database migrates from the shipped import-ledger schema without losing or changing import receipt behavior.
- [x] #2 Validated root, binding, representation, cursor, journal, recovery, migration, and bounded setting records support multiple local roots and illegal state transitions fail closed.
- [x] #3 Active note scope, relative path, and stable file identity ownership are transactionally unique; candidate roots may not become active without a logical folder owner.
- [x] #4 New lasting-sync public projections and diagnostics expose opaque IDs and bounded reason codes only; paths, content, hashes, recovery bytes, and exception text stay private, while the existing import-receipt digest fields remain API-compatible.
- [x] #5 The owner remains backup- and export-excluded, is registered once in the private SQLite inventory, and all migration tests use isolated temporary databases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the historical v1 receipt schema and write RED v0/v1/current/newer/failure migration tests.\n2. Introduce NotesDeviceStateStore as the single notes.sync_state owner while preserving the TASK-19003 read-only lookup and every receipt API.\n3. Add frozen privacy-safe sync models plus root/binding/representation/cursor/journal/recovery/migration/settings records and transactional uniqueness/state constraints.\n4. Move the exact private-owner inventory entry without adding an owner, backup, export, or raw SQLite path.\n5. Run receipt/executor/private SQLite/model/store gates, perform spec/quality review, update inventory/task notes, and close only with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md\nReason: ADR-059/073 already decide private ownership, binding uniqueness, representation, journaling, recovery, migration, privacy, and backup exclusion; this task implements that inert registry without activating sync.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the schema-v2 private lasting-sync registry while preserving the literal shipped v1 receipt schema, receipt APIs, SQLite-enforced read-only lookup, and single `notes.sync_state` owner.
- Added validated root, binding, journal, recovery, migration, and allowlisted-setting records with transactional lifecycle, scope, path, identity, and operation constraints. Migration and current-schema adoption reject unexpected objects and corrupt recovery/settings data without changing source databases.
- Kept public projections path/hash/recovery-free and retained the existing receipt digests required by import execution. Updated the private-owner inventory; backup/export remain disabled and the runtime remains inert.
- Commits: `cd32f18ed`, `e392841c5`, `fe03b3fd2`. Final gate: 615 passed, 1 skipped, 1 dependency warning. Ruff, formatting, and diff checks passed. Independent review: Ready with no findings.
- ADR check: no new ADR; implementation follows ADR-059 and ADR-073.
<!-- SECTION:NOTES:END -->
