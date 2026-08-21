---
id: TASK-19008
title: Migrate legacy Notes sync into paused candidates
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:50'
updated_date: '2026-08-21 03:59'
labels:
  - notes
  - sync
  - migration
dependencies:
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translate legacy root and per-note sync evidence into reviewable paused lasting-root candidates without reading absence as deletion or mutating notes files or legacy history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Distinct safe legacy roots become separate paused candidates with recognizable bindings, bounded reports, and a source fingerprint.
- [x] #2 Config-only, row-only, missing, overlapping, duplicate, out-of-root, unsafe, and invalid evidence is represented honestly without inferring deletion.
- [x] #3 Migration writes only the private device store and never creates folders, managed memberships, notes, files, receipts, watchers, or active roots.
- [x] #4 Legacy conflict winners and auto-sync settings are historical evidence only and never become lasting-root policy.
- [x] #5 Repeated migration is idempotent, preserves legacy history read-only, and every candidate requires a complete current dry-run plus explicit activation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED legacy-evidence matrices for multiple/config-only/row-only/missing/overlap/duplicate/unsafe/crash/idempotent cases.
2. Implement a read-only legacy snapshot and pure migration plan that treats conflict winners and auto-sync as historical evidence only.
3. Persist paused candidates and bounded reports in one private transaction, with no note/file/folder/watcher/receipt mutation and no deletion inference.
4. Prove every candidate requires a fresh TASK-19005 check and explicit later activation, then run focused gates, static checks, independent review, and task hygiene.

ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-059/073 already define legacy evidence as paused candidate state, prohibit policy inheritance or deletion inference, and require a fresh dry-run plus explicit activation; this task implements that migration only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a read-only legacy snapshot and deterministic pure migration plan covering config, per-note metadata, and sync-session evidence. Conflict winners, directions, and auto-sync values affect only the private source fingerprint and never become new policy.
- Safe distinct filesystem roots become paused roots with candidate bindings. Missing/unsafe/private/overlapping/duplicate/out-of-root evidence remains bounded report-only state; no absence becomes deletion and no missing file receives fabricated identity.
- Persisted the fingerprint, paused candidates, and bounded machine summary in one private transaction. Exact deterministic reports are returned on first/idempotent runs; crash injection rolls back all private writes. Activation recomputes and binds the complete fresh TASK-19005 plan and requires explicit approval; read-only Folder-to-Notes remains supported.
- Hardened canonical/filesystem identity handling for lexical/case aliases, ancestor symlinks, application-private roots, file presence/freshness fingerprints, and exact 200/201 report bounds. Removed unused raw evidence fields after review.
- Commits: `529d49526`, `a99b11d4d`, `14018768b`, `91bc2d480`, `e772e9ee1`. Final migration/store gate: 67 passed, 1 dependency warning. Ruff, formatting, and diff checks passed. Final review: Ready with no remaining findings.
- ADR check: no new ADR; implementation follows ADR-059 and ADR-073.
<!-- SECTION:NOTES:END -->
