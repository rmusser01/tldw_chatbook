---
id: TASK-97
title: Notes lasting-sync private state foundation
status: Done
assignee:
  - '@codex'
created_date: '2026-06-11 17:05'
updated_date: '2026-08-22 20:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the device-private storage foundation required by ADR-059 and ADR-060
before replacing the legacy Database Notes sync flow. Upgrade the existing
`notes.sync_state` import-receipt owner so it can also persist paused local sync
roots, provisional bindings, and idempotent legacy-migration receipts without
granting filesystem or note mutation authority.

The original task proposed an interrupting file/app/skip conflict modal. That
contract is superseded by the accepted lasting-sync design: later slices pause
affected bindings as Needs attention and offer Keep file, Keep note, or Keep
both. This task is the first atomic prerequisite and intentionally contains no
conflict UI or active sync behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing v1 Notes import-receipt databases upgrade atomically to a shared v2 private schema, and receipt behavior remains compatible when import and sync repositories initialize in either order or concurrently through separate SQLite connections.
- [x] #2 The private owner persists paused local Database Notes candidate roots and provisional bindings with optimistic versions, redacted typed projections, one non-disconnected binding per note, and conditional per-root uniqueness for non-null path keys.
- [x] #3 Legacy configuration and per-note metadata can be captured idempotently as paused provisional candidates without accessing migrated candidate paths or mutating files, notes, configuration, or legacy rows; fresh pre/post source snapshots bound drift claims, duplicate-note classes choose no winner, and drift/conflicts/malformed items remain marked for rescan/review.
- [x] #4 The exact ceilings of 64 live roots and 100,000 live bindings reject the whole request atomically, and paths, note IDs, identities, digests, and raw exception text remain confined to the backup-excluded `notes.sync_state` owner and absent from diagnostic representations.
- [x] #5 This slice exposes no activation, watcher, reconciliation, conflict-content, resolver, journal, UI, or server-backed lasting-sync behavior, and the legacy engine remains the only active sync owner.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Centralize `notes.sync_state` schema ownership and preserve every existing
   import-receipt behavior while upgrading to the exact v2 schema.
2. Prove fresh/upgrade parity, malformed-schema rejection, initialization
   ordering, real two-connection convergence, and the read-only v2 fast path.
3. Add the narrow paused-root and provisional-binding repository with redacted
   projections, optimistic versions, fixed capacities, and closed ownership
   invariants.
4. Add the read-only legacy source seam, exact canonical digests/locators, and
   operand-scoped proof that candidate paths are never accessed.
5. Persist deterministic migration generations with duplicate preflight,
   global capacity rejection, idempotent replay, fresh A/B drift detection, and
   immutable compare-and-set finalization.
6. Run the bounded related compatibility/static matrix, complete independent
   cumulative review, update task evidence, and close out through Backlog CLI.

Executable plan:
`Docs/superpowers/plans/2026-08-22-task-97-notes-lasting-sync-state-foundation.md`.

ADR required: no. This task directly implements
`backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
and
`backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the device-private TASK-97 foundation governed by ADR-059 and
ADR-060. `notes_sync_state_schema.py` is now the sole `notes.sync_state`
connection owner: it preserves the receipt v1 DDL, takes the writer lock only
for v0/v1 initialization, re-reads the version under that lock, validates the
exact supported census, and writes v2 last. Existing receipt operations now use
that coordinator without changing their public behavior.

Added `notes_sync_state.py` for paused roots, provisional bindings, redacted
typed projections, optimistic versions, atomic root/child disconnect, live-root
and live-binding ceilings, and the one-live-binding-per-note/conditional-path
uniqueness invariants. Added `notes_sync_legacy_migration.py` plus one bounded
read-only `ChaChaNotes_DB` source query. Migration uses deterministic preflight,
snapshot A, one committed candidate generation, fresh snapshot B, and immutable
CAS finalization; replay is idempotent, drift and duplicate-note classes choose
no winner, and candidate paths are never accessed. No activation, watcher,
reconciliation, conflict payload, resolver, journal, UI, server, backup/export,
or legacy-engine authority was added.

Focused implementation and governance coverage lives in
`Tests/Notes/test_notes_sync_state_schema.py`,
`Tests/Notes/test_notes_sync_state.py`,
`Tests/Notes/test_notes_sync_legacy_migration.py`, the existing import receipt
and executor modules, and both related private-owner modules. The settled
bounded matrix passed **762 tests**, with **1 Windows-only skip** and **1
RequestsDependencyWarning** from the installed Requests dependency stack, in
**77.32 seconds** (77.84 seconds wall time). Scoped Ruff, Ruff format, MyPy,
compileall, and `git diff --check` passed. Independent cumulative review at
`c05c8c6fc` found no remaining P0-P2 issues.

Review-driven corrections tightened transaction rollback and version-overflow
handling, exact schema census/claimed-v1 validation, SQLite analysis/internal
name handling, source capture bounds, privacy ratchets, and separation of the
independent schema oracle from the production owner. These corrections did not
expand the accepted scope. The governing records remain
`backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
and
`backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`;
no new ADR was required. No new incident generalized beyond the existing
testing-evidence, live-verification, or backlog-hygiene lessons, so no lessons
document was changed.
<!-- SECTION:NOTES:END -->
