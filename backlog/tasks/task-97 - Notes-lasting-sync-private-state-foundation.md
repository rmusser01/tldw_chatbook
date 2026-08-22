---
id: TASK-97
title: Notes lasting-sync private state foundation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-06-11 17:05'
updated_date: '2026-08-22 15:57'
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
- [ ] #1 Existing v1 Notes import-receipt databases upgrade atomically to a shared v2 private schema, and receipt behavior remains compatible when import and sync repositories initialize in either order or concurrently through separate SQLite connections.
- [ ] #2 The private owner persists paused local Database Notes candidate roots and provisional bindings with optimistic versions, redacted typed projections, one non-disconnected binding per note, and conditional per-root uniqueness for non-null path keys.
- [ ] #3 Legacy configuration and per-note metadata can be captured idempotently as paused provisional candidates without accessing migrated candidate paths or mutating files, notes, configuration, or legacy rows; fresh pre/post source snapshots bound drift claims, duplicate-note classes choose no winner, and drift/conflicts/malformed items remain marked for rescan/review.
- [ ] #4 The exact ceilings of 64 live roots and 100,000 live bindings reject the whole request atomically, and paths, note IDs, identities, digests, and raw exception text remain confined to the backup-excluded `notes.sync_state` owner and absent from diagnostic representations.
- [ ] #5 This slice exposes no activation, watcher, reconciliation, conflict-content, resolver, journal, UI, or server-backed lasting-sync behavior, and the legacy engine remains the only active sync owner.
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
