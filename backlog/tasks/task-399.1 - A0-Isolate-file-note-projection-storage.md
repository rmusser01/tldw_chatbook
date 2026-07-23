---
id: TASK-399.1
title: A0 Isolate file-note projection storage
status: To Do
assignee: []
created_date: '2026-07-23 14:22'
labels:
  - notes
  - library
  - storage
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish deterministic, owner-protected File Notes storage and bootstrap boundaries so linked-file state cannot alter, delay, or impair the existing Database Notes store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canonical configured main-database path is domain-hashed into a stable, non-secret storage-instance ID, and all File Notes state is confined to its deterministic owner-only `<user-data>/file_notes/<storage-instance-id>/` directory.
- [ ] #2 Roots, bindings, projections, indexing state, and file FTS are stored only in that instance's owner-protected file_notes.db; its database, WAL/SHM sidecars, diagnostics, and bootstrap-marker locations cannot be shared across storage instances.
- [ ] #3 File Notes coordinator and future mutation leases use an owner-only fixed application runtime namespace independent of configurable user-data, main-database, repository, cache, and log paths.
- [ ] #4 The existing ChaChaNotes schema version, tables, triggers, constructors, backup/restore behavior, and Database-note CRUD remain unchanged.
- [ ] #5 Missing, corrupt, incompatible, unavailable, or mismatched File Notes storage produces only a source-scoped Files diagnostic and leaves Database Notes startup, first paint, and use available.
- [ ] #6 A pristine profile with no file_notes.db or recovery evidence opens or creates no File Notes database and starts no File Notes connection, worker, watcher, scan, or lease.
- [ ] #7 Only after the Database canvas reaches first paint, an existing file_notes.db may receive one off-critical-path, read-only detached-evidence query with zero lock wait and a 100 ms hard result budget, then closes when no root is active; timeout, lock, corruption, or cancellation cannot delay or replace Database Notes.
- [ ] #8 Existing corrupt, incompatible, partial, or orphan-indicating databases, sidecars, markers, root/binding rows, and recovery evidence are preserved for diagnosis; no integrity error or apparent unpaired state may trigger a silent rename, replacement, or clean rebuild.
- [ ] #9 Routine File Notes diagnostics contain neither absolute root paths nor note bodies.
<!-- AC:END -->
