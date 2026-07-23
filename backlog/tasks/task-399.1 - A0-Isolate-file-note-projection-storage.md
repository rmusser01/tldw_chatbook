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
Establish a dedicated File Notes persistence boundary so linked-file indexing cannot alter or impair the existing Database Notes store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Roots, bindings, projections, indexing state, and file FTS are stored only in owner-protected file_notes.db.
- [ ] #2 The existing ChaChaNotes schema version, tables, triggers, constructors, backup/restore behavior, and Database-note CRUD remain unchanged.
- [ ] #3 Missing, corrupt, incompatible, or unavailable file_notes.db disables only the Files source and leaves Database Notes usable.
- [ ] #4 A pristine profile with no file_notes.db or recovery evidence opens/creates no File Notes database and starts no File Notes worker, watcher, scan, or lease; an existing detached file_notes.db may receive only one bounded read-only bootstrap query before closing.
- [ ] #5 File Notes schema creation, integrity failure, and clean unpaired rebuild with no recovery evidence pass without touching source files or ChaChaNotes.
- [ ] #6 Routine diagnostics contain neither absolute root paths nor note bodies.
<!-- AC:END -->
