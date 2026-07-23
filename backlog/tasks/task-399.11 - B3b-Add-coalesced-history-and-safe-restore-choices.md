---
id: TASK-399.11
title: B3b Add coalesced history and safe restore choices
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - recovery
  - history
dependencies:
  - TASK-399.9
  - TASK-399.10
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide minimal per-note recovery history without turning every autosave into a revision or expanding into general backup management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A protected editing session seals at most one distinct checkpoint at approved boundaries; ordinary autosaves update the current replica without checkpoint spam.
- [ ] #2 History lists exact path, revision kind, time, expiry, verification state, and logical size for one note without opening ChaChaNotes or file_notes.db.
- [ ] #3 Users can verify and exact-export a revision before choosing restore.
- [ ] #4 Restore to an absent original or alternate path uses no-replace; overwrite requires separate hash-bound confirmation, displaced-target preservation, and 30-day retention of overwritten bytes.
- [ ] #5 Original/alternate-path tombstone restore reuses its UUID only when no live binding owns it; Restore as copy always creates a new UUID.
- [ ] #6 Retention enforces at most 50 checkpoints per note and 30 days without collecting current, guaranteed, referenced, pending, or unresolved content.
- [ ] #7 Checkpoint, expiry, reference-safe collection, occupied destination, alternate path, overwrite race, security restoration, and crash tests pass.
- [ ] #8 Recovery-only in-place mutation, general purge, guarantee waiver, configurable quota, and paired-store backup/restore remain outside this task.
- [ ] #9 Forget preserves guaranteed deletion payloads through expiry, removes protection pins, makes ordinary current replicas/checkpoints pruning-eligible, and reports retained logical bytes before confirmation.
<!-- AC:END -->
