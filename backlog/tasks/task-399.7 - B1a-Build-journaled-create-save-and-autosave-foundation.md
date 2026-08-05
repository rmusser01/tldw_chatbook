---
id: TASK-399.7
title: B1a Build journaled create save and autosave foundation
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.5
  - TASK-399.6
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roll-up tracker for the three PR-sized B1a children that establish writable ownership and recovery pairing, implement journaled create/save publication, and integrate autosave, startup classification, recovery-only access, and controlled shutdown. This tracker does not own a separate implementation PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-399.7.1, TASK-399.7.2, and TASK-399.7.3 are complete and their combined release tests pass on the exact packaged B0 writable matrix.
- [ ] #2 Recovery pairing, installed-manifest admission, owner-only storage, exclusive mutation ownership, complete legacy-pass blocking, and fixed recovery capacity fail closed before any mutation is admitted.
- [ ] #3 Blank create and body save preserve exact supported file representation and displaced content through fresh hashes, durable intent, verified safety bytes, path-safe native publication, projection completion, and non-blocking retryable FTS.
- [ ] #4 Autosave, startup classification, recovery-only enumerate/verify/exact-export, coalesced session changes, navigation guards, and the controlled-shutdown barrier retain the only copy of every live draft.
- [ ] #5 Rename, move, delete, folder mutation, and public writable controls remain unavailable throughout B1a.
- [ ] #6 The B1 release gate remains default-off until all three B1a and all three B1b children pass together.
<!-- AC:END -->

## Child Tasks

- [TASK-399.7.1](task-399.7.1%20-%20B1a1-Pair-recovery-storage-and-acquire-writable-ownership.md) — recovery pairing, capability admission, leases, and capacity
- [TASK-399.7.2](task-399.7.2%20-%20B1a2-Implement-journaled-create-and-save-publication.md) — journaled create/save mutation core
- [TASK-399.7.3](task-399.7.3%20-%20B1a3-Integrate-autosave-recovery-classification-and-controlled-shutdown.md) — autosave, recovery classification/access, and shutdown
