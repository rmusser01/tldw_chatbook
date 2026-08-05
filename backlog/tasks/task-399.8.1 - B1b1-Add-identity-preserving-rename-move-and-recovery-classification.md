---
id: TASK-399.8.1
title: B1b1 Add identity-preserving rename move and recovery classification
status: To Do
assignee: []
created_date: '2026-07-23 15:36'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.7.3
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add journaled same-root rename and move with identity preservation and extend the shared startup classifier to interrupted relocation states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rename and move preserve note UUID only after re-resolving both paths beneath the pinned writable root and verifying an existing destination directory plus the B0-proven no-replace primitive.
- [ ] #2 Existing, normalized-colliding, cross-device, unsafe, symlinked, mounted, unsupported, or changed destinations are rejected without changing either source or destination.
- [ ] #3 Each relocation records durable intent, hashes and metadata fingerprints for both paths, preserves any bytes or metadata that could be displaced, publishes disk and projection state in the specified order, and closes the journal only after verification.
- [ ] #4 The shared startup classifier handles interrupted create, save, rename, and move from observed paths, hashes, fingerprints, journal state, and exact-owned artifacts; it never blindly replays a filesystem operation.
- [ ] #5 Watcher echoes, Git rename storms, duplicate identity candidates, offline transitions, and crashes at every relocation boundary converge to one unambiguous canonical path or durable Attention without losing a draft or disk side.
- [ ] #6 Folder creation, folder rename or move, cross-root movement, recursive mutation, delete, and public writable controls remain unavailable.
<!-- AC:END -->
