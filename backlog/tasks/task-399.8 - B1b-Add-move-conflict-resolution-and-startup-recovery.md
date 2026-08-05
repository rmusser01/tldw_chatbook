---
id: TASK-399.8
title: B1b Add move conflict resolution and startup recovery
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.7
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roll-up tracker for the three PR-sized B1b children that add identity-preserving relocation, bounded three-sided conflict UX, writable lifecycle barriers, and the all-or-nothing B1 release gate. This tracker does not own a separate implementation PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-399.8.1, TASK-399.8.2, and TASK-399.8.3 are complete after the three B1a children and their combined release tests pass on the exact packaged B0 writable matrix.
- [ ] #2 Same-root rename/move preserves identity through no-replace publication and deterministic shared startup classification without guessing across changed, unsafe, colliding, or cross-device paths.
- [ ] #3 Focused-clean and dirty external changes retain exact Base, Draft, and Disk sides and provide bounded, cancellable comparison plus save-as-new, keep-editing, overwrite, and discard actions that cannot erase the only copy.
- [ ] #4 Read/write transitions, controlled lifecycle changes, writable Unlink, detached-root management, and Forget cross the complete operation/editor/lease barrier while Database Notes remains usable.
- [ ] #5 Folder mutation, recursive mutation, delete, and Git staging/commit/push controls remain unavailable.
- [ ] #6 Create, save, autosave, rename, move, read/write mode, Unlink, and Forget appear only when the all-or-nothing B1 release gate verifies all six B1 children; every other platform or capability result remains read-only.
<!-- AC:END -->

## Child Tasks

- [TASK-399.8.1](task-399.8.1%20-%20B1b1-Add-identity-preserving-rename-move-and-recovery-classification.md) — rename/move and classifier expansion
- [TASK-399.8.2](task-399.8.2%20-%20B1b2-Build-bounded-conflict-comparison-and-resolution-UX.md) — bounded three-sided conflict UX
- [TASK-399.8.3](task-399.8.3%20-%20B1b3-Integrate-writable-lifecycle-and-open-the-B1-release-gate.md) — lifecycle barriers and release gate
