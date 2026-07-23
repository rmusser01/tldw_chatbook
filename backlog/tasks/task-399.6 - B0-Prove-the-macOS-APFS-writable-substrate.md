---
id: TASK-399.6
title: B0 Prove the macOS APFS writable substrate
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
labels:
  - notes
  - filesystem
  - macos
dependencies:
  - TASK-399.2
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish a release-blocking executable go/no-go result for safe native file mutation before any writable action is implemented or exposed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A packaged-app probe verifies the actual root volume and reports every required primitive separately.
- [ ] #2 The probe demonstrates pinned no-follow traversal, atomic no-replace/displaced-target exchange, file/directory durability barriers, and required full-fsync behavior on the supported macOS/APFS matrix.
- [ ] #3 Nested mounts, cross-device paths, network/cloud volumes, symlink substitution, hardlinks, unsafe names, and unsupported primitives fail closed with specific reasons.
- [ ] #4 Files carrying ACLs, extended attributes, flags, unusual ownership, or other unround-trippable metadata remain read-only.
- [ ] #5 Packaged Linux and Windows probes report the writable primitives unsupported and expose no writable action.
- [ ] #6 backlog/docs/file-backed-notes-apfs-capability-matrix.md records runner hardware, supported macOS/APFS versions, probe results, the named power-cut/reboot and crash/fsync methods plus observed durability results, build, and explicit go/no-go.
- [ ] #7 No create, save, rename, move, restore, or delete control is implemented or exposed by this task.
<!-- AC:END -->
