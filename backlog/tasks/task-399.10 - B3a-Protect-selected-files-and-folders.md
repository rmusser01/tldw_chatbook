---
id: TASK-399.10
title: B3a Protect selected files and folders
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - recovery
  - storage
dependencies:
  - TASK-399.8
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Maintain an independent exact current replica only for selected files and folder prefixes, with explicit capacity and coverage state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Protection defaults off and resolves by file override, deepest folder-prefix rule, then default.
- [ ] #2 Selection preview reports affected files, logical bytes, estimated compressed recovery cost, physical-store caveats, and inherited/explicit state.
- [ ] #3 Coverage is claimed only after each current replica independently decompresses and verifies against current disk hash.
- [ ] #4 Every completed protected Chatbook save and successfully reconciled external edit leaves exactly one verified current replica enumerable without opening ChaChaNotes or file_notes.db.
- [ ] #5 Protection pending and Recovery copy behind prevent Chatbook writes only to the affected note.
- [ ] #6 The fixed 1 GiB live-data cap counts compressed content plus encoded manifests and the 256 MiB post-reservation free-space floor fails closed; guaranteed, current, or unresolved content is never silently evicted.
- [ ] #7 Disabling protection is confirmed and retains latest current bytes under the retention policy rather than promising immediate space recovery.
- [ ] #8 Selection, inheritance, external lag, corruption, capacity, free-space, and independent verify/export tests pass.
<!-- AC:END -->
