---
id: TASK-399.2
title: A1 Preview and link one read-only notes root
status: To Do
assignee: []
created_date: '2026-07-23 14:22'
labels:
  - notes
  - library
  - filesystem
dependencies:
  - TASK-399.1
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users safely evaluate and link one existing local folder without changing its files or interfering with legacy Database-note sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Packaged macOS, Linux, and Windows builds preview one root and report supported, ignored, read-only, inaccessible, collision, and special-file counts.
- [ ] #2 Preview is memory-only: it writes no projection row, creates no watcher/recovery store, acquires no lease, and leaves every source byte unchanged.
- [ ] #3 Confirmation discloses that readable bodies are cached/indexed as local plaintext and estimates that storage separately from recovery storage.
- [ ] #4 Link read-only publishes a resumable root only after canonical-path, filesystem-identity, unsafe-name, and normalized-collision checks pass.
- [ ] #5 A configured legacy folder-sync overlap is rejected with a concrete remedy.
- [ ] #6 Read-only activation uses only coordinator election; it does not acquire the mutation lease, drain/pause legacy sync, or expose an editable-but-unsavable buffer.
- [ ] #7 Unlink stops election and monitoring without modifying the linked folder.
<!-- AC:END -->
