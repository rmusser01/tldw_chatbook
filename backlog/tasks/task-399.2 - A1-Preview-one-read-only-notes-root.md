---
id: TASK-399.2
title: A1 Preview one read-only notes root
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
Let users safely evaluate one existing local folder without activating it, changing its files, or interfering with legacy Database-note sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Packaged macOS, Linux, and Windows builds preview one root and report supported, ignored, read-only, inaccessible, collision, and special-file counts.
- [ ] #2 Preview is memory-only: it persists no root, projection, or recovery row; creates no database, watcher, reconciliation/index worker, or recovery store; acquires neither coordinator election nor mutation lease; and leaves every source byte unchanged.
- [ ] #3 Preview discloses that readable bodies will be cached/indexed as local plaintext after activation and estimates that projection/index storage separately from future recovery storage.
- [ ] #4 Candidate validation rejects canonical-path or filesystem-identity failures, unsafe names, normalized collisions, and any candidate that is an ancestor or descendant of the File Notes instance directory, user-data directory, Chatbook databases/sidecars, fixed runtime namespace, configuration, cache, or log paths.
- [ ] #5 A candidate that overlaps a configured legacy folder-sync root in either ancestor/descendant direction is rejected with a concrete remedy.
- [ ] #6 This task exposes no public Link or Unlink behavior and starts no activation scan, event capture, watcher, worker, or durable root publication; confirmed read-only activation belongs to A2 and remains behind the default-off release gate.
<!-- AC:END -->
