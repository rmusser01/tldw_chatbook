---
id: TASK-35780
title: Character scope fingerprint can match a different database after id() reuse
status: To Do
assignee: []
created_date: '2026-10-20 06:00'
labels:
  - console
  - database
  - correctness
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Character display decides whether its saved-chat list is current by comparing a scope fingerprint. The fingerprint names the database by id(database), and CPython reuses an object's id once the object is garbage-collected. After a profile swap or a backup restore, a new database object with the same local authority id and an equal search revision, allocated at the address of the old one, compares as unchanged, so the display can keep a list read from the previous database object. This predates TASK-33620.15.6 (dev's probe compared the same tuple) and was found while reviewing it. The fingerprint needs a database identity that is never reused within a process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A swap to a different database object with the same authority id and revision is detected as a scope change even when the new object reuses the old object's id()
- [ ] #2 A deterministic test reproduces the id() reuse case and fails on the current dev build
- [ ] #3 Unchanged checks against the same database object still compare as unchanged, with no extra SQLite handle or helper start per check
<!-- AC:END -->
