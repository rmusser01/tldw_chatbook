---
id: TASK-31943
title: Library rail - Media count stays stale after a bulk-delete Undo
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR E Task 3 live finding (2026-09-05): after a bulk delete followed by Undo in the Library Media list, the rail's 'Media N' count still shows the post-delete number until some later refresh re-reads it. Reproduced identically at PR E's base, so it is pre-existing rather than E's regression: the Undo path restores the rows but never re-publishes the rail count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a bulk delete and Undo, the rail Media count matches the restored row count without leaving the screen
- [ ] #2 A regression pin asserts the painted rail count after delete then Undo, not just the controller's row state
<!-- AC:END -->
