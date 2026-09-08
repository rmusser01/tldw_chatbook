---
id: TASK-31943
title: Library rail - Media count stays stale after a bulk-delete Undo
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:34'
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
- [x] #1 After a bulk delete and Undo, the rail Media count matches the restored row count without leaving the screen
- [x] #2 A regression pin asserts the painted rail count after delete then Undo, not just the controller's row state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted pin: delete 2 of 3, Undo → the rail reads Media (3) again without leaving the screen. 2. Trace why the Undo path never re-read the count.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the bulk-delete prune compared canvas row ids (`local:media:<n>`) against source records keyed by the backing id, so nothing was ever pruned while the count dropped; Undo's de-dup guard then read rows that had never left and added 0. Fixed at the one prune site by matching both spellings; Undo untouched; the count is right after the delete and after the Undo. Collateral repair: the same prune feeds the hub's recent items and the study sample, which had kept trashed media until a refresh. Rider: the selection reconcile in the same delete path has the identical id-shape mismatch (it unchecks failed rows on a partial failure; task-3020 AC3).
<!-- SECTION:NOTES:END -->
