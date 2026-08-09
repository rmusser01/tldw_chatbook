---
id: TASK-4025
title: >-
  Library media Trash view is unreachable — build a browsable, restorable
  surface
status: To Do
assignee: []
created_date: '2026-08-09 22:56'
updated_date: '2026-08-09 22:57'
labels:
  - library
  - media
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-4022 fixed the two acute defects (a deleted file could never be re-imported, and bulk delete had no receipt or undo) but deliberately scoped out a persistent, browsable Trash surface: today, an item moved to trash (mark_as_trash/is_trash=1) has no rail entry, no type: filter value, and no canvas anywhere it can be listed or restored from once its at-point-of-action Undo receipt is dismissed or the session ends. The only way back at that point is re-importing the exact same file (now honest and functional, but not available for content that isn't a re-importable file, and not discoverable for a user who doesn't remember what they deleted). This task is to design and ship that surface: a place to see everything currently in trash and restore it, using the existing DB-layer restore_from_trash/MediaDatabase.restore_from_trash and Media/local_media_reading_service.py's already-implemented restore_media_item (currently unwired to any UI).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Trash surface exists somewhere reachable from the Media canvas (rail entry, type: filter value, or dedicated canvas) listing every item with is_trash=1
- [ ] #2 Each trashed item can be restored from that surface via the existing restore_from_trash/restore_media_item seam, with the list and rail counts updating in place
- [ ] #3 The Media delete confirmation copy (bulk and single-item) is updated to point at the new surface instead of describing an Undo-only/re-import-only recovery path
- [ ] #4 Live verification of the full cycle: delete an item, dismiss or lose its Undo receipt, find and restore it from the new Trash surface
<!-- AC:END -->
