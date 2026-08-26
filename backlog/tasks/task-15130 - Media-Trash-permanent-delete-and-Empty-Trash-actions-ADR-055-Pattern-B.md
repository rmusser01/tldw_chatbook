---
id: TASK-15130
title: 'Media Trash: permanent delete and Empty Trash actions (ADR-055 Pattern B)'
status: To Do
assignee: []
created_date: '2026-08-11 12:41'
labels:
  - library
  - media
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-4025 shipped the browsable, restorable Media Trash view (Library ▸ Media ▸ Trash) with per-item Restore as its only operation, so today nothing ever leaves the Trash except by restoring it — it grows without bound. The store-level seams for the destructive half already exist and are policy-gated but unwired to any UI: LocalMediaReadingService.permanently_delete_media_item (refuses non-trashed rows) and empty_media_trash, both exposed through MediaReadingScopeService. Wiring them is deliberately out of task-4025's ACs: a hard delete owes ADR-055 Pattern B (permanence stated in the confirm copy — 'cannot be undone' — naming exactly what is removed), which deserves its own copy/confirm design and live verification rather than riding along on the restore surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A trashed item can be permanently deleted from the Trash view via the existing permanently_delete_media_item seam, with a confirm that states permanence per ADR-055 Pattern B
- [ ] #2 An Empty Trash action uses the existing empty_media_trash seam with a permanence-stating confirm naming the count
- [ ] #3 Neither action leaves a receipt (a receipt whose Undo cannot exist would be a lie — ADR-055 Pattern B); list, Trash count, and rail counts update in place
<!-- AC:END -->
