---
id: TASK-32651
title: >-
  EnhancedFileSave never got FileSave's filename-field focus
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - picker
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606. PROVEN by construction: `EnhancedFileSave` extends
`EnhancedFileDialog` -> `BaseFileDialog`, not `FileSave`, so
`_focus_initial_widget` resolves straight to `FileSystemPickerScreen` and the
dialog opens on the directory listing. task-1479's reasoning -- a keyboard
user should be able to press Enter once to confirm the seeded filename --
applies to it identically.

Out of scope for task-32606: saving is not that task's route, and the dialog
is not folder-returning, so `RETURNS_A_FOLDER` does not reach it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 EnhancedFileSave opens with its filename field focused and the seeded name selected
- [ ] #2 A negative control pins that EnhancedFileOpen still opens on its listing
<!-- AC:END -->
