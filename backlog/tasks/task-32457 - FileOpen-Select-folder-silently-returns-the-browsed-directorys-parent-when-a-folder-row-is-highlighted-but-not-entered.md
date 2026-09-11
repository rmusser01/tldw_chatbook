---
id: TASK-32457
title: >-
  FileOpen Select folder silently returns the browsed directory's parent when a
  folder row is highlighted but not entered
status: To Do
assignee: []
created_date: '2026-09-11 17:14'
labels:
  - library
  - notes
  - picker
  - ux
  - critique-notes-2026-09
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Import once and Keep a folder synced both open the vendored `FileOpen(offer_select_folder=True)`. Its `Select folder` action resolves through `_resolve_select_folder_target` (`Third_Party/textual_fspicker/base_dialog.py:550-589`), which reads the input-bar field and otherwise falls back to `DirectoryNavigation.location` -- the directory being LISTED. Highlighting a folder row changes neither.

So a user who moves the highlight onto `vault` and presses **Select folder** imports `vault`'s PARENT, with no feedback that anything else happened: the confirmation line simply names a folder they did not choose. Reproduced live twice in the task-4 fix round; the only thing on screen that says so is the hint line, `Enter Open · Select folder to use this folder`.

This is the dialog's design, not a regression, and it is outside every AC in the Library ▸ Notes wave -- `fix/library-notes-w3-pickers-git` changes the path field and where a picker opens, neither of which touches what Select folder resolves to. Filed so the behaviour is decided rather than inherited.

Two options worth weighing: (a) resolve the highlighted directory row when there is one, so highlight and button agree; or (b) leave the resolution alone and remove the ambiguity in the surface -- mirror the highlighted row into the field, or name the button `Select this folder` so it reads as "the one I am in".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Highlighting a folder row and pressing Select folder either uses that folder, or the surface says which folder will be used before the press
- [ ] #2 The chosen folder is confirmed by name on the surface that follows, so a wrong pick is visible without reopening the picker
- [ ] #3 Covered by a test over the real dialog's folder-resolution seam
<!-- AC:END -->
