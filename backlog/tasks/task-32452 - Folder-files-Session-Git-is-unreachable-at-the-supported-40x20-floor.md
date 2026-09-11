---
id: TASK-32452
title: >-
  Folder files: Session Git is unreachable at the supported 40x20 floor
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
  - file-notes
  - layout
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 40x20 -- the size `test_folder_files_and_session_git_use_supported_40x20_navigator`
calls supported -- the Folder files WORK pane is resolved off-screen, so
Session Git (and everything else the work pane hosts) is mounted but never
composited. Pressing it opens the route and moves focus, and the screen does
not change at all: a reader at that width gets no feedback and no way back.

Measured while repairing task-32294 (the painted-text reds), at
dev ff2dc03145:

- On entry: `#file-notes-work` resolves to `Region(x=40, y=7, width=1,
  height=12)` -- entirely past the right edge of a 40-column screen --
  and `#file-notes-session-changes` to `Region(x=40, y=28, width=1,
  height=1)`. `session_git in app.screen._compositor.visible_widgets` is
  False.
- After pressing Session Git: `#file-notes-git-back` reports
  `display == True` and takes focus, its region is `Region(0, 0, 0, 0)`,
  and the painted screen is byte-identical to before the press.
- Collapsing the navigator through `#library-file-notes-items-grip` (which
  IS composited) does give the work pane the width -- `Region(x=10, y=7,
  width=30, height=12)` -- but Session Git then lands on row 19, below the
  pane's own 12 rows, and is still not composited.

task-32294 corrected the test to assert the route rather than the pixels and
recorded this as the product half; nothing here is fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 40x20, pressing Session Git in Folder files puts the Session Git view on screen (its Back control composited), or the control is not offered at a width where it cannot be shown
- [ ] #2 The same holds for every other work-pane task Folder files offers at that width (Details, the editor), so the fix is the stage/priority rule rather than one control
- [ ] #3 `test_folder_files_and_session_git_use_supported_40x20_navigator` re-asserts paint (not just route state) once the pane can hold it, and its task-32294 comment is retired
<!-- AC:END -->
