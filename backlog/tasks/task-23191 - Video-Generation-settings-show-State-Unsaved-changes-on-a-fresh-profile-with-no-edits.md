---
id: TASK-23191
title: >-
  Video Generation settings show 'State: Unsaved changes' on a fresh profile
  with no edits
status: To Do
assignee: []
created_date: '2026-08-29 02:25'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a newly created profile, opening Settings -> Video Generation reports unsaved changes before the user has edited anything, so the dirty indicator cannot be trusted and a Revert appears to be needed on a page nobody touched. Observed during the TASK-23109 verification pass on an isolated profile; the draft for this category appears to initialize dirty rather than adopting the persisted values. The State banner is the sole carrier of the save contract (task-1717, TASK-23104), so a false dirty state undermines the mechanism the whole screen relies on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Video Generation on a fresh profile with no user edits reports no unsaved changes
- [ ] #2 The dirty state appears only after an actual user edit and clears after save or revert
- [ ] #3 A test mounts the category on a fresh profile and asserts the clean state, so the regression cannot return silently
<!-- AC:END -->
