---
id: TASK-32643
title: 'Library Notes: make the folder picker vault-aware'
status: To Do
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
  - picker
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 9, ACCEPTED in task-32627. The folder picker is used for
three different Notes decisions and gives the user nothing to choose with:
folders and files interleave, no folder says how many notes it holds, and a
root chosen last week must be navigated to again from scratch.

Sits with the picker work (task-32606, task-32611) rather than on its own
branch — it is the same dialog.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Folders sort first, name-ascending, in every picker that can only return a folder.
- [ ] #2 Each folder shows how many notes it holds, computed without walking the whole tree on open.
- [ ] #3 Recently chosen roots are offered without navigation, and choosing one is a single keystroke away.
- [ ] #4 A vault is marked as such in the listing, using the same detection task-32641 surfaces.
<!-- AC:END -->
