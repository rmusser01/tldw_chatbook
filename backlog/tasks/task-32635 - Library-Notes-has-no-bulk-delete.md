---
id: TASK-32635
title: 'Library Notes: has no bulk delete'
status: To Do
assignee: []
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 rider. Notes can be deleted one at a time only. A user who has
just imported a vault and wants to undo it, or who is clearing out a folder,
has no way to remove more than one note per interaction — and the import
paths can create dozens in a single action, so the asymmetry is the user's
problem before it is anyone else's.

The Media screen's selection model is the precedent to follow rather than
invent against.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A user can select several notes and delete them in one action.
- [ ] #2 The action is reversible or confirms first, with the count in the confirmation — deletion is the one place this screen cannot be casual.
- [ ] #3 Notes bound to a lasting-sync root are handled explicitly: the user is told what happens to the files on disk before the deletion runs, and what happens matches what they were told.
- [ ] #4 The selection affordance matches the Media screen's rather than introducing a second grammar.
<!-- AC:END -->
