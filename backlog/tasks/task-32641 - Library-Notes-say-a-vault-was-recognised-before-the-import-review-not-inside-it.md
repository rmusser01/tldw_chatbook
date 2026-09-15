---
id: TASK-32641
title: >-
  Library Notes: say a vault was recognised before the import review, not inside it
status: To Do
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 3, ACCEPTED in task-32627. On folder selection, detect an
Obsidian vault and say so on the confirmation line — with the counts and what
will be skipped — instead of leaving the user to discover it inside the review.

The detection already exists; only the surfacing is missing. This turns the
Obsidian toggle from a checkbox the user has to interpret into a recognised
handshake, and it is the moment at which the duplicate-vault class of bug
(task-32605, task-32637) becomes visible to the person who can still change
their mind.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Selecting a folder that is an Obsidian vault says so before the review, naming what was detected.
- [ ] #2 The line carries the counts and names what will be skipped (`.obsidian/`, `.trash/`, `Templates/`, empty files) in user terms.
- [ ] #3 A folder that is NOT a vault says nothing extra — no empty "0 detected" row.
- [ ] #4 If any of those files were already imported or already bound, the line says so here, where the user can still choose differently.
<!-- AC:END -->
