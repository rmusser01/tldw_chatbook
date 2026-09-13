---
id: TASK-32539
title: >-
  Library Notes: focus vanishes after a confirmed delete, and Undo is reachable
  only by / then Tab×6
status: To Do
assignee: []
created_date: '2026-09-13 06:46'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, personas Sam and Jordan, Edit/delete workflow. D7.

**What happened.** Info → Delete → Tab → Enter (delete confirmed) → "✓ deleted · Jordan first note" receipt with Undo / Dismiss and "Recently deleted (1)". Then: no focus mark anywhere; twelve Tabs boxed nothing; Enter did nothing. Undo was reached only by `/` (focus the filter) then Tab×6 — eight keystrokes to the one recovery action (B 19, 20, 21, 22). A's delete → Undo round trip used the mouse (A 19–21). Captures: B 19–22.

**Cause.** INFERRED: the post-delete focus is not parked on the receipt or a list row; task-32132 / 32268 fixed the prompt's placement and Tab cycle, task-32255 the restored row's reveal, none of them the focus target after confirmation. The docs describe the receipt but not where focus goes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Delete is confirmed, focus lands on the receipt's Undo (or, if the receipt is absent, the next list row) with a visible shape-based cue and a footer chip naming it
- [ ] #2 Undo from the post-delete state costs at most two keystrokes
- [ ] #3 A test pins the post-delete focus target and the chip
<!-- AC:END -->
