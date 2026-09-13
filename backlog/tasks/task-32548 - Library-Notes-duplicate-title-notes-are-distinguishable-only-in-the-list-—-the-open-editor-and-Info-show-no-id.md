---
id: TASK-32548
title: >-
  Library Notes: duplicate-title notes are distinguishable only in the list —
  the open editor and Info show no id
status: To Do
assignee: []
created_date: '2026-09-13 06:47'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Alex and Riley. Residual of task-32254 (list-row tie-break shipped in #2611).

**What happened.** Two "Reading list" rows are tie-broken in the list ("· #8d61" / "· #0a89", B 29; "#0dc8 / #c4e8", A 41). Opening one: the editor header reads "Reading list", Info reads "reading, study · v1 · 12 words", and nothing identifies which of the two is open (A 43; B 37, 38). Captures: A 41, 43; B 29, 37, 38.

**Cause.** INFERRED: `note_row_tiebreak_labels()` (task-32254) is a list-row projection; the editor header and Info do not consult it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the open note's title collides with another visible note, the editor header or Info Properties carries the same tie-break suffix the list row shows
- [ ] #2 A test opens the second of two same-titled notes and asserts the suffix is rendered in the editor
<!-- AC:END -->
