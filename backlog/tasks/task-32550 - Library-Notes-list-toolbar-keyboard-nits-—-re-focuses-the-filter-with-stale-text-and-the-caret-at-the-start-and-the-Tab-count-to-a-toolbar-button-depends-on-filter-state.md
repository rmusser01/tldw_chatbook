---
id: TASK-32550
title: >-
  Library Notes: list-toolbar keyboard nits — "/" re-focuses the filter with
  stale text and the caret at the start, and the Tab count to a toolbar button
  depends on filter state
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
Critique #3 (dev 5fd502dbac), both assessors, persona Alex. Two small keyboard defects on the same toolbar.

1. `/` on an already-filtered list re-focuses the field with the old text and the caret at the start, so typing "scaling" produced "scalingReading"; End + Ctrl+U are needed first (A 41).
2. Disabled Sort is skipped in the Tab order while a filter shows, so the same recipe `/` + Tab×4 lands on Add from files… on an unfiltered list and on Export on a filtered one — B's power run opened the Export bundle canvas by accident (B D15). Captures: A 41; B §3.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 "/" on a filtered list selects the existing filter text (or clears it) so typing replaces it
- [ ] #2 Tab counts to a toolbar button do not change with filter state, or notes.md's keyboard recipes name the state they assume
- [ ] #3 Tests pin both
<!-- AC:END -->
