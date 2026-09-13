---
id: TASK-32544
title: >-
  Library Notes: the list toolbar clips to "Remove pl" whenever a note is open
  at 235x52, and the New note view's status wraps into a 52-column pane beside a
  130-column empty list
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
Critique #3 (dev 5fd502dbac), assessor A, everyone, Edit and Create workflows. P2 #9 + minor.

**What happened.** With a note open at 235x52 the list's second toolbar row reads "New folder  Add to folder  Move note  Remove pl" (A 05, 09, 36) — task-32127 pinned two full rows (`test_notes_toolbar_fits_two_rows_with_every_action_visible`) only for the list with NO note open. In the New note view the status "Ready · Next: Press Blank note, or choose a template." wraps inside a 52-column work pane while the empty list keeps ~130 columns (A 04). Captures: A 04, 05, 09, 36.

**Cause.** INFERRED (a width threshold below the pinned case; the guide already describes a third-row wrap for narrow panes). Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a note open at 235x52 every list toolbar label paints whole (third row, overflow menu or shorter labels — the guide's narrow-pane rule applied at this width)
- [ ] #2 The New note view gets the work-pane width its status line needs; the empty list does not keep more than half the canvas while the New note view is the task in hand
- [ ] #3 Tests pin both widths through the production layout resolver
<!-- AC:END -->
