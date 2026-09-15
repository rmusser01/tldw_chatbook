---
id: TASK-32649
title: >-
  Library Notes: the mode strip is Shift+Tab-only and the Notes rail row advertises no footer chip
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606, two small keyboard-discoverability gaps observed
live at 235x52.

1. From the Notes list, Tab forward never reaches the
   **Library notes | Folder files** strip -- the work-pane Tab region is
   closed by design (task-32540 AC#3) and the strip sits above it. Only
   Shift+Tab x3 gets there. A keyboard user reading "switch to Folder files"
   in the empty-list copy has no forward route to the control it names.
2. The rail rows for Media and Conversations show `enter open media` /
   `enter open conversations` on the footer; the **Notes (10)** row shows no
   chip at all.

Both INFERRED as to mechanism.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Folder files is reachable from the Notes list by Tab, not only Shift+Tab
- [ ] #2 The Notes rail row advertises its Enter action the way its siblings do
<!-- AC:END -->
