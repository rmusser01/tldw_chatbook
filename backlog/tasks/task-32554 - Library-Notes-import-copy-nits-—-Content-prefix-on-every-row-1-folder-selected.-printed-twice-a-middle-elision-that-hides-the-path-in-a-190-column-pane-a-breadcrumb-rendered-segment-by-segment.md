---
id: TASK-32554
title: >-
  Library Notes: import copy nits — "Content:" prefix on every row, "1 folder
  selected." printed twice, a middle-elision that hides the path in a 190-column
  pane, a breadcrumb rendered segment-by-segment
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A (B's captures show the same strings), persona Jordan, Import once workflow. Minor observations grouped by surface.

1. Every New review row starts "Content: create 1 new note:" — "Content:" carries nothing (A 31; B 29).
2. The setup pane says "1 folder selected." then "1 folder selected: /Users/…/crit3/…vault" — the same fact twice, and the middle-elision hides `A/fresh/` in a 190-column pane that had room for the whole path (A 30; B 28).
3. The picker breadcrumb renders a 100+ character path segment-by-segment (A 27).

**Cause.** PROVEN copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review rows drop the "Content:" prefix
- [ ] #2 The setup pane states the selection once, with the full path when it fits and a middle-elision only when it does not
- [ ] #3 The picker breadcrumb elides from the middle to one line
<!-- AC:END -->
