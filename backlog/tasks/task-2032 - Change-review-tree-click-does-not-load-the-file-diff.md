---
id: TASK-2032
title: 'Change review: tree click does not load the file diff'
status: To Do
assignee: []
created_date: '2026-08-03 00:45'
labels:
  - change-review
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT. On the Review screen, clicking a file row in
the changed-file tree does not load that file's diff into the right pane —
only keyboard navigation (j/k) switches the diff. A mouse user clicks
summary.md, keeps reading notes.md's diff, and has no signal anything is
wrong. Likely the diff loader listens to a Tree event the mouse path does
not emit (NodeHighlighted-via-key vs NodeSelected-on-click, or focus
gating in `_load_file`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking a file row loads that file's diff (parity with j/k)
- [ ] #2 A test pins the mouse-selection path
<!-- AC:END -->
