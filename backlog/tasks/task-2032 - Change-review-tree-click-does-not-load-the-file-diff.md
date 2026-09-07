---
id: TASK-2032
title: 'Change review: tree click does not load the file diff'
status: Done
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
- [x] #1 Clicking a file row loads that file's diff (parity with j/k)
- [x] #2 A test pins the mouse-selection path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test: move the tree cursor to a second file and select it (the same
   NodeSelected path a mouse click takes) — the diff pane must switch
2. Fix: leaf nodes carry their leaf index as node.data; a Tree.NodeSelected
   handler calls _focus_leaf(index); group nodes (data None) ignored
3. Sabotage-verify
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the screen had NO Tree.NodeSelected handler — j/k
(`action_next_file`/`action_previous_file` → `_focus_leaf`) was the only
diff loader, so a mouse click selected the row visually and loaded nothing.

Fix: leaf nodes carry their `_leaves` index as `node.data` at add time;
an `@on(Tree.NodeSelected, "#change-review-tree")` handler calls
`_focus_leaf(index)`; group nodes (data None) keep their expand/collapse
click untouched. Test drives Tree's own cursor-select action — the same
NodeSelected event a mouse click produces — and watched the exact live
failure first (diff pinned to the first file). 14 screen tests +
change-review regression green.
<!-- SECTION:NOTES:END -->
