---
id: TASK-14900
title: Library Media canvas side-by-side list and detail at wide widths
status: To Do
assignee: []
created_date: '2026-08-10 17:20'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

Filed from task-4023 AC#7 (re-critique 2026-08-09, layout heuristic #8). The bounded
half shipped there: canvas rows no longer inherit the rail's 20-cell title cap, so
titles render in full at wide widths. The structural half remains: the Media canvas
stacks its preview/detail BELOW the list, so on a 170-column terminal the right
~half of the canvas is blank while the user scrolls vertically between list and
preview. A side-by-side (list | detail) split above a width breakpoint — the shape
Collections' workbench already uses (`#library-collections-workbench`) — is a
layout redesign with focus-order, compact-mode, and select-mode implications, too
large to ride a copy/grammar batch.

## Acceptance Criteria

- [ ] At wide widths the Media list and its preview/detail render side by side; below the breakpoint the current stacked layout is preserved
- [ ] Keyboard traversal (rows, preview actions, viewer entry) works in both layouts and is advertised honestly by the footer
- [ ] Select mode and the bulk-action toolbar remain fully usable in both layouts
