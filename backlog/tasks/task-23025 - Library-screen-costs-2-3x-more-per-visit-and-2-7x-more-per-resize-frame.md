---
id: TASK-23025
title: >-
  Library screen costs 2.3x more per visit and 2.7x more per resize frame
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - ui
  - screens
  - regression
priority: medium
---

## Description

`library_screen.py` grew +6,340 lines in the review window. It did **not** add recompose sites -
those went **down**, 126 -> 96. It made each existing one materially more expensive.

- **Widgets constructed per visit: 67 -> 157 (+134%)**. Library is the only pre-existing screen whose
  cost changed materially; every other destination moved by 1-6 widgets.
- **DOM queries per resize frame: 31.6 -> 84.2** (153 `query` + 689 `query_one` per 10 frames). The
  handler carries a recent optimisation comment (TASK-22228) yet net query volume still grew 2.7x -
  the width-crossing early return sits *after* the query work.
- Focus-change queries per Tab: 22.4 -> 25.8, and that path can reach `refresh(recompose=True)`.

One whole-screen recompose now constructs 103 widgets at ~640-790 ms.

## Acceptance Criteria

- [ ] Per-visit widget count and per-resize-frame query count measured before and after, interleaved
- [ ] The resize handler returns early **before** doing query work when the layout cannot have changed
- [ ] `research_workspace_screen.py:252` -> `_apply_pane_layout` gets the same gate; it is currently ungated
- [ ] No behaviour change to layout at any terminal width, including the width-crossing cases the current code handles

## Evidence

Interleaved A/B, 2 runs per arm, identical every run. Console is worse in absolute terms
(171-185 queries/frame) but **improved** slightly - pre-existing, not this delta.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
