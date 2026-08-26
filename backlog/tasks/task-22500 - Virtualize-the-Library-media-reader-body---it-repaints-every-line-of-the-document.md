---
id: TASK-22500
title: >-
  Virtualize the Library media reader body - it repaints every line of the document
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - performance
  - library
priority: high
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Measured by TASK-22209's implementer while verifying its own (much smaller) win: the reader
body is a raw auto-height `Static` inside a `VerticalScroll`, so a 2.5 MB / 24,000-line
document gives the widget `height=45000` and `Widget._render_content` re-renders ALL 45,000
lines on EVERY repaint. `_render_content` alone measured **1.08-1.83 s**, and the click a
user actually feels is **~1.4-1.5 s both before and after** 22209's fix (whose 45 ms saving
is ~2% of the real cost).

This is distinct from TASK-22228 items 6-7 (recompose COUNT, now fixed) — this is paint
VOLUME per repaint. After the whole reader burn-down (22207 traversal, 22208 no-change
syncs, 22209 match nav, 22210 progress writes, 22228 press scope), this is the single
largest remaining Library reader cost and it dominates everything already fixed.

## Acceptance Criteria

- [ ] A large document (>=2 MB) repaints in time proportional to the VIEWPORT, not the document: measure `_render_content` and end-to-end click latency before/after at 100 KB / 1 MB / 2.5 MB
- [ ] Scrolling, match navigation, highlight styling and search-scroll-into-view keep working across the window boundary (the existing reader gates stay green)
- [ ] The approach is stated: Textual line-api/virtualized widget, chunked mounting, or an explicit window with the trade-offs recorded
- [ ] A guard pins the property (render cost independent of document size) so it cannot regress silently
