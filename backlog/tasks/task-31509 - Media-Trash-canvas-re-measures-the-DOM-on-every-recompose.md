---
id: TASK-31509
title: Media Trash canvas re-measures the DOM on every recompose
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - library
  - ui
dependencies: []
priority: low
---

## Description (the why)

`Widgets/Library/library_media_trash_canvas.py:167-244` schedules
`call_after_refresh(_measure_after_layout)` from `on_mount`, `on_resize` AND
`_after_recompose`; each pass runs 3+ `query_one` lookups plus per-child
geometry math and may schedule a second capping pass. The screen constructs a
brand-new canvas instance per state transition (page load, filter change,
mutation commit), so the chain fires on every transition and multiplies with
the known per-visit fresh-screen-instance cost (task-24452). Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 7.

## Acceptance Criteria (the what)

- [ ] A single state transition of the Trash canvas triggers at most one measurement pass (coalesced), and unchanged geometry skips the second capping pass
- [ ] Fold/cap visual behavior is unchanged (existing trash-canvas tests stay green)
