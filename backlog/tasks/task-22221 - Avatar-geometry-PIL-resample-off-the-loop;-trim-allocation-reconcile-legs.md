---
id: TASK-22221
title: >-
  Avatar geometry: PIL resample off the loop; trim allocation-reconcile legs
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - console
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22221).

New with PR #2034. (a) `UI/Console_Modules/left_rail.py:1037-1094` +
`Chat/console_image_view.py:103-107`: each distinct rail viewport size triggers
`image.copy()` + LANCZOS `thumbnail` synchronously on the UI thread (drag-resize burst
cost; the memo prevents steady-state cost). (b) `_run_allocation_reconcile` gained three
per-pass legs since the pin (`left_rail.py:944-1035`): avatar geometry reconcile,
`set_allocation(None)`+height reset across all 7 sections before every measurement pass,
and `_measure_outer_content_height` iterating every outer child — plus
`_refresh_workspace_tree_after_reflow` (`:1030`) clears the tree hover row on EVERY pass
(~5 Hz during runs: hover flicker + 2 repaints + tooltip per tick).

## Acceptance Criteria

- [ ] The avatar resample runs off the loop with the existing memo and race fences intact
- [ ] The allocation reconcile does not clear tree hover when the hover row is unaffected
- [ ] Per-pass query/measure counts recorded before/after; reconcile passes converge in the same number of frames
- [ ] Resize-drag cost measured before/after with a high-resolution character card
