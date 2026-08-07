---
id: TASK-3010
title: Console mount window runs the control-bar sync 11× and rebuilds inspector state 28×
status: To Do
assignee: []
created_date: '2026-08-07 23:30'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
cProfile of a single ChatScreen push (task-2902 round 2, standard harness, 235x52): `_sync_console_control_bar` ran **11 times at ~102ms each — 1.12s** of the ~2.5s first paint, and `_build_console_inspector_state` was rebuilt **28 times (0.75s)**. Each caller is individually justified (mount hooks, session activation, restore, resize, watchers) but nothing coalesces them, so the mount window pays the same expensive sync repeatedly against near-identical state.

This is the top lever for Console's switch latency: task-2902's widget-deferral experiment moved only 4–8% because the cost is in these repeated syncs plus per-query DB connections (task-3011), not hidden-widget mounting. Candidate shapes: a dirty-flag + `call_after_refresh` coalescer for the control-bar sync (collapse the mount-window burst into one trailing run), and memoizing `_build_console_inspector_state` on its inputs within a frame.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] One screen push runs the control-bar sync a small constant number of times (target ≤3), verified by the same profiling method.
- [ ] Console push first-paint improves measurably in an interleaved A/B against dev (same probe as task-2902's notes).
- [ ] No behavioral regression across the 31-file console mount-path surface (list in task-2902's notes).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
