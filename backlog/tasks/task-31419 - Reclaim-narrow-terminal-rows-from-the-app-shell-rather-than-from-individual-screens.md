---
id: TASK-31419
title: >-
  Reclaim narrow-terminal rows from the app shell rather than from individual
  screens
status: To Do
assignee: []
created_date: '2026-09-04 22:42'
labels:
  - ui
  - responsive
  - shell
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Schedules workbench now holds an 80x24 floor with every spec-named operation reachable (redesign PR-4 Task 6, pinned by `Tests/UI/test_schedules_responsive_floor.py`). Reaching it consumed the screen's own slack: the four filter chips collapse to one cycling control, the rail degrades to a single row of flat buttons, the detail region pushes full-screen on Enter instead of blank-hiding.

PR-4 Task 6's accounting of the remaining budget: of the 24 rows, 13 are app-shell chrome that the schedules screen neither owns nor can reclaim — navigation (3), destination header (5), scheduler liveness (1) and the status strip (4) — leaving 11 rows for the queue, its header and its content. Further floor gains are therefore a SHELL question, not a schedules one, and any future "make schedules work at a smaller floor" request should be routed here rather than spent squeezing the screen again.

Note the measurement's provenance before acting on it: the floor test harness (`BundledCSSWorkbenchApp`) mounts the workbench under a bare `ConsolidatedCSSApp`, not the real app shell, so it does not itself measure the chrome. The 13-row figure comes from PR-4 Task 6's own analysis against the real shell and should be re-measured in the real app before any row is traded away.

This is a placeholder for the shell-side conversation, not a commitment to shrink any specific element: several of those rows are deliberate (the destination header is a navigation affordance, the status strip carries the conflicts badge).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 13-row app-shell chrome figure is re-measured in the real app shell at 80x24 and recorded per element
- [ ] #2 Each chrome element is classified as reducible, conditionally reducible at the floor, or deliberate, with the reason stated
- [ ] #3 Any reduction applies at the shell so every destination benefits, not as a per-screen override
- [ ] #4 The Schedules floor test still passes unchanged, proving the screen was not asked to absorb the change
- [ ] #5 If the conclusion is that no row can be reclaimed, that is recorded as the answer and the task closes rather than staying open indefinitely
<!-- AC:END -->
