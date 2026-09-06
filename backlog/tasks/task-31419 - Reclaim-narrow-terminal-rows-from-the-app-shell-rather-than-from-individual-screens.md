---
id: TASK-31419
title: >-
  Reclaim narrow-terminal rows from the app shell rather than from individual
  screens
status: Done
assignee: []
created_date: '2026-09-04 22:42'
updated_date: '2026-09-06 14:15'
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
- [x] #1 The 13-row app-shell chrome figure is re-measured in the real app shell at 80x24 and recorded per element
- [x] #2 Each chrome element is classified as reducible, conditionally reducible at the floor, or deliberate, with the reason stated
- [x] #3 Any reduction applies at the shell so every destination benefits, not as a per-screen override
- [x] #4 The Schedules floor test still passes unchanged, proving the screen was not asked to absorb the change
- [x] #5 If the conclusion is that no row can be reclaimed, that is recorded as the answer and the task closes rather than staying open indefinitely
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Outcome: AC#5 branch — no row is reclaimable from the app shell; recorded as the answer and closed.** The one real narrow-terminal lever found is a shared-widget-layer change, filed as TASK-31825.

**Measurement (AC#1):** real app shell, dev tip, 80×24, tmux with a scratch
`TLDW_CONFIG_PATH` profile (real user profile untouched), verified on
Schedules plus Home and Console. Per-element record:

| Rows | Element | Owner | Source | Classification (AC#2) |
|------|---------|-------|--------|------------------------|
| 1–3 (3) | Top navigation (`MainNavigationBar`) | **shell** (`BaseAppScreen.compose()`, `base_app_screen.py:225`, `height: 3`) | identical on every destination | deliberate — primary navigation |
| 24 (1) | Footer (`AppFooterStatus`) | **shell** (`base_app_screen.py:243`, `height: 1`) | identical on every destination | deliberate — already minimal |
| 4–8 (5) | Destination header | **screen** (`schedules_workbench.py:596` via shared `DestinationHeader`, `UI/Workbench/workbench_widgets.py:164`) | absent on Home/Console | conditionally reducible — a dormant `density="compact"` CSS rule exists that no caller triggers (→ TASK-31825) |
| 9 (1) | Scheduler-liveness line | **screen** (`schedules_workbench.py:628`) | schedules-only | deliberate, but a screen concern, not shell |
| 20–23 (4) | Status strip (`#scheduling-status-strip`) | **screen** (`schedules_workbench.py:759`) | carries the conflicts badge; schedules-only; distinct from `AppFooterStatus` | deliberate, but a screen concern, not shell |

**Key correction to the premise:** PR-4 Task 6's per-element arithmetic (3+5+1+4=13)
reproduces exactly, but its *attribution* was wrong — only **4 rows**
(navigation 3 + footer 1) are true, unconditional app-shell chrome. The
destination header, liveness line, and status strip are composed inside
`SchedulesWorkbench.compose_content()` and do not exist on other destinations,
so "13 rows the schedules screen neither owns nor can reclaim" overstated the
shell's share by ~10 rows. Any future floor work on those 10 rows is
screen/shared-widget work, not shell work.

**AC#3:** consequently no shell-level reduction ships — both true shell elements
are already minimal and deliberate. **AC#4:** `Tests/UI/test_schedules_responsive_floor.py`
untouched and green (27/28; the 1 failure,
`test_the_docked_task_detail_pane_scrolls_to_reveal_history_past_the_fold`,
fails identically at the branch base — pre-existing dev debt, attributed in a
throwaway worktree). **AC#5:** this note is the recorded answer; the task closes.

Full raw measurement (per-row table, captures) in the SDD workspace
(`.superpowers/sdd/plan-2026-09-06-schedules-deferred-burndown/t31419-measurement.md`,
session-local).
<!-- SECTION:NOTES:END -->
