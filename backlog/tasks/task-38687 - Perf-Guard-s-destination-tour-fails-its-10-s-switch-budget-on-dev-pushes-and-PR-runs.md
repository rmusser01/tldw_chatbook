---
id: TASK-38687
title: "Perf Guard's destination tour fails its 10 s switch budget on dev pushes and PR runs"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- ci
- performance
- ci-flake
priority: medium
references:
- Tests/Performance/test_ui_latency_guardrails.py
- .github/workflows/perf-guard.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down. Dev push Perf Guard run 41610209372 (dev 9bbd5c2515, 2026-11-03 09:20Z) failed "Run latency guardrails": `test_destination_tour_stays_under_switch_budgets` reported `ctrl+4 -> PersonasScreen: 10.1s > budget`, and the step's 12 other tests passed.

**It recurs.** A review count of the 37 failed Perf Guard runs between 2026-11-02 09:23Z and 2026-11-03 10:36Z found 19 that failed this test. In 11 of them one destination was over budget, in 6 two were, and in 2 three were, each at 10.0-11.0 s. Settings dominates:
- `f4 -> SettingsScreen`: 14 times, including dev pushes 41420242795 and 41591904036 (dev 66155632ed);
- `ctrl+9 -> SettingsScreen`: 7;
- `ctrl+7 -> SchedulesWorkbench`: 7;
- `ctrl+4 -> PersonasScreen`: 1 (this dev push).

The other failed runs failed other Perf Guard tests. The dev push at 18482c1cf7 passed.

**What the numbers do not say yet.** The reported time (test_ui_latency_guardrails.py:135-146) runs from before `pilot.press(key)` to after `_wait_for_screen` returns. So it covers the press, the arrival and the six settle passes (:109-111), not the arrival alone. Each `pilot.pause()` can wait up to 1 s for idle (Textual's `wait_for_idle` max), and up to 30 s inside Pilot's own screen wait. So a slow settle after a prompt arrival gives the same reading as a slow switch. One hypothesis is that the event loop or a pause's idle wait was held for seconds. It is unproven; nothing yet splits the time by phase. Perf Guard is not the required check, so these reds do not block merges, but they hide real latency regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The over-budget time is attributed to its phase (press, arrival or settle) and its cause is recorded
- [ ] #2 The destination tour passes 10 of 10 Perf Guard runs (CI, or local runs that match the job) with its 10 s budget unchanged, or any budget change is justified with measurements in the task notes
<!-- AC:END -->
