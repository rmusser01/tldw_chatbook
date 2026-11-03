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

**It recurs.** I read the logs of 33 failed Perf Guard runs between 2026-11-02 09:23Z and 2026-11-03 10:36Z. 16 of them failed this test. Each time one destination took 10.0-11.0 s:
- `f4 -> SettingsScreen` (dev pushes 41420242795 and 41591904036, the second at dev 66155632ed);
- `ctrl+9 -> SettingsScreen`, `ctrl+7 -> SchedulesWorkbench` and `ctrl+4 -> PersonasScreen` (PR runs and this dev push).

The other failed runs failed other Perf Guard tests. The dev push at 18482c1cf7 passed.

**What the numbers say.** Every reported time sits just over the budget, and the screen did arrive. `_wait_for_screen` (test_ui_latency_guardrails.py:101-112) checks the screen after each `pilot.pause()` and gives up at the 10 s deadline. A result of `arrived=True` with elapsed at least 10 s means one `pilot.pause()` that started before the deadline returned after it, with the switch already done. So the event loop, or the pause's idle wait, was held for seconds; this is not a switch that ran slightly slow. Perf Guard is not the required check, so these reds do not block merges, but they hide real latency regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the multi-second hold is found and recorded, naming the destination and what held the loop or the pause
- [ ] #2 The destination tour passes 10 of 10 Perf Guard runs (CI, or local runs that match the job) with its 10 s budget unchanged, or any budget change is justified with measurements in the task notes
<!-- AC:END -->
