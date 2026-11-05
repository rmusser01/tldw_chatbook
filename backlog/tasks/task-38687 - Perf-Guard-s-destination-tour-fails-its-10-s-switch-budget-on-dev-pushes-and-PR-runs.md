---
id: TASK-38687
title: "Perf Guard's destination tour fails its 10 s switch budget on dev pushes and PR runs"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-05 10:05'
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
- [x] #1 The over-budget time is attributed to its phase (press, arrival or settle) and its cause is recorded
- [x] #2 The destination tour passes 10 of 10 Perf Guard runs (CI, or local runs that match the job) with its 10 s budget unchanged, or any budget change is justified with measurements in the task notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recount the failing CI runs from their logs (read-only gh).
2. Split each destination's reading into press, arrival and settle with a local probe of the tour, counting how often a `pilot.pause()` idle wait runs to its 1 s ceiling.
3. Find what keeps the process busy while the app is idle (cProfile of an idle window, timer and worker-start counters) and fix that cost where its callers converge.
4. Pin the fix with a deterministic counter test, shown red on dev, in a file a PR lane runs.
5. Re-run the tour 10 times locally (load recorded) with the 10 s budget unchanged; compare per-destination phases before and after.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: an idle Console fed itself, so every `pilot.pause()` saw a busy process.** The switches themselves were within budget. One Console sync calls `ConsoleReadinessConfigProjection.tokenizer_ready` twice, with the provider spelled two ways: the chat's setting gives "OpenAI" and the display gives the config key "openai". The projection keeps a single tokenizer request. Each spelling replaced the other's request and reset its retry, which started a refresh worker. Each refresh then found the request changed and published a full `_sync_native_console_chat_ui`, and that sync asked again. The loop ran 16-28 full Console syncs a second for as long as the app ran, on every screen, because the Console stays mounted.

Textual's `wait_for_idle` counts the process as busy when it uses more than 1 ms of CPU per 20 ms, across all threads. So each `pilot.pause()` in the tour waited its full 1 s.

**CI census** (read-only `gh`): over the last 100 Perf Guard runs (2026-11-03 00:48Z to 11-04 00:26Z), 46 failed, 27 of them on this test. Those 27 runs had 40 over-budget readings, every one between 10.0 and 11.0 s:
- `f4 -> Settings`: 15
- `ctrl+7 -> Schedules`: 12
- `ctrl+9 -> Settings`: 11
- `ctrl+4 -> Personas`: 1
- `f2 -> LLM`: 1

Settings was over budget in 20 of the 27 runs (the coordinator's 14-of-19 count covered a narrower window). Another sign: in run 41610209372 the child's tour test took 104.7 s.

**Phase attribution (AC#1).** A temporary probe (not committed) timed press, arrival and settle, counted idle waits that hit the 1 s ceiling, and measured idle CPU with `process_time`/`thread_time` over a 2 s idle window.
- On dev, at load 11-18:
  - press: 0.6-2.6 s
  - arrival: 0 extra pauses (the screen was already current when the press returned)
  - settle: 3.3-6.5 s, with 2 to 6 of its 6 pauses at the 1 s ceiling
  - idle process CPU: 82-99 % of a core (the event-loop thread alone 65-81 %)
- cProfile of the idle window, plus a worker-start census, found 39-60 `console-readiness-config` refreshes and as many `console-readiness-publication` syncs per 2 s. A per-call trace showed the provider alternating between "OpenAI" and "openai".
- Settings led the counts because it has the costliest press (about 2-3 s) and the most idle-time timers of its own, the MCP workbench save status and subscription readiness, both every 0.25 s. Its settle hit the ceiling most often: 4-6 of 6 pauses.

**Fix: landed with TASK-38703 (#3396, dev 447628d08b).** That PR fixes the same loop. `tokenizer_ready` keys its preparations by `(model, normalize_provider_config_key(provider))` and keeps one slot per key, so "OpenAI" and "openai" share one preparation. It also covers two different models in one sync, which my single-slot normalization (24c8ccd8dd, dropped in the rebase) did not. Its unit pin is `test_provider_label_and_config_key_share_one_preparation`.

Measured with my earlier, equivalent fix (probe, load 27-53):
- idle CPU 0.8-5.3 % of a core
- settle 0.46-1.37 s, with no idle wait at its ceiling
- per-destination totals 1.2-4.5 s

**This branch adds the tour's idle-sync guard.** After the tour, `test_destination_tour_stays_under_switch_budgets` (Perf Guard) counts full Console syncs during 1 s of idle, with a limit of 1. It counts rather than times. With the loop present (dev before #3396) it fails: 16 syncs at load 30, 28 at load 17. With the fix it records 0.

**AC#2: 10 of 10 on this branch's HEAD** (47cc9a22ff, on dev 447628d08b, i.e. with #3396's fix). Command: the Perf Guard step, run serially as in the job (`pytest Tests/Performance/test_ui_latency_guardrails.py Tests/Utils/test_ui_responsiveness_stall_persist.py -v --timeout=300`), at load 6.8-10.5.
- All 10 green, 13/13 tests each; the 10 s budget is unchanged.
- Step 58-69 s; tour test 22.4-27.2 s.
- A temporary, uncommitted probe printed what the test checks: 0 idle Console syncs in every run, and the slowest single switch was 1.58-2.18 s (`ctrl+9` or `ctrl+4`), against the 10 s budget.

Before #3396, the same command on dev sources took 199 s (load 17.8), with the tour test at 75 s. On CI, #3396's own runs and the dev push 41994911006 took 85-117 s for the latency step; the median over the previous 40 green runs was 179 s.

No `/verify` live launch: the brief forbids launching the TUI, and this branch changes no product code.

**Files:** `Tests/Performance/test_ui_latency_guardrails.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
