---
id: TASK-33644
title: Census the trace-maintenance GC and compaction pass
status: Done
assignee:
  - '@claude'
created_date: '2026-09-30 18:12'
labels:
  - perf
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2888's storage-unit census measures legacy trace maintenance per 1 Hz tick through run_batch only. The production loop (console_runtime) also runs a physical pass after a logically complete batch -- graph-epoch read, garbage collection and compaction through owned database calls -- at most once per GC interval, and PERF-10 (TASK-33269) wakes parked maintenance for it every interval. Qodo noted on #2888 that storage admissions or helper spawns added to that pass cannot fail any ratchet today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census drives one deterministic eligible GC interval through the production maintenance loop (after PERF-10's parking) and records its config admissions, storage admissions, helper spawns and os.open calls
- [x] #2 Those counts are pinned as ceilings per GC pass, and an added admission in the pass fails the ratchet (negative control recorded)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Keep the real maintenance scheduler the census captures, and the runtime it was armed on
2. Add a `gc` census step: run the real loop with a 2 s GC interval, let its first pass collect uncounted and park, advance the graph epoch with no exchange signal, bill the interval wake's pass from its epoch read through compaction
3. Pin the measured units; negative control
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The census's `gc_pass` step runs the production loop (the scheduler the census otherwise captures) with a 2 s GC interval and no ready delay. Its first pass collects at once (uncounted); the moment that pass returns, the interval is raised out of reach so the loop parks and stays parked. The graph epoch is advanced by SQL with no exchange signal and billing armed, and only then does the interval drop to zero, so the next park poll wakes exactly one GC pass (no race with the arm). A wrapper around `console_runtime.run_owned_db_call` bills that pass from its own `current_graph_epoch` call until `run_after_gc` returns, so the owned connections and helpers each call opens are included; any call in the pass raising, or an unusable collection, fails the census with the call's name. The loop task is cancelled afterwards.

On the census's small database the first compaction defers with `database_threshold`, which the loop treats as retryable, so it keeps that collection pending: the billed pass is the epoch read plus the compaction retry, the steady state of an idle small profile. (Observation, not changed here: while a database stays under the compaction threshold the loop never collects a newer epoch, because the pending collection is only cleared when compaction completes.)

Measured at dev `2612fc56b2` (macOS, three runs, both evidence variants): 0 config / 5 storage / 2 helpers / 34 opens every time. `MAX_TRACE_GC_PASS_STORAGE_UNITS` pins exactly that ("trace GC pass" row in the census).

Re-pinned 34 -> 36 opens on 2026-10-03 before merge, after the rebase onto dev `420b53a63d` read 36 in every run. A per-open trace showed it is exact, not jitter: each of the two helper starts walks the profile directory chain (one `os.open` per component, plus `/` and `/dev/null`), so the count is `2 x (components + 2)`. The 34 came from a scratch `--basetemp` one component shallower. macOS's default pytest temp dir gives 16 components (36); the Linux runner gives 11 (26, the Linux pin). The same rebase run also read 73 once: the trace showed the extra 37 opens were the 1 Hz backup-maintenance probe (`storage_admission._local_pause_requested`, own thread) landing inside the billed window, about one run in ten. `gc_pass` now holds that probe still for the whole phase (the census's convention for wall-clock loops, like the scheduler) and restores it after; 10/10 runs then read exactly 36.

Negative control: one extra guarded `get_user_data_dir()` inside `run_after_gc` fails the ratchet -- config 1 > 0, storage 7 > 5, opens 148 > 34 x 1.05.

File: `Tests/Performance/test_console_keystroke_work_census.py`.
<!-- SECTION:NOTES:END -->
