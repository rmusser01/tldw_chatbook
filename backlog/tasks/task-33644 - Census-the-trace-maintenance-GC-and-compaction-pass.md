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
The census's `gc_pass` step runs the production loop (the scheduler the census otherwise captures) with a 2 s GC interval and no ready delay. Its first pass collects uncounted and the loop parks; the graph epoch is then advanced by SQL with no exchange signal, so the interval wake's pass collects and compacts. A wrapper around `console_runtime.run_owned_db_call` bills exactly that pass: counting starts when it calls `current_graph_epoch` and stops when `run_after_gc` returns, so the owned connections and helpers each call opens are included. The loop task is cancelled afterwards.

Measured at dev `2612fc56b2` (macOS, three runs, both evidence variants): 0 config / 5 storage / 2 helpers / 34 opens every time. `MAX_TRACE_GC_PASS_STORAGE_UNITS` pins exactly that ("trace GC pass" row in the census).

Negative control: one extra guarded `get_user_data_dir()` inside `run_after_gc` fails the ratchet -- config 1 > 0, storage 7 > 5, opens 148 > 34 x 1.05.

File: `Tests/Performance/test_console_keystroke_work_census.py`.
<!-- SECTION:NOTES:END -->
