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
2. Add a `gc` census step: start the real loop (no ready delay) and bill its first GC pass -- epoch read, collection, compaction attempt -- holding the 1 Hz backup-maintenance probe still for the phase
3. Pin the measured units; negative control
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The census's `gc_pass` step starts the production maintenance loop, which the census otherwise holds back by capturing its scheduler, with no ready delay. It bills the loop's first GC pass: a wrapper around `console_runtime.run_owned_db_call` counts from that pass's `current_graph_epoch` call until `run_after_gc` returns, so the owned connections and helpers each call opens are included. The census asserts the billed calls are exactly `current_graph_epoch`, `collect` and `run_after_gc`. Any call raising, or an unusable collection, fails the census with the call's name. The batch normalization before the pass is not billed (the per-tick trace row covers it), and the loop task is cancelled afterwards.

On the census's small database compaction defers with `database_threshold`, which the loop treats as retryable, so it keeps that collection pending. Every later pass is the epoch read plus a compaction retry with no collection, which makes the first pass the superset. (Observation, not changed here: while a database stays under the compaction threshold the loop never collects a newer epoch.)

The 1 Hz backup-maintenance probe (`storage_admission._local_pause_requested`, on its own thread) is held still for the phase. Its ~37-open walk landed in the short billed window about one run in ten. `gc_pass` replaces the probe and waits for its first held call before starting the loop: the monitor probes one at a time, so a probe already in flight has finished.

Pinned at dev `0409592a2d`: 0 config / 9 storage / 3 helpers / 54 opens in every run on macOS, and 39 opens on Linux (`LINUX_OS_OPENS_CEILINGS`; perf-guard read 39 on two PR heads). `os_opens` is exact, not jitter: each helper start walks the profile directory chain (one `os.open` per component, plus `/` and `/dev/null`), so it is `3 x (components + 2)`. That is 16 components under macOS's default pytest temp dir and 11 on the Linux runner. A helper reused from a live connection reads one helper (18 or 13 opens) fewer; that is the documented downward jitter.

Negative control: one guarded `get_user_data_dir()` inside `collect` fails it (config 1 > 0, storage 11 > 9, opens 175 > 54 x 1.05).

History, superseded within this PR: the first version billed a later pass. The loop's boot pass collected uncounted and parked, the graph epoch was advanced, then one wake was billed. That design pinned 0/5/2/34 (34 measured under a scratch `--basetemp` one component shallower; 36 at the default depth). Qodo found that the later pass never runs `collect`, so collection costs escaped the ratchet; billing the first pass fixed that.

File: `Tests/Performance/test_console_keystroke_work_census.py`.
<!-- SECTION:NOTES:END -->
