---
id: TASK-41602
title: "The logging pipeline's production-wiring test fails its 2.0 microsecond dropped-debug bound under host load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 19:40
updated_date: 2026-11-19 15:17
labels:
- follow-up
- tests
- flaky
- logging
priority: low
references:
- Tests/test_logging_pipeline_single_pass.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during a round-15 covering run (2026-11-17). `Tests/test_logging_pipeline_single_pass.py::test_production_wiring_drops_debug_before_evaluating_lazy_arguments` measured 2.39 microseconds at load about 50-80 and failed. It passed 3 of 3 alone, at base and at head.

**The check** (read at 212f015a91):
- The child script (`:437-463`) times 20,000 dropped `logger.debug` calls with `time.perf_counter()`.
- The test asserts that the mean per call is under 2.0 microseconds (`:505`).
- The comment at `:504` records about 0.15 microseconds with the level gate and about 7-8 without it.
- The bound is wall-clock time on a shared host, so it measures the host's load as well as the gate.

The file runs in `test.yml`'s full suite and in `nightly-deep.yml`, not on a PR lane. TASK-33262 added the test and is Done. No open task covers this failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test still fails when the level gate is removed (the 7-8 microsecond path), shown red under a temporary source edit
- [x] #2 It passes 10 of 10 at `-n 4` with host load above 60, and the notes record the load
- [x] #3 The notes state the speed assertion and the measurement and load it rests on
<!-- AC:END -->

## Implementation Plan

1. Measure, in the production-wiring child, wall time and process CPU time of the same dropped-debug loop, with and without the level gate, under host load.
2. Name the cause from that measurement before touching the check.
3. Change the measure (not the bound) so host load is not counted; show the gate-removed path still fails and the test passes 10 of 10 at `-n 4` above load 60.

## Implementation Notes

- **Cause, measured.** The child timed the loop with `time.perf_counter()`, i.e. wall time, which also counts every moment the child waits for a CPU. A probe ran the test's child wiring 12 times, four at once, at load 58, timing the same 20,000 dropped `logger.debug` calls both ways: process CPU time was 0.13-0.16 us per call in all 12; wall time was 0.13-0.29 us in 11 and **3.44 us** in one, whose CPU time was 0.14 us (a 2,000-call batch in that run took 8.9 us per call). The gate's cost did not change; the host took the CPU away.
- **Change.** The loop is timed with `time.process_time()` (CPU time of the child). The **2.0 us bound is unchanged**: with the gate removed the same probe measured 8.1-8.3 us of CPU per call (wall 38-48 us), so the bound sits about 12x above the gated cost and 4x below the ungated one. The comment above the assertion now records these numbers and the load.
- **The speed assertion and what it rests on:** `dropped_debug_us < 2.0`, the mean process-CPU time of 20,000 dropped `logger.debug` calls under `configure_application_logging` at INFO; 0.13-0.16 us measured with the gate and 8.1-8.3 us without it, at load ~58 on this 18-core host.
- **AC #1 (red with the gate removed):** temporary edit, `_loguru_forward_level` returning 0 (TRACE, the pre-PERF-03 forwarder). The test failed at `lazy_evaluated == 0` (`assert 1 == 0`); with that assertion also removed (temporary) it failed on speed, `assert 7.8287 < 2.0`, at load 66. Both edits reverted.
- **AC #2:** 10 of 10 passed at `-n 4` (the node temporarily parametrized x10) at load 63.3-64.2, with 8 extra `yes` CPU burners running; then 30 of 30 at load 62.3-62.8. For comparison the unchanged wall-clock test passed 30 of 30 at load 66-73 and 10 of 10 at 57-58: the failure is rare, which is why the cause is shown by the probe's paired measurement, not by a red run.
- Windows note: `process_time` advances in scheduler ticks there; the gated loop then reads 0 or ~0.8 us and the ungated one ~8 us, so both outcomes stay on the right side of 2.0.
- Files: `Tests/test_logging_pipeline_single_pass.py`. Covering run: that file plus `Tests/Backup_Recovery/test_home_citation_retirement.py`, 41 passed at `-n 4`.

