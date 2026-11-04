---
id: TASK-38712
title: "Speculative-voice history gate: the fresh-commit p95 paired-delta limit fails under host load"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- performance
- perf-guard
priority: medium
references:
- Packaging/speculative_voice_history_gate.py
- Tests/Performance/test_speculative_voice_latency.py
- .github/workflows/perf-guard.yml
dependencies:
- TASK-36510
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down, while TASK-36510 was being implemented (PR #3395, into #3345's branch; not on dev at c2cdb6b82e). Code references checked on dev c2cdb6b82e, 2026-11-03.

`Tests/Performance/test_speculative_voice_latency.py::test_completed_pair_history_gate_meets_large_history_thresholds` is a strict xfail on dev (:71-78) until TASK-36510 indexes the eight message-locator columns. perf-guard's "Run Console tab-publication and voice gates" step (perf-guard.yml:171) runs it, so it runs for real once TASK-36510 lands and its marker is removed.

The gate (`Packaging/speculative_voice_history_gate.py`) builds a 1,000-row baseline and a 100,000-row history and times 40 ABBA-ordered trials per operation. `_operation_report` (:455-481) passes an operation only if the large p95 is at most 100 ms, the median paired delta is at most 2 ms, and the p95 paired delta is at most 10 ms (`P95_DELTA_THRESHOLD_MS`, :32). Both the fresh-commit and the uncertain-retry reports use it (:563-564).

On TASK-36510's branch, the fresh-commit p95 paired delta exceeded 10 ms in 7 of 11 direct gate runs at host load 27-42, with deltas of 11.8-24 ms. The noise is symmetric and is not a slope: baseline maxima reached 918 ms, deltas went as low as -785 ms, and the median deltas were 0.06-0.7 ms. `_percentile` takes the 38th of 40 sorted paired deltas, so three scheduler stalls on the large side of a pair decide it. perf-guard's runners are shared, so the node will flake there in the same way.

The gate's purpose is to catch a locator column that is scanned instead of indexed, which grows with history size. The pass rule should catch that and not catch host load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With TASK-36510's indexes in place, the history gate passes in 10 of 10 direct runs at a host load of 25 or more, with the load and the per-run deltas recorded in the task notes
- [ ] #2 With one of the eight locator indexes removed (a real regression), the gate fails in 10 of 10 runs under the same pass rule, recorded the same way
- [ ] #3 The pass rule and its thresholds are documented next to the constants, with the measurements that chose them
<!-- AC:END -->
