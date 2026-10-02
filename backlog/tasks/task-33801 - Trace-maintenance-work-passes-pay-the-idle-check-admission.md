---
id: TASK-33801
title: Trace maintenance work passes pay the idle-check admission
status: Done
assignee:
  - '@claude'
created_date: '2026-10-02 17:30'
labels:
  - perf
  - console
dependencies:
  - TASK-33269
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PERF-10 (TASK-33269, PR #2914) gave legacy trace maintenance a read-only idle check that runs before the write transaction. The check costs a storage admission of its own, so a pass that has work paid it only to fall through to the write path: three admissions where a pass used to pay two. In production every pass woken by an exchange write is such a pass. The Console storage-unit census showed it as "trace maintenance (per tick) storage_admissions: 2.125 > ceiling 2" (its first tick has work), failing the UI latency guardrails job on dev and on unrelated PRs (#2944, #2945, #2920) about two runs in three.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pass whose work is already known (the first pass, or one woken by an exchange write) opens only its write transaction, with no idle check
- [x] #2 Idle passes still answer from the read-only check without the write lock
- [x] #3 The census trace-maintenance ratchet holds at its existing ceiling across repeated runs (no ceiling raised)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument the census per tick to locate the extra admission (tick 0: idle check returned False, then the write path)
2. Add an expect_work flag to LegacyTraceMaintenance, true at first and consumed by run_batch
3. Set it in the runtime loop when the exchange-write generation changes
4. Tests: transactions opened per pass on a real database; the loop flags a signal-woken pass; census repeated
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Per-tick instrumentation of the census (uncommitted) showed tick 0 always cost +3 with the idle check returning False, and idle ticks +2 (occasionally +1); 17 admissions over 8 ticks failed the ceiling of 2, and a run passed only when an idle tick happened to cost 1. Bisect: 0/4 failures at 92a95170a5 (before #2914), failing from ebf7dc4b06 (#2914) on.

LegacyTraceMaintenance.expect_work (True at construction) makes run_batch skip the idle check once; the runtime loop sets it when the exchange-write generation differs from the value it read before its last pass. A GC-interval wake leaves it False, so an idle interval pass still uses the read-only check.

Census after the fix: trace-maintenance breach 0/8 runs (dev 5/8). A separate typing-burst flake ("typing (whole burst) helper_spawns: 1 > ceiling 0") occurs with and without this change and is not addressed here.

Files: tldw_chatbook/Chat/console_trace_maintenance.py, tldw_chatbook/Chat/console_runtime.py, Tests/Chat/test_console_trace_legacy_migration.py (test_a_pass_with_known_work_skips_the_idle_check), Tests/Chat/test_console_trace_maintenance_parking.py (test_a_signal_wake_tells_the_pass_it_has_work).
<!-- SECTION:NOTES:END -->
