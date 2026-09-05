---
id: TASK-31501
title: Legacy trace maintenance polls a write-lock transaction at 1 Hz forever
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - console
  - chat
dependencies: []
priority: high
---

## Description (the why)

`Chat/console_runtime.py:994-1152` (`_schedule_legacy_trace_maintenance`)
loops `asyncio.to_thread(maintenance.run_batch)` + `asyncio.sleep(1.0)` for
the life of the process, for every real profile. `run_batch`
(`Chat/console_trace_maintenance.py:735`) opens
`db.transaction(immediate=True)` -- acquiring the ChaChaNotes WRITE lock --
and runs 3 SELECTs even in the fully-migrated steady state (the
logical-complete check lives inside the transaction). There is no idle gate,
no visibility gate, and no config off-switch. This breaks the zero-SQL-at-idle
property the 2026-08-27 review established, wakes the CPU every second
(battery), and contends with user writes on the same lock -- it also pays the
TASK-31502 per-statement tax every tick. No task records the 1 Hz cadence as
a deliberate decision. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 2.

## Acceptance Criteria (the what)

- [ ] In the steady state (migration logical_complete, no pending legacy rows, provider idle) the maintenance loop performs no write-lock acquisition more often than once per 30 s (event-driven wake on new legacy rows, or an equivalent backoff)
- [ ] Steady-state completeness checks run without `immediate=True` (read path), or without any transaction at all
- [ ] Migration progress behavior while legacy rows ARE pending is unchanged (existing maintenance tests stay green)
- [ ] A test pins the steady-state cadence so a future change cannot silently restore the 1 Hz write-lock poll
