---
id: TASK-31507
title: Scheduler tick does synchronous file IO on the event loop
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - scheduling
dependencies: []
priority: low
---

## Description (the why)

Every scheduler tick (30 s default, runs for every user from boot),
`_record_heartbeat` (`Scheduling/scheduler/loop.py:380`) synchronously does
mkdir + mkstemp + write + rename on the event loop
(`scheduler_heartbeat.py:94`), and `_emergency_stopped` (`loop.py:411`) does
a blocking `read_text` per tick. Sub-ms on a healthy SSD, but it is blocking
I/O on the render thread and degrades on slow or network-mounted home directories. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 7.

## Acceptance Criteria (the what)

- [ ] Heartbeat writes and emergency-stop reads on the tick path run off the event loop (e.g. `asyncio.to_thread`)
- [ ] Heartbeat "never raises / never breaks the loop" contract and TASK-26025 diagnostics behavior are preserved (existing tests stay green)
