---
id: TASK-31507
title: Scheduler tick does synchronous file IO on the event loop
status: Done
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

- [x] Heartbeat writes and emergency-stop reads on the tick path run off the event loop (e.g. `asyncio.to_thread`)
- [x] Heartbeat "never raises / never breaks the loop" contract and TASK-26025 diagnostics behavior are preserved (existing tests stay green)

## Implementation Plan (the how)

1. Offload the heartbeat write in `tick()`'s `finally` via `asyncio.to_thread`, awaited so a reader observing a finished tick always sees its heartbeat.
2. Make `_emergency_stopped` async and offload the stop-state read the same way; failure reads as stopped (fail-safe preserved).
3. Run the full Tests/Scheduling suite.

## Implementation Notes

Both blocking file operations on the scheduler tick path now run via
`asyncio.to_thread`, awaited in place so ordering and observable behavior are
unchanged: a finished `tick()` has always written its heartbeat (the
TASK-26025 tests pin this), and an erroring tick still records liveness before
re-raising. `_emergency_stopped` became async (its only caller is
`_dispatch_due`); an offload failure returns True, holding work on doubt per
TASK-26004 AC#4. No API surface outside `Scheduling/scheduler/loop.py`
changed. Verified: full `Tests/Scheduling/` -- 1,029 passed.
Files: `tldw_chatbook/Scheduling/scheduler/loop.py`.
