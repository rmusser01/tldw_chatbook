---
id: TASK-26025
title: 'Scheduling: scheduler liveness heartbeat'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 19:47'
labels:
  - scheduling
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A dead scheduler loop is indistinguishable from an idle one. Verified on origin/dev: a named grep for heartbeat across tldw_chatbook/Scheduling returns zero; the loop reports unhandled handler types at startup and increments scheduler_tasks_unhandled (Scheduling/scheduler/loop.py:349-362), but once running there is no liveness signal - if the loop dies, reminders simply stop and nothing says so. Hermes persists tick heartbeat, last-success age and last error, so its status command can state that the scheduler has not ticked in three hours.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each scheduler tick records a durable timestamp
- [x] #2 The Schedules surface shows scheduler liveness, and a stale heartbeat is visibly distinct from an empty queue
- [x] #3 The last error encountered by the loop is retained and surfaced rather than only logged
- [x] #4 Staleness is judged against the configured poll interval, so a long interval does not read as a stall
- [x] #5 The heartbeat write is cheap enough to run every tick without measurable overhead - measured and recorded
- [x] #6 A never-started scheduler is distinguishable from a stalled one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: pure classifier (never_started/live/stale, poll-scaled window) + durable store round-trip + never-raise; loop-level (healthy tick = live, erroring tick retains last_error); summary line 3-state\n2. scheduler_heartbeat.py: SchedulerHeartbeat + classify_scheduler_liveness + atomic read/write + scheduler_liveness_line\n3. Loop tick() finally writes heartbeat, captures dispatch error; ctor heartbeat_path kwarg (None=default path)\n4. Workbench liveness Static, refreshed on mount + the existing next-run timer (runs even on empty queue); measure write cost
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Durable heartbeat file (atomic JSON in the user data dir, mkstemp+os.replace) rewritten by SchedulerLoop.tick()'s finally each tick with last_tick_at, last_success_at, last_error, poll_interval, tick_count (AC#1). persist_event was NOT usable — it is write-only diagnostics to a log, not a readable state store; a heartbeat must be read back to judge liveness. classify_scheduler_liveness is pure: None/no-tick => never_started (AC#6, distinct from stalled), else the last tick's age vs a window = max(90s floor, poll_interval*3) => live/stale — so a long poll interval is not read as a stall (AC#4, pinned both directions). AC#3: tick() catches the dispatch exception, records it as last_error (retained across ticks), then re-raises so the run loop's existing handling is unchanged — even a failed tick records liveness. Surface (AC#2): a Static under the sync bar shows scheduler_liveness_line's 3-state summary (not started / live · last tick Xago / STALLED — last tick Xago · last error: …), refreshed on mount AND on the existing next-run timer, which now runs even on an empty queue (a stall with nothing queued is exactly the distinguish-from-idle case). AC#5 measured: 0.120 ms/write vs the 30,000 ms poll interval — negligible. All writes/reads swallow errors (a diagnostics write never breaks the loop it observes). heartbeat_path ctor kwarg (None = default_heartbeat_path) is injectable for tests. 11 new tests; scheduler suite 379 + schedules UI 101 green.
<!-- SECTION:NOTES:END -->
