---
id: TASK-1212
title: Scheduler should not silently drop tasks whose type has no registered handler
status: In Progress
assignee: []
created_date: '2026-07-27 23:05'
labels:
  - scheduling
  - observability
dependencies:
  - TASK-1210
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SchedulerLoop.tick` pops a due task, finds no handler registered for its `type`, logs a warning and
continues. That is the exact mechanism by which watchlist checks did nothing for the entire life of
the feature (TASK-1210): `app.py` registered the `watchlist_job` handler only behind a flag that
shipped false, so every due watchlist task was queued, dequeued and discarded, once per poll, with
a log line nobody read.

A per-task warning is the wrong shape for this failure. It is emitted at the point of loss rather
than the point of misconfiguration, it repeats forever without escalating, and it looks identical
to a task type that was deliberately retired.

Raised by Qodo's review of PR #1054, which recommended a startup check as a follow-up.

Confirmed while verifying TASK-1210 against the running app: a full boot with a watchlist source
present writes **zero** lines matching `scheduler` or `scheduling` to `tldw_cli_app.log`. The
scheduler reports neither which handlers it registered nor that it started, so there is no way to
tell a working scheduler from a completely unwired one by observation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At startup, the scheduler reports any task type it can be handed but has no handler for - specifically, a queued watchlist projection with no watchlist_job handler
- [x] #2 The report is visible without enabling debug logging, and states the consequence rather than only the fact
- [x] #3 A metric distinguishes tasks dropped for want of a handler from tasks that ran
- [x] #4 Deliberately unregistered task types can be declared so they do not produce a warning on every run
- [x] #5 A test asserts that wiring a projection without its handler is reported, and that the fully wired case is silent
- [x] #6 Scheduler startup logs which handlers were registered, so a running app can be told apart from an unwired one
<!-- AC:END -->

## Implementation Notes

`SchedulerLoop.report_configuration()` runs once from `run()`, before the first poll. It logs how
many handlers are registered and which, plus the poll interval and queue depth -- so a wired
scheduler is recognisable at all -- and warns separately when queued work has no handler, naming the
task types and stating the consequence ("discarded on every poll, schedules will never fire").

`expected_unhandled_types` lets a deliberately-retired task type be declared, so it does not warn on
every launch. Without that, the warning becomes noise and gets ignored, which is how the original
per-task warning failed.

Dropped tasks now emit `scheduler_tasks_unhandled` with a `task_type` label, so "busy" and
"discarding everything" are distinguishable in metrics rather than only in a log line that repeats
every poll.

**Live verification produced a better result than the logging change.** Chasing why the new startup
line was not visible in the running app, I seeded an overdue source with a 5-second poll interval
and ran the real app without touching the UI. `last_checked` advanced from a 2-hour-stale value to
now, `last_error` stayed None, and **5 real items from summitroute.com were persisted**. That is the
first true end-to-end proof that automatic scheduled checking works -- stronger evidence than
TASK-1210 shipped with, which rested on unit tests plus the projection's `next_run_at` in the UI.

I had briefly suspected the scheduler worker was not running at all, which would have meant
TASK-1210's fix never took effect. It does; the suspicion was wrong.

**Known limitation, filed as TASK-1240.** The report is emitted correctly at INFO/WARNING through
the production loguru stack (directly observed), but a user on a fresh profile currently cannot read
it anywhere: the file log is zero bytes for a new profile, and the in-app Logs screen only buffers
from the point its persistent handler is installed, which is after early startup logging. This
change is still the right one -- the record has to exist before it can be surfaced -- but the
observability goal is not fully met until 1240 lands.

Modified: `tldw_chatbook/Scheduling/scheduler/loop.py`.
Added: `Tests/Scheduling/test_scheduler_observability.py`.
