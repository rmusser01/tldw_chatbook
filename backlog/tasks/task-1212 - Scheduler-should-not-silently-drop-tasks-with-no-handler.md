---
id: TASK-1212
title: Scheduler should not silently drop tasks whose type has no registered handler
status: To Do
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At startup, the scheduler reports any task type it can be handed but has no handler for - specifically, a queued watchlist projection with no watchlist_job handler
- [ ] #2 The report is visible without enabling debug logging, and states the consequence rather than only the fact
- [ ] #3 A metric distinguishes tasks dropped for want of a handler from tasks that ran
- [ ] #4 Deliberately unregistered task types can be declared so they do not produce a warning on every run
- [ ] #5 A test asserts that wiring a projection without its handler is reported, and that the fully wired case is silent
<!-- AC:END -->
