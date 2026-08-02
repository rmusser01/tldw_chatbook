---
id: TASK-1810
title: Scheduled briefings are invisible to the Scheduling screen
status: To Do
assignee: []
created_date: '2026-08-01 18:25'
labels:
  - watchlists
  - briefings
  - scheduling
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed at close-out of the Watchlists briefings phase 4 plan (spec #2), per the phase 3 task's own
recorded follow-up.

Phase 4 wired a `briefing_job` task type through the scheduler's dispatch seam directly:
`app.py` constructs a `BriefingProjection` and a `BriefingJobHandler` and threads them into
`SchedulerLoop`/`PriorityQueue` (`Scheduling/scheduler/loop.py`, `Scheduling/scheduler/queue.py`)
beside the existing watchlist-check wiring, gated on `[scheduling] briefing_schedules_enabled`.
Scheduled briefings genuinely fire on this path -- the dispatch and generation both work.

They are, however, invisible to the Scheduling screen. `SchedulingService.list_tasks`
(`Scheduling/services/scheduling_service.py:136-145`) only extends its unified task list with
`self.watchlist_projection.list_jobs(...)` (line 140) -- it has no equivalent call into a
`BriefingProjection`, and `app.py` never constructs `SchedulingService` with one (only
`watchlist_projection` is passed at its construction site, `app.py:4717-4721`). A user who opens
the Scheduling screen therefore sees their reminders and watchlist checks, but never a scheduled
briefing job, even though one is due, running, or has just completed -- the only way to confirm a
briefing schedule exists at all today is the Watchlists Artifacts pane's own cadence picker and
scope-label copy (`UI/Watchlists_Modules/artifacts_pane.py`), not the screen whose entire purpose
is to list scheduled work.

This was deliberately out of phase 4 task 3's scope (it only touched the scheduler-dispatch seam,
`queue.py`/`app.py`), not an oversight discovered here for the first time -- see that task's own
report for the same observation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A watchlist with a non-NULL `briefing_cadence_seconds` shows up as a scheduled task on the Scheduling screen, alongside reminders and watchlist checks
- [ ] #2 The projected briefing task's next-run time on the Scheduling screen matches `BriefingProjection.list_jobs`'s own calculation (last completed run + cadence, or immediate if never run)
- [ ] #3 No new `persist_event` names are introduced (mirror the existing `log_counter`/`log_histogram`-only observability this stream already uses for briefings and watchlist checks)
- [ ] #4 A watchlist with a NULL cadence (scheduling off) does not appear on the Scheduling screen
<!-- AC:END -->
