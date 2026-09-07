---
id: TASK-1810
title: Scheduled briefings are invisible to the Scheduling screen
status: Done
assignee: []
created_date: '2026-08-01 18:25'
updated_date: '2026-08-02 08:44'
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
- [x] #1 A watchlist with a non-NULL `briefing_cadence_seconds` shows up as a scheduled task on the Scheduling screen, alongside reminders and watchlist checks
- [x] #2 The projected briefing task's next-run time on the Scheduling screen matches `BriefingProjection.list_jobs`'s own calculation (last completed run + cadence, or immediate if never run)
- [x] #3 No new `persist_event` names are introduced (mirror the existing `log_counter`/`log_histogram`-only observability this stream already uses for briefings and watchlist checks)
- [x] #4 A watchlist with a NULL cadence (scheduling off) does not appear on the Scheduling screen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add optional `briefing_projection` to `SchedulingService.__init__`; `list_tasks` extends with its jobs the same way `watchlist_projection` does (`scheduling_service.py:136-145`).
2. Wire it at the `app.py` construction site (`~:4717`), passing the already-constructed `BriefingProjection` when briefing schedules are enabled.
3. Tests: cadence watchlist appears in the unified list with next-run matching `BriefingProjection.list_jobs`; NULL cadence absent; no new `persist_event` names (log_counter/log_histogram only).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Mirrored `watchlist_projection` exactly, per the plan:

- `SchedulingService.__init__` (`Scheduling/services/scheduling_service.py`) gains an optional
  `briefing_projection: BriefingProjection | None = None`, stored as `self.briefing_projection`.
- `list_tasks` grows one more guarded branch, parallel to the existing watchlist one:
  `if self.briefing_projection is not None: tasks.extend(self.briefing_projection.list_jobs(owner_id=self.owner_id))`.
  No re-derivation of `next_run_at` -- the projection's own `ScheduledTask` objects are
  extended straight into the unified list, so AC #2 ("matches `BriefingProjection.list_jobs`'s
  own calculation") holds by construction, not by coincidence.
- AC #3: no new `persist_event`/metrics calls were added -- `list_tasks` doesn't emit any
  observability today (for either projection), so there was nothing to mirror.

**Construction-order fix (app.py).** The task's own brief flagged that this exact method
(`_wire_watchlists_and_notifications_services`) had just shipped a construction-order bug
elsewhere (the kept-briefings branch) where a projection built *after* the consumer it needed
to feed froze `None` in forever. The same shape existed here: `briefing_projection` was built
several lines *after* `SchedulingService(...)` was already constructed. Fixed by moving
`briefing_schedules_enabled`'s read and `briefing_projection`'s construction up, immediately
before the `SchedulingService(...)` call, and passing it in directly. This is a pure reorder --
`briefing_projection` only ever depended on `subscriptions_db`, already created above both the
old and new call sites -- so `briefing_handler`'s construction (which does need to stay late,
since `chachanotes_db_getter`'s closure timing matters) was left exactly where it was, now
gated on `if briefing_projection is not None:` instead of re-reading the flag a second time.
`SchedulerLoop`'s own `briefing_projection if briefing_handler is not None else None` ternary
was left untouched (now a tautology given the shared gate, but harmless and lower-diff to leave
alone).

**Tests** (`Tests/Scheduling/test_scheduling_service.py`): mock-based structural mirrors of the
existing watchlist-projection tests (`test_list_tasks_includes_briefing_projection`,
`test_list_tasks_filters_briefing_by_owner`, plus a combined-both-projections test), and three
AC-driving tests against a real `SubscriptionsDB`/`BriefingProjection`: AC #1
(`test_list_tasks_includes_a_cadenced_briefing_schedule_ac1`), AC #2
(`test_briefing_schedule_next_run_at_matches_the_projection_ac2` -- asserts equality against a
second, independent `projection.list_jobs()` call, not a hand-derived expected value; uses a
seeded `complete` briefing so the comparison is deterministic rather than `now`-dependent), and
AC #4 (`test_null_cadence_watchlist_absent_from_scheduling_screen_ac4`). A seam-level liveness
test (`test_app_wiring_briefing_projection_is_live_not_a_frozen_none`) boots the real app via
`Tests/UI/app_factory._build_test_app` and asserts `app.scheduling_service.briefing_projection`
is a live `BriefingProjection`, identical to the one `SchedulerLoop`'s queue holds -- proving the
app.py reorder, not just the `SchedulingService` unit behavior.

`Tests/Scheduling/test_config_flags.py::test_briefing_projection_is_only_wired_when_the_flag_is_on`
pinned the *old* literal code pattern (`briefing_projection = None` inside
`if briefing_schedules_enabled:`) via source inspection; updated to pin the new (still
flag-gated) pattern instead, preserving the same behavioral intent (flag off -> both
`briefing_projection` and `briefing_handler` are `None`).

**Mutation checks** (Edit-revert cycles, `git status --short` clean before/after each):
- Dropped the new `list_tasks` branch entirely -> AC #1 test, the mock-based briefing-inclusion
  test, and the combined-projections test all REDed (3 failures); AC #4 unaffected. Restored.
- Broke the next-run "passthrough" by zeroing `next_run_at` on each projected task after
  `list_jobs()` returned (simulating a future re-derivation bug) -> only AC #2 REDed; AC #1 and
  AC #4 stayed green, confirming AC #2 is the one that actually discriminates a broken
  passthrough from a merely-present task. Restored.
- Hardcoded `briefing_projection=None` at the `SchedulingService(...)` call site (the exact
  frozen-None bug this task's brief described) -> the app.py wiring liveness test REDed.
  Restored.

**Files modified:**
- `tldw_chatbook/Scheduling/services/scheduling_service.py`
- `tldw_chatbook/app.py`
- `Tests/Scheduling/test_scheduling_service.py`
- `Tests/Scheduling/test_config_flags.py`

Full `Tests/Scheduling/` suite: 264 passed.
<!-- SECTION:NOTES:END -->
