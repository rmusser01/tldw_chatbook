---
id: TASK-1812
title: Briefing schedule gate/UX residuals
status: To Do
assignee: []
created_date: '2026-08-01 19:08'
labels:
  - watchlists
  - briefings
  - scheduling
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed during the whole-branch review fix wave for the Watchlists briefings phase 4 branch
(spec #2), bundling three minors the reviewer parked rather than blocking the wave on.

1. **The cadence picker ignores the kill switch.** `[scheduling] briefing_schedules_enabled`
   (`config.py:2370`, read only at `app.py:4738-4743`) gates whether `app.py` ever constructs the
   `BriefingProjection`/`BriefingJobHandler` pair that makes a schedule actually fire. Nothing gates
   the UI side: `cadence_scope_phrase` (`UI/Watchlists_Modules/artifacts_pane.py:331-361`) turns any
   non-NULL `briefing_cadence_seconds` into "scheduled &lt;cadence&gt; while the app is open"
   unconditionally, and the cadence `Select` itself is never disabled when the flag is off. There is
   no UI control for this flag today (hand-edit-only), so the gap is currently latent, but a
   watchlist can be fully configured to look scheduled while the process that would ever dispatch it
   was never wired up.
2. **A cadence pick has an undocumented activation delay.** `set_watchlist_briefing_settings`
   writes `briefing_cadence_seconds` synchronously the moment the picker changes, but the running
   `SchedulerLoop`'s `PriorityQueue` only re-reads `list_briefing_schedules` (via
   `BriefingProjection`) every `queue_reload_interval_ticks` ticks (`Scheduling/scheduler/loop.py:31`,
   default 60 -- roughly the ~30-minute reload cadence this same review's FIX 1 reasons about). A
   freshly-picked schedule can therefore sit inert for up to one reload cycle before the scheduler
   ever sees it. Neither the picker's own copy nor `Docs/User_Guide/watchlists.md`'s "Scheduled
   briefings" section states this.
3. **The zombie sweep's `exclude` is watchlist-granular, not row-granular.** `fail_interrupted_
   briefings`'s `exclude` (`Subscriptions/briefing_service.py:785-815`) skips every `generating` row
   for a watchlist id present in the collection, on the reasoning that such a row is "a LIVE,
   in-process generation" (the docstring's own words). That reasoning only holds if a watchlist can
   have at most one `generating` row at a time -- but a genuine crash-zombie row from a PRIOR process
   can coexist with a freshly-claimed live generation for the SAME watchlist (the crash predates the
   claim). When that happens, the live claim's presence in `exclude` incidentally shields the old
   zombie row too, so it survives until a sweep runs while that watchlist is NOT claimed -- the
   docstring over-claims what `exclude` actually protects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 When `[scheduling] briefing_schedules_enabled` is `false`, the Artifacts cadence picker and the scope label no longer imply an active schedule (disabled control, and/or copy stating scheduling is off at the app level), for a watchlist that already has a stored cadence
- [ ] #2 The cadence picker's UI copy or the user guide's "Scheduled briefings" section states that a newly picked cadence can take up to one queue-reload cycle to reach the running scheduler
- [ ] #3 `fail_interrupted_briefings`'s `exclude` (or its docstring) is corrected so a crash-zombie row is swept even when its watchlist has an unrelated live claim -- either by scoping the exclusion to the actual claimed briefing row rather than the whole watchlist, or by an accurate docstring plus a regression test pinning the coexistence case (a zombie row and a live claim on the same watchlist in the same sweep)
<!-- AC:END -->
