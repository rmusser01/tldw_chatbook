---
id: TASK-26028
title: 'Scheduling: per-handler preflight validation before dispatch'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:46'
updated_date: '2026-09-01 20:39'
labels:
  - scheduling
  - reliability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing checks whether a task can succeed before firing it. Verified on origin/dev: validation happens at creation, and a grep for preflight across tldw_chatbook/Scheduling returns zero - at dispatch the loop calls the handler directly (Scheduling/scheduler/loop.py:342-363), so a watchlist whose source was deleted or a briefing whose provider key was removed fails at run time, repeatedly, on schedule. Hermes validates provider key, delivery target and skill availability before firing and alerts once per condition rather than once per occurrence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A handler may declare a preflight check that runs immediately before dispatch
- [x] #2 A failed preflight records a distinct outcome from a handler failure, so the cause is legible
- [x] #3 A failed preflight does not consume the occurrence in a way that hides the problem - the task remains visible as needing attention
- [x] #4 The user is told once per condition, not once per occurrence, composing with the incident grouping
- [x] #5 Handlers without a preflight check dispatch exactly as today
- [x] #6 Preflight is bounded in time and cannot itself wedge the loop
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: failed preflight skips handler + distinct outcome; repeats group; passing preflight dispatches; no preflight = today\n2. Loop _run_preflight (duck-typed handler.preflight, sync/async, time-bounded, never-raise) + _record_preflight_incident\n3. dispatch_reminder runs preflight after ledger-begin, before handler; on fail: ledger 'preflight_failed' + incident + mark failed + skip handler\n4. guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A handler may declare an optional preflight(task) -> (ok, reason) (sync or async); SchedulerLoop._run_preflight runs it duck-typed, bounded by _PREFLIGHT_TIMEOUT_SECONDS=10s (AC#6), and never raises out (a broken/erroring preflight proceeds to normal dispatch — fail-open, so a bug in a check can't block real work). In dispatch_reminder the preflight runs after the run-ledger opens and BEFORE the handler: on failure it finishes the ledger row as 'preflight_failed' (AC#2 — distinct from a handler 'failed'), records a grouped incident via record_task_failure with signature 'preflight: <reason>' (AC#4, composes with 26027 — repeats group into one incident, pinned), marks the reminder dispatched=False so it stays visible as needing attention (AC#3), and returns without running the handler (pinned: handler never runs). Server-scoped rows excluded from the incident (ADR-077). AC#5: no .preflight attribute => byte-identical dispatch (pinned). No migration — reuses the 26027 incident table. No handler ships a preflight yet; this is the seam handlers opt into (watchlist source-exists / briefing provider-key checks are the intended first consumers, a small follow-on). 4 new tests; scheduler suite 402 green.
<!-- SECTION:NOTES:END -->
