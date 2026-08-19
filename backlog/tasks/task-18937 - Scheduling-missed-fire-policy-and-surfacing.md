---
id: TASK-18937
title: 'Scheduling: missed-fire policy and surfacing'
status: To Do
assignee: []
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - parity
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the missed-fire gap found by the TASK-18936 parity audit (hermes ships missed-fire surfacing and cron continuity flags; chatbook has neither). Verified current behavior (probe evidence in TASK-18936): a reminder due while the app was closed fires once, late, and records `last_status="completed"` — lateness is invisible; an overdue recurring task collapses N owed occurrences into one late dispatch with next-run re-derived from dispatch time; `missed_at` exists in the schema but no code path writes it; `"missed"` status means only "handler raised".

Decide and implement a missed-fire policy: on reopen, tasks whose stored `next_run_at` is materially in the past should be surfaced honestly — record the actual owed-occurrence count and lateness, write `missed_at` when occurrences elapsed undispatched, and show a "missed while away" state in the Schedules Queue tab and task-detail pane (distinct from failed: the work never ran, as opposed to ran and raised). Catch-up semantics (re-running every missed occurrence vs run-once-then-continue, the current implicit behavior) are a product decision to record in the task before implementation; the recommendation is run-once-then-continue for reminders (matches user expectation of "at least it told me") with the missed count surfaced, not replayed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The missed-fire policy is decided and recorded in this task before implementation (catch-up vs run-once; what counts as materially late; one_time vs recurring rules)
- [ ] #2 On reopen, a task with elapsed undispatched occurrences records honest state: owed-occurrence count and/or lateness persisted, and `missed_at` is written on that path (the column stops being dead schema)
- [ ] #3 The Queue tab and task-detail pane show a distinct "missed while away" state that is visually and semantically separate from failed (never-ran vs ran-and-raised), with the owed count where applicable
- [ ] #4 The recurring next-run re-derivation rule is made a deliberate, documented choice (from dispatch time vs from schedule) rather than an accident of `mark_reminder_dispatched`'s implementation
- [ ] #5 Behavior is pinned by tests using the real `ScheduledTasksDB` + `PriorityQueue` + dispatch path (seeded overdue one_time and recurring cases per the TASK-18936 probe), not a reimplementation
- [ ] #6 Docs updated: `Docs/User_Guide/schedules.md` (still a stub — extend it) documents the policy and the missed/failed distinction
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: execution-policy refinement within ADR-018's existing local-first module; no schema/interface boundary change beyond populating an existing column. If the chosen policy turns out to require new sync semantics (missed-state reconciliation with the server), raise it then.

1. Decide the policy (catch-up vs run-once; lateness threshold; one_time/recurring rules) and record it here
2. Startup/first-tick detection of elapsed-owed occurrences; persist count + `missed_at`
3. Queue-tab and task-detail "missed while away" rendering
4. Real-path tests from the TASK-18936 probe shape; docs
<!-- SECTION:PLAN:END -->
