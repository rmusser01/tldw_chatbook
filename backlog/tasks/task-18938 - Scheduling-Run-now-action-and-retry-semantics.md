---
id: TASK-18938
title: "Scheduling: \"Run now\" action and honest retry semantics"
status: To Do
assignee: []
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - parity
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the manual-run gap found by the TASK-18936 parity audit (hermes has manual-run with attachments; chatbook has none). Add a "Run now" action to the Schedules workbench: dispatch the selected task immediately through the same handler path the scheduler loop uses (`SchedulerLoop.tick`'s dispatch sequence — handler → `mark_reminder_dispatched`), bypassing the poll wait. "Run now" on a recurring task must compute and persist the next occurrence (i.e. it is a real dispatch, not a preview); on a one_time task it consumes the task exactly as a scheduled firing would.

This also gives the workbench honest retry semantics: a task whose `last_status` is `missed` (handler raised) offers Run now as its retry — the deprecated `SchedulesScreen` rendered disabled "Retry run" buttons that were never wired; the routed workbench should ship the real thing. Keyboard binding per ADR-031 conventions, footer hint only for the implemented action. The action must respect the same guards as scheduled dispatch (enabled check is NOT one of them — running a disabled task manually is a legitimate user intent; decide and pin that explicitly).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A "Run now" action exists in the workbench (keyboard + button) and dispatches the selected task through the same handler/`mark_reminder_dispatched` path as the scheduler loop — no parallel dispatch code path
- [ ] #2 Run now on a recurring task persists the next occurrence (real dispatch semantics); on a one_time task it consumes the task identically to a scheduled firing; both pinned by tests using the real DB path
- [ ] #3 The disabled-task question is decided and pinned (recommendation: Run now works on disabled tasks, labeled honestly, since manual intent outranks the schedule)
- [ ] #4 Tasks with `last_status="missed"` surface Run now as their retry affordance, replacing the never-wired "Retry run" concept from the deprecated screen
- [ ] #5 Binding follows ADR-031 (single-letter screen action, no terminal-convention keys); footer hint added only for this implemented action
- [ ] #6 Tests cover recurring/one_time consumption, disabled-task behavior, missed-retry flow, and no-duplicate-dispatch (a manual run does not double-fire the pending scheduled occurrence)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: new user action over the existing dispatch path; no schema/boundary change.

1. Factor the loop's dispatch sequence into a reusable unit (handler + mark dispatched) shared by tick and manual run
2. Workbench action + binding + detail-pane affordance; missed-retry surfacing
3. No-duplicate-dispatch guard for the pending occurrence
4. Tests + schedules.md docs
<!-- SECTION:PLAN:END -->
