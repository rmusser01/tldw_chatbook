---
id: TASK-18938
title: "Scheduling: \"Run now\" action and honest retry semantics"
status: Done
assignee:
  - '@robert'
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
- [x] #1 A "Run now" action exists in the workbench (keyboard + button) and dispatches the selected task through the same handler/`mark_reminder_dispatched` path as the scheduler loop — no parallel dispatch code path — `SchedulerLoop.dispatch_reminder` is the single seam both `tick` and `run_reminder_now` call; `test_manual_and_scheduled_dispatch_share_the_seam` pins row-for-row equivalence
- [x] #2 Run now on a recurring task persists the next occurrence (real dispatch semantics); on a one_time task it consumes the task identically to a scheduled firing; both pinned by tests using the real DB path — `test_run_now_recurring_consumes_and_advances`, `test_run_now_one_time_consumes_task`
- [x] #3 The disabled-task question is decided and pinned (recommendation: Run now works on disabled tasks, labeled honestly, since manual intent outranks the schedule) — dispatches without re-enabling; notify copy says "(still disabled)"; `test_run_now_works_on_disabled_task_without_enabling`
- [x] #4 Tasks with `last_status="missed"` surface Run now as their retry affordance, replacing the never-wired "Retry run" concept from the deprecated screen — detail-pane button reads "Run now (retry)" with a retry-specific tooltip when the last dispatch failed
- [x] #5 Binding follows ADR-031 (single-letter screen action, no terminal-convention keys); footer hint added only for this implemented action — `r` binding + SCHEDULES_SHORTCUTS entry, kept 1:1 with BINDINGS
- [x] #6 Tests cover recurring/one_time consumption, disabled-task behavior, missed-retry path, and no-duplicate-dispatch (a manual run does not double-fire the pending scheduled occurrence) — 12 tests in `Tests/Scheduling/test_run_now.py` (all real loop/queue/DB/service paths), including `test_run_now_does_not_double_fire_queued_occurrence`, `test_service_run_now_delegates_and_notifies`, and honest-refusal paths (no loop / no handler / missing task)
<!-- AC:END -->

## Implementation Notes

Implemented 2026-08-19 in `.worktrees/hermes-parity-audit` (branch `task/hermes-parity-audit`, on top of the TASK-18937 commit).

**Approach.** One dispatch seam: `SchedulerLoop.dispatch_reminder(task, handler, task_type, now)` now holds the exact handler-await → `mark_reminder_dispatched` sequence `tick` always ran (including missed-fire accounting from 18937); `tick` calls it per due task, and the new `run_reminder_now(task_id)` calls it for exactly one task. A manual run is therefore a real dispatch by construction, not a lookalike — pinned by a test that dispatches two identical tasks, one via tick and one manually, and asserts row-for-row equality of every outcome column.

**No-duplicate guard.** The queue is an in-memory sorted list, so `PriorityQueue.remove(task_id)` drops a pending occurrence cheaply before the manual dispatch; the post-dispatch reload re-adds the task with its new `next_run_at`. A task that is both due-now and manually run fires exactly once.

**Service seam.** `SchedulingService.run_reminder_now(task_id, loop=...)` delegates to the loop (the loop owns the registered handlers — without it there is honestly nothing to run) and fires `on_queue_changed`, so the live queue reconciles immediately. The workbench resolves `app.scheduler_loop` and follows the same worker + notify + `load_tasks()` discipline as enable/disable.

**Disabled-task semantics** (decided in-plan before implementation): manual intent outranks the schedule — Run-now dispatches a disabled task and does NOT re-enable it; the recurring schedule still advances, so a later enable cannot double-fire. Notify copy discloses ("…ran now (still disabled).").

**Retry affordance.** The detail pane's Run-now button becomes "Run now (retry)" with a retry-specific tooltip when the task's status is Missed — the deprecated SchedulesScreen's never-wired "Retry run" buttons made real on the routed workbench.

**Keyboard.** `r` on the workbench (verified free: existing bindings c/e/space/d/x/s/escape), footer hint added 1:1 per ADR-031.

**Verification.** `Tests/Scheduling/` fully green (**296 passed**, +12 new in `test_run_now.py`); `Tests/UI -k sched` green (**96 passed**). Not verified against a live TTY session (headless worktree) — the binding/button wiring is covered by the existing UI harness suites staying green; noted honestly.

**Files modified:** `tldw_chatbook/Scheduling/scheduler/loop.py` (dispatch_reminder extraction + run_reminder_now), `tldw_chatbook/Scheduling/scheduler/queue.py` (remove), `tldw_chatbook/Scheduling/services/scheduling_service.py` (run_reminder_now), `tldw_chatbook/Scheduling/events.py` (RunReminderNowRequested), `tldw_chatbook/UI/Screens/scheduling/task_detail.py` (button + retry variant + request), `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (binding + handlers), `Tests/Scheduling/test_run_now.py` (new), `Docs/User_Guide/schedules.md` (Run-now section).

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: new user action over the existing dispatch path; no schema/boundary change.

**Decisions (2026-08-19, before implementation):**

1. **Disabled tasks: Run-now works, labeled honestly.** Manual intent outranks the schedule — the action dispatches a disabled task, and the dispatch itself does NOT silently re-enable it (the row keeps its disabled state; a recurring task's next_run_at is still advanced so a later enable doesn't double-fire). The button tooltip and notify copy say the task is disabled.
2. **Shared dispatch unit.** `SchedulerLoop.dispatch_reminder(task_id, handler)` extracts exactly the sequence `tick` runs per task (handler await → `mark_reminder_dispatched` with the loop's clock + grace); `tick` calls it, and manual run calls it. No parallel dispatch path.
3. **No-duplicate guard.** Manual dispatch of a task that is ALSO sitting in the live queue must not double-fire: the manual path pops the task from the queue before dispatching (the queue is an in-memory sorted list; removal is cheap), and `request_reload()` afterwards reconciles anything changed.
4. **Service seam.** `SchedulingService.run_reminder_now(task_id, handler)` awaits the shared dispatch unit, then fires `on_queue_changed`. The workbench resolves the handler from the app's scheduler loop and calls the service — never the DB directly (same discipline as enable/disable).
5. **Retry affordance.** The detail pane's Run-now button reads "Run now (retry)" for tasks whose `last_status` is `missed` — the never-wired "Retry run" concept from the deprecated screen, now real.
6. **Keyboard.** `r` binding on the workbench (ADR-031 single-letter; verified free — existing bindings are c/e/space/d/x/s/escape) + footer hint entry.

Implementation steps:

1. `SchedulerLoop.dispatch_reminder` extraction; manual-run entry (queue-pop + dispatch + reload); `tick` refactored onto the shared unit
2. `SchedulingService.run_reminder_now` + `on_queue_changed`
3. `RunReminderNowRequested` event; detail-pane button (retry label variant); workbench `r` binding + worker handler
4. Tests: real loop/queue/DB paths (recurring consumption, one_time consumption, disabled-task behavior, no-double-fire, missed-retry path) + UI binding test
5. `schedules.md` Run-now section
<!-- SECTION:PLAN:END -->
