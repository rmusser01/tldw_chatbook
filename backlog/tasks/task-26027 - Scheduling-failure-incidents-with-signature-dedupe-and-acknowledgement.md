---
id: TASK-26027
title: 'Scheduling: failure incidents with signature dedupe and acknowledgement'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:46'
updated_date: '2026-09-01 20:36'
labels:
  - scheduling
  - ux
dependencies:
  - TASK-26026
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A repeatedly failing task re-notifies on every occurrence. Verified on origin/dev: a failed dispatch writes last_status and dispatches a notification through the handler path (Scheduling/scheduler/handlers/reminder_handler.py:20-30); a named grep for incident across tldw_chatbook/Scheduling and UI/Screens/scheduling returns zero, and the only aggregation is a count on the Home dashboard (Home/dashboard_state.py:1030). A task failing hourly for a week is a week of identical notifications with no way to say "I know". Hermes groups failures by job and normalized error signature into durable incidents with detected, alerted and closed states plus per-signature acking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated failures of the same task with the same error signature are grouped into one incident rather than notified individually
- [x] #2 An incident can be acknowledged, which suppresses further notifications for that signature until it recurs after a resolution
- [x] #3 A different error signature on the same task opens a distinct incident rather than folding into the acknowledged one
- [x] #4 An incident closes automatically when the task next succeeds
- [x] #5 Error signatures are normalized so that varying details (timestamps, ids, paths) do not defeat grouping - asserted by tests
- [x] #6 Incident state is durable across restarts
- [x] #7 Acknowledging suppresses notification only; it never disables the task or hides it from the queue
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure normalize_error_signature (strip ts/uuid/hex/paths/numbers, bounded)\n2. v4_to_v5 migration: task_incidents table + partial-unique open index\n3. State machine on the DB: record_task_failure (open/group + should_notify), record_task_success (close), acknowledge_incident, list_task_incidents\n4. Briefing handler: gate _notify_error via should_notify, close on complete; app.py wires the DB as recorder\n5. version pins, guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
normalize_error_signature (Scheduling/task_incidents.py) strips ISO timestamps, uuids/hex, posix+windows paths, and durations/numbers, then bounds to 200 chars (hash tail for over-long) — so same-shape failures group and a different error class doesn't (AC#5, pinned). Durable task_incidents table (ScheduledTasks v4->v5) with a PARTIAL UNIQUE index on (task_id, signature) WHERE status!='closed' enforcing one open incident per signature at the DB level. State machine on the DB: record_task_failure returns (incident_id, should_notify) — new signature opens 'alerting' + notifies (AC#1), a repeat of any open incident (alerting OR acknowledged) bumps occurrence_count and does NOT re-notify (AC#1/#2), a different signature opens its own (AC#3); record_task_success closes all open incidents for the task (AC#4); acknowledge_incident sets 'acknowledged' touching ONLY the incident row (AC#7 — no task mutation); durable across restart (AC#6, pinned). Wired into BriefingJobHandler (the confirmed per-occurrence failure notifier via _notify_error): _should_notify_failure records+decides, _close_incident on STATUS_COMPLETE; default (no recorder) = today's always-notify (pinned). app.py injects self.scheduling_service.db as the recorder. NEAR-MISS CAUGHT: handler first called recorder.record_failure but the DB method is record_task_failure — a silent no-op in prod (except->always-notify) that the test's fake adapter hid; fixed by aligning names AND passing the REAL ScheduledTasksDB as the recorder in the handler test (matches app.py exactly). Migration not in the DB/ index-pin census (Scheduling/db scope). Remaining follow-on: an ack UI button (the AC surface for acknowledging) — the durable ack path + state machine are complete and tested; a Task Detail 'Acknowledge' button is a small addition. 13 new tests; scheduler 398 + schedules UI 41 green.
<!-- SECTION:NOTES:END -->
