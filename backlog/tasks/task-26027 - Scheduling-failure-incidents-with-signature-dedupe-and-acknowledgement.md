---
id: TASK-26027
title: 'Scheduling: failure incidents with signature dedupe and acknowledgement'
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
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
- [ ] #1 Repeated failures of the same task with the same error signature are grouped into one incident rather than notified individually
- [ ] #2 An incident can be acknowledged, which suppresses further notifications for that signature until it recurs after a resolution
- [ ] #3 A different error signature on the same task opens a distinct incident rather than folding into the acknowledged one
- [ ] #4 An incident closes automatically when the task next succeeds
- [ ] #5 Error signatures are normalized so that varying details (timestamps, ids, paths) do not defeat grouping - asserted by tests
- [ ] #6 Incident state is durable across restarts
- [ ] #7 Acknowledging suppresses notification only; it never disables the task or hides it from the queue
<!-- AC:END -->
