---
id: TASK-35564
title: The Schedules queue does not log a local-mode server refusal as an error
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- schedules
priority: low
references:
- tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
- tldw_chatbook/Scheduling/services/sync_engine.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

On a fresh local profile, opening Schedules logs an ERROR with two chained tracebacks: "Failed to load server automations for the Queue list (server_id=http://127.0.0.1:8000)". The cause is a `PolicyDeniedError`: "scheduler.automations.list.server requires server mode". `SchedulesWorkbench._load_queue_definitions` treats this expected "not applicable" outcome as a failure.

`_server_configured` is true there because the template's placeholder [tldw_api] URL is set, which is the task-31798 case. Scheduling/services/sync_engine.py already logs the same refusal as "not applicable" (the task-2722 precedent). The queue still shows the local rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Schedules in local mode logs no ERROR and no traceback for the server-mode refusal; the refusal is logged at most once at info level as not applicable
- [ ] #2 The queue still shows the local rows, and a real server failure is still logged as an error
- [ ] #3 A test covers the refusal: only local rows return, and nothing is logged at ERROR level
<!-- AC:END -->
