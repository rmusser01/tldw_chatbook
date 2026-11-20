---
id: TASK-42074
title: "A quit during a Console session close does not wait for the wake deliveries the close cancelled"
status: To Do
assignee: []
created_date: 2026-11-20 02:35
updated_date: 2026-11-20 02:35
labels:
- follow-up
- console
priority: low
references:
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Chat/console_runtime.py
- Tests/Chat/test_console_fleet_wake_safety.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-42028 (round 17). TASK-42028 makes quit wait, within the shared dispose deadline, for wake deliveries that `ConsoleFleetWakeCoordinator.dispose()` cancelled, before it retires AgentRuns.

`fence_conversation` (session close) still removes the deliveries it cancels from `_delivery_tasks` straight away. `Tests/Chat/test_console_fleet_wake_safety.py` pins that removal. So when the app quits while a closed session's cancelled delivery is still settling its attempt, the quit does not wait for it. The settlement can then race the AgentRuns retirement: at quit this is logged as "SQLite retirement incomplete", and the attempt may be left for the next start's `recover()`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A quit that starts while a session close's cancelled wake delivery is still settling waits for that delivery, within the dispose deadline, before AgentRuns is retired
- [ ] #2 A test drives a quit during that settlement window and fails without the fix
- [ ] #3 A session close still releases its fence on a graceful close and still keeps no strong reference to a delivery that never settles; `test_session_fence_releases_uncooperative_delivery_task_reference` stays green
<!-- AC:END -->
