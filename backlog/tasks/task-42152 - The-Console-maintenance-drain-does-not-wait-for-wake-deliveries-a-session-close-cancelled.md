---
id: TASK-42152
title: "The Console maintenance drain does not wait for wake deliveries a session close cancelled"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- console
- backup
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consolerun stream (TASK-42074, round 18). `ConsoleChatController.maintenance_drain` (`tldw_chatbook/Chat/console_chat_controller.py:6012`) reports drained once `_maintenance_calls`, `_maintenance_background`, `_fleet_wake._delivery_tasks` and `_chat_start.tasks()` are empty. `ConsoleFleetWakeCoordinator.fence_conversation` (`Chat/console_fleet_wake.py:1651`) removes the deliveries a session close cancels from `_delivery_tasks` straight away (`:1676`), while each still settles its attempt in AgentRuns from its `finally`. So a backup's maintenance drain can report drained while such a delivery is still writing.

TASK-42074 (round-18 consolerun branch, not yet on dev) closes the same gap for quit: fenced deliveries go into a `weakref.WeakSet` (`_fenced_delivery_tasks`) that `ConsoleRuntime._dispose_owned` joins. `maintenance_drain` does not read it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A maintenance drain that starts while a session close's cancelled wake delivery is still settling waits for it, within the drain's deadline, before it reports drained
- [ ] #2 The drain still holds no strong reference to a delivery that never settles, and returns False at its deadline as before if one never does
- [ ] #3 A test holds a fenced delivery's settlement and shows the drain waiting for it; it is red before the change
<!-- AC:END -->
