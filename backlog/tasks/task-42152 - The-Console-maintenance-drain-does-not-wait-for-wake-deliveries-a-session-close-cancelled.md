---
id: TASK-42152
title: "The Console maintenance drain does not wait for wake deliveries a session close cancelled"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:51
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
- [x] #1 A maintenance drain that starts while a session close's cancelled wake delivery is still settling waits for it, within the drain's deadline, before it reports drained
- [x] #2 The drain still holds no strong reference to a delivery that never settles, and returns False at its deadline as before if one never does
- [x] #3 A test holds a fenced delivery's settlement and shows the drain waiting for it; it is red before the change
<!-- AC:END -->

## Implementation Plan

1. Make `ConsoleChatController.maintenance_drain` also wait on the coordinator's `_fenced_delivery_tasks` members that are not done, read under the coordinator's registry lock, without keeping a reference between polls.
2. Add a test beside TASK-42074's fence pins in `Tests/UI/test_console_launch_wake.py` (UI census lane) that holds a fenced delivery's settlement and shows the drain waiting for it, plus the deadline case for a delivery that never settles; show it red first.
3. Run the touched files at -n 4, the size ratchets, preflight and the ruff gates.

## Implementation Notes

`ConsoleChatController.maintenance_drain` now also waits while `ConsoleFleetWakeCoordinator.fenced_delivery_settling()` is true. That new method reports whether any member of `_fenced_delivery_tasks`, the weak set TASK-42074 added, is still not done. It reads the set under the coordinator's `_registry_lock`, the lock `fence_conversation` fills it under, and keeps nothing between polls. Finished members are skipped: the weak set keeps a settled task while something else still holds it, which would otherwise hold the drain until its deadline.

**Test.** `Tests/UI/test_console_launch_wake.py::test_maintenance_drain_waits_for_a_delivery_a_session_close_cancelled` (UI census lane), beside TASK-42074's weak-handle pin. It uses a real controller after `maintenance_close_admission()`:
- A delivery whose `finally` waits on an event is registered and fenced. The drain (5 s deadline) must still be running 0.2 s later, and returns True once the event is set and the delivery is done.
- A delivery that swallows every cancellation is fenced. A 0.1 s drain returns False, and a weak reference to the task is dead after the test drops its own (with the `sleep(0)` the TASK-42074 lesson describes).

**Red/green.**
- At base: red on its first assertion ("the drain reported drained while the delivery a session close cancelled was still settling its attempt"; the drain task had already finished with True). With the fix: green.
- Negative control for AC#2: `fenced_delivery_settling` temporarily keeping `list(self._fenced_delivery_tasks)` on the coordinator made the weak-reference assertion fail ("the drain kept a strong reference to the delivery"); reverted. A first version of the test let that failure hang its teardown until pytest-timeout, because the stuck task ignored the loop's final cancellation. The task now stops once the test's `finally` has run.

**Checks.**
- `test_console_launch_wake.py`, `test_console_fleet_wake_safety.py`, `test_fleet_wake_retry.py`, and the UI maintenance-drain files (`test_console_final_poll_completion.py`, `test_console_sync_worker_admission.py`, `test_console_tab_publication_maintenance.py`) at `-n 4`: 41 passed (load 45-53).
- `Tests/Architecture/test_module_size_ratchet.py` serially: 209 passed. The controller grows one line, inside its existing budget, so no row changes.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` serially: 79 passed, 1 failed. The failure, `test_reviewed_diagnostic_changes_are_metadata_only`, is about `Agents/agent_service.py` ("expected one diagnostic containing 'on_child_settled consumer raised', found 0"), which this task does not touch. It also fails with the two Console modules checked out at base. A first run of this file at load 50 reported 2 failed.
- `ruff check tldw_chatbook --select F`: clean.

**Files:** `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_fleet_wake.py`, `Tests/UI/test_console_launch_wake.py`.
