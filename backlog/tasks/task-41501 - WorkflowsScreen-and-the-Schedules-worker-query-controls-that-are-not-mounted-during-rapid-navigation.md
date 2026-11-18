---
id: TASK-41501
title: "WorkflowsScreen and the Schedules worker query controls that are not mounted during rapid navigation"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:21
labels:
- follow-up
- ui
- workflows
- schedules
priority: medium
references:
- tldw_chatbook/UI/Screens/workflows_screen.py
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15). `WorkflowsScreen._show_status` (introduced by c7fe03ab08) queries controls that are not mounted and loops on `NoMatches`. In the libtests fix round the Schedules worker raised `NoMatches #scheduling-task-table` during tab-switch storms too, so `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` had to stay off its lane.

**Evidence.** `test_overlapping_navigate_requests_complete_in_fifo_order` fails 2 of 3 on the Workflows `NoMatches` loop; `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` fails on the Workflows and Schedules `NoMatches`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A status update that arrives before or after its screen's controls are mounted is dropped or deferred without raising or looping
- [x] #2 `test_overlapping_navigate_requests_complete_in_fifo_order` passes 3 of 3 under a plain run
- [x] #3 `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` passes 3 of 3 and is admitted to a PR lane
<!-- AC:END -->

## Implementation Plan

1. Confirm the defect at base and locate the window (running vs shutdown) with a temporary state probe at each failing query (reverted).
2. Workflows: one liveness check (attached and not pruning) at the entry points that touch the authoring controls after an await or from a deferred callback: `_show_status`, `_render_authoring`, `_show_library`, `_layout_panes`, the mount-load worker after each await, and its init failure branch. Replace the `is_mounted` checks there, which never turn False.
3. Schedules: the same check at the top of `_render_table`, the query the load worker reaches after its awaits. State assignments in `load_tasks` stay before it.
4. Deterministic tests in `Tests/UI/test_screen_navigation.py`, red at base: entry points on a removed Workflows screen; the mount worker with the exit fence set while authoring init is held; Schedules `_render_table` on a removed screen.
5. Storm and FIFO nodes 3 of 3 under a plain run; lane the storm (and FIFO) with measured serial times.

## Implementation Notes

**Where the `NoMatches` came from.** A temporary probe (reverted) at each failing query, base 67583dfa72:
- **Storm: while the app runs.** Workflows' `call_after_refresh(self._refresh_authoring)` ran after Workflows had unmounted (`attached=False`, `pruning=True`, `closed=True`, current screen Home or Schedules, `_exit` False). Textual hands `call_after_refresh` callbacks to `app.screen`, so a screen's deferred callback runs on the screen that replaced it, and its `is_mounted` checks (which never turn False) let it query `WorkflowLibrary`. The app died mid-storm, so the last hop never landed (`'LibraryScreen' == 'HomeScreen'`), 3 of 3 at load 15.
- **FIFO: at shutdown.** The test ends with Workflows still loading. The exit fence makes the screen read detached, so `recompose` returns without building the authoring panes, and the worker carried on with a controller and the unavailable layout. The load failed on the missing editor, reported "Definitions could not be loaded", and the deferred `_show_status` raised `NoMatches '#workflows-editor'` at teardown (1 of 4 base runs).
- **Schedules: while the app runs.** Once the Workflows crash no longer ended the storm first, Schedules showed the same class. Its load and reachability workers reached `_render_table`, `_refresh_owner_select` and the follow button after their awaits while the screen was being removed. Textual cancels a node's workers only at its own Unmount, after its children are gone. A Schedules screen removed before it composed also ran `on_mount` against no children (`NoMatches '#scheduling-sync-status'`).
- **No loop.** Each `NoMatches` was a single unhandled exception that ended the app.

**Fix.** Liveness checks; nothing catches an exception. Each screen gets a `_controls_live()` check (`is_attached and not _pruning`; Workflows also keeps `is_mounted`).
- Workflows: `_show_status`, `_render_authoring`, `_show_library`, `_layout_panes` and the mount-load worker after each await use it in place of `is_mounted`.
- Schedules: `on_mount` returns before starting timers or workers on a screen removed before it composed. `_render_table`, `_refresh_owner_select` and `_update_follow_button_state` return early. `load_tasks` still records its rows, before the render.
- Deviation from the plan: no check on Workflows' init-failure status line or on its session attach after a failed load. Neither was seen, and that worker runs with `exit_on_error=False`.

**Tests** (`Tests/UI/test_screen_navigation.py`):
- `test_workflows_and_schedules_updates_after_removal_touch_no_controls` removes both screens through real navigation, then calls each update. Red at base: `NoMatches` from 7 of its 8 calls (Workflows refresh, library and layout; Schedules table, follow, owner and mount). Green after.
- `test_workflows_load_after_the_exit_fence_posts_no_status` holds authoring init, sets `_exit`, then releases it. Red at base: `NoMatches '#workflows-editor'`. Green after.

**Acceptance runs.**
- AC#2 and AC#3: storm, FIFO and both new nodes, 3 serial rounds with the lane flags: 6 of 6 each round, load 14-19.
- The same four nodes ×10 each (a temporary parametrize, removed afterwards) at `-n 4`: 40 passed, load 10-15. Before the splash change, storm and FIFO ×10 each also passed 20 of 20 at load 16-17.
- Lane (AC#3): storm and FIFO plus both new nodes are at the census tail. Serial lane-flag call times: storm 7.1-7.3 s, FIFO 3.4-4.3 s, removal 3.8-4.3 s, exit fence 2.7-3.3 s; setup and teardown under 0.01 s.

**Covering runs.**
- `Tests/UI/test_schedules_*`, `test_workflows_*`, `test_destination_*`, `test_ux_batch5-7`, `test_console_live_work_handoffs`, `Tests/Scheduling` (two files), `Tests/Workflows`, `test_ui_latency_guardrails`, the worker-contract script tests and `test_application_state_ownership`, all at `-n 4`: 1752 passed, 430 failed, 6 errors. 369 of the 377 failure sections are the local `RecoveryRequired` trap. The 14 other nodes fail the same way, with the same messages, at base.
- Because RR hid most of those files, they ran again under a temporary `bootstrap_profile` marker (removed afterwards), at head and at base: 33 failed at head, 40 at base (load 11-33). The six head-only nodes, rerun 3× on each side: the same four flake at base too, and the other two passed 3 of 3 at head.
- Serial: the size ratchets, the Architecture inventories that name these modules, and the census and shard tests. 498 passed, 2 failed. The two failures are `test_timer_path_static_update_inventory.py`'s `project_index_binding_card.py` clock root, a file unchanged since dev 97f8c58c0c (proposed as a follow-up).
- `preflight.sh`: all checks passed, with the worker contract at "none new".

**Ratchet.** `schedules_workbench.py`: 5530 -> 5544 lines (+14, measured); the budget moves 5532 -> 5544 with a dated note.

**Files.** `tldw_chatbook/UI/Screens/workflows_screen.py`, `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`, `Tests/UI/test_screen_navigation.py`, `Tests/Architecture/test_module_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md` (a `call_after_refresh` entry gains the "outlives its own screen" paragraph).
