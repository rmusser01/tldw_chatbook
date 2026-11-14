---
id: TASK-40716
title: "An unproven tool cleanup parks a worker forever, refusing the run's later tool calls and holding an app-wide tool slot"
status: In Progress
created_date: 2026-11-13 06:28
labels:
- follow-up
- agents
- console
- bug
priority: high
references:
- tldw_chatbook/Agents/agent_service.py
- tldw_chatbook/Agents/execution_capacity.py
- Tests/Chat/test_console_ssh_custody.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** When a tool operation's cleanup cannot be proven, the code keeps the worker alive on purpose by waiting on an event or future that nothing ever completes. The intent is that a resource that may still be live is never reused.
- Every later tool call in that run is then refused with `previous_tool_still_running`, because `Agents/execution_capacity.py:336-337` refuses while any tool operation is stopping.
- The parked operation keeps counting toward the app-wide `max_tool_workers` until the app restarts.

It happens with the workflow feature off. The `agent_service` park needs the ordinary tree's source scope (:2236-2242). Windows skips that tree (`Chat/console_ordinary_tree.py:67`), so there `:2375-2380` finishes the operation instead of parking it. Whether the other five parks are reachable on Windows has not been checked. A probe on macOS on 2026-11-12 (dev e0186fa10a) confirmed both effects. The five `Event().wait()` parks arrived with #3480.

**The parks:**
- `Agents/agent_service.py:2254-2258`, `retain_unknown_return()`. It is reached from :2322, :2333, :2369, :2371 and :2373 when a source scope or `operation.finish()` exits other than through the tool's own error. Ordinary Console tool calls set that scope (`Chat/console_agent_preparation.py:2306`).
- `Chat/console_raw_cli_custody.py:299-305`, `retain_unproven()`, for Raw CLI.
- `Tools/raw_cli_executor.py:1210-1213`, in `activate()`'s finally when the physical stop is not proven.
- `Tools/workspace_tool_executor.py:1622-1630`, in SSH `activate()`'s finally.
- `Workspaces/change_revert_ownership.py:292-294`, `retain_unknown()`, for change-review revert.
- `Workspaces/native_git.py:63-71`, `while True: await asyncio.shield(never)` when native Git cleanup proof fails.

**Tests that pin the park:**
- `Tests/Chat/test_console_ssh_custody.py:1217-1220` asserts that the worker is alive and `tool_workers == 1`.
- `Tests/Chat/test_console_raw_cli_custody.py:552`.
- `Tests/Workspaces/test_native_change_review_git_custody.py:302,362`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One unproven cleanup no longer makes every later tool call in the same run fail with `previous_tool_still_running`. Later calls either run, or the run ends in a terminal state that names the unproven cleanup.
- [ ] #2 The app-wide tool-worker slot held for an unproven operation is returned once the operation is proven finished or the user explicitly abandons it, without restarting the app.
- [ ] #3 No thread or task waits on an event or future that nothing can complete, and the app can still exit while a cleanup is unproven.
- [x] #4 A resource whose cleanup is unproven, including its held storage leases (the SSH test's `storage.live` and its `db.agent_runs` source), is still never reused while it may be live. The SSH, Raw CLI and native Git tests assert the new behaviour instead of a living parked worker.
<!-- AC:END -->

## Implementation Plan

ADR required: yes. The parks implement ADR-285's and ADR-084's "no proof retains the original worker/scopes/operation; this can hold disposal indefinitely", and AC#3 contradicts that sentence. ADR-134 (what a retained operation counts toward) and ADR-155 (the drain outcome after an abandon) need amendments too. Design first: the owner rules before any production change.

1. Reproduce (done): `Tests/Agents/test_unproven_tool_cleanup_run.py`, a strict xfail. An ordinary-tree source scope whose return fails parks the `tool-*` worker. The first call reports a timeout only after its whole deadline, the next call in the same run is refused `previous_tool_still_running`, and `tool_workers` stays 1 after the root finishes.
2. Owner ruling on the design: keep the resource held and release the run. One shared retention helper replaces the six parks. It marks the owner `cleanup_unproven` and registers a runtime-visible record. The worker waits on that record's event, which something can complete, so its scope stack (leases, scratch, claim) stays held exactly as today. The call returns at once with a terminal reason that names the unproven cleanup, and the run ends. The slot stays counted until a later proof or an explicit user abandon. Abandon finishes only the operation, so the drain reports unproven (ADR-155). Disposal stops waiting on retained records. Two sites run on the loop's default executor, which joins its threads at exit: Raw CLI (a Textual worker) and change-revert (`to_thread`). Those two move their scopes into the record instead of blocking the thread.
3. After the ruling: amend the ADRs, implement the helper, change the six sites, add the run-terminal reason and the abandon action, and convert the SSH, Raw CLI and native Git subprocess tests to assert the new behaviour. Remove the xfail from the reproduction test.
4. Owner ruling (2026-11-13): option A with four decisions. (1) The run ends at once with a new termination reason, for every run. (2) Abandon releases only the slot; the resource, leases and claim stay held until restart. (3) The abandon action sits on the run's explanation row. (4) At exit, disposal records retained items and stops waiting. The change-revert site also marks its owner unproven.
5. Deviation found while implementing step 2: the joined-pool sites cannot safely move their scopes into the record and return. Their scopes leave thread-bound state on the pool thread: Textual workers run in the pool thread's persistent context, so `_current_owner`/storage ContextVars would stay set for the next task on that thread, and `operation_owned_connection` keeps a thread-local owner set. Instead, Raw CLI and Undo now run their custody on a daemon worker, and the pool thread or awaiting coroutine returns once the call is retained. The two Undo sites whose scopes have already unwound retain the call without holding any thread.

## Implementation Notes

Implements the owner's 2026-11-13 ruling (option A).

**Behaviour.**
- One helper, `OwnedOperation.retain_unproven(label)` in `Agents/execution_capacity.py`, replaces all six parks. It marks the owner unproven, keeps the operation stopping (it still counts toward `max_tool_workers`), and registers an `UnprovenRetention` record on the `RuntimeCapacity`.
- A worker that still holds the original scopes calls `record.hold()`. This is always a daemon thread. Exit ends it. `prove()` sets the record's event for a late proof, but no producer calls it today.
- `abandon()` finishes only the operation, so the drain is unproven (ADR-155).
- `_call_with_timeout` returns within one poll slice with `RunTerminationReason.CLEANUP_UNPROVEN`. `agent_runtime` ends every run on that reason with "Agent stopped: …".
- The controller remembers the explanation row (`unproven_cleanup_runs`). `Chat/console_unproven_cleanup.py` maps rows to retained records and releases them. `Console_Modules/agent.py` pushes the rows on the sync tick. The transcript shows the Release tool slot button from `Widgets/Console/console_cleanup_release.py` under the row; the button handles its own press and then reads "Tool slot released …". Both new modules load only once such a row exists, so the UI-ready module census (dev is at its 1040 limit) is unchanged.
- Disposal stops waiting for retained items: `_drain_ordinary_turns` uses `OrdinaryTurn.unproven_only()`, the Raw CLI and change-tracking drains use `wait_retired_unless_retained`, and the change-review init join uses `join_unless_retained`.
- Change-revert now marks its owner unproven.

**Deviation (see plan step 5).** Raw CLI user commands and Undo now run their custody on a daemon worker instead of moving scopes into the record. Leaving their scopes behind on a reused pool thread would leak `_current_owner`/storage ContextVars (Textual workers keep the pool thread's context) and thread-local connection ownership into the next task on that thread.

**ADRs.** Owner-decided amendments appended to ADR-285, 084, 093, 094 and 134. ADR-155 and 356 §6 fit as they are.

**Tests.**
- `Tests/Agents/test_unproven_tool_cleanup_run.py` was the strict-xfail repro; it now passes. It covers the immediate return, the slot held until abandon, the unproven drain, the scopes still held, a late proof that unwinds them, and the run ending for any run.
- The SSH (6 modes), Raw CLI (3), native Git and Undo (2) subprocess tests assert the run end, the retention record, the release (SSH goes through the controller's row), the slot returning to 0, prompt disposal, and storage/claim still held.
- New `Tests/UI/test_console_unproven_cleanup_release.py`: the row, the sync push, and a mounted press. Mutation: a no-op press fails it.

**Mutation checks.**
- Without the immediate return and the run end: the call waits its 30 s deadline, and the run makes a second call.
- Without the disposal predicate (`unproven_only`) and the Raw CLI drain change: the SSH and Raw CLI children hang until their 40 s limit.

**Base vs head, same files, `-n 4`.** Identical results for test_execution_capacity, test_agent_runtime, test_agent_service, test_fleet_budget_boundary_probes (5F 1E on both), test_console_raw_cli_custody (22 passed), test_console_first_send_raw_cli_native (2F on both), test_console_raw_cli_persistence, test_console_runtime_shutdown, test_raw_cli_executor_contract/_process, test_workspace_tool_executor (1F on both), and the UI files (send_admission_off_pump 8F and turn_activity_line 21F on both). Differences:
- test_console_ssh_custody: 4F on head, 10F on base; the 6 retention tests fail on base under load.
- test_native_confirmed_undo: the failures are flaky on both (failed_path 1 of 3 on each).
- test_console_ordinary_tree: identical outcomes until both hang at `test_original_stop_hook_holds_same_ordinary_tree`, a pre-existing hang on base.
- The native Git files flake under load on both (managed_descendant, pins_helpers). `test_final_snapshot_child_and_original_publisher_both_hold_custody[final-command]` failed 2 of ~25 times on head and 0 of 9 on base. It passed 8 of 8 serially on head and has no retention or disposal in its body.

**Size ratchets (size waiver requested, not granted).** These are exact deltas with dated comments: console_chat_controller 28743 -> 28752 (+9), console_transcript 8534 -> 8555 (+21), chat_screen 24568 -> 24569 (+1). The review-selection projection pin was re-pinned to match. The remaining hooks are as small as I could make them; everything else moved to new modules. agent_service stays under its budget (9714 of 9720) because the result helper moved to execution_capacity.

**Preflight** passes on the final head. The diagnostic inventory was read and regenerated first: +1 constant warning each in execution_capacity and console_runtime. **Tests/Architecture** (serial, final head): 3 failed, 1348 passed. The same three fail on base: diagnostic metadata-only and two timer-path.

**Not done.**
- AC#2: the release action exists only for agent runs. The Raw CLI, Undo and change-review snapshot records can only be abandoned through the API.
- AC#3, part one: a daemon custody worker waits on the record's event, and nothing in production sets it; it ends with the process.
- AC#3, part two: workflow planning and close (`console_workflow_preparation.py:1977,2628`) still `join()` `_tool_workers` unconditionally. A workflow run's unproven tool cleanup therefore still holds its planning driver and disposal.
- `test_actual_child_joins_original_claim_before_its_row[*]` (the `disposal_pending` test) fails on base for all five params before it reaches disposal (the gateway is never called). Its expectation was updated, but neither base nor head runs it to completion.

**Files.** Agents/{execution_capacity,agent_service,agent_runtime,agent_models,ordinary_tree}.py, Chat/{console_raw_cli_custody,console_raw_cli,console_runtime,console_chat_controller}.py, Tools/{raw_cli_executor,workspace_tool_executor}.py, Workspaces/{native_git,change_revert_ownership,change_tracking,change_review_consent}.py, UI/Console_Modules/raw_cli.py, UI/Screens/chat_screen.py, Widgets/Console/console_transcript.py, new Chat/console_unproven_cleanup.py and Widgets/Console/console_cleanup_release.py, UI/Console_Modules/agent.py, css (+bundle), the size ratchets, the diagnostic inventory, the User Guide (agent-runs-and-tools), ADRs, tests.

