---
id: TASK-40716
title: "An unproven tool cleanup parks a worker forever, refusing the run's later tool calls and holding an app-wide tool slot"
status: Done
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
- [x] #2 The app-wide tool-worker slot held for an unproven operation is returned once the operation is proven finished or the user explicitly abandons it, without restarting the app. (Owner scope, 2026-11-14: the Release UI covers agent runs; Release UI for Raw CLI, Undo and change-review records is a follow-up.)
- [x] #3 A worker retained for an unproven cleanup is a daemon whose wait ends on a late proof or at process exit (the resource is held until restart, owner decision 2026-11-14); no exit path joins or waits on it, and the app can still exit while a cleanup is unproven.
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

Implements the owner's 2026-11-13 ruling (option A) and the 2026-11-14 review round.

**Behaviour.**
- One helper replaces all six parks: `OwnedOperation.retain_unproven(label)` in `Agents/execution_capacity.py`.
  - It marks the owner unproven, keeps the operation stopping (still counted toward `max_tool_workers`), and registers an `UnprovenRetention` on `RuntimeCapacity`.
  - A worker that holds the original scopes calls `hold()`. That worker is always a daemon thread, enforced by an assert. A late proof (`prove()`) or process exit ends its wait.
  - `abandon()` finishes only the operation, so the drain stays unproven (ADR-155).
- How the run ends.
  - The call returns within one poll with `RunTerminationReason.CLEANUP_UNPROVEN`. `agent_runtime` ends every run on it ("Agent stopped: …").
  - A retention that arrives after a timeout, cancel or per-call abandon makes `_reserve` raise `CleanupUnprovenRefused`. `_call_with_timeout` maps that to the same result, so the run ends at its next call.
  - The goal ledger treats `CLEANUP_UNPROVEN` as uncertain, which leads to `recovery_required`.
- Release.
  - The controller remembers every run's terminal row (`run_terminal_rows`). The map stays bounded: each sync forgets a row whose session closed, or whose run drained without a retained record (which includes a late proof). A released record stays listed while it is still retained, so its "released" note keeps showing.
  - `Chat/console_unproven_cleanup.py` shows **Release tool slot** for rows whose run has a retained record, whatever the run's outcome. The button is in `Widgets/Console/console_cleanup_release.py`, loaded lazily so the UI-ready census does not change.
  - The release runs off the UI thread because drain callbacks write the agent-runs DB. The released note is styled and protected from selection clicks.
- Disposal stops waiting for retained items:
  - ordinary turns: `OrdinaryTurn.unproven_only()`;
  - Raw CLI and change-tracking drains: `wait_retired_unless_retained`;
  - change-review init joins: `join_unless_retained`;
  - workflow planning and native close: `_tool_workers_returned`. On False the lifetime stays held and the phase fails its physical-return proof.
- Daemon custody instead of joined pool threads. Raw CLI user commands, Undo and @git reference expansion now run their custody on a daemon worker (`run_on_custody_thread` for Raw CLI and references, a daemon thread for Undo). Leaving scopes on a reused pool thread would leak `_current_owner`/storage ContextVars and thread-local connection ownership into the next task. A retained reference keeps its operation, root and claim. The reference is refused with the existing "reference source unavailable" record.
- Other fixes. Undo marks its owner unproven. `OrdinaryTurn.host_returned` makes host return and durability visible together. The native Git and @git reference test fixtures publish their pid files atomically.

**ADRs.** Amendments to ADR-285, 084, 093, 094 and 134 separate the owner's decision from an implementation note.

**Tests (all mutation-checked unless noted).**
- `Tests/Agents/test_unproven_tool_cleanup_run.py`:
  - immediate return; slot held until abandon; unproven drain; late proof; every run ends;
  - the timeout and per-call-abandon paths end the run at the next call;
  - the daemon assert;
  - host-return atomicity (written to fail on the old ordering, not mutation-run).
- `test_goal_progress.py`: CLEANUP_UNPROVEN leads to `recovery_required`.
- `test_workflow_tool_worker_custody.py`: the joins skip a retained worker, and native close refuses.
- `test_console_reference_custody.py`: an unproven @git cleanup holds a daemon worker, the Send settles, and the slot and claim stay held.
- `Tests/UI/test_console_unproven_cleanup_release.py`: rows keyed by retained run, the bounded row map (mutation-checked), sync, an off-thread press, the assistant-card row, and protected classes. A mounted ConsoleTranscript test was dropped because this environment fails even the existing citation harness with `RecoveryRequired: raw_source_selection_changed`.
- The SSH (6), Raw CLI (3), native Git and Undo (2) custody tests assert the new behaviour.

**Size ratchets.** Raised for TASK-40716 unproven-cleanup retention, with dated notes: console_chat_controller +3, console_transcript +22, chat_screen +1. agent_service sits exactly at its 9720 budget.

**Follow-ups.**
- Release UI for Raw CLI, Undo, change-review snapshot, reference-preparation and child-agent records (coordinator to file).

**Files.** Agents/{execution_capacity,agent_service,agent_runtime,agent_models,ordinary_tree}.py, DB/goal_runs.py, Chat/{console_raw_cli_custody,console_raw_cli,console_runtime,console_chat_controller,console_unproven_cleanup,console_reference_preparation,console_workflow_preparation}.py, Tools/{raw_cli_executor,workspace_tool_executor}.py, Workspaces/{native_git,change_revert_ownership,change_tracking,change_review_consent}.py, UI/Console_Modules/{raw_cli,agent}.py, UI/Screens/chat_screen.py, Widgets/Console/{console_transcript,console_cleanup_release}.py, css (+bundle), the diagnostic inventory, the User Guide, ADRs, the ratchets and tests.
