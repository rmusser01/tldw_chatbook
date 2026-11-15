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

Implements the owner's 2026-11-13 ruling (option A), the 2026-11-14 review round and the delta review round below.

**Behaviour.**
- One helper replaces all six parks: `OwnedOperation.retain_unproven(label)` in `Agents/execution_capacity.py`.
  - It marks the owner unproven, keeps the operation stopping (still counted toward `max_tool_workers`), and registers an `UnprovenRetention` on `RuntimeCapacity`.
  - A worker that holds the original scopes calls `hold()`. That worker is a daemon thread; `hold()` never unwinds entered custody (see the delta round). A late proof (`prove()`) or process exit ends its wait.
  - `abandon()` finishes only the operation, so the drain stays unproven (ADR-155).
- How the run ends.
  - The call returns within one poll with `RunTerminationReason.CLEANUP_UNPROVEN`. `agent_runtime` ends every run on it ("Agent stopped: …").
  - A retention that arrives after a timeout, cancel or per-call abandon makes `_reserve` raise `CleanupUnprovenRefused`. `_call_with_timeout` maps that to the same result, so the run ends at its next call.
  - The goal ledger treats `CLEANUP_UNPROVEN` as uncertain, which leads to `recovery_required`.
- Release.
  - The controller remembers every run's terminal card (`run_terminal_rows`, by persisted id once it has one). The map stays bounded: each sync forgets a row whose run is neither live nor retained (which includes a late proof). A released record stays listed while it is still retained, so its "released" note keeps showing.
  - `Chat/console_unproven_cleanup.py` shows **Release tool slot** for rows whose run has a retained record, whatever the run's outcome. The button is in `Widgets/Console/console_cleanup_release.py`, loaded lazily so the UI-ready census does not change.
  - The release runs off the UI thread because drain callbacks write the agent-runs DB. The released note is styled and protected from selection clicks.
- Disposal stops waiting for retained items:
  - ordinary turns: `OrdinaryTurn.unproven_only()`;
  - Raw CLI and change-tracking drains: `wait_retired_unless_retained`;
  - change-review init joins: `join_unless_retained`;
  - workflow planning and native close: `_tool_workers_returned`. On False the lifetime stays held and the phase fails its physical-return proof.
- Daemon custody instead of joined pool threads. Raw CLI user commands, Undo, @git reference expansion and the startup SSH instruction read now run their custody on a daemon worker (`run_on_custody_thread` for Raw CLI, references and the instruction read, a daemon thread for Undo). Leaving scopes on a reused pool thread would leak `_current_owner`/storage ContextVars and thread-local connection ownership into the next task. A retained reference keeps its operation, root and claim. The reference is refused with the existing "reference source unavailable" record.
- Other fixes. Undo marks its owner unproven. `OrdinaryTurn.host_returned` makes host return and durability visible together. The native Git and @git reference test fixtures publish their pid files atomically.

**Delta review round (2026-11-14).**
- `UnprovenRetention.hold()` no longer asserts. Its scopes are already entered, so on a joined thread it logs an error and still holds; unwinding could free a resource that may be live (and an `assert` vanishes under `-O`). The test that pinned the fail-open assert is replaced.
- The Console's startup SSH instruction read (`run_ordinary_preparation` with `ssh=`) now runs its custody on a daemon worker via `run_on_custody_thread`. It used to run on a joined pool thread. The new test also found that `_PreparationSshCall` had no `name`, so its retention raised `AttributeError` and unwound the leases; it is now labelled "a remote SSH read".
- `hold()`/`retain_unproven` callers and their threads:
  - `agent_service._call_with_timeout` (`retain_unknown_return`): the `tool-<name>` daemon.
  - `ModelSshCall.activate`: the `tool-*` daemon; the startup instruction read (custody daemon, this round); the recovery probe (`ssh-recovery-probe-*` daemon); `agent_request_execution`'s DEFINITIVE_AFTER_START and no-timeout paths on the run thread. Those two are unreachable in the Console: SSH calls exist only for `fs_*`/`virtual_cli`, which are never DEFINITIVE_AFTER_START, and the Console maps a 0 tool timeout to a century deadline, so every call goes through `_call_with_timeout`. Another engine caller with a literal 0 would hold its run thread, which `hold()` now keeps, logged.
  - `RawCliCall.activate`: the `tool-*` daemon; the same two unreachable run-thread paths (`shell_exec` only).
  - `RawCliCustody.retain_unproven` (Raw CLI user commands): the custody daemon from `run_on_custody_thread`. Never-started calls keep every proof flag True, so their `finish()` on a pool or UI thread never holds.
  - `_UndoCall.retain_unknown`: the Undo daemon thread; the two unwound sites pass `hold=False`.
  - `native_git.run_native_git`: @git references (custody daemon); Change Review jobs on the `change-review-fs-*`/`change-review-init-*` daemons; Undo's daemon.
- The two unwrapped dispatch paths in `agent_request_execution` (DEFINITIVE_AFTER_START, no timeout) end the run on `CleanupUnprovenRefused` instead of reporting blocked/not-started.
- Change Review snapshot admission on a retained owner stays a terminal refusal (unadmitted), deliberately not a capacity wait: waiting would admit it after the slot is released and read a workspace that may still be changing. Documented in ADR-084 and pinned by a test.
- Release rows: records are grouped per run, so two retentions on one run show one row and Release frees every slot. Rows are no longer dropped when their conversation closes; a card is recorded by its persisted id and found again by it when the conversation is reopened. A turn that ends before its run is created (no run id) still has no row; the User Guide says so.
- `_ReferenceRead.finish` runs once a late `prove()` arrives: the custody worker finishes the operation (and, outside an ordinary tree, the root and claim) through a once-only guard. The claim's release still follows the owner's drain, which stays unproven (ADR-155).
- Raw CLI: `_execute` treats a `RuntimeError` as a failed thread start only when the custody worker never ran; a worker's own error is re-raised, not refused and finished a second time.
- ADR-084/093/094/134/285 owner blocks now say: the run ends; abandon releases only the slot; the button sits on the run's terminal card (its reply, which the explanation follows); exit stops waiting. The implementation notes drop the assert and describe the ordinary-tree preparation case.
- User Guide: the paragraph moved out of the middle of the live-line paragraph and covers @git/startup-instruction retention, reopened conversations and the no-run case.
- `test_native_confirmed_undo.py` head-only failures were load flakes, not this change: over 5 runs at load 25-45, `test_native_ticket_exists_before_original_worker_enqueue` failed 2/5 on head and 3/5 on base (a `git archive 1455ceeb96` tree); `test_applying_engine_requires_exact_unconsumed_selection` and `test_original_provider_refusal_reaches_review_as_per_path_outcome` failed on both, and `test_historical_smudge_helper_retains_writer_after_ui_unmount[cancel]` failed on base too. The one failure traced, in `test_applying_engine_requires_exact_unconsumed_selection`, is its UI click returning False (`assert await pilot.click("#change-revert-yes")`).
- `test_unknown_peer_proof_keeps_original_model_worker_and_storage` (all 6 modes) failed at e64f2bbc65 too: it looked for the explanation text on the Release row, but the row is the run's failed reply card and the explanation is a separate system row. The test now asserts both; all 6 pass.

**Final review round (2026-11-14).**
- Exit with a retained planning tool worker. `ConsoleWorkflowCoordinator.close` now raises `WorkflowShutdownRetained` (a `RuntimeError`, same `workflow_original_shutdown_unresolved` message) when every unresolved run's driver returned and its prepared reply holds unproven custody. `ConsoleRuntime.dispose` then runs the rest of disposal (planning/wave fences, `_dispose_owned`: controller, gateway, capacity close, voice, Raw CLI and Undo drains) and reports the error afterwards. A run with live unresolved custody still raises the plain error before capacity closes, as the existing scheduling/admission tests pin. `_shutdown_app_owned_lifecycles` catches the Console shutdown error like the lab/workflow errors, runs change review, Buddy, llama.cpp, audio, the meeting owner, image edits, file notes and the egress pools, then re-raises; the `on_unmount` fallback goes through the same function.
- The two run-thread dispatch paths are closed by construction: a custody-bearing call (SSH or Raw CLI) on the no-timeout path goes through `_call_with_timeout` with a century deadline, and on the DEFINITIVE_AFTER_START path it is refused before start. Their dead activation code is gone.
- ADRs and docstrings now say custody is held until restart (no proof path exists yet; nothing in production calls `prove()`).
- Reopening a retained run's conversation replaces its alias row instead of adding one.
- Tests: the planning journey also asserts that capacity closed and the retention is still held; `test_a_console_shutdown_error_still_shuts_down_every_later_owner`; `test_custody_bearing_calls_never_hold_the_run_thread` (both paths); the reopen test checks aliases stay at one per run. Each fix was mutation-checked (6 of 6 fail when reverted).
- Size ratchet: console_runtime 7043 -> 7048 (+5), dated note.

**ADRs.** Amendments to ADR-285, 084, 093, 094 and 134 separate the owner's decision from an implementation note.

**Tests (all mutation-checked unless noted).**
- `Tests/Agents/test_unproven_tool_cleanup_run.py`:
  - immediate return; slot held until abandon; unproven drain; late proof; every run ends;
  - the timeout and per-call-abandon paths end the run at the next call;
  - `hold()` on a joined thread keeps custody instead of unwinding;
  - the DEFINITIVE_AFTER_START and no-timeout dispatch paths end the run;
  - Change Review snapshot admission on a retained owner fails closed (pins the decision, no code change);
  - host-return atomicity (written to fail on the old ordering, not mutation-run).
- `test_goal_progress.py`: CLEANUP_UNPROVEN leads to `recovery_required`.
- `test_workflow_tool_worker_custody.py`: the joins skip a retained worker, and native close refuses. A subprocess journey drives the real planning `finally`: the lifetime stays held, `_planning_physical_proof` raises, and disposal returns (reporting `workflow_original_shutdown_unresolved`) instead of hanging.
- `test_console_reference_custody.py`: an unproven @git cleanup holds a daemon worker, the Send settles, and the slot and claim stay held; a late proof then retires the operation.
- `Tests/UI/test_console_unproven_cleanup_release.py`: rows keyed by retained run, the bounded row map, a closed and reopened card, two retentions on one run, sync, an off-thread press, the assistant-card row, and protected classes. A mounted ConsoleTranscript test was dropped because this environment fails even the existing citation harness with `RecoveryRequired: raw_source_selection_changed`.
- The SSH (6), Raw CLI (3), native Git and Undo (2) custody tests assert the new behaviour.
- `test_console_ssh_custody.py::test_unproven_startup_instruction_read_holds_a_daemon_worker` (subprocess): the startup read holds a daemon worker with its leases and `db.agent_runs` source.
- `test_console_raw_cli_custody.py`: a custody worker's own `RuntimeError` is not refused twice; a thread that cannot start refuses once.
- Delta round mutation checks (each reverted fix fails its test): the hold assert, both dispatch paths, the per-run grouping, the persisted-id lookup, the reference late-proof finish, the Raw CLI start check, the planning join, the startup custody wrapper and the preparation SSH label.

**Size ratchets.** Raised for TASK-40716 unproven-cleanup retention, with dated notes: console_chat_controller +3, console_transcript +22, chat_screen +1. agent_service sits exactly at its 9720 budget.

**Follow-ups.**
- Release UI for Raw CLI, Undo, change-review snapshot, reference-preparation and child-agent records (coordinator to file).
- A retention during turn preparation (startup SSH read, @git reference) that the turn hits before its run is created surfaces as "accepted-turn settlement failed" with no explanation row and no Release row (no run id). It could map to the same CLEANUP_UNPROVEN explanation.

**Files.** Agents/{execution_capacity,agent_service,agent_runtime,agent_models,ordinary_tree,agent_request_execution}.py, DB/goal_runs.py, Chat/{console_raw_cli_custody,console_raw_cli,console_runtime,console_chat_controller,console_unproven_cleanup,console_reference_preparation,console_workflow_preparation,console_ordinary_tree,console_ssh_custody}.py, Tools/{raw_cli_executor,workspace_tool_executor}.py, Chat/console_workflow_runs.py, app_lifecycle.py, Workspaces/{native_git,change_revert_ownership,change_tracking,change_review_consent}.py, UI/Console_Modules/{raw_cli,agent}.py, UI/Screens/chat_screen.py, Widgets/Console/{console_transcript,console_cleanup_release}.py, css (+bundle), the diagnostic inventory, the User Guide, ADRs, the ratchets and tests.
