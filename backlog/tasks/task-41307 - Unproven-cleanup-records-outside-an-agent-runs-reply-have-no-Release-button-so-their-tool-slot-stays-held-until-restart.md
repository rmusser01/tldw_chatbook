---
id: TASK-41307
title: "Unproven-cleanup records outside an agent run's reply have no Release button, so their tool slot stays held until restart"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
- ux
priority: medium
references:
- tldw_chatbook/Chat/console_unproven_cleanup.py
- tldw_chatbook/Agents/execution_capacity.py
- tldw_chatbook/Chat/console_chat_controller.py
- Docs/User_Guide/console/agent-runs-and-tools.md
- backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
- https://github.com/rmusser01/tldw_chatbook/pull/3545
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. These are the follow-ups TASK-40716 (PR #3545) listed. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Docs/`.

**Why.** TASK-40716 keeps a tool whose cleanup is unproven held until restart. It lets the user free the tool-worker slot with **Release tool slot**, but only on an agent run's terminal reply. Every other retained record holds a slot against the app-wide tool limit until the app restarts, and the user cannot release it. ADR-084, 093, 094, 134 and 285 carry TASK-40716's amendments.

**Evidence.**
- Rows come only from agent-run reply cards:
  - `Chat/console_unproven_cleanup.py:18-26` groups `capacity.retained()` by `record.run_id` and drops records without one.
  - `:29-58` shows a row only for a card in `controller.run_terminal_rows`.
  - That map is written only in `ConsoleChatController._record_run_assistant_message` (`Chat/console_chat_controller.py:27213-27214`).
- `UnprovenRetention.run_id` is the owner's bound run (`Agents/execution_capacity.py:459-461`).
- Retentions that never get a row, per TASK-40716's follow-up list:
  - Raw CLI user commands (`Chat/console_raw_cli_custody.py:303`);
  - Undo (`Workspaces/change_revert_ownership.py:312`);
  - change-review snapshot jobs and `@git` reference preparation (both `Workspaces/native_git.py:67`);
  - child-agent records.
- A retention that arrives before the run exists has no run id at all:
  - the startup SSH instruction read;
  - `@git` in an ordinary tree when MCP setup reserves a slot.

  The turn then ends with only the logged error "Console accepted-turn settlement failed" (`console_chat_controller.py:13226-13230`), and the transcript gets no explanation row and no Release row.
- The User Guide documents the gap (`Docs/User_Guide/console/agent-runs-and-tools.md:280-282`): "A turn that stops before its run starts has no run to release, so its slot stays counted until you restart the app."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The user can release the tool-worker slot of every retained unproven-cleanup record: Raw CLI user commands, Undo, change-review snapshots, `@git` reference preparation and child-agent records. Each is shown by a test that is red on dev
- [x] #2 A turn whose retention arrives before its run exists shows the cleanup-unproven explanation naming the tool, and a way to release the slot, instead of only "accepted-turn settlement failed"
- [x] #3 Release still frees only the tool-worker slot; the workspace, storage and claim stay held until restart (ADR-134), pinned by a test
- [x] #4 The User Guide and the affected ADR amendments describe the new release paths
<!-- AC:END -->

## Implementation Plan

ADR required: yes. The owner decision in ADR-084/093/094/134/285 puts the abandon action "on the run's terminal card". A record with no run card needs a new owner decision, recorded as an amendment (AC #4).

**Owner ruling, 2026-11-17: option 3, all five recommendations.** (1) A Console-wide "Held tool slots" list behind a status segment shown only while an unreleased record exists; it says once what stays held. (2) A turn stopped before its run shows the cleanup-unproven explanation naming the tool instead of "Response recovery failed", with Release under it. (3) Raw CLI: Release under its "Cleanup: Unproven" row, plus the list. (4) Sub-agents: the list only. (5) The list includes run records that also have a card.

**Evidence check** (read on fbf1c9ea74, which is dev 5e3c8b2bd4 plus the filing commits; not run). Every cited line holds. Refinements:
- The pre-run case is not log-only. The reply placeholder goes into dispatch recovery and the status line reads "Response recovery failed. Try again or discard." (`_restore_dispatch_recovery_after_settlement_failure`). That names the wrong cause and no tool. Two pre-run reservations trigger it: MCP capture (`Chat/console_ordinary_tree.py:286`) and skill-catalog capture (`:229`).
- Raw CLI records do have a run id: the `local_command` run, bound at `console_raw_cli_custody.py:242`. Only the card mapping is missing. Their tool marker (`raw-cli-<invocation>`) already reads "Cleanup: Unproven".
- A change-review snapshot job that borrows an ordinary turn's owner carries that turn's run id, so the existing reply row already covers it. A job with its own owner (outside an ordinary tree, e.g. Windows) has no run id, and neither does a consent initialization job (`conversation_id=""`).
- An `@git` reference inside an ordinary tree either ends the run at its first tool call (existing row) or stops the turn before the run exists (AC #2). Outside a tree it has its own owner, which is never bound to a run.
- Child records are bound to the child run id (`agent_service.py:5290`, `agent_native_execution.py:1064/1205`). Only root runs get a card in `run_terminal_rows`.
- Trap: `console_unproven_cleanup._capacity` reads the bridge's private `_runtime_capacity`. That stays None until an agent turn allocates it, because `bind_runtime_capacity` stores `existing_capacity`, which can be None. In a session that only ran Raw CLI or Undo, it would find no records. Read `runtime._execution_capacity` instead.
- `UnprovenRetention.abandon()` already works on any record. The capacity API needs only a stable key: `run_id`, else `operation.owner.execution_id`.

**Where each kind shows today**

| Kind | Run id on the record | What the user sees | Possible Release spot |
|---|---|---|---|
| Raw CLI user command | yes (`local_command`) | Raw CLI tool marker: "Cleanup: Unproven" | under that marker |
| Undo (turn card Undo All, Change Review Revert) | no | a warning toast, which fades | **none** |
| Change-review snapshot (own owner), consent init | no | turn file card hides the root; init shows nothing | **none** |
| `@git` outside a tree | no | `@-references:` row "REFUSED — reference source unavailable" (wrong cause) | that row |
| Child agent | child run id, no card | Inspect rail Agents row; parent's `⤷ spawned sub-agent` | fleet row, or parent's card (needs a child-to-root map) |
| Startup SSH read / `@git` in tree, pre-run | no | the recovery prompt above; SSH read itself: log only | a new explanation row under that reply |

**Options**
1. *Per-surface rows.* Map each kind's own row to its record. Covers Raw CLI, `@git` and pre-run. Undo, the consent and own-owner snapshot jobs, and children (without a child-to-root map) have no row. That needs new buttons in the Change Review screen, the turn file card and the fleet panel. Release stays next to the failure, but this touches about 6 files and the controller, chat_screen and transcript size ratchets, and still leaves gaps.
2. *One Console-wide "Held tool slots" list.* A status chip appears only while an unreleased record exists. It opens a small modal that lists every `capacity.retained()` record (label, released state) with the reused `cleanup_release_widget`, and one line saying what stays held. AC #2 adds an explanation row naming the tool that points to the list. This covers every kind, including the future ones, with the smallest diff. It is detached from context, though: the user has to notice the chip. Labels are coarse ("native Git" covers snapshot, `@git` and Undo's Git).
3. *Recommended: option 2 plus rows where a transcript row already names the failure.* Key records by run id, else execution id. Put the existing Release row under the new pre-run explanation row (AC #2) and under the Raw CLI marker. Everything else goes through the list. The run card stays as it is.

**Steps after the ruling (option 3)**
1. Red tests on base, one per kind, each using the real retention site (Raw CLI, Undo, native Git snapshot, `@git` outside a tree, a child run), plus the pre-run SSH/`@git` turn, plus an AC #3 pin (after Release: slot free, claim held, drain unproven).
2. Group records by `run_id or owner.execution_id`. `CleanupUnprovenRefused` carries its retention. Read capacity from the runtime.
3. Settlement catch: when the cause is `CleanupUnprovenRefused`, append the `cleanup_unproven_result` explanation naming the label instead of the recovery prompt, and map its row to the record.
4. Raw CLI: map the marker id to the record when the marker finishes (on the UI thread).
5. Chip and modal for the list. Release calls `record.abandon()` off the UI thread, as the button does now.
6. User Guide paragraph and ADR-084/093/094/134/285 amendments. Size ratchets get dated notes.

## Implementation Notes

Implements the owner's 2026-11-17 ruling (option 3). Verified on fu/41307-release-outside-runs (base fbf1c9ea74), 2026-11-17, load 60-83.

**Behaviour.**
- Records are keyed by their run, else by their owner's execution id (`Chat/console_unproven_cleanup.py::_key`). Capacity is read from the runtime (`_hooks_v2_runtime._execution_capacity`), with the bridge as a fallback: the bridge allocates its own reference only once an agent turn runs, so a Raw CLI- or Undo-only session listed nothing.
- Status line: a "Held tool slots: N" segment (unreleased records only, hidden at 0) opens `ConsoleHeldToolSlotsModal`, which lists every retained record by key with what held it, the copy saying what stays held, and the existing Release widget (now taking a release callable). The segment syncs on each Console transcript sync; closing the list resyncs.
- Pre-run stop: `CleanupUnprovenRefused` carries its retention. The accepted-turn failure handler calls `explain_pre_run_stop`, which sets the turn's block copy to "Agent stopped: the cleanup of <tool> could not be proven…" (replacing "Response recovery failed") and maps the reply to the record. The reply's row shows that explanation with Release under it. The dispatch recovery itself is unchanged (only its copy), so the accepted turn keeps its durable recovery.
- Raw CLI: when `run_on_custody_thread` reports the retention, the command's marker (`raw-cli-<invocation>`) is mapped to the record on the UI thread and the Console resyncs.
- Release is still `UnprovenRetention.abandon()`: slot only (ADR-134).

**Decisions.** Kept the accepted turn's dispatch recovery rather than settling it failed: settling an accepted durable turn from the generic failure handler would duplicate the pre-provider-stop settlement path, and the owner asked only to replace the prompt. No push channel from capacity to the view: a Change Review job retained in the background between syncs appears at the next sync (noted in the ADR amendment). Row values became `(released, explanation)`; the TASK-40716 tests were updated to the new shape.

**Tests (red on base, mutation-checked).**
- Real retention sites, each now releasing through the new path and asserting what stays held: Raw CLI (`test_unproven_required_terminal_keeps_real_worker_and_storage`, 3 modes: marker row + list), Undo (`test_unknown_required_settlement_retains_original_scopes`, 2 modes: list), Change Review init job (`test_unavailable_cleanup_proof_retains_original_worker_in_isolated_process`: list), `@git` outside a turn (`test_unproven_reference_git_cleanup_holds_a_daemon_worker_not_the_send`: list read with no agent bridge; AC #3 pin: slot freed, claim held, custody worker alive, record still listed, drain unproven), startup SSH pre-run (`test_unproven_startup_instruction_read_holds_a_daemon_worker`: stop copy, reply row, Release frees one slot, storage still live).
- Unit: runtime-capacity list with no bridge, sub-agent listed and not on the parent card, pre-run explanation row and widget, unrelated failure untouched (`Tests/UI/test_console_unproven_cleanup_release.py`); segment visibility and activation, the list releasing off the UI thread, opening the list (`Tests/UI/test_console_held_tool_slots.py`).
- Mutation checks (9), each failing its tests: runtime capacity read, execution-id key, Raw CLI marker mapping, pre-run copy call, retention on `_reserve`'s and on `run_on_custody_thread`'s refusal, segment visibility, sync gate, Release unwinding custody (AC #3).
- Covering files, head vs base: Chat custody files 107/107 on head after re-running load flakes alone (base: 6 failures, the same 6 on head's first run); Workspaces custody files: head 2 failures, base 7, all `test_native_confirmed_undo.py` load flakes listed in TASK-40716; UI/status/capacity files 86/86.

**Diagnostic inventory.** One added statement, reviewed with `--statements`: `console_unproven_cleanup.explain_pre_run_stop` logs "Unproven-cleanup stop copy failed | exception_type={}" with only the exception's type name (content-free, like the accepted-turn handler's own log). Inventory regenerated.

**Size ratchets.** console_chat_controller 28753 -> 28757 (+4, after the review round); chat_screen 24569/742 -> 24570/743 (+11 lines on 24,559, +1 method), dated notes; `test_console_review_selection_controller_boundary.LIVE_RATCHET_ROW` re-pinned with the row, as TASK-40716 did.

**Tests/Architecture (serial).** 3 failures on head, the same 3 on base fbf1c9ea74: `test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only` (agent_service "on_child_settled consumer raised") and `test_timer_path_static_update_inventory` x2 (`project_index_binding_card._poll`). A timeout of `test_task_15103_review_ledger_canonical_schema_and_arithmetic` in the full run (load 80) passed when re-run.

**Files.** Agents/execution_capacity.py, Chat/{console_unproven_cleanup,console_chat_controller}.py, UI/Console_Modules/{agent,raw_cli}.py, UI/Screens/chat_screen.py, Widgets/Console/{console_cleanup_release,console_held_tool_slots (new),console_status_chips,console_transcript}.py, ADR-084/093/094/134/285, the User Guide, the ratchets and tests.

**Review round (2026-11-17, Opus review of adfa6332b4: no Critical/Important; every Minor fixed).**
1. A retried reply kept losing its Release row: "Try again" reuses the reply id, and `_record_run_assistant_message` overwrote the row's key with the retry's run. A row now carries a set of keys (`run_terminal_rows: dict[str, set[str]]`); the sync drops only keys that are neither live nor retained. Test: `test_a_retried_reply_keeps_its_pre_run_release_row` (drives the real `_record_run_assistant_message`).
2. A card's Release left the status segment stale (the 0.2 s poll stops when idle). The button now calls `ConsoleAgentController.resync_held_tool_slots()`, the same console-sync worker the list's close callback uses. Test: the existing button test asserts one resync.
3. The pre-run copy leaked on early returns. `explain_pre_run_stop` now runs just before each of the two recovery restores in the accepted-turn handler, and ignores a cancellation. Tests in the new `Tests/Chat/test_console_pre_run_cleanup_stop.py` drive the real handler through `submit_draft`: the stop names the tool and maps Release with no copy left over; a chat closed mid-turn leaves no copy (fails with the old placement); an unrelated failure keeps "Response recovery failed".
4. Test gaps: ChatScreen's `_console_held_slots_activated` (subscription and call), the real Console sync pushing the count (`test_the_console_sync_counts_held_slots_on_the_status_line`, a mounted Console), and a user Raw CLI marker's Release row in the grouped `_transcript_rows`. Not added: a real-site `@git`-in-tree (MCP capture) test. It needs a new rig joining the ordinary-tree MCP harness (`ssh_console`/`mcp_case`) to a hanging-git wrapper and a held-until-exit subprocess. It takes the same path as the startup-SSH test (the retained tree owner's MCP reservation at `console_provider_composition.py:163`, then the same handler), and its retention site (`run_native_git` through reference custody) is covered by the `@git`-outside-a-turn test.
5. ADR wording: Change Review jobs "with their own owner" are never bound to a run; a snapshot job that borrows a turn's owner carries its run id. The User Guide says the same.
6. Flake check of `test_unproven_required_terminal_keeps_real_worker_and_storage[cleanup]`: 3 of 3 passed on head and 3 of 3 on base, run alone at load 80-107 (27-39 s each). Its child process has a 40 s timeout, so the earlier failure is most likely that timeout under heavier load.
- Covering files, head vs base (load 70-100): 13 failures each. The 5 that failed only on head (`test_unknown_peer_proof_keeps_original_model_worker_and_storage[no_terminal]`, a subprocess timeout, and 4 `test_native_confirmed_undo.py` tests) all passed when re-run alone; base had its own 5 Undo/SSH failures that head did not.
- Mutation checks for the new tests (each fails with its fix reverted, files restored from memory, never git): overwrite instead of add (retry test), no resync (button test), no `project_held_tool_slots` call (mounted sync test), no `open_held_tool_slots` call (handler test), copy written at the top of the handler (closed-chat test).
- After the round: preflight passes (`PYTHON=.venv/bin/python`). Serial Tests/Architecture: the same 3 base failures, plus `test_surface_swap_guard::test_a_child_swap_that_can_finish_is_left_alone[#outgoing]` and a 300 s timeout in `test_task_15103_review_ledger_canonical_schema_and_arithmetic` (load ~90); both files pass alone on head and on base (16 passed each, load 36-46). Controller size pin 28757 (+4 on 28,753); chat_screen unchanged this round.

**Delta review round (2026-11-17, review of 377adb7db9).**
1. Important: `resync_held_tool_slots` ran an unforced transcript pass, which the TASK-34958 gate skips after an abandon (nothing it reads changes), so the segment stayed "Held tool slots: 1". It now forces `_sync_native_console_transcript(force=True)` in its own worker group. Test: `test_a_release_resync_hides_the_segment_while_the_console_idles` (mounted Console, waits out the poll, abandons, calls `resync_held_tool_slots()`).
2. Minor: `_preflight_block_copies` is popped before `mark_dispatch_recovery_needed`, so a raise leaves no stop copy keyed by the reply. Test: `test_a_failed_recovery_mark_leaves_no_stop_copy_behind`.
3. Minor: `unproven_cleanup_rows` computes `pre_run` over the run-less records only, so a retried reply that also retains keeps its pre-run explanation. Test: `test_a_retried_reply_that_also_retains_still_names_its_pre_run_cleanup`.
- Mutation checks: each fix reverted alone (file from 377adb7db9) makes its test fail.
- Custody files (Chat custody/raw-cli, execution capacity, the three new-test files) on head: 274 passed, 10 failed (received/reference/ssh custody); the same 10 fail identically on base fbf1c9ea74 (fu-base).
