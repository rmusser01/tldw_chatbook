---
id: TASK-41307
title: "Unproven-cleanup records outside an agent run's reply have no Release button, so their tool slot stays held until restart"
status: In Progress
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
- [ ] #1 The user can release the tool-worker slot of every retained unproven-cleanup record: Raw CLI user commands, Undo, change-review snapshots, `@git` reference preparation and child-agent records. Each is shown by a test that is red on dev
- [ ] #2 A turn whose retention arrives before its run exists shows the cleanup-unproven explanation naming the tool, and a way to release the slot, instead of only "accepted-turn settlement failed"
- [ ] #3 Release still frees only the tool-worker slot; the workspace, storage and claim stay held until restart (ADR-134), pinned by a test
- [ ] #4 The User Guide and the affected ADR amendments describe the new release paths
<!-- AC:END -->

## Implementation Plan

ADR required: yes. The owner decision in ADR-084/093/094/134/285 puts the abandon action "on the run's terminal card". A record with no run card needs a new owner decision, recorded as an amendment (AC #4).

**Status: DRAFT, design first.** No product change until the owner rules on where Release lives.

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
