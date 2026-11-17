---
id: TASK-41307
title: "Unproven-cleanup records outside an agent run's reply have no Release button, so their tool slot stays held until restart"
status: To Do
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
