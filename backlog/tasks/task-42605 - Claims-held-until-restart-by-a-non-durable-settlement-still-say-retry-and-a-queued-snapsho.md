---
id: TASK-42605
title: "Claims held until restart by a non-durable settlement still say retry, and a queued snapshot waits behind one forever"
status: To Do
assignee: []
created_date: '2026-11-25 17:59'
labels:
- follow-up
- workspaces
priority: medium
references:
- tldw_chatbook/Workspaces/worktree_activity.py
- tldw_chatbook/Workspaces/change_review_ownership.py
- tldw_chatbook/Workspaces/change_review_finalization.py
- tldw_chatbook/Agents/ordinary_tree.py
- backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
dependencies:
- TASK-41950
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the third review of PR #3776 (TASK-41950, 2026-11-25). Checked on branch `fu3/41950-retained-init-claim` (base dev 2959b48314). Paths are under `tldw_chatbook/`.

**Why.** TASK-41950 gives a Send, a tool's widening and a Change Review initialization the restart message when a conflicting claim has a member whose owner's cleanup is unproven (`WorktreeActivity.restart_refusal`). That is not every claim held until restart. `WorktreeActivity._retire_if_settled` (`Workspaces/worktree_activity.py`, the `member.drained is True and member.durable` check) also needs each member's durable settlement. Two production paths drain a member proven but never settle it durable:

- (a) `_SnapshotTicket.returned` settles `durable=False` once and retires, when the write failed and there is no workflow wave (`Workspaces/change_review_ownership.py:300-303`). The producer is the UNRESOLVED handoff in `Workspaces/change_review_finalization.py:1136-1141`.
- (b) `OrdinaryTurn.host_returned` sets `member.durable = False` when the run-row read mismatches or raises (`Agents/ordinary_tree.py:893-929`).

Those claims are held until restart, yet a Send they refuse still says "Workspace activity is busy; retry after it finishes." The registry cannot tell a member that is not yet durable from one that never will be, so the fix needs a recorded "settled non-durable" state, not a one-line predicate change.

- (c) Also: `_SnapshotSource._admit_owned`'s `_snapshot_prelaunch` refusal returns `(None, "activity")` (`Workspaces/change_review_ownership.py:216-218`). The snapshot queue then waits for an activity release. When the refusing claim is held until restart, that release never comes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Send refused by a claim whose member was settled non-durable after a failed snapshot write (no workflow wave) tells the user that a restart clears it; a test that is red on the current code pins it
- [ ] #2 A Send refused by an ordinary turn's claim whose run-row durability check failed tells the user that a restart clears it; a test that is red on the current code pins it
- [ ] #3 A Change Review snapshot refused by a claim held until restart reports that a restart is needed instead of waiting for a release; a test that is red on the current code pins it
- [ ] #4 A claim whose members have not yet settled keeps the "busy; retry" message (control test)
<!-- AC:END -->
