---
id: TASK-42605
title: "Claims held until restart without an unproven owner cleanup still say retry, and a queued snapshot waits behind one forever"
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
- tldw_chatbook/Agents/legacy_hook_ownership.py
- backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
dependencies:
- TASK-41950
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the third review of PR #3776 (TASK-41950, 2026-11-25). Checked on branch `fu3/41950-retained-init-claim` (base dev 2959b48314). Paths are under `tldw_chatbook/`.

**Why.** TASK-41950 gives a Send, a tool's widening and a Change Review initialization the restart message when a conflicting claim has a member whose owner's cleanup is unproven (`WorktreeActivity.restart_refusal`). That is not every claim held until restart. `WorktreeActivity._retire_if_settled` (`Workspaces/worktree_activity.py`, the `member.drained is True and member.durable` check) also needs each member's durable settlement. Known production paths that leave a claim held without an unproven owner cleanup include:

- (a) `_SnapshotTicket.returned` settles `durable=False` once and retires, when the write failed and there is no workflow wave (`Workspaces/change_review_ownership.py:300-303`). The producer is the UNRESOLVED handoff in `Workspaces/change_review_finalization.py:1136-1141`.
- (b) `OrdinaryTurn.host_returned` records the turn's member non-durable (`member.durable = proven`, `Agents/ordinary_tree.py:893-929`). This happens when the run-row read mismatches or raises. It also happens when the row started but capture failed or no terminal was recorded, because `proven = not member.row_started` is then never upgraded. `_retire_scratch` (`Agents/ordinary_tree.py:~750-775`) settles the turn's claim only when every member is durable, so that registry member is never settled at all.
- (d) A legacy hook event outside an ordinary tree takes its own opaque claim (`Agents/legacy_hook_ownership.py:238-255` on dev b534feabd3).
  - When its required write fails, `_EventCustody.host_done(required_write_ok=False)` sets `failed_write` (`:281-285`). The failure arrives through `Chat/console_native_commit.py:890` (`prompt_handed_off(required_write_ok=False)`), then `Agents/run_hooks.py:124-127`, then `:98-105` or `:84`.
  - `_reconcile_active` then never releases a target (`:321`) and never settles (`:332`), so `claim.settle` and `owner.finish_root` (`:337-338`) never run. The owner's cleanup is not marked unproven either, so the claim is held until restart with the busy message.
  - PR #3778 (TASK-41943, armed) changes `legacy_hook_ownership.py`, so re-check these line numbers after it lands.

Those claims are held until restart, yet a Send they refuse still says "Workspace activity is busy; retry after it finishes." The registry cannot tell a member that is not yet durable from one that never will be, so the fix needs a recorded "will never be durable" state, not a one-line predicate change.

- (c) Also: `_SnapshotSource._admit_owned`'s `_snapshot_prelaunch` refusal returns `(None, "activity")` (`Workspaces/change_review_ownership.py:216-218`). The snapshot queue then waits for an activity release. When the refusing claim is held until restart, that release never comes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Send refused by a claim whose member was settled non-durable after a failed snapshot write (no workflow wave) tells the user that a restart clears it; a test that is red on the current code pins it
- [ ] #2 A Send refused by an ordinary turn's claim whose run-row durability check failed, or whose started row captured no terminal, tells the user that a restart clears it; a test that is red on the current code pins it
- [ ] #3 A Change Review snapshot refused by a claim held until restart reports that a restart is needed instead of waiting for a release; a test that is red on the current code pins it
- [ ] #4 A Send refused by a legacy hook event's claim whose required write failed (outside an ordinary tree) tells the user that a restart clears it; a test that is red on the current code pins it
- [ ] #5 A claim whose members have not yet settled keeps the "busy; retry" message (control test)
<!-- AC:END -->
