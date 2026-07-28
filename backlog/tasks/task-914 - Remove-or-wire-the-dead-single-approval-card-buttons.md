---
id: TASK-914
title: Remove or wire the dead single-approval card buttons
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 03:55'
updated_date: '2026-07-27 17:27'
labels:
  - console
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatApprovalCard's single-approval body renders "Allow once" (#approval-allow-once) and "Deny" (#approval-deny) buttons that are not wired in on_button_pressed and can never emit ApprovalDecided — pre-existing dead UI confirmed during the parallel-agents train review. All production flows use the batch body.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The single-approval buttons either resolve their round correctly or the dead body is removed.
- [x] #2 No unreachable button handlers remain on the card.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reachability sweep: grep ChatApprovalCard usages, set_approval callers, and TaskResumeState.pending_approval producers to confirm no production path builds the legacy non-batch payload.
2. Remove the single-approval body (compose(), set_approval()) from ChatApprovalCard; collapse ChatTaskCards.sync_state to always call set_batch.
3. Fix the two duplicated UI-thread fallbacks (chat_screen.py, console_status_chips.py) that referenced the now-removed #approval-allow-once id.
4. Update/replace tests pinning the dead body; add a regression test asserting the retired ids never render.
5. Run targeted pytest suites; update backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verdict: REMOVE (not wire). Reachability sweep traced every write to
TaskResumeState.pending_approval back to ConsoleChatController.request_mcp_approvals,
whose payload always carries a "calls" key (or is None) -- no production code ever
builds the legacy {"summary","details"} shape ChatApprovalCard.set_approval expected.
The only caller of that legacy API, the pre-task-649 Chat_Window_Enhanced composition,
was already fully retired (commit 94b2c558f), which deleted its dedicated pinning
suite (Tests/UI/test_chat_approvals_and_resume.py) but left the now-orphaned widget
code behind -- that gap is what this task closes.

Removed: ChatApprovalCard's #approval-single-body Container (Allow once/Deny/Review
details buttons) and its set_approval() method. ChatTaskCards.sync_state now always
calls set_batch(...), since it already treats an empty/absent "calls" list as "clear".
Two duplicated UI-thread fallbacks (chat_screen.py's handle_console_inspector_review_
approval, console_status_chips.py's _focus_pending_approval_card) dropped their
batch_visible ternary that fell back to the now-gone #approval-allow-once -- both now
focus #approval-submit unconditionally, the card's only possible action target.

Tests: removed the one test that synthesized the legacy payload directly
(test_console_approvals_chip_activation_focuses_pending_approval_card); fixed a stale
id assertion in its neighbor; added test_legacy_single_approval_api_was_removed and
test_card_never_renders_the_retired_single_approval_buttons (Tests/UI/
test_console_mcp_approval.py) covering the card's default/batch/cleared states.

Verification: test_console_mcp_approval.py + test_chat_approval_card.py -> 41 passed,
2 pre-existing failures (confirmed via git stash against unmodified HEAD, unrelated).
test_console_parallel_runs.py (12/12) and most of test_console_workbench_contract.py
(36/58) are currently blocked by an unrelated, freshly-landed regression: commit
1df0c4cb4 made TldwCli.current_runtime_backend a read-only property without updating
the shared test helper Tests/UI/test_screen_navigation.py:800, which still assigns it
directly. Confirmed pre-existing (identical on unmodified HEAD via git stash) and
out of scope for this task; flagged for the fleet since it will block other
follow-up tasks' Console UI test verification too. Full detail in
.superpowers/sdd/fleet-followups-910-915/task-914-report.md.
<!-- SECTION:NOTES:END -->
