---
id: TASK-40097
title: "Two approval-card tests still pin the high-risk reads or changes local data copy that the approval UX merge replaced with current policy requires approval"
status: To Do
created_date: 2026-11-09 23:55
labels:
- follow-up
- tests
- console
- approvals
priority: low
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/UI/test_console_mcp_approval.py
- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py
- Tests/UI/test_chat_approval_card.py
dependencies:
- TASK-32278
- TASK-34565
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (coordinator, after #3036 merged). Checked on dev 3a6498ae14, 2026-11-09 (run.sh, `-n 4`, load ~5-25). Line numbers are that dev's.

**The red.** `Tests/Chat/test_console_chat_controller.py::test_review_hook_gives_a_mutating_builtin_the_mutation_effect` (`:3787`) fails at `:3818`:
- expected `High risk: this tool changes local data and always asks first.`
- got `High risk: current policy requires approval for this call.`

The test drives the real review hook with the real `WriteFileTool` and `ReadFileTool`. It pins task-32278 AC#3: the row's blast-radius sentence says "changes" for a mutating tool and "reads" for a reading one. Its `effects` checks (`("mutates_local",)` and `()`) still pass.

**The change.** a2fcbe61bc ("Integrate captured approval controls with current Console owners", 2026-10-15) made this change on purpose. It reached dev with #3036 (merge b498bc3aec, 2026-11-09), the approval-presentation work of TASK-34565.
- `_REASON_COPY["risk_floored"]` became the policy sentence (`Widgets/Chat_Widgets/chat_approval_card.py:281`).
- `_RISK_FLOORED_MUTATES_COPY` and its `effects` branch in `format_approval_reason` were deleted.
- The blast radius now lives on a separate line, `format_approval_effects` (`:165-175`). For `mutates_local` it reads `Effects: may modify local data` (`:159`).
- The new copy is pinned for both rows on purpose by `Tests/UI/test_chat_approval_card.py::test_the_high_risk_explanation_describes_current_policy` (`:526`, asserts `:536-537`). It is also pinned in `test_approval_presentation.py`, `test_approval_row_information_budget.py` and `test_console_approval_first_open_render.py`.

**A second stale pin.** `Tests/UI/test_console_mcp_approval.py::test_reason_badges_carry_a_visible_why_line_not_a_tooltip` (`:384`) asserts the old reading sentence at `:421-423`. It fails the same way plain and under a temporary `bootstrap_profile` marker (a scratch plugin, not committed).

**No other old-copy pin.** No other test or production file holds "always asks first" for this card.

**Lanes.** Neither file runs on a PR lane, which is how the drift went unseen. TASK-39138 (To Do on this dev) re-pins 34 other controller nodes; this one was green when that list was made.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both nodes pin the current `risk_floored` sentence, and the controller test still proves the mutating row's blast radius through the row's effects line (`Effects: may modify local data` for `write_file`, none for `read_file`); the red at `:3818` and `:421` is recorded as the before evidence
- [ ] #2 The re-pin is recorded as test drift from a2fcbe61bc (#3036), with no production change
- [ ] #3 Both nodes run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
