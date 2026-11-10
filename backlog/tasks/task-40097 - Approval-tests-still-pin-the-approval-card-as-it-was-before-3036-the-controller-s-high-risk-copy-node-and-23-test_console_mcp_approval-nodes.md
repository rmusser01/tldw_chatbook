---
id: TASK-40097
title: "Approval tests still pin the approval card as it was before #3036: the controller's high-risk copy node and 23 test_console_mcp_approval nodes"
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
- tldw_chatbook/css/features/_console_approvals.tcss
- Tests/UI/test_chat_approval_card.py
dependencies:
- TASK-32278
- TASK-34565
- TASK-39154
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (coordinator after #3036 merged; widened after review r12fr). Checked on dev 3a6498ae14, 2026-11-09 (run.sh, `-n 4`, load ~5-25). Line numbers are that dev's.

**The cause.** #3036 (merge b498bc3aec, 2026-11-09), the approval-presentation work of TASK-34565, rewrote the approval card. It is the only merge since dev b590a15afd that changed `chat_approval_card.py` or the approval CSS.
- a2fcbe61bc ("Integrate captured approval controls with current Console owners", 2026-10-15) replaced `_REASON_COPY["risk_floored"]` with the policy sentence (`Widgets/Chat_Widgets/chat_approval_card.py:281`). It also deleted `_RISK_FLOORED_MUTATES_COPY` and its `effects` branch in `format_approval_reason`. The blast radius now lives on a separate line, `format_approval_effects` (`:165-175`); for `mutates_local` it reads `Effects: may modify local data` (`:159`).
- The new copy is pinned on purpose by `Tests/UI/test_chat_approval_card.py::test_the_high_risk_explanation_describes_current_policy` (`:526`, asserts `:536-537`), and in `test_approval_presentation.py`, `test_approval_row_information_budget.py` and `test_console_approval_first_open_render.py`.
- The merge also moved the approval rules out of `_agentic_terminal.tcss` into `css/features/_console_approvals.tcss` (for example `#approval-batch-rows {` at `:42`), and renamed the fast-button labels.

**The controller red.** `Tests/Chat/test_console_chat_controller.py::test_review_hook_gives_a_mutating_builtin_the_mutation_effect` (`:3787`) fails at `:3818`:
- expected `High risk: this tool changes local data and always asks first.`
- got `High risk: current policy requires approval for this call.`

The test drives the real review hook with the real `WriteFileTool` and `ReadFileTool`. It pins task-32278 AC#3: the row's blast-radius sentence says "changes" for a mutating tool and "reads" for a reading one. Its `effects` checks (`("mutates_local",)` and `()`) still pass.

**`Tests/UI/test_console_mcp_approval.py`.** Under a temporary `bootstrap_profile` marker (a scratch plugin, not committed), two whole-file runs gave 31 failed / 92 passed each. Seven of those were red before #3036 too (on dev b590a15afd and on the round-12 stack); TASK-39154 lists them. The 23 newly red in both runs:
- copy and labels: `test_reason_badges_carry_a_visible_why_line_not_a_tooltip` (`:384`, the old reading sentence at `:421-423`); `test_single_row_batch_renders_fast_approve_and_deny_buttons` (`:1084`) and `test_two_option_save_row_renders_once_and_deny_only` (`'Allow once' == 'Approve once'`);
- CSS: `test_approval_batch_rows_height_rule_pinned_in_bundle_source_and_bundle` (`:2104`, `_agentic_terminal.tcss is missing '#approval-batch-rows {'`);
- bulk actions: `test_approve_all_and_deny_all_bulk_set_every_row`, `test_approve_all_leaves_raw_shell_row_on_deny_and_flags_needs_decision`, `test_bulk_actions_flag_a_row_they_could_not_apply_to`, `test_bulk_approve_and_deny_all_are_row_aware_and_never_raise` (`'approve_session' == 'deny'`), `test_deny_all_sets_raw_shell_row_to_deny_and_clears_needs_decision_prefix`, `test_changing_a_flagged_rows_select_clears_needs_decision`, `test_changing_raw_shell_select_after_approve_all_clears_needs_decision_prefix`;
- single-row fast buttons: `test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` (`NoMatches '.approval-row-fast-approve'`);
- raw-shell row: `test_raw_shell_row_defaults_to_deny_and_enter_does_not_submit` (`:352`, focus on `approval-request-summary`, not `approval-row-decision-1-0`), `test_raw_shell_row_has_no_generic_scope_static_but_an_mcp_row_does`;
- Alt+A: `test_alt_a_focuses_the_pending_approval_decision_select` (`:1944`), `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed`;
- submit and rounds: `test_submit_posts_approval_decided_with_per_row_decisions` (`:772`), `test_submit_echoes_back_the_round_id_stamped_by_set_batch`, `test_new_batch_reenables_submit_after_a_prior_round_disabled_it`, `test_back_to_back_changed_rounds_leave_only_latest_controls`, `test_changed_ordinary_one_row_reuses_only_noncommitting_widgets`, `test_unrelated_select_changed_does_not_touch_batch_rows`;
- `test_approved_definitive_tool_stays_mounted_until_real_terminal[success]` ("approved handler never started"; `[crash]` was red before).

Two more failed in one run each: `test_stale_generation_fast_button_press_is_a_noop` and `test_fast_approve_button_disables_after_press_to_prevent_double_submit` (`assert True is False` on the fast-approve button's disabled state).

**Ownership.** TASK-39154 owns this file's profile-redirect reds, its pre-#3036 own reds, and its PR lane (the whole file). The census forbids a single-test entry beside the same file's whole-file entry, so this task adds no lane for this file.

**Lanes.** Neither file runs on a PR lane, which is how the drift went unseen. TASK-39138 (To Do on this dev) re-pins 34 other controller nodes; the controller node here was green when that list was made.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The controller node pins the current `risk_floored` sentence and still proves the mutating row's blast radius through the row's effects line (`Effects: may modify local data` for `write_file`, none for `read_file`); its red at `:3818` is the before evidence
- [ ] #2 Each of the 23 `test_console_mcp_approval.py` nodes, and the two intermittent ones, passes under the file's command against the post-#3036 card, or is retired with the #3036 commit that removed what it pinned named; each is classified as test drift or a product defect, and a product defect gets a test that is red before its fix
- [ ] #3 No pin is loosened to pass: each re-pin asserts the post-#3036 behaviour that replaced the old one (label, rule location, focus target, bulk result, submit payload)
- [ ] #4 The controller node runs on a PR lane (never the fast PR contract), with three timings recorded; `test_console_mcp_approval.py` is laned by TASK-39154, not here
<!-- AC:END -->
