---
id: TASK-40097
title: "Approval tests still pin the approval card as it was before #3036: the controller's high-risk copy node and 22 test_console_mcp_approval nodes"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 20:35'
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

**`Tests/UI/test_console_mcp_approval.py`.** Under a temporary `bootstrap_profile` marker (a scratch plugin, not committed), two whole-file runs gave 31 failed / 92 passed each. Eight of those were red before #3036 too (on dev b590a15afd or on the round-12 stack 9091820bf2, which has no #3036); TASK-39154 owns them. They include both `test_approved_definitive_tool_stays_mounted_until_real_terminal` cases, `[crash]` and `[success]`, which fail on the stack with "approved handler never started". The 22 newly red in both runs:
- copy and labels: `test_reason_badges_carry_a_visible_why_line_not_a_tooltip` (`:384`, the old reading sentence at `:421-423`); `test_single_row_batch_renders_fast_approve_and_deny_buttons` (`:1084`) and `test_two_option_save_row_renders_once_and_deny_only` (`'Allow once' == 'Approve once'`);
- CSS: `test_approval_batch_rows_height_rule_pinned_in_bundle_source_and_bundle` (`:2104`, `_agentic_terminal.tcss is missing '#approval-batch-rows {'`);
- bulk actions: `test_approve_all_and_deny_all_bulk_set_every_row`, `test_approve_all_leaves_raw_shell_row_on_deny_and_flags_needs_decision`, `test_bulk_actions_flag_a_row_they_could_not_apply_to`, `test_bulk_approve_and_deny_all_are_row_aware_and_never_raise` (`'approve_session' == 'deny'`), `test_deny_all_sets_raw_shell_row_to_deny_and_clears_needs_decision_prefix`, `test_changing_a_flagged_rows_select_clears_needs_decision`, `test_changing_raw_shell_select_after_approve_all_clears_needs_decision_prefix`;
- single-row fast buttons: `test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` (`NoMatches '.approval-row-fast-approve'`);
- raw-shell row: `test_raw_shell_row_defaults_to_deny_and_enter_does_not_submit` (`:352`, focus on `approval-request-summary`, not `approval-row-decision-1-0`), `test_raw_shell_row_has_no_generic_scope_static_but_an_mcp_row_does`;
- Alt+A: `test_alt_a_focuses_the_pending_approval_decision_select` (`:1944`), `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed`;
- submit and rounds: `test_submit_posts_approval_decided_with_per_row_decisions` (`:772`), `test_submit_echoes_back_the_round_id_stamped_by_set_batch`, `test_new_batch_reenables_submit_after_a_prior_round_disabled_it`, `test_back_to_back_changed_rounds_leave_only_latest_controls`, `test_changed_ordinary_one_row_reuses_only_noncommitting_widgets`, `test_unrelated_select_changed_does_not_touch_batch_rows`.

Two more failed in one run each: `test_stale_generation_fast_button_press_is_a_noop` and `test_fast_approve_button_disables_after_press_to_prevent_double_submit` (`assert True is False` on the fast-approve button's disabled state).

**Ownership.** TASK-39154 owns this file's profile-redirect reds, its pre-#3036 own reds, and its PR lane (the whole file). The census forbids a single-test entry beside the same file's whole-file entry, so this task adds no lane for this file.

**Lanes.** Neither file runs on a PR lane, which is how the drift went unseen. TASK-39138 (To Do on this dev) re-pins 34 other controller nodes; the controller node here was green when that list was made.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The controller node pins the current `risk_floored` sentence and still proves the mutating row's blast radius through the row's effects line (`Effects: may modify local data` for `write_file`, none for `read_file`); its red at `:3818` is the before evidence
- [x] #2 Each of the 22 `test_console_mcp_approval.py` nodes, and the two intermittent ones, passes under the file's command against the post-#3036 card, or is retired with the #3036 commit that removed what it pinned named; each is classified as test drift or a product defect, and a product defect gets a test that is red before its fix
- [x] #3 No pin is loosened to pass: each re-pin asserts the post-#3036 behaviour that replaced the old one (label, rule location, focus target, bulk result, submit payload)
- [x] #4 The controller node runs on a PR lane (never the fast PR contract), with three timings recorded; `test_console_mcp_approval.py` is laned by TASK-39154, not here
<!-- AC:END -->

## Implementation Plan

1. Show the controller node red at `:3818`-equivalent and the 22 (+2 intermittent) `test_console_mcp_approval.py` reds at base ae9f9e8d21 under the file's marker.
2. Re-pin the controller node to the current `risk_floored` sentence and prove the blast radius through `format_approval_effects` (write_file: `Effects: may modify local data`; read_file: none).
3. For each approval node, read #3036's replacement (label, rule file, focus target, bulk result, submit payload) and re-pin to it, or fix a product defect with a red-first test.
4. Time the controller node serially three times (the parent wires its lane).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-10 (round 13, branch fu/r13-cred, base ae9f9e8d21). All runs through run.sh at `-n 4`.

**Red first, same command** (`pytest Tests/UI/test_console_mcp_approval.py <controller node> -n 4`, the approval file under TASK-39154's module marker, load ~55): 31 failed. The controller node failed `'High risk: current policy requires approval for this call.' == 'High risk: this tool changes local data and always asks first.'` (now `:3894`). After: 123 passed, and 215 passed twice with TASK-39154's two Chat files (load 70-88). No product defect found; every red is drift from #3036 (merge b498bc3aec), whose first commit a2fcbe61bc rewrote the card and a41da77591 added deferred decision readiness.

**Controller node (AC#1).** Both rows pin the policy sentence; the blast radius is pinned on the effects line: `format_approval_effects` gives `Effects: may modify local data` for `write_file` and `""` for `read_file`, beside the unchanged `effects` checks.

**`test_console_mcp_approval.py` (AC#2, AC#3).** Two helpers: `_until_actionable` waits for `card._information_ready` (a41da77591), and `_open_staged_choices` presses the visible More options and asserts the Selects and Apply are shown (a2fcbe61bc hides both until then). Per node, what the re-pin asserts now:
- Labels: `test_single_row_batch_renders_fast_approve_and_deny_buttons`, `test_two_option_save_row_renders_once_and_deny_only`: "Allow once" / "Deny"; Select+Apply reached through More options, with the same submit payload.
- Copy: `test_reason_badges_carry_a_visible_why_line_not_a_tooltip`: the policy sentence, and no effects line for a call that declares none.
- Rule location: `test_approval_batch_rows_height_rule_pinned_in_bundle_source_and_bundle` (and TASK-39154's three CSS pins): `features/_console_approvals.tcss` plus the bundle, values resolved from tokens.
- Bulk result (a2fcbe61bc's `_commit_bulk`: one press sets every row AND commits, all or nothing): `test_approve_all_and_deny_all_bulk_set_every_row` (each on its own round, payload and round id); `test_bulk_approve_and_deny_all_are_row_aware_and_never_raise` and `test_bulk_actions_flag_a_row_they_could_not_apply_to` (Deny all with an undeniable row commits nothing; Allow all reads "Review individually" and opens the staged choices; values unchanged, nothing raises, no row flagged); `test_approve_all_leaves_raw_shell_row_on_deny_and_flags_needs_decision` (no commit, shell row stays Deny, Apply disabled, hint "Review required. Select each shell choice, even to keep Deny."); `test_changing_raw_shell_select_after_approve_all_clears_needs_decision_prefix` (a deliberate shell choice, `ApprovalChoice.Deliberate`, clears that hint and enables Apply, whose payload keeps Deny); `test_deny_all_sets_raw_shell_row_to_deny_and_clears_needs_decision_prefix` (Deny all commits both rows from the staged review); `test_unrelated_select_changed_does_not_touch_batch_rows` (a foreign `Select.Changed` leaves every scope line, the count summary and the values unchanged).
- Retired: `test_changing_a_flagged_rows_select_clears_needs_decision`. a2fcbe61bc removed every production setter of `needs-decision` (`_set_all_batch_decisions` has no caller), so the flagged row it starts from cannot exist.
- Fast buttons: `test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`: the buttons sit in `#approval-single-actions` below the row, each at least its label wide (`width: auto` there), in order Allow once, Deny, More options; row height bound kept. `test_changed_ordinary_one_row_reuses_only_noncommitting_widgets`: a2fcbe61bc enables a fast button only for a legal decision, so on the Until-Chatbook-exits/Deny row Allow once is disabled and Deny commits.
- Focus target: `test_raw_shell_row_defaults_to_deny_and_enter_does_not_submit`, `test_alt_a_focuses_the_pending_approval_decision_select`, `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed`: the neutral `ApprovalSummary` (`first_focus_widget_id()`), never a committing control; Enter submits nothing.
- Scope line: `test_raw_shell_row_has_no_generic_scope_static_but_an_mcp_row_does`: a2fcbe61bc gives every row one current-consent line (production's captured view words the shell's session scope); the shell row has exactly one, stating its Deny default, beside its own session statement.
- Submit payload and rounds: `test_submit_posts_approval_decided_with_per_row_decisions`, `test_submit_echoes_back_the_round_id_stamped_by_set_batch`, `test_new_batch_reenables_submit_after_a_prior_round_disabled_it`, `test_back_to_back_changed_rounds_leave_only_latest_controls`: Apply through More options; payloads and round ids unchanged.
- Intermittent: `test_stale_generation_fast_button_press_is_a_noop` (seen red once here: the fresh press beat readiness) and `test_fast_approve_button_disables_after_press_to_prevent_double_submit` now wait for readiness, as do the two fast-submit tests that share the race.

**Lane timings** (AC#4). The node's file is module bootstrap_profile. Serial timings, run.sh, `pytest <node> --timeout=180 --tb=short -q`, wall seconds (pytest seconds; 1-min load before each run): 13 (3.1; 62), 10 (2.6; 71), 11 (2.4; 56).

**Lane (AC#4).** `Tests/Chat/test_console_chat_controller.py::test_review_hook_gives_a_mutating_builtin_the_mutation_effect` runs by node id in `recovered-suites` (bootstrap_profile, last block); no other step lists the controller file. Node timings, serial: 13 s, 10 s and 11 s wall. `recovered-suites`' whole pytest command, serial through run.sh: 943 s, 1088 s and 1020 s wall, 871 passed and 1 skipped each (1-minute load 32-44). In CI the same step before this round ran 733 passed in 307 s (run 42841528703). This round's additions take about 330 s serial locally at most, so the job should stay near half its 20-minute cap. `test_console_mcp_approval.py` is laned by TASK-39154 (UI census, whole file).

**Review r13c-rev1.**
- **M4.** Four tests named or described a needs-decision flag nothing produces. They are renamed to what they assert, docstrings rewritten:
  - `test_bulk_actions_commit_nothing_when_a_row_cannot_take_the_bulk_choice`
  - `test_review_individually_keeps_raw_shell_row_on_deny_and_requires_its_choice`
  - `test_deliberate_raw_shell_choice_after_review_individually_enables_apply`
  - `test_deny_all_after_review_individually_denies_every_row_and_commits`

  The vacuous `not row.has_class("needs-decision")` asserts are dropped. The `test_chat_approval_card.py` comment points at the new name. No census entry named them; the file is laned whole.
- **(c) Dead production code removed.** Since a2fcbe61bc nothing produced the needs-decision flag. Removed: `_set_all_batch_decisions`, `_mark_row_needs_decision`, `_clear_row_needs_decision`, `NEEDS_DECISION_PREFIX`, the flag-only `_batch_base_headers` list and the two `remove_class("needs-decision")` calls, plus the `.approval-row.needs-decision` rule in `css/features/_console_approvals.tcss`. The bundle was rebuilt and `check_bundle_sync.py` passes.
- **Runs.** `git grep` finds no reference outside history. The approval files (`test_console_mcp_approval.py`, `test_chat_approval_card.py`, `Tests/UI/test_approval*.py`, `test_console_approval*.py`) at `-n 4` gave 363 passed, 2 failed:
  - `test_the_longest_decision_label_paints_on_one_line` passed 2 of 2 on rerun.
  - `test_first_open_paints_an_answerable_card_nav_away_path` fails "screens are never cached" with the pre-deletion card too, so this change did not cause it.
<!-- SECTION:NOTES:END -->
