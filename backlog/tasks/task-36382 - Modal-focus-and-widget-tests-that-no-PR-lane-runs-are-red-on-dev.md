---
id: TASK-36382
title: Modal, focus and widget tests that no PR lane runs are red on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: '2026-10-23 16:51'
labels:
- follow-up
- tests
- css
priority: medium
references:
- Tests/UI/test_modal_wide_tier.py
- Tests/UI/test_non_obscuring_focus_contract.py
- Tests/UI/test_css_staleness_manifest.py
- Tests/UI/test_llamacpp_setup_refresh_gate.py
- Tests/UI/test_evals_skill_eval_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the implementation and reviews of TASK-35648/35649 (PR #3217, 2026-10-22). These are red on dev 165e31fab8, and no PR lane runs them:
- **`test_modal_wide_tier.py`:**
  - `test_every_modalscreen_class_is_tiered_or_explicitly_skipped` fails on three stale skip names for deleted classes (`ConfirmDialog`, `ConfirmDisableDialog`, `DeleteConfirmDialog`).
  - `test_every_registered_anchor_has_a_wide_tier_rule[path0]` and `[path1]` fail because `#console-hooks-review` has no wide-tier rule.
- **`test_non_obscuring_focus_contract.py`**, 7 failures:
  - Three come from the deleted `_tools-settings.tcss`: two `FileNotFoundError`s, and one missing `.tools-nav-pane .ts-nav-button.active-nav` block.
  - Four are stale token or geometry pins: `test_shared_button_hover_uses_non_obscuring_surface_contract`, `test_console_selected_message_actions_keep_clickable_hit_targets`, `test_chat_rag_focus_within_uses_non_semantic_container_cue`, and `test_bundled_native_choice_and_tree_states_match_source_contracts` (TASK-35572 already lists that last one for lane placement).
- **`test_css_staleness_manifest.py`**: its end-to-end unicode-checkout (漢) test is red.
- **`test_llamacpp_setup_refresh_gate.py`**: 3 tests fail because a `SimpleNamespace` fake has no `.renderable` (`llamacpp_setup_view.py` around :632).
- **`test_evals_skill_eval_screen.py`**:
  - `test_skill_eval_bench_can_be_deleted` fails with `NoMatches #evals-delete-bench` in 3 of 3 deterministic reruns.
  - In two review runs at load 13-18, `test_unfed_remount_restores_depth_and_running_state` failed both times and `test_launch_picks_survive_navigation_away_and_back` failed once.

`test_css_class_coverage_contract` is red too, but TASK-31860 covers it (the contract reads only the pre-split bundle).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each named test passes on dev, or is retired with a recorded reason
- [x] #2 No test in these files references a deleted class or a deleted sheet
- [x] #3 Each named file runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory: run the five files at base 52b3207ff1 and record each failure with its cause.
2. For each failure decide which side is wrong: update stale pins (deleted classes, the deleted `_tools-settings.tcss`, old tokens or geometry) to the current CSS after reading it; fix small clear product bugs (the `#console-hooks-review` wide-tier rule) with a `run_test` render check at 211x44 and 235x52.
3. llamacpp gate: make the `SimpleNamespace` fake match the current widget API.
4. Evals: fix `test_skill_eval_bench_can_be_deleted`; run the two load-flaky tests serially 10 times and fix any race with a bounded wait.
5. Lanes: append the five files to `scripts/ui_pr_gate_census.txt` at the tail with `MINIMUM_FILES` at the new count; measure runtime against the job caps; run `Tests/CI -n 4`.
6. Preflight, ratchets, notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Inventory at base 52b3207ff1** (all five files, `-n 4`): 15 failed, 272 passed.

| File | Failure | Cause | Side that was wrong |
|---|---|---|---|
| `test_modal_wide_tier.py` | reverse sweep: stale skips `ConfirmDialog`, `ConfirmDisableDialog`, `DeleteConfirmDialog` | classes deleted (TASK-33081, tier-2 batches) | test |
| same | reverse sweep, second assertion (hidden by the first): 15 ModalScreen subclasses neither anchored nor skipped | modals added after wave 2 | test registry (see below) |
| same | `test_every_registered_anchor_has_a_wide_tier_rule[path0/1]`: no `#console-hooks-review` rule | 1256af2d39 moved the rule (with the modal's CSS) to `features/_settings.tcss` / `screen_agentic_settings.tcss`, which the modal loads by `CSS_PATH`; the contract read only `_agentic_terminal.tcss` and the boot bundle | test |
| `test_non_obscuring_focus_contract.py` | two `FileNotFoundError` + missing `.tools-nav-pane .ts-nav-button.active-nav` | `_tools-settings.tcss` and its screen deleted (TASK-33081/33244); no selector survives anywhere | test (retired) |
| same | `test_shared_button_hover_...` | ADR-150 `Button:hover` uses `$ds-hover-bg`/`$ds-hover-fg` (= `$surface-lighten-1`/`$text`) | test |
| same | `test_console_selected_message_actions_...` | rules carry `!important` (the int parser rejected it); second sheet read was the library sheet, but the console rules ride the boot bundle; min-width is 4 with no padding since PR #2152 | test |
| same | `test_chat_rag_focus_within_...` | `$panel` tokenized to `$ds-surface-panel` | test |
| same | `test_bundled_native_choice_...` | Switch model's list is a bar (TASK-33004.4), and three compact-list focus cues (`#console-workspace-tree:focus`, the two Settings workspace pickers) were never listed | test |
| `test_css_staleness_manifest.py` | unicode-checkout end-to-end: `FileNotFoundError core/_variables.tcss` | `build_widget_defaults()` now reads the token file; the scratch tree lacked it (not a unicode problem) | test |
| `test_llamacpp_setup_refresh_gate.py` | 3 x `SimpleNamespace` has no `renderable` | the view compares `Static.renderable` before `update()` | test fake |
| `test_evals_skill_eval_screen.py` | `test_skill_eval_bench_can_be_deleted` NoMatches | waited on `_selection.kind`, which `select()` sets at once; the inspector swap comes later | test wait |

**Product truth checked, no product change.** The hook-review wide tier renders: a temporary `run_test` probe (deleted) measured `#console-hooks-review` at 170 columns at 211x44 and 235x52 and 108 at 120x40; with the screen-sheet rule disabled it measured 110 at both wide sizes. The same disabled rule makes the new contract's built-sheets parameter fail (source parameter passes).

**Changes.**
- Modal contract: anchors are read from the source modules and the built sheets they produce (boot bundle plus the Settings screen sheet), with no anchor in two homes; the orphan check reads all four. Registry: three dead skip names removed; the 15 unlisted modals triaged by base width. Five are tiny or already wider than the tier (receipt 60, draft-save 60, session summary 44, theme leave 60, response rules 95%). Ten carry fixed widths the ladder would relax, so they join the registry's existing "follow-up wave" convention rather than a silent gap; rolling the tier out to them is product work (proposed follow-up).
- Focus contract: `_tools-settings.tcss` checks and the deleted `_unified_sidebar.tcss` source-only entry retired; hover helper accepts the ADR-150 tokens, and the button-hover test pins their definitions; `css_int_declaration` accepts `!important`; bar list gains `ConsoleModelPopover`; a focus-cue list with its own contract covers the three compact-list cursors.
- Staleness test writes a one-token `_variables.tcss`; llamacpp fake gains `renderable=""`.
- Evals waits target the control the next line touches (delete button, run-again, subject-dir, detail Stop, a mounted generator `Select`), the press-then-check sites wait for the effect (`dispatched`, `token.cancelled`), and the launch-picks wait polls all three restored values.

**Flaky evals tests.** Serial x10 at base, load 28-35: `test_launch_picks_...` failed 3/10, `test_unfed_remount_...` 0/10. An instrumented trace of a failing launch-picks run showed the compose-time callback restoring depth and the test asserting before the swap's safety net restored the model picks; the persisted picks were intact. After the fix: 20/20 serial and 24/24 across four concurrent loops (load 20-45). `test_unfed_remount_...` reproduced only under heavier contention, as `NoMatches 'SelectOverlay'` on the generator `Select` (queryable before it composed); waiting for `is_mounted` gave 32/32 concurrent. Whole-file runs also caught `test_report_rerun_running_state_and_stop` (1/5 `-n 4` runs, `again.disabled` False; then 1/24 `token.cancelled` after one pause); 32/32 concurrent after its waits. The `again.disabled` failure was not reproduced in 52 instrumented runs, so its cause is unconfirmed; the control wait closes the compose window Textual allows.

**Lanes (AC#3).** Appended at the census tail: the four smaller files whole, and `test_modal_wide_tier.py` by three node ids (the anchor contract, the reverse sweep, the orphan check). The whole modal file took 66 s serially, over the lane's 60 s per-file rule; its ~130 mounted spot-geometry cases stay out. A replay of three 2026-10-23 PR runs' shard logs puts every UI Fast Lane shard at 14.0-17.7 pytest minutes (job wall 12.7-19.2 of 20, before this change). The entries are ordered so the two ~27 s ones land on the lightest shards in that replay and the 1-7 s ones on the heaviest; no shard's projected maximum moves. Serially, in census order, the seven entries ran 154 passed in 87 s (load ~30).

**Evidence.** All five files `-n 4`: 286 passed (one parametrize case retired with the tools sheet). `Tests/CI -n 4`: 1025 passed, 1 skipped. Preflight passed. Ruff: no new findings in touched files (two unused imports this change created were removed).

**Not verified.** The files' first CI run; CI timings are estimated from local serial runs. The remaining Evals file tests that use one-pause press checks (`test_compare_with_previous_run_group`, `test_middle_report_...`, `test_artifact_listing_...`, `test_new_skill_eval_button_...`, `test_subject_picker_product_path_run_completes`) failed only at load 50-60 with three concurrent `-n 4` runs; they passed in every run at load 20-45.

Files: `Tests/UI/modal_wide_tier_registry.py`, `Tests/UI/test_modal_wide_tier.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, `Tests/UI/test_css_staleness_manifest.py`, `Tests/UI/test_llamacpp_setup_refresh_gate.py`, `Tests/UI/test_evals_skill_eval_screen.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
