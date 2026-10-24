---
id: TASK-36382
title: Modal, focus and widget tests that no PR lane runs are red on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: '2026-10-24 15:47'
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

**Flaky evals tests.** Serial x10 at base, load 28-35: `test_launch_picks_...` failed 3/10, `test_unfed_remount_...` 0/10. An instrumented trace of a failing launch-picks run showed the compose-time callback restoring depth and the test asserting before the swap's safety net restored the model picks; the persisted picks were intact. After the fix: 20/20 serial and 24/24 across four concurrent loops (load 20-45). `test_unfed_remount_...` reproduced only under heavier contention, as `NoMatches 'SelectOverlay'` on the generator `Select` (queryable before it composed); waiting for `is_mounted` gave 32/32 concurrent. Whole-file runs also caught `test_report_rerun_running_state_and_stop` (1/5 `-n 4` runs, `again.disabled` False; then 1/24 `token.cancelled` after one pause); 32/32 concurrent after its waits. `again.disabled` (not reproduced in 52 instrumented runs) has its cause in the code, found in review: `_set_skill_eval_running_ui` swallows `QueryError` for `#skill-eval-run-again`, so a call that lands after the detail mounts but before it composes its buttons changes nothing; waiting for the detail's Stop button closes that window.

**Lanes (AC#3).** Appended at the census tail: the four smaller files whole, and `test_modal_wide_tier.py` by node id: all five static contracts (anchor rules, reverse sweep, orphan check, and, added in review, the renamed-modal-id guard over every anchor and the block check). The whole modal file took 66 s serially, over the lane's 60 s per-file rule; its 40 mounted spot-geometry cases and the toggle MRO check stay out. A replay of three 2026-10-23 PR runs' shard logs puts every UI Fast Lane shard at 14.0-17.7 pytest minutes (job wall 12.7-19.2 of 20, before this change). After the rebase onto dev's 664-entry census, the order was re-derived for the new positions: the Evals file lands on shard 8 and the reverse sweep on shard 7, the two lightest in the run that set dev's census (12.9 and 15.3 min), with only sub-second nodes on the near-cap shards 3 and 9 (19.2 min). Any tail append ahead of this block shifts every position, so `--shard` is re-checked after each rebase. Serially, in census order, the entries ran 245 passed in 83 s (load ~25).

**Evidence.** All five files `-n 4`: 286 passed (one parametrize case retired with the tools sheet). Evals file serially: 32 passed. `Tests/CI -n 4`: 1025 passed, 1 skipped. Census checker OK; preflight passed. Ruff: no new findings in touched files.

**Review round 1.** Five more Evals tests synced by timing: `test_new_skill_eval_button_...` waited on `_selection.kind` and then queried the panel (an unused query, deleted; its depth and estimate asserts now poll), and the compare, middle-report, artifacts-"more" and subject-pick tests asserted after one pause (now bounded waits on the asserted state). Under the reviewer's probe plugin, which makes `pilot.pause()` see one 20 ms idle window, the previous commit's `test_new_skill_eval_button_...` failed 3/3 with `NoMatches 'SkillEvalPanel'`. After the fix, a 10-run check passed all five, but that was too few for the subject-pick test: review round 2 saw it fail 4 of 37 probe runs. The registry's skip-list comment now says "follow-up wave" entries are queued gaps, not verdicts.

**Review round 2.** The subject-pick failures were in the shared helper `_pick_via_overlay` (`test_evals_skill_eval_panel.py`). Its coordinate click could land before layout settled, leaving the Select closed and focus elsewhere. The helper now opens the dropdown from the keyboard: it focuses the Select, presses Space (which opens Select and SetSelect alike), and waits for the dropdown to open and take focus. It then presses Home plus `downs - 1` downs and Enter, and waits for the dropdown to close. A keyboard open highlights the NULL padding row where a click highlighted nothing, and without Home every pick moved one row: the first rewrite failed both screen tests 20/20 under the probe and three panel tests normally. The subject test's Run is now pressed, not clicked, after a wait for the persisted picks. Under the probe, the subject-pick and launch-picks tests passed 20/20 each. The panel file's other callers pass. Under the probe, `test_directory_path_subject_enables_run` fails 3 of 5 at its own coordinate Run click both before and after this change; that file runs on no lane.

**Post-rebase round (onto dev after TASK-36512's census re-deal).** PR run 40007935895 showed `test_every_anchor_targets_real_code`'s ~90 parametrized cases costing about 1 s of teardown each. It is now one test that checks every anchor and names every one that no longer resolves in a single assertion; the census node id is unchanged. Renaming one registry anchor made it fail naming that anchor and its owner, and it was then reverted. Measured alone, its setup, call and teardown take 0.16 s, in a 1.2 s session. A whole-census `--collect-only --keep-duplicates` collects 89 fewer tests than at the previous head (the 90 cases became one), with no duplicate node ids. The block is now in the order the review's measured slack calls for (heavy entries on the shards with the most slack, sub-second ones on the near-cap shards), and the census and floor comments no longer claim the anchor check is cheap in total. Five files `-n 4`: 197 passed; `Tests/CI -n 4`: 1029 passed, 1 skipped; census checker OK; preflight passed.

**Not verified.** This round's placement has not yet run in CI.

Files: `Tests/UI/modal_wide_tier_registry.py`, `Tests/UI/test_modal_wide_tier.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, `Tests/UI/test_css_staleness_manifest.py`, `Tests/UI/test_llamacpp_setup_refresh_gate.py`, `Tests/UI/test_evals_skill_eval_screen.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
