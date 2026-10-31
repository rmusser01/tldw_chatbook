---
id: TASK-36382
title: Modal, focus and widget tests that no PR lane runs are red on dev
status: To Do
created_date: 2026-10-22 23:58
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
- [ ] #1 Each named test passes on dev, or is retired with a recorded reason
- [ ] #2 No test in these files references a deleted class or a deleted sheet
- [ ] #3 Each named file runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->
