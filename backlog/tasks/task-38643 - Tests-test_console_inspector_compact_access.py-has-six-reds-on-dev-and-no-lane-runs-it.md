---
id: TASK-38643
title: 'Tests: test_console_inspector_compact_access.py has six reds on dev, and no lane runs it'
status: Done
assignee:
  - '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 00:38
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_inspector_compact_access.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Widgets/Console/console_setup_modal.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-36385
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). Through the scratch-profile wrapper, the file gives 6 failed, 6 passed with `-n 4`, and the same six fail serially. The five that are not a NameError also fail alone. None is the local RecoveryRequired trap. No PR lane or census entry runs the file, and Nightly Deep last ran on 2026-09-27, which is why A and C went unnoticed. Three causes:

**A. Undefined name.** `test_tools_chip_opens_inspector_at_80x24_single_pane` calls `line.post_message(...)` at :305. `line` exists only inside the helper `_activate_status_segment` (:212). This has been red since d102ce9cb5 (PR #3145, 2026-10-17, segment-contract conversion). A read-only probe that bound `line` to `#console-status-text` made it pass.

**B. Clicks land on the setup modal.** In `test_visible_attach_context_action_switches_rails_without_file_picker` (:158) and `test_inspector_handle_opens_and_collapses_rail_at_140_cols` (:187), `pilot.click` lands on the blocking first-run `#console-setup-modal`. The test harness has no ready provider, and the modal blocks in card mode (console_setup_modal.py:118-120; chat_screen.py `_sync_console_setup_modal`). So the click returns False and the rail never moves. The file's keyboard tests pass. A probe that made the modal non-blocking made both pass. This is the compact-access half of TASK-31900. This task owns that fix, and TASK-31900 keeps its other items. TASK-35160 had called it environmental, but it fails every time.

**C. Retained first-use seeds pre-empt the source records.** These three tests stage a workspace or legacy rail record after mount:
- `test_console_rail_scope_seeding_is_lossless_one_time_and_responsive_safe` (:524): `{'left_open': True} != {'left_open': False}`
- `test_console_rail_writes_cannot_finish_with_stale_seed` (:622): `seed_started` is never set
- `test_console_rail_scope_seed_preserves_explicit_left_open_intent` (:700): `KeyError: 'left_open_explicit'`

Since 71ddbecdbb ("retain first-use rail defaults across restarts", TASK-36385, in #3225, 2026-10-31), the mount latches a first-use seed. `_console_rail_state_config()` (chat_screen.py:11808) re-injects it whenever the key is absent, so `_ensure_console_rail_scope_seed` returns early (:12128) and never copies the staged record. A probe that disabled the first-use seeds made all three pass. Which one should win, the retained seed or a source record that appears after mount, is a product rule. It belongs to TASK-36385 and ADR-083.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file passes on dev under the plain one-file command, with -n 4 and serially, with no assertion dropped
- [x] #2 The 80x24, 120-column and 140-column tests reach their controls the way a user can (a defined status-line widget; a ready provider or keyboard activation when the setup modal is up)
- [x] #3 The three seeding tests pin the precedence between a retained first-use seed and a later source record, and that rule is recorded in TASK-36385 or ADR-083
- [x] #4 A PR lane or census entry runs the file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Show the six reds at dev 2dcf038cb8 (`-n 4`).
2. A: bind `line` to `#console-status-text` (the widget whose segment activation posts `ConsoleToolsChip.OpenRequested`).
3. B: the 120- and 140-column click tests select the send-ready llama.cpp fixture (`_select_llamacpp_console`) and wait until no `ConsoleSetupModal` blocks before clicking, as a user with a configured provider would.
4. C: trace every writer of `console.rail_state` and the ADR-083/TASK-36385 rule. If a later source record cannot cost the user an explicit preference, the retained seed's precedence is the rule: the three seeding tests stage their records before the seed latches, the seeding test pins both directions (a retained seed beats a source record that appears later; an explicit record for the same key beats the retained seed), and ADR-083's first-use paragraph states it. Otherwise stop and report NEEDS_CONTEXT.
5. Run the file plain, `-n 4` and serially; then append one whole-file census entry at the tail and raise `MINIMUM_FILES`.
6. Preflight, size ratchets for touched modules, task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only fix plus one ADR-083 paragraph and a census entry; no product code changed.

**A.** The 80x24 test binds `line` to `#console-status-text`, the widget whose segment activation posts `ConsoleToolsChip.OpenRequested` (`_post_for`), so posting from it is what a click or Enter produces.

**B.** A new `_ready_provider` helper selects the send-ready llama.cpp fixture (`_select_llamacpp_console`) and waits until no `ConsoleSetupModal` blocks. The 120- and 140-column click tests use it, so their clicks reach the controls as a configured user's would. The four keyboard tests use it too: under `-n 4` the file failed in 1 of 4 runs at this head before that (`test_sources_chip_opens_inspector_at_140_cols`: the cursor stayed on `provider`, and `test_left_handle_opens_left_rail_at_90_cols`), and a repeat run failed the sources test 1 in 8 times. Each setup sync moves focus to the blocking modal's action, so a key press can land there. TASK-36385's notes saw the same sources flake.

**C. Stale test setup, not a product precedence problem.** Writers of `console.rail_state`:
- `_set_console_rail_preference` saves only the selected key (the shared key in Global, the workspace `:layout` key in Per workspace).
- `_ensure_console_rail_scope_seed` saves only the selected key, and only when it copied a source.
- `take_console_rail_seed_flush` writes held defaults ahead of a record.
- `_prune_console_rail_preferences` deletes orphaned conversation and session scopes and keeps `layout`, legacy `global` and `shared-layout-v1`.
- No writer creates a legacy `:global` record; those exist only in older configs.

So a source record can appear after a seed latched: a Per-workspace edit followed by a later launch in Global, or a Global edit after a workspace's first use. The owner's own tests pin that such a record must not reseed (`test_a_per_workspace_edit_leaves_the_global_rail_layout_at_its_defaults`, `test_untouched_workspace_first_use_survives_later_global_edit`). No explicit preference is lost: the later record keeps its own scope, and an explicit edit of the seeded scope writes that scope's key, which beats the seed. The three tests now stage their records ahead of the latch (`_stage_before_seed` clears the harness mount's latch, as a profile holding those records sees at its first mount). The seeding test pins both directions:
- a retained seed beats a workspace or legacy record that appears later, with nothing written and the sources unchanged;
- an explicit record under the seed's own key beats the seed.

ADR-083's first-use paragraph now states this rule.

**Red/green.** At dev 2dcf038cb8, `-n 4`: 6 failed, 6 passed (A NameError, B two click misses, C three seeding failures). The new precedence assertions were red under temporary source edits, both reverted:
- with the seed re-injection in `_console_rail_state_config` disabled, the later workspace record was copied (`{'left_open': False} != {'left_open': True}`, line 618);
- with its `key not in rail_state_config` guard removed, the seed overrode the explicit record (`{'model_open': False} != {'model_open': True}`, line 629).

**Runs** (through the scratch wrapper, load 29-43):
- The file at this head: 12 passed serially; 12 passed under `-n 4` in each of 4 runs after the keyboard fix; 72 passed with every test repeated 6 times under `-n 4` (scratch repeat plugin, deleted).
- Census and CI pin tests (7 files, `-n 4`): 422 passed, 1 failed. The failure is `test_ui_lane_shard_count_statements.py`'s ADR-103 statement ("38" against a 39-job matrix). It reads only ADR-103 and the workflow, which this change leaves alone, and it fails at dev too.
- Preflight passed.

**AC#4:** one whole-file entry appended at the census tail; `MINIMUM_FILES` is now the measured entry count.

**Files:** `Tests/UI/test_console_inspector_compact_access.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`.
<!-- SECTION:NOTES:END -->
