---
id: TASK-38643
title: 'Tests: test_console_inspector_compact_access.py has six reds on dev, and no lane runs it'
status: To Do
created_date: 2026-11-02 21:52
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
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). Through the scratch-profile wrapper, the file gives 6 failed, 6 passed with `-n 4`, and the same six fail serially. The five that are not a NameError also fail alone. None is the local RecoveryRequired trap. No PR lane or census entry runs the file, and Nightly Deep last ran on 2026-09-27, which is why these went unnoticed. Three causes:

**A. Undefined name.** `test_tools_chip_opens_inspector_at_80x24_single_pane` calls `line.post_message(...)` at :305. `line` exists only inside the helper `_activate_status_segment` (:212). This has been red since d102ce9cb5 (PR #3145, 2026-10-17, segment-contract conversion). A read-only probe that bound `line` to `#console-status-text` made it pass.

**B. Clicks land on the setup modal.** In `test_visible_attach_context_action_switches_rails_without_file_picker` (:158) and `test_inspector_handle_opens_and_collapses_rail_at_140_cols` (:187), `pilot.click` lands on the blocking first-run `#console-setup-modal`. The test harness has no ready provider, and the modal blocks in card mode (console_setup_modal.py:118-120; chat_screen.py `_sync_console_setup_modal`). So the click returns False and the rail never moves. The file's keyboard tests pass. A probe that made the modal non-blocking made both pass. This is the compact-access half of TASK-31900. TASK-35160 had called it environmental, but it fails every time.

**C. Retained first-use seeds pre-empt the source records.** These three tests stage a workspace or legacy rail record after mount:
- `test_console_rail_scope_seeding_is_lossless_one_time_and_responsive_safe` (:524): `{'left_open': True} != {'left_open': False}`
- `test_console_rail_writes_cannot_finish_with_stale_seed` (:622): `seed_started` is never set
- `test_console_rail_scope_seed_preserves_explicit_left_open_intent` (:700): `KeyError: 'left_open_explicit'`

Since 71ddbecdbb ("retain first-use rail defaults across restarts", TASK-36385, in #3225, 2026-10-31), the mount latches a first-use seed. `_console_rail_state_config()` (chat_screen.py:11808) re-injects it whenever the key is absent, so `_ensure_console_rail_scope_seed` returns early (:12128) and never copies the staged record. A probe that disabled the first-use seeds made all three pass. Which one should win, the retained seed or a source record that appears after mount, is a product rule. It belongs to TASK-36385 and ADR-083.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file passes on dev under the plain one-file command, with -n 4 and serially, with no assertion dropped
- [ ] #2 The 80x24, 120-column and 140-column tests reach their controls the way a user can (a defined status-line widget; a ready provider or keyboard activation when the setup modal is up)
- [ ] #3 The three seeding tests pin the precedence between a retained first-use seed and a later source record, and that rule is recorded in TASK-36385 or ADR-083
- [ ] #4 A PR lane or census entry runs the file
<!-- AC:END -->
