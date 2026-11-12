---
id: TASK-39129
title: "The timer-path update inventory is red on dev: Plugins, first-run provider detection and the composer's raw-CLI sync reach updates without layout= from repeating clocks, and one classified row is stale"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 16:48
labels:
- follow-up
- tests
- architecture
- dev-red
priority: low
references:
- Tests/Architecture/test_timer_path_static_update_inventory.py
- tldw_chatbook/UI/Screens/plugins_screen.py
- tldw_chatbook/UI/Wizards/first_run_provider_detection.py
- tldw_chatbook/Widgets/Console/console_composer_bar.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. No round-11 branch touches these files.

`Tests/Architecture/test_timer_path_static_update_inventory.py` (serial) has 3 deterministic reds on dev, at `:2462`, `:2509` and `:2595`. No PR lane runs the file.

**`update()` calls without `layout=`, reached from repeating clocks:**
- `plugins_screen.py` (lines 289, 327, 388, 426, 443, 444), through `PluginsScreen._check_scope_generation`;
- `first_run_provider_detection.py` (lines 387, 412), through `ProviderStep._refresh_subscription_readiness`;
- `console_composer_bar.py:1244` (`_sync_raw_cli_state`), through `ChatScreen._reconcile_console_after_attach`;
- `project_index_binding_card.py` (lines 167, 194, 356, 365), through `_poll`. TASK-36191 (To Do) already owns these, with an AC to lane the file.

**Other drift:**
- One stale `CLASSIFIED_SITES` row: `ConsoleComposerBar.set_voice_status` `'chip'`.
- Three new clock roots: `PluginsScreen._check_scope_generation`, `ProviderDetectionStep._quiet_recheck` and `ProjectIndexBindingCard._poll`.

**Older task.** TASK-35059 (To Do) lists 13 older sites, none of which the test reports any more.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every reported Plugins, provider-detection and composer site either passes `layout=` (with a geometry A/B check) or has a `CLASSIFIED_SITES` row giving its reason
- [x] #2 The new clock roots are pinned and the stale row is removed; with TASK-36191's sites, the file passes on dev, and it is shown red first
- [x] #3 TASK-35059's status is reconciled in its notes, since its sites are no longer reported
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run `Tests/Architecture/test_timer_path_static_update_inventory.py` serially on the base (ae9f9e8d21) to get the current reds.
2. For each reported Plugins, provider-detection and composer site, read the path from its clock root; classify it in `CLASSIFIED_SITES` with its gate (or pass `layout=` where size cannot change).
3. Remove the stale `set_voice_status` / `chip` row; pin the new Plugins and provider-detection clock roots in `EXPECTED_CLOCK_ROOTS`. Leave `ProjectIndexBindingCard` (sites and `_poll` root) to TASK-36191.
4. Classify any other site the base reports that is not TASK-36191's, so the file passes once TASK-36191's sites are handled; prove that with a temporary stand-in for TASK-36191, then revert it.
5. Reconcile TASK-35059 in its notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only change: `Tests/Architecture/test_timer_path_static_update_inventory.py`. Every reported site is gated, so each gets a `CLASSIFIED_SITES` row naming its gate; no `layout=` was added, so no geometry A/B was needed.

- **Plugins** (6 rows: `_load` scope + detail, `_publish` page, `_status`, `_render_details` detail + readme): `PluginsScreen._check_scope_generation` (0.5 s) reloads only past `if generation != self._scope_generation`, then through the `plugins-load` worker. NOT-PER-TICK.
- **Provider detection** (`ProviderDetectionStep._refresh_endpoint_status` `status`): `ProviderStep._refresh_subscription_readiness` reaches it only through `_refresh_auth_readiness`, behind the same subscription-status gate the existing `first_run_provider_step.py` row names. NOT-PER-TICK.
- **Composer** (`ConsoleComposerBar._sync_raw_cli_state` `status`): equality-gated on `label == self._raw_cli_status_visible`; a changed label also flips display/width/height, so it keeps layout. The stale `set_voice_status` / `chip` row is replaced by it (`set_voice_status` has no `chip.update(` since 653463ed15/cc7969a280).
- **Clock roots pinned:** `PluginsScreen._check_scope_generation`, `ProviderDetectionStep._quiet_recheck` (it only starts a quiet discovery worker). `ProjectIndexBindingCard._poll` and its 4 sites are left to TASK-36191, which owns them (its AC #2 pins the root).
- **Drift found since filing (dev 2684e5453f -> ae9f9e8d21):** five `ChatApprovalCard` sites reached from `BuddyConversationModal._on_poll_tick` were also red and are not TASK-36191's, so AC #2 needed them: `_save_presentation` `state` is the presentation dict (NOT-A-WIDGET); `_sync_batch_apply` hint, `_sync_control_layout` `count_summary`, `_update_mounted_single_row` `withheld_scope` and `set_batch` `#approval-request-summary` sit behind the poll's `gate_key` equality and `set_batch`'s unchanged-round return (NOT-PER-TICK), as the card's existing rows say.
- **Red first (base ae9f9e8d21, serial, load ~13):** 3 failed / 14 passed: `test_no_timer_path_update_defaults_to_layout_true`, `test_classified_sites_are_not_stale` (the chip row), `test_clock_root_set_is_pinned` (3 new roots).
- **After:** 2 failed / 15 passed (load ~62), both TASK-36191's only (`_poll` root; its 4 card sites). **With TASK-36191 stood in** (temporary copy of the file with `_feedback`/`_paint` rows and the `_poll` root, plus `layout=True` on the two `"Keyword index unavailable"` writes; reverted after): 17 passed, 0 failed (serial, load ~60). Trap for TASK-36191: the census names the nested `run` closure's calls (lines 356/365) under both `_launch` and `run`, and a `_launch` row then reads as stale, so those two writes need `layout=` rather than rows.
- AC #3: TASK-35059 reconciled in its own notes (its 13 sites are no longer reported).
- The file is still on no PR lane (serial `Tests/Architecture`); TASK-36191's AC #3 owns laning it.
<!-- SECTION:NOTES:END -->
