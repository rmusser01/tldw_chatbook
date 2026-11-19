---
id: TASK-39129
title: "The timer-path update inventory is red on dev: Plugins, first-run provider detection and the composer's raw-CLI sync reach updates without layout= from repeating clocks, and one classified row is stale"
status: To Do
created_date: 2026-11-07 02:03
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
- [ ] #1 Every reported Plugins, provider-detection and composer site either passes `layout=` (with a geometry A/B check) or has a `CLASSIFIED_SITES` row giving its reason
- [ ] #2 The new clock roots are pinned and the stale row is removed; with TASK-36191's sites, the file passes on dev, and it is shown red first
- [ ] #3 TASK-35059's status is reconciled in its notes, since its sites are no longer reported
<!-- AC:END -->
