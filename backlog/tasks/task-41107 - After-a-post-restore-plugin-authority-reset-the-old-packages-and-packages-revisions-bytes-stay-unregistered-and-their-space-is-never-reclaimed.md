---
id: TASK-41107
title: "After a post-restore plugin authority reset, the old packages/ and packages-revisions/ bytes stay unregistered and their space is never reclaimed"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:25
labels:
- follow-up
- plugins
- storage
priority: low
references:
- tldw_chatbook/Plugins/coordinator.py
- Docs/Plugins/operations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The set-aside plugins' package files keep counting against plugin storage capacity, and the only way out is deleting folders by hand. TASK-40308's review (M11) recorded this as the reviewer's follow-up.

**Evidence.** `coordinator.py:141-163` (`_set_restored_plugins_aside`) moves `staging/` and `acquisition-custody/` into `reset-<id>/`; its docstring says "Package bytes under ``packages/`` and ``packages-revisions/`` stay too, unregistered, until removed by hand." `Docs/Plugins/operations.md:149-151` tells users the same.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The app offers a reviewed way to remove package bytes no registry names (showing their size first), or reset moves them aside with the rest
- [x] #2 Plugin storage capacity stops counting them once removed or set aside
- [x] #3 A test covers the reclaim, and the Plugins operations page is updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Take the AC's second option: the post-restore setup (`_set_restored_plugins_aside`) also moves `packages/` and `packages-revisions/` into `reset-<id>/`, so `managed_usage` (which counts only packages, packages-revisions, cache and staging) stops counting them; nothing is deleted.
2. Test: the lost-passphrase fallback asserts the old package bytes moved aside and managed usage dropped, red first; the reinstall still materializes.
3. Copy and docs: the Reset dialog's consequence, operations.md, User Guide, owner inventory, coordinator docstring.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** The criterion's second option: the post-restore setup (`_set_restored_plugins_aside`, now through the shared `_set_aside`) also moves `packages/` and `packages-revisions/` into `reset-<id>/`, beside the restored registry, download records and downloads. No registry names those bytes after the set-aside, and `retention.managed_usage` counts only `packages/`, `packages-revisions/`, `cache/` and `staging/`, so plugin storage capacity stops counting them (AC#2). Nothing is deleted. A reinstall recreates `packages/` (`_materialize` makes its containers through descriptors). Chose this over a separate reviewed removal: no new screen, no deletion path, and the set-aside folder is already excluded from backups and kept by restores.
- Ceiling: the bytes still use disk space inside `reset-<id>/` until the user deletes that folder; the copy says so. A reviewed delete of `reset-<id>/` is not built (the folder also holds the set-aside registry, the only hint of which kept saved-data folder belonged to which plugin).
- Red: `test_lost_passphrase_reset_sets_restored_plugins_aside_and_keeps_data` now measures the restored package bytes and plugin storage before the reset and asserts, after setup, `reset-*/packages/<installation>` exists, `packages/` is gone and `managed_usage(plugin_root) == 0`. With the set-aside list reverted to `("staging", "acquisition-custody")` it fails at `assert (aside / "packages" / installation).is_dir()`; green with the change, and the rest of the test (reinstall, attach kept data, run) passes.
- Copy and docs (AC#3): the Reset dialog's post-restore consequence ("Their package files move there too and stop counting against plugin storage; the reset-… folder keeps its disk space until you delete it."), its UI pin in `Tests/UI/test_plugins_browse.py::test_reset_after_restore_sets_plugins_aside_and_kept_data_reattaches`, Docs/Plugins/operations.md (fallback step 2), the User Guide's fallback step 2, the coordinator docstrings and the owner inventory's `reset-<id>/` row.
- Files: `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Widgets/plugin_review_modal.py`, `Tests/Plugins/test_recovery.py`, `Tests/UI/test_plugins_browse.py`, `Docs/Plugins/operations.md`, `Docs/User_Guide/plugins.md`, `backlog/docs/backup-recovery-owner-inventory.md`.
**Covering run** (TASK-41106/41107/41108 together, before TASK-41105; `-n 4`, load 62→107): `Tests/Plugins/{test_recovery,test_coordinator,test_authority_store,test_surviving_process_recovery,test_source_service}.py`, `Tests/UI/{test_plugin_review,test_plugins_browse,test_backup_data_groups,test_backup_restore_screen}.py`, `Tests/Backup_Recovery/{test_group_cli,test_launcher,test_recovery_cli_actions,test_restore_plan_excluded_directories}.py`, end to end deselected: 476 passed, 6 failed.
- `test_surviving_process_recovery.py::test_actual_native_fields_stable_in_separate_processes` fails on base sources too (TASK-40308's notes). `test_plugin_review.py::test_installed_manager_collects_all_51_exact_root_refs` hit its 300 s timeout (passes alone, TASK-40712's notes).
- `test_plugin_review.py::test_candidate_reload_requires_fresh_discard_review[next]` and three `test_backup_restore_screen.py` nodes (`test_actual_archive_inspection_survives_navigation`, `test_replacement_omissions_require_explicit_review_after_untouched_abort[destination]`, `test_retention_review_survives_target_omissions_only_for_same_selection[restore]`) are load-flaky: a rerun of those three tests and the reload test at head (load ~100) failed 4 of 13, a different parameter set; the same command with every changed `tldw_chatbook/` file swapped to base failed 3 of 13 (`…survives_navigation`, `…untouched_abort[profile]`, `…discard_review[next]`), then the changes were restored.
**Review r15pa-rev1 (2026-11-16).** M7: the `Plugins/backup.py` docstrings, `Docs/Architecture/backup-recovery.md` and `Docs/Backup-and-Recovery.md` now say package bytes move into `reset-<id>/` after a restore.
<!-- SECTION:NOTES:END -->
