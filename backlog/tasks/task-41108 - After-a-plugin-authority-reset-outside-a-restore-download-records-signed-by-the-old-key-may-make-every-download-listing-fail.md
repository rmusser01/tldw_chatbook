---
id: TASK-41108
title: "After a plugin authority reset outside a restore, download records signed by the old key may make every download listing fail"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:25
labels:
- follow-up
- plugins
- investigation
priority: low
references:
- tldw_chatbook/Plugins/coordinator.py
- tldw_chatbook/Plugins/acquisition.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Reported by the round-14 plugin stream as unverified. If true, a user who reset authority and set it up again cannot open Candidates… or update a plugin until they delete files by hand.

**Evidence (read from the code; not reproduced).**
- Setup outside a restore (`coordinator.py:117-136`) refuses only when installations or data roots remain or `data/` exists; it does not move `acquisition-custody/` aside (only the post-restore branch does, `:155-157`). Download records with no installation behind them stay.
- `acquisition._inventory` (`acquisition.py:105-128`) verifies every record's MAC with the current authority (`authority.verify_acquisition_custody`) and raises for the whole inventory on any failure. Six callers list through it (`:200`, `:275`, `:357`, `:389`, `:402`, `:424`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test resets authority with a download record present, sets authority up again, and records whether listing fails
- [x] #2 If it fails, old-key records are set aside or refused one by one so listing and new downloads keep working, with a test red before the fix
- [x] #3 If it does not, the notes name what removes or re-signs the records
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: a test bootstraps, downloads a candidate, resets authority, sets authority up again, then lists candidates and starts a new download, recording whether listing fails.
2. If it fails: setup outside a restore, with nothing installed, sets the old download records and downloads aside into `reset-<id>/` (the post-restore layout), since every record there is signed by the old key; test red before the fix.
3. If not: record what removes or re-signs the records.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Reproduced** (AC#1). `Tests/Plugins/test_recovery.py::test_setup_after_a_reset_outside_a_restore_keeps_downloads_listable`, real `PluginService` with a fake secure keyring and real git acquisition: bootstrap, download a candidate (`op-old`), Reset authority… (`retained` and `restored` both false), set authority up again, then list downloads and download again.
- At base cc6006fcf5 the listing after setup failed: `ValueError: acquisition custody invalid` from `acquisition._inventory` → `verify_acquisition_custody` → `sign_acquisition_custody`, whose namespace check refuses the old record. With that listing removed, the new download failed the same way in `_reserve`. So every download listing (Candidates…, cleanup, recovery's `verify_reference`) and every new download failed until the files were deleted by hand.

AC#3 does not apply: listing failed.

**Fix** (AC#2). Setup outside a restore, with nothing installed and no saved data (the only case setup allows there) and posture `needs_setup`, moves `staging/` and `acquisition-custody/` into a new `reset-<id>/` (`PluginCoordinator._set_aside`, shared with the post-restore set-aside). Every record there is signed by an earlier authority: before setup no authority exists to sign one. Nothing is deleted; backups exclude `reset-<id>/` whole and restores keep it (TASK-40308's rules). No folder is made when there is nothing to move. The return value stays `None`, so the service does not treat it as a restore approval (review r14plg-rev1 I1). Chose set-aside over refusing records one by one: a skipped record's download folder would then be unknown to `_reserve` (counted as unsettled against `MAX_ACQUISITIONS`) and to `_protected` (`acquisition_dependencies_unknown`), so listing and new downloads would still break.
- Red: the test fails at base with `ValueError: acquisition custody invalid` (line of the first post-setup listing); green with the fix. It also asserts the old record and download are in `reset-*/`.
- Copy: the Reset dialog's no-retained consequence (`reset_consequence`) now says setup moves old-authority downloads and records into a `reset-…` folder; Docs/Plugins/operations.md and the User Guide's Reset authority… row say the same.
- Files: `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Widgets/plugin_review_modal.py`, `Tests/Plugins/test_recovery.py`, `Docs/Plugins/operations.md`, `Docs/User_Guide/plugins.md`, `backlog/docs/backup-recovery-owner-inventory.md` (the two `_set_restored_plugins_aside` producer rows now name `_set_aside`, which holds the `mkdir`/`os.rename`; the `reset-<id>/` row names this case).
**Covering run** (TASK-41106/41107/41108 together, before TASK-41105; `-n 4`, load 62→107): `Tests/Plugins/{test_recovery,test_coordinator,test_authority_store,test_surviving_process_recovery,test_source_service}.py`, `Tests/UI/{test_plugin_review,test_plugins_browse,test_backup_data_groups,test_backup_restore_screen}.py`, `Tests/Backup_Recovery/{test_group_cli,test_launcher,test_recovery_cli_actions,test_restore_plan_excluded_directories}.py`, end to end deselected: 476 passed, 6 failed.
- `test_surviving_process_recovery.py::test_actual_native_fields_stable_in_separate_processes` fails on base sources too (TASK-40308's notes). `test_plugin_review.py::test_installed_manager_collects_all_51_exact_root_refs` hit its 300 s timeout (passes alone, TASK-40712's notes).
- `test_plugin_review.py::test_candidate_reload_requires_fresh_discard_review[next]` and three `test_backup_restore_screen.py` nodes (`test_actual_archive_inspection_survives_navigation`, `test_replacement_omissions_require_explicit_review_after_untouched_abort[destination]`, `test_retention_review_survives_target_omissions_only_for_same_selection[restore]`) are load-flaky: a rerun of those three tests and the reload test at head (load ~100) failed 4 of 13, a different parameter set; the same command with every changed `tldw_chatbook/` file swapped to base failed 3 of 13 (`…survives_navigation`, `…untouched_abort[profile]`, `…discard_review[next]`), then the changes were restored.
**Review r15pa-rev1 (2026-11-16).** M7: the same docstrings and docs now say setup after a reset outside a restore also creates `reset-<id>/`.
<!-- SECTION:NOTES:END -->
