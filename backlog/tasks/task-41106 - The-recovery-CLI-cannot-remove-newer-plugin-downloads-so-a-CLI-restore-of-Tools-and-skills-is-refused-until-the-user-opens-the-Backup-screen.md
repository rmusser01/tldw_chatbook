---
id: TASK-41106
title: "The recovery CLI cannot remove newer plugin downloads, so a CLI restore of Tools and skills is refused until the user opens the Backup screen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 19:14
labels:
- follow-up
- plugins
- backup
- cli
priority: low
references:
- tldw_chatbook/Backup_Recovery/launcher.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** `tldw-chatbook recovery restore` is the path for a user who cannot start the app. TASK-40309 added the removal to the Backup screen's review only, and the CLI's refusal sends the user to that screen.

**Evidence.** `recovery_service.py:168-175`: `plugin_fetch_removal_required` says "Review the restore on the Backup screen, which lists them and removes them when you confirm; or restore without the Tools and skills group." The `restore` sub-command (`launcher.py:644-668`) has no flag for the newer download folders.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The recovery CLI can list the newer plugin download folders a restore would remove and remove them on an explicit flag, with the same no-safety-copy warning the screen gives
- [x] #2 Without the flag the refusal names the flag as well as the Backup screen
- [x] #3 A CLI test covers refusal, listing and removal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `tldw-chatbook recovery restore` catches `FetchRemovalRequired`: it prints the folders and the members a restore would remove.
2. Without `--remove-newer-plugin-downloads` it refuses with `plugin_fetch_removal_required`, whose message now names the flag as well as the Backup screen.
3. With the flag it prints the screen's no-safety-copy warning, asks the user to type `remove`, removes exactly those members (`remove_fetch_members`) and reviews the restore again.
4. One CLI test (fake recovery service over a real plugin root) covers refusal, listing and removal, red first.
5. Docs: Backup-and-Recovery and operations.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.**
- `tldw-chatbook recovery restore` gains `--remove-newer-plugin-downloads`. When the restore review raises `FetchRemovalRequired`, the CLI prints the folders and each `Delete:` member, with the Backup screen's warning (no safety copy, a rollback cannot bring them back, your own git configuration and `~/.ssh` untouched, snapshot and record go into the safety copy at Start). The text is one helper, `Plugins/backup.fetch_removal_review(members, removal)`, now used by both the screen and the CLI, so they cannot drift.
- Without the flag it prints the refusal message and exits with `Recovery refused: plugin_fetch_removal_required`. The message (`recovery_service._ISSUE_MESSAGES`) now names the flag as well as the Backup screen and restoring without Tools and skills (AC#2).
- With the flag it asks the user to type `remove` (the CLI's pattern for every destructive step; the flag alone was given before the list was seen), removes exactly the listed members through `remove_fetch_members` (unchanged: runtime lock, no-follow descriptors, never `snapshot/`), previews the target again and reviews the restore again, which then continues to the usual "Type restore" prompt.
- Test (AC#3): `Tests/Plugins/test_recovery.py::test_recovery_cli_lists_newer_plugin_downloads_and_removes_them_on_its_flag[False|True]` runs `launcher.recovery_main` over a real plugin root with a download folder the archive lacks; the fake recovery service calls the planner's real `refuse_added_fetch_folders` over the target inventory it is given. `[False]`: listing, warning, the refusal line and message naming the flag and the Backup screen, nothing deleted. `[True]`: `remove` deletes `host-git-config` and `repository.git` and keeps `snapshot/`; the restore is reviewed again over a fresh target inventory. It sits in the plugin-qualification lane's file (`test_group_cli.py` is in no PR lane).
- Red: with base `launcher.py` and `recovery_service.py`, `[False]` fails on `assert 'Folder: …/staging/zz-new' in …` (base printed only the refusal code) and `[True]` fails (unknown argument); both green with the change.
- The screen's last review line changed from "Review restore runs again after the removal." to "The restore is reviewed again after the removal." (shared text); no test pinned it.
- Docs: Docs/Backup-and-Recovery.md, Docs/Plugins/operations.md, Docs/Architecture/backup-recovery.md.
- Files: `tldw_chatbook/Backup_Recovery/launcher.py`, `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/Plugins/backup.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `Tests/Plugins/test_recovery.py`, the docs above.
**Covering run** (TASK-41106/41107/41108 together, before TASK-41105; `-n 4`, load 62→107): `Tests/Plugins/{test_recovery,test_coordinator,test_authority_store,test_surviving_process_recovery,test_source_service}.py`, `Tests/UI/{test_plugin_review,test_plugins_browse,test_backup_data_groups,test_backup_restore_screen}.py`, `Tests/Backup_Recovery/{test_group_cli,test_launcher,test_recovery_cli_actions,test_restore_plan_excluded_directories}.py`, end to end deselected: 476 passed, 6 failed.
- `test_surviving_process_recovery.py::test_actual_native_fields_stable_in_separate_processes` fails on base sources too (TASK-40308's notes). `test_plugin_review.py::test_installed_manager_collects_all_51_exact_root_refs` hit its 300 s timeout (passes alone, TASK-40712's notes).
- `test_plugin_review.py::test_candidate_reload_requires_fresh_discard_review[next]` and three `test_backup_restore_screen.py` nodes (`test_actual_archive_inspection_survives_navigation`, `test_replacement_omissions_require_explicit_review_after_untouched_abort[destination]`, `test_retention_review_survives_target_omissions_only_for_same_selection[restore]`) are load-flaky: a rerun of those three tests and the reload test at head (load ~100) failed 4 of 13, a different parameter set; the same command with every changed `tldw_chatbook/` file swapped to base failed 3 of 13 (`…survives_navigation`, `…untouched_abort[profile]`, `…discard_review[next]`), then the changes were restored.
<!-- SECTION:NOTES:END -->
