---
id: TASK-40308
title: "After a restore, plugins whose data roots or authority changed stay off with no way back in the app: add an authority reset that keeps and re-attaches data, and a reviewed rebind of restored roots"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 07:55
labels:
- follow-up
- plugins
- backup
- recovery
- feature
priority: medium
references:
- tldw_chatbook/Plugins/coordinator.py
- tldw_chatbook/Plugins/data_cleanup.py
- tldw_chatbook/Plugins/acquisition.py
- Docs/Plugins/operations.md
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132; owner decision D2, recorded there: "Truthful copy now, feature next"). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** After a restore where plugins changed, or where their data roots were recreated, the plugins stay off with their data on disk, and the app offers no way out. TASK-40132 shipped the truthful copy; this is the feature the owner asked for next.

**Evidence.**
- Reset keeps data roots but never re-attaches them: `Plugins/coordinator.py:131` (`root_reset_retained_custody`). `Docs/Plugins/operations.md:125-126`: "Resetting plugin authority does not help... setup refuses while the restored registry still lists installed plugins."
- A recreated root or fetch record fails identity checks: `Plugins/data_cleanup.py:209`, `:213`, `:884` (`root_anchor_changed`), `:232` (`root_incarnation_changed`); `acquisition.py:131` `_qualify` fails recreated custody. `operations.md:139-144`: "Chatbook has no way to rebind restored folders yet". TASK-40132 lists it under Known limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 From the Plugins screen, a user recovers previously installed plugins after such a restore, with their saved data re-attached; an end-to-end test covers it
- [x] #2 A restored data root or fetch record with a new identity can be rebound after a review step, and the plugin runs again
- [x] #3 Docs/Plugins/operations.md describes the recovery instead of the current limit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checked on fu/r14-plugins at 0afda812ba (dev 26b2e677dd plus the docs stream and this stream's commits), 2026-11-13.

**The premise holds:**
- `coordinator.bootstrap` refuses while the registry lists installs or roots, or `data/` exists.
- `open_root` refuses a changed anchor or leaf even with `rebind=True`, which waives only the boot id.
- `acquisition._qualify` refuses changed root, anchor or leaf identity.

The owner decisions (2026-11-12) are recorded in the notes.

1. **Passphrase adoption.**
   - `PluginAuthorityStore.restored_snapshots(passphrase)` opens every retained snapshot with the passphrase's keys.
   - `PluginService.review_restored_authority(passphrase)`, offered only while a restored generation awaits plugin review:
     - picks the newest authenticated snapshot the restored registry matches;
     - counts the generations the protected marker is ahead (the changes undone), or reports that the marker is missing;
     - lists the plugins it brings back.
   - `adopt_restored_authority(review)`:
     - pins that marker;
     - records a clean runtime checkpoint when the restored registry has no live process or held root user (a backup is never captured with live plugin writers);
     - runs recovery, which approves the restored generation.
   - UI: Unlock and **Authority…** route the restore cases to a "Recover plugins from this backup" dialog that shows the review and asks for confirmation.
2. **Reset fallback.**
   - After a restore whose authority was reset, setup:
     - moves the restored registry, packages, downloads and download records into `reset-<id>/` in the plugin folder;
     - keeps every saved-data folder;
     - approves the restored generation.
   - Plugins are then reinstalled from their sources.
   - A new "Attach kept saved-data folder" review in Manage plugin re-attaches a kept folder to an installed plugin.
   - The Reset dialog's copy says this. Its success text stops sending users to a refused step.
3. **Rebind a data folder.** A per-folder review in Manage plugin for a root whose folder identity changed. It shows the plugin, path, file count and size, with a "replaced since last bound" warning. Confirming records the folder's new identity through the existing reconcile commit.
4. **Rebind a download record.** A per-record review in **Candidates…** for a custody record whose folder identity changed. It shows the plugin, path, file count and size, and the same warning. Confirming re-signs the record with the current identities; the snapshot bytes are still checked against the reviewed package digest.
5. **Tests.**
   - Service tests with the real plugin service and a fake secure keyring: adopt after changes, adopt with the marker missing, rebind both kinds, reset fallback with reattach.
   - The end-to-end restore's changed-plugins child adopts.
   - UI tests for the dialogs.
6. **Docs.** `Docs/Plugins/operations.md`, and the User Guide's restore section.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decisions (2026-11-12, AskUserQuestion via the coordinator), recorded verbatim:**
1. **AC#1, plugin trust after a restore:** "Passphrase adoption, reinstall fallback (Recommended)".
   - The option as put to the owner: "the user unlocks with the backup's plugin passphrase; Chatbook verifies the restored authority, lists every plugin change made since the backup that this undoes, and re-pins the marker; saved data stays attached. If the passphrase is lost: reset, reinstall from sources, and re-attach the kept data roots. A plugin whose source is gone cannot come back that way."
   - With it, per the coordinator:
     - The fallback means Reset authority must work after a restore while installs or roots remain. Change the bootstrap refusal only for the reviewed post-restore case.
     - The reset must never delete saved data.
     - Fix the Reset dialog's success text.
2. **AC#2, rebinding a recreated data folder:** "Per-folder review (Recommended)". "One review per folder, showing the plugin, path, file count and size, with a "replaced since last bound" warning. Keep it inside Plugins."

**Deviation, flagged for review: what "lists every plugin change made since the backup" can show.**
- A restore replaces the plugin authority folder, so the snapshots and transition records written after the backup are gone. The keyring marker records only the newest generation's number and an opaque operation id.
- So the review counts the changes it undoes and names their generation range ("This undoes N plugin changes made after this backup (authority generations X to Y): installs, updates, configuration, enable or disable changes and removals made since then. A restore keeps no record of what each one was.").
- When the marker is missing (another computer, a reset keyring), it says it cannot count them and that all are undone.
- Listing each change by name would need a record written before the restore, which the plugin owner has no hook for.

**Approach.**
1. **Adoption.**
   - `PluginAuthorityStore.restored_snapshots(passphrase)`:
     - derives the keys from the restored salt;
     - authenticates every retained snapshot, each naming its own marker;
     - verifies the v2 legacy cutover, so nothing is written.
   - `PluginService.review_restored_authority(passphrase)`, offered only while a restored generation awaits plugin review (`restore_review_required`):
     - picks the newest authenticated snapshot the restored registry's projection equals;
     - reports `plugin_restore_authority_unmatched` when none matches;
     - returns a `RestoredAuthorityReview`: the marker, the current marker, `undone`, the plugins, and the count of saved-data roots.
   - `adopt_restored_authority(review)`:
     - re-checks that the review is current;
     - pins the marker (`adopt_marker`);
     - when the restored registry has no unsettled process and no held root user (capture refuses otherwise, so this is how a backup ends), records a clean runtime checkpoint for the adopted marker;
     - then recovers, which approves the restored generation.
   - Without that checkpoint, an in-place restore leaves saved data `root_prior_session_unqualified` (negative control below).
2. **UI for adoption.**
   - When unlock fails, the unlock dialog tries the adoption review with the same passphrase. On success it hands the review to the new "Recover plugins from this backup" dialog (**Confirm recovery** / **Close**).
   - **Authority…** opens the unlock dialog for `recovery_required` while a restore is pending. With no restore it keeps a truthful status: no in-app recovery without a restored backup.
   - The unlock-mismatch copy now points to the backup's passphrase. An unmatched authority points to Reset authority….
3. **Per-folder rebinds** (AC#2).
   - **Saved-data folders.** `data_cleanup.review_rebind(root)`:
     - takes one root;
     - proves the folder's identity really changed (an anchor or the leaf; a reboot alone keeps boot reconciliation's clean-checkpoint rule);
     - proves no live grant, pending claim, running operation or unsettled process uses it;
     - reuses `review_reconciliation`/`reconcile_data` with a new `open_root(reidentify=True)`.
     - In Manage plugin it is the action **Review replaced saved-data folder**. The review shows the plugin, folder, file count and size (`folder_summary`, no-follow walk) and the "Replaced since last bound" warning.
   - **Download records.** `acquisition.rebind_custody`:
     - re-signs one terminal record whose root, `staging/` or download-folder identity changed;
     - changes only those three fields. The snapshot bytes still have to match the reviewed package digest when the record is used.
     - In **Candidates…** such a record reads "folder replaced since last bound; Review rebind". **Review rebind** shows the plugin(s) whose revisions name it, folder, file count and size, and the warning. **Confirm rebind** re-signs it and runs recovery.
4. **Fallback** (lost passphrase).
   - Reset no longer fails when the marker is missing: there is no checkpoint to mark dirty, and every root is still kept.
   - Setup while a restored generation is pending runs `_set_restored_plugins_aside`. It moves the restored registry (`PluginRegistry.set_aside`, which then starts empty in place), `acquisition-custody/` and `staging/` into `<root>/reset-<id>/`. (Review r14plg-rev1 moved this out of `staging/.reset-<id>/`.)
     - Backups exclude `reset-<id>/` whole, so the downloads' git and SSH copies never reach an archive, and replace restores keep it whole.
     - Nothing is deleted; `data/` stays. Package bytes stay unregistered in `packages/` and `packages-revisions/` until removed by hand.
     - The bootstrap refusal is unchanged outside a pending restore.
   - Setup then approves the restored generation.
   - Manage plugin ▸ **Attach kept saved-data folder** (`review_kept_attachment`/`attach_kept_data`) registers a kept `data/<id>` folder for a reinstalled plugin after the same per-folder review, through the authority commit (phase `attached`).
   - The Reset dialog's review and success text say what setup will do in each case: after a restore; plugins installed with no restore (still refused); or nothing retained. The old "Use Authority to set up again" contradiction is gone.

**Tests** (real `PluginService`, real git acquisition, a fake secure keyring; only the restore gate is monkeypatched, as `Tests/Plugins/test_recovery.py` already does).
- **Service tests, `Tests/Plugins/test_recovery.py`** (plugin-qualification lane):
  - `test_backup_passphrase_adopts_restored_plugins_and_rebinds_replaced_folders`:
    - `[same-computer]` and `[new-keyring]`, over recreated folders: wrong passphrase refused; undone 1 or unknown; adoption leaves `recovery_required` until the download record is rebound; data folder rebound; plugin runs; a reboot-only reason is refused by the rebind.
    - `[in-place]`: no rebind needed, and data stays attached.
  - `test_lost_passphrase_reset_sets_restored_plugins_aside_and_keeps_data`.
- **End to end** (`test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`, the app's real backup and replace restore): its changed-plugins child now adopts, with undone 1, no `recovery_required`, the plugin running, and "ADOPTED".
- **UI** (`Tests/UI/test_plugins_browse.py`):
  - `test_restored_plugins_recover_from_the_plugins_screen`: Authority… → unlock → recovery review → confirm → Candidates… rebind → Manage plugin folder rebind → plugin runs with data.
  - `test_reset_after_restore_sets_plugins_aside_and_kept_data_reattaches`: Reset copy before and after → setup → reinstall → attach kept folder.
- **Updated copy pins:** `test_unlock_mismatch_copy_is_truthful_and_has_no_dead_end_step`, `test_unlock_mismatch_shows_restore_copy_not_passphrase_advice` (its fake service now refuses adoption), `test_authority_recovery_copy_is_truthful_without_a_reset_step`.

**Red → green.**
- Every new test is red at base: the service methods do not exist (`AttributeError`).
- Negative controls, each reverted afterwards:
  - Adoption without the clean checkpoint: `[in-place]` fails with `assert 'root_prior_session_unqualified' is None`.
  - Setup without the post-restore set-aside: the fallback fails with "existing installation or root custody requires reviewed recovery".
  - `open_root(reidentify=True)` ignored: both recreated cases fail with `root_anchor_changed`.
- The reboot-only refusal is pinned in the adoption test.
- End to end, from a temporary copy with relaxed limits, deleted afterwards: 1 passed in 777 s at load ~40. Children: first 401 s, unchanged 7 s, again 348 s, changed 6 s, which printed ADOPTED.
  - The test as committed keeps its limits: 360 s and 300 s per child, 480 s overall. At load 30–50 its restore children exceed them, as the 40309 run also showed. Round 13 measured 4 min at load 15–22.

**Covering runs.**
- `Tests/Plugins`, `Tests/UI/test_plugins_browse.py` and `Tests/UI/test_plugin_review.py` at `-n 4`, load 35–40: 1358 passed, 8 failed, 4 skipped, end to end deselected.
  - Five fail with the base sources too, run the same way with this branch's `tldw_chatbook/` changes swapped out and then restored: `test_declared_variables.py::test_bridge_validates_captured_requirements_without_availability_rows`, `test_native_components.py::test_actual_console_registers_selected_managed_agent` and `::test_actual_console_dependency_failure_blocks_owned_skill_only`, `test_native_skill_flow.py::test_storage_worker_retires_its_workspace_lookup_connection`, `test_surviving_process_recovery.py::test_actual_native_fields_stable_in_separate_processes`.
  - `test_installed_manager_collects_all_51_exact_root_refs` hit its 300 s timeout; it passes alone (TASK-40712's notes).
  - `test_acquisition_limits.py::test_actual_git_tree_cancel_retains_slot_until_terminal[True]` passed when re-run.
  - The eighth was this task's `test_restored_plugins_recover_from_the_plugins_screen`, which pressed a dialog button before the dialog had composed it. Its waits now require each dialog's last widget. Since then: 15 of 15 in three concurrent `-n 4` runs, and 14 of 14 in two more.
- One earlier run of the adoption service test failed once (`[same-computer]`) with its output not kept. It did not recur in 46 runs since, at load 30–45.
- Architecture, serial: `test_layer_direction`, the module and screen size ratchets and `test_tier2_screens_offload_guards` pass. `test_timer_path_static_update_inventory` fails twice on `Widgets/Settings_Widgets/project_index_binding_card.py`. That file and the test are unchanged since base, and this change adds no timer.
- `./scripts/preflight.sh` passes, and `check_markup_interpolation.py` is OK.
  - The worker-contract check first counted five new post-await lookups in the Candidates dialog's handler. They moved into `_review_rebind`/`_confirm_rebind`, which look their widgets up before any await.

**PR lane.**
- The two new UI tests are appended at the UI census tail; `MINIMUM_FILES` 2033 → 2035.
- Serial times with the lane flags: 30 s (load 41→43) and 34 s (load 43→42).
- The service and end-to-end tests are on the plugin-qualification lane.

**Docs.**
- `Docs/Plugins/operations.md`: "After restoring a backup" describes adoption, both rebinds and the fallback instead of the limit (AC#3).
- `Docs/User_Guide/plugins.md`: the restore section's steps, the Reset authority… and Candidates… rows, and the two Manage plugin actions.
- `Docs/Backup-and-Recovery.md`.

**Files.**
- Plugins: `tldw_chatbook/Plugins/{authority_store,service,coordinator,data_cleanup,acquisition,source_service,registry,review}.py`.
- UI: `tldw_chatbook/UI/Plugins_Modules/{review,operations}.py`, `tldw_chatbook/UI/Screens/plugins_screen.py`, `tldw_chatbook/Widgets/plugin_review_modal.py`.
- The tests and docs above, the census and its checker.
**Review r14plg-rev1 (2026-11-13), every item fixed.**
- **I1, an unreadable restore witness unlocked restore-only paths.** `restore_review_required` fails closed to True, which is right for keeping plugins off but wrong for what only a restore allows. `Plugins/backup.restore_pending(root)` now returns `bool(_restored_generations(root))` and lets errors propagate. It is used in three places:
  - adoption review (raises);
  - the Reset review's post-restore copy flag (unreadable means not restored);
  - setup over restored plugins (unreadable means not pending, so the existing refusals apply).
  The Plugins screen's routing (`restore_review_pending`) uses it too. Setup approves the restored generation only when it actually set plugins aside.
  - With the reviewer's `r14plg_witness_error` probe, `test_dirty_orphan_and_reset_never_mint_clean_over_retained_data` now passes; the reviewer showed it failing with DID NOT RAISE.
  - New: `test_unreadable_restore_witness_never_unlocks_adoption_or_setup`. Negative control (`restore_pending` delegating to `restore_review_required`): DID NOT RAISE.
- **I2, the set-aside folder was listed as newer downloads.** It moved out of `staging/` to `<root>/reset-<id>/`. The plugins owner excludes it whole, as one row never walked. `Plugins/backup.kept_whole` adds it to `restore_plan`'s kept directories, so a replace restore keeps it even from an archive made before it, and `refuse_added_fetch_folders` now only considers `staging/<name>/` members.
  - The `.reset-` skips in `_reserve` and `_protected` were dropped (nothing set aside lives in `staging/` now).
  - Tests: `test_reset_aside_is_excluded_whole_and_never_listed_as_newer_downloads`, and `test_replace_keeps_a_plugin_reset_aside_whole_even_when_the_archive_predates_it[archived|added]` (planner).
  - Negative control (old predicate, no `kept_whole`): both fail with `FetchRemovalRequired`.
  - The reviewer's `probe.py` part 1 still lists a `staging/.reset-…` folder, the old layout setup no longer writes.
- **I3, kept folders had to be typed.** In attach mode, Manage plugin now offers the kept folders as a choice. Each is labelled with the installation id and alias the set-aside registry records (read with `PluginRegistry(read_only=True)`), marked "(not proven)". The review repeats the label. The reference field is not used, so another action's draft is kept.
  - Pinned in `test_reset_after_restore_sets_plugins_aside_and_kept_data_reattaches`. User Guide step 4 updated.
- **M6, a folder swapped after review was rebound.** The download-record review stores the root, `staging/` and folder identities; `rebind_custody` refuses `acquisition_rebind_review_stale` if any differs.
  - Pinned with a swap between review and confirm. Negative control: DID NOT RAISE.
- **M7, a foreign marker was counted.** When the runtime checkpoint's marker is the current marker and its namespace differs from the restored authority's, `undone` is `None` ("cannot count").
  - `[foreign-marker]` case. Negative control: `assert 1 == None`.
- **M8, the adoption status named one cause.** It is hedged: "recovery is not finished. If Candidates… marks a download record … rebind it …; otherwise the restored plugin files need attention".
- **M10, the docs overclaimed.** Operations and User Guide now say the dialog names the generation, counts the undone changes when it can, lists the plugins and revisions, and gives the number of saved-data folders.
- **M11, unregistered package bytes.** After the set-aside, `packages/<old-id>` and `packages-revisions/` stay unregistered and still count toward plugin storage capacity. The plan did not say so. The Reset copy and the docs now say this space is kept until removed by hand; reclaiming it is your follow-up.
- **M12.** The adoption review text goes through `display_text(..., limit=16384)`.
- **Also.** `Docs/Backup-and-Recovery.md`, `Docs/Architecture/backup-recovery.md` and the owner inventory now describe the `reset-<id>/` exclusion.

**Checks after the review fixes** (`-n 4`, load 12–23):
- `Tests/Plugins/{test_recovery,test_surviving_process_recovery,test_acquisition_limits,test_source_service}.py`, `Tests/Backup_Recovery/test_restore_plan_excluded_directories.py`, `Tests/UI/{test_backup_data_groups,test_plugins_browse,test_plugin_review}.py`, end to end deselected: 288 passed, 1 failed.
- The one failure, `test_actual_native_fields_stable_in_separate_processes`, also fails on the base sources.
- End to end with the committed limits, at load 19→41: 1 passed in 326 s, ADOPTED printed. Children: first 143 s, unchanged 4 s, again 168 s, changed 5 s.
- `./scripts/preflight.sh` passes; `check_markup_interpolation.py` is OK.
- Serial: the layer, module-size, screen-size and offload guards pass (221). The `Tests/CI` census, derived-artifacts and plugin-qualification pins pass (158).
<!-- SECTION:NOTES:END -->

