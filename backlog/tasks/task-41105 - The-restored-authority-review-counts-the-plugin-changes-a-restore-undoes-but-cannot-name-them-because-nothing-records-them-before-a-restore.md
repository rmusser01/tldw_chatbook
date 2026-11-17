---
id: TASK-41105
title: "The restored-authority review counts the plugin changes a restore undoes but cannot name them, because nothing records them before a restore"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:25
labels:
- follow-up
- plugins
- backup
- ux
priority: low
references:
- tldw_chatbook/Plugins/service.py
- tldw_chatbook/Widgets/plugin_review_modal.py
- Docs/Plugins/operations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A user adopting a restored plugin authority is told "This undoes N plugin changes made after this backup … A restore keeps no record of what each one was." To judge the restore they need to know which installs, updates, configuration and enable changes are lost. TASK-40308's owner ruling accepted a count for now; naming them needs new state.

**Evidence.** `Plugins/service.py:660-681` computes `undone` as `current.generation - marker.generation` (None when the marker is missing or foreign); `plugin_review_modal.py:423` says no record is kept; `Docs/Plugins/operations.md:119-124` documents the count. TASK-40308's notes: "Listing each change by name would need a record written before the restore, which the plugin owner has no hook for."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Before a restore replaces plugin state, the plugin owner keeps a list of the authority changes made since the backup (kind, plugin, generation), outside the restored files
- [x] #2 The restored-authority review lists those changes by name when the list is available and falls back to the count otherwise
- [x] #3 A test restores over two named changes and checks the review lists both
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record each authority change when it commits: after the marker advances, `PluginCoordinator.commit` appends (namespace, generation, operation id, kind, plugin id and alias, enable/disable intent) to `<plugins>/authority-changes.json`. Bounded to the last 1000; entries of another namespace, or at or past the new generation (an undone timeline), are dropped on append. No restore-time work.
2. Keep it outside the restored files: the plugins backup owner excludes the file (one `intentionally_excluded` row, capture refused), so a replace restore preserves it like `runtime.lock`.
3. `review_restored_authority` names the undone changes when the journal covers exactly generations marker+1..current in the restored namespace and its newest entry carries the protected marker's operation id; otherwise the review keeps the count (`changes=None`).
4. The adoption dialog lists them by name; the count copy stays as the fallback.
5. Tests: two named changes after a backup, restored in place, review lists both (red first); planner/preview pins that backups exclude the journal and a replace restore preserves it; e2e reopen's state() ignores it.
6. Docs: operations.md, User Guide restore section, owner inventory.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Record each change as it commits, so the list exists before any restore with no restore-time work, and keep it where a restore cannot replace it.
- **Recording** (AC#1). After the marker advances and the operation is marked complete, `PluginCoordinator.commit` (every install, update, enable/disable, selection, configuration, trust, revocation, removal and saved-data change routes through it) calls `Plugins/backup.record_change`, which appends `{namespace_id, generation, operation_id, parent, kind, installation_id, alias, detail}` to `<plugins>/authority-changes.json` (`parent` added in review r15pa-rev1). `detail` is the enable/disable intent or the saved-data action; configuration values are never recorded. The last 1000 entries are kept (after review r15pa-rev1 nothing else is dropped on append; see below). A failure to write is swallowed: the list is advisory, and the commit already succeeded.
- **Outside the restored files.** The plugins backup owner excludes the file (`_staging_excluded`), so discovery lists it as one `intentionally_excluded` row and capture refuses it. A replace restore preserves an excluded target item, so the target's list survives the restore, as `runtime.lock` does.
- **Review** (AC#2). `review_restored_authority` reads it only when it can count (`undone` > 0, same namespace) through `Plugins/backup.changes_since`, which returns the entries only when they form an unbroken parent chain from the backup's marker to the protected keyring marker's operation, within the restored namespace (review r15pa-rev1). `RestoredAuthorityReview.changes` holds `(generation, kind, installation, alias, detail)`; otherwise `None`. The dialog lists each change ("Generation 6: disable · review-helper · <id>", after review r15pa-rev1); without the list it keeps the count, now worded "Chatbook has no complete list of them here, so it cannot name them." instead of "A restore keeps no record of what each one was."
- Falls back to the count when: the plugin folder was deleted before the restore (the list goes with it), a commit's list write failed, recovery completed an interrupted commit (`recovery.py` advances the marker without recording; rare), or the changes predate this list.

**Tests.**
- AC#3: `Tests/Plugins/test_recovery.py::test_restored_authority_review_names_the_changes_it_undoes`. `_restored_after_change` gains `changes=` (activation intents made after the backup) and its simulated backup now leaves the list out, as the real one does. Two changes (disable, enable), in-place restore: the review lists both with generation, kind, alias, plugin and intent, and the dialog text shows both lines. With the list deleted the review falls back to the count, and adoption still recovers.
- `test_reset_aside_is_excluded_whole_and_never_listed_as_newer_downloads` asserts discovery excludes the list; `test_capture_refuses_an_included_item_at_an_excluded_staging_path[authority-changes.json]` asserts capture refuses it (the parametrization now takes root-relative paths).
- `Tests/Backup_Recovery/test_restore_plan_excluded_directories.py::test_replace_keeps_the_plugin_change_list_no_archive_holds` (Architecture guard lane): a replace restore from an archive without the list preserves the target's copy.
- End to end (`test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`, the app's real capture, archive and replace restore): the first backup's preview shows the list `intentionally_excluded`; the changed reopen asserts `review.changes` names the one disable; `state()` ignores the list, which changes with every commit.

**Red → green.**
- The new tests fail at base on import (`CHANGES` does not exist). On assertions, under temporary source edits, each reverted:
  - `record_change` call replaced with a no-op: `assert None == ((6, 'activate', …, 'review-helper', 'disabled'), (7, 'activate', …, 'enabled'))`.
  - `relative == CHANGES` dropped from `_staging_excluded`: `assert 'included' == 'intentionally_excluded'` and capture `DID NOT RAISE`.
  - `restore_plan` not preserving excluded `plugins` items: `assert ('changes', …/authority-changes.json) in ()`.
- All green with the change.
**Covering run** (every plugin commit now records, so all of `Tests/Plugins`): `Tests/Plugins`, `Tests/UI/{test_plugin_review,test_plugins_browse,test_backup_data_groups}.py` and `Tests/Backup_Recovery/test_restore_plan_excluded_directories.py` at `-n 4`, load 87→114, end to end deselected: 1391 passed, 10 failed, 4 skipped.
- Rerun of the 10 at load ~8: 4 passed (`test_owned_hook_acceptance.py::test_installed_owned_mcp_post_compaction[cold]`, `test_owned_mcp_tools.py::test_pending_launch_is_owned_before_scope_disable`, `test_plugin_review.py::test_review_growth_keeps_focused_action_painted[size0]`, `::test_candidate_reload_requires_fresh_discard_review[next]`).
- The other 6 fail the same way with every changed `tldw_chatbook/` file swapped to base cc6006fcf5 (same command, then restored): `test_surviving_process_recovery.py::test_actual_native_fields_stable_in_separate_processes`, `test_native_components.py::test_actual_console_registers_selected_managed_agent` and `::test_actual_console_dependency_failure_blocks_owned_skill_only`, `test_declared_variables.py::test_bridge_validates_captured_requirements_without_availability_rows`, `test_native_skill_flow.py::test_storage_worker_retires_its_workspace_lookup_connection`, `test_plugin_review.py::test_installed_manager_collects_all_51_exact_root_refs` (the first five are TASK-40308's known base failures).
- Serial Architecture: `test_backup_owner_inventory` (new `record_change` row), `test_layer_direction`, `test_module_size_ratchet`, `test_screen_size_ratchet`: 283 passed. `test_timer_path_static_update_inventory` fails 2 on `Widgets/Settings_Widgets/project_index_binding_card.py`, untouched here and red at base (TASK-40308's notes).
- End to end, from a temporary copy with load-tolerant child limits (`_child(..., idle=120)`), deleted afterwards, at load ~30-50: 1 passed in 456 s. Children: seed 31 s, first 213 s, unchanged 19 s, again 177 s, changed 9 s. So the app's real capture left the list out, the real replace restore kept it, and the changed reopen's review named the one disable before adopting (ADOPTED). The committed test keeps its limits (360/300 s per restore child, 480 s overall), which this load exceeds, as round 14 also found.
**Review r15pa-rev1 (2026-11-16), every item fixed.**
- **I1, a rewind plus one unrecorded commit named a stale change** (the reviewer's `probe.py`: timeline a1..a10, adoption back to a5, b6 committed unrecorded, b7 recorded; the review named a6 and b7). Each entry now records its parent operation (`review.authority_marker.operation_id`). `changes_since` names the changes only when each undone generation has an entry of the restored namespace whose parent is the previous entry's operation (the first one's is the backup marker's), and the last is the keyring marker's operation. The probe now prints `None`. Stale entries no longer need dropping on append (the newest of a generation wins), so `record_change` only appends and caps.
- **I2, the complete-list check had no tests.** `test_change_list_names_only_an_unbroken_chain_from_the_backup` (pure, about 3 s) covers a complete chain, a mismatched newest entry, another namespace, a gap, the probe's rewind, and the 1000-entry cap. Each guard went red with it removed, then was restored:
  - pre-fix `backup.py` (HEAD): the rewind assertion, `assert ((6, 'activate', …), (7, 'uninstall', …)) is None`;
  - namespace filter: the other-namespace assertion;
  - parent-chain check: the gap assertion, `assert ((6, …), (None, None, None, None, None), (8, …)) is None`;
  - newest check: the mismatch assertion;
  - cap: `assert 1001 == 1000`.
- **I3, a long list hid the plugin rows.** The dialog now puts the plugin rows, saved-data count and confirmation sentence first, names at most 20 changes and adds "…and N more (generations a to b)". `test_a_long_change_list_never_hides_the_plugins_or_the_confirmation` uses 1000 changes. Red with the 20-line cap removed (`assert not True`, truncated) and with the changes listed before the plugins (`assert 1641 < 108`). The probe's 200-change case is no longer truncated.
- **M6, internal kind names.** Changes read as the User Guide words them: "enable", "disable", "configuration change", "removal", "delete saved data", "old revisions cleaned up", and so on (`_change_words`). Red with raw kinds.
- **M7, stale text.** The `Plugins/backup.py` module docstring ("Left out:" now lists the change list and every `reset-<id>/`), the `_reset_aside` docstring, `Docs/Architecture/backup-recovery.md` and `Docs/Backup-and-Recovery.md` now say setup outside a restore also creates `reset-<id>/`, and that package bytes move there after a restore.
- **R8, the durable write.** `record_change` now writes `authority-changes.json.tmp` (0600, `O_EXCL`, `O_NOFOLLOW`) and `os.replace`s it, with no fsync. The `.tmp` name is excluded from backups too. The reviewer's `timing.py`, load about 80: at 1000 entries, median 91.2 ms and max 250 ms before; median 12.5 ms and max 18.5 ms after. The first call (609 ms after, 741 ms before) is mostly the lazy imports, which the running plugin service already holds. The census gained the matching `open`/`write`/`os.replace`/`os.unlink` rows.
- Checks after the fixes (load 62-80): `Tests/Plugins/{test_recovery,test_coordinator,test_source_service,test_authority_store}.py`, `Tests/UI/{test_plugin_review,test_plugins_browse,test_backup_data_groups}.py`, `Tests/Backup_Recovery/{test_group_cli,test_launcher,test_restore_plan_excluded_directories}.py` at `-n 4`, end to end deselected: 433 passed, 1 failed. The failure, `test_installed_manager_collects_all_51_exact_root_refs`, also fails with the base sources (see above). Owner-inventory census, serial, `--timeout=1500`: 47 passed. Ruff gate OK; `./scripts/preflight.sh` passes.
<!-- SECTION:NOTES:END -->
