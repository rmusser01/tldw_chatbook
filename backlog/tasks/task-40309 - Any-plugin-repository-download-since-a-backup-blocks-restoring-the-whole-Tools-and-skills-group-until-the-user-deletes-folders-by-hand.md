---
id: TASK-40309
title: "Any plugin repository download since a backup blocks restoring the whole Tools and skills group until the user deletes folders by hand"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 15:43
labels:
- follow-up
- plugins
- backup
- recovery
priority: medium
references:
- tldw_chatbook/Plugins/backup.py
- tldw_chatbook/Backup_Recovery/restore_plan.py
- tldw_chatbook/Backup_Recovery/data_groups.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132; owner ruling "Refuse early, feature next"). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** `Plugins/backup.py:83-111` `refuse_added_fetch_folders` raises `plugin_fetch_added_since_backup` (`:40`), called from `restore_plan.py:1179-1181`. Any repository download since the backup, even an update check, therefore blocks a replace restore of the Tools and skills group. The workaround is to restore without that group, or delete everything but `snapshot/` by hand.

**Open question for the owner.** Plugins sit in the `tools` data group with MCP and skills (`Backup_Recovery/data_groups.py:179-187`), so "restore without Tools and skills" (`recovery_service.py:166-175`) also drops MCP and skills. Whether Plugins should get its own data group is undecided; record the decision here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A replace restore over plugin fetch folders added since the backup completes without manual steps, and installed plugins still verify afterwards; a test covers it
- [x] #2 plugin_fetch_added_since_backup no longer fires for that case
- [x] #3 The owner's decision on a separate Plugins data group is recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checked on fu/r14-plugins at 4af56eb459 (dev 26b2e677dd plus the docs stream), 2026-11-13. The premise held: `restore_plan.plan_restore` called `Plugins/backup.refuse_added_fetch_folders`, which raised `plugin_fetch_added_since_backup`.

The restore engine forces the order. Publication re-checks each retired directory's metadata, and the plan fingerprints the target. Deleting members inside a planned retirement would fail `retirement_identity_changed` or `target_changed`. So the uncaptured members are deleted after the user confirms in the review, and the restore is then planned again over a snapshot-only folder.

1. `Plugins/backup.py`:
   - The planner check raises `FetchRemovalRequired` (`plugin_fetch_removal_required`), naming every member to delete.
   - `remove_fetch_members` deletes exactly those, under the runtime lock, never following a link and never touching `snapshot`.
2. Backup screen:
   - Review restore shows the folders, the members and the required copy, with **Remove newer plugin downloads**.
   - Confirming removes the members and reviews again.
3. Tests: the planner/helper unit test, the screen test, and the end-to-end restore. Then docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decisions (2026-11-12, AskUserQuestion via the coordinator), recorded verbatim:**
- Fetch folder added since the backup: "Remove it after the user confirms (Recommended)". The option as put to the owner: "The restore review lists these folders. On confirm, snapshot/ and the custody record go into the safety copy as today, and repository.git, content/, host-git-config and host-ssh-* are deleted with no safety copy. They must never enter an archive. A rollback cannot bring them back, and ~/.gitconfig and ~/.ssh themselves are untouched. The review copy must say so before the user confirms."
- AC#3, data group: "Keep it in Tools and skills (Recommended)". No group split.
  - Why: plugin authority (`skills/trust/plugins/`) is captured by the skills owner in the same group. A Plugins-only group could restore a registry without its matching authority, or the reverse, and land plugins in `recovery_required`; avoiding that would need a select-both-or-neither rule.
  - The split was considered only because this refusal made "restore without Tools and skills" the way out. The removal review removes that reason.

**Approach.**
- **Planner check** (`Plugins/backup.refuse_added_fetch_folders`):
  - It now raises `FetchRemovalRequired` instead of the old refusal. Its `.members` lists every `staging/<name>/<member>` other than `snapshot` in a fetch folder the archive lacks.
  - That covers the four named kinds (`repository.git`, `content`, `host-git-config`, `host-ssh-*`) and any other excluded member (a lock residue, a link). "Remove it" removes the whole folder, so every listed name is shown to the user.
- **Removal** (`remove_fetch_members(members)`):
  - Deletes exactly the reviewed paths.
  - Each must be a direct member of `<root>/staging/<name>/`, not `snapshot`, with neither `staging` nor the folder a link, under a root that holds `registry.sqlite3`.
  - It holds the plugin runtime lock (refuses with `plugin_writers_active` while a plugin owner runs). Directories go through `shutil.rmtree`, refused unless `avoids_symlink_attacks` holds; links, files and FIFOs are unlinked, never followed.
  - The members are gone before the restore is planned, so they never reach the safety copy or any archive.
  - Planning again retires the snapshot-only folder and its custody record into the safety copy, as before.
- **Backup screen** (`backup_restore_screen.py`):
  - The review worker catches `FetchRemovalRequired` and lists each folder and its "Delete:" members.
  - Before the button, the copy says the deletion happens now with no safety copy, a rollback cannot bring the files back, and your own git configuration and `~/.ssh` are untouched. It also says the snapshot and download record move into the before-replacement safety copy when the restore starts, and installed plugins keep working.
  - **Remove newer plugin downloads** runs the removal in a worker, then runs Review restore again. A refusal shows "Removal refused: …".
  - Any form change (`_invalidate`) hides the button and forgets the list.
- **Copy for other callers.** The issue message for `plugin_fetch_removal_required` (CLI and any other caller) points to the Backup screen's review, or to restoring without Tools and skills. `plugin_fetch_added_since_backup` no longer exists.

**Red → green.**
- With the source reverted, both new tests fail (`AttributeError: … no attribute 'FetchRemovalRequired'`); with the change they pass.
  - Unit: `test_replace_restore_lists_and_removes_a_fetch_folder_added_since_the_backup[fetched|snapshot-only]`. It also checks the link member is unlinked without touching its target, and that `snapshot`, the registry and an outside path are refused.
  - Screen: `test_restore_review_lists_newer_plugin_downloads_and_removes_them`.
- End to end, `test_plugin_backup_restores_in_place_and_waits_for_authority_recovery` (the app's real capture, archive and replace restore), first child:
  - Before Start, the review raised `FetchRemovalRequired` for a planted newer fetch folder plus custody record. The members were exactly `host-git-config` and `repository.git`; removing them left `snapshot/` and the record. The restore then succeeded and retired both ("PHASE newer fetch removed 24 … restored 256").
  - The unchanged reopen then re-verified the git-installed plugin and ran it.
  - The test then hit its own 480 s limit in the later "again" child at load ~30 (8 + 256 + 5 + 210 s). That child does not exercise this change. It was re-run with TASK-40308; see that task's notes.

**Covering run** (`-n 4`, load 32–36): 102 passed, e2e deselected. Files: `Tests/UI/test_backup_data_groups.py`, `Tests/UI/test_backup_restore_screen.py`, `Tests/UI/test_backup_restore_safety_scope.py`, `Tests/Plugins/test_recovery.py`, `Tests/Backup_Recovery/test_restore_plan_excluded_directories.py`.

**Lanes.** No new test file. The screen test is in `test_backup_data_groups.py`, already a whole-file UI census entry. The unit and end-to-end tests are in `Tests/Plugins/test_recovery.py`, on the plugin-qualification lane.

**Docs.**
- `Docs/Backup-and-Recovery.md`, `Docs/Plugins/operations.md`, `Docs/User_Guide/plugins.md` (one paragraph in "After restoring a backup").
- `Docs/Architecture/backup-recovery.md`, `backlog/docs/backup-recovery-owner-inventory.md`.

**Files.** `tldw_chatbook/Plugins/backup.py`, `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, the tests and docs above.
**Review r14plg-rev1 (2026-11-13), fixes.**
- **M4, a `..` member deleted the data tree.** `remove_fetch_members` refuses a member path that is not normalized, holds an empty, `.` or `..` component, is `snapshot`, or is not `staging/<name>/<member>`. The unit test's refused cases add `root/staging/../data`, and the data tree survives; the reviewer's `probe.py` part 2 now prints "dotdot refused".
- **M5, checks ran outside the lock and paths were re-resolved.** Validation is now done under the runtime lock. Each removal goes through no-follow directory descriptors, as `Plugins/retention.py` does:
  - plugin root, then `registry.sqlite3` checked as a regular file through it;
  - `staging/`, then the folder;
  - `os.stat`/`os.unlink(dir_fd=…)` and `shutil.rmtree(dir_fd=…)` for the member.
  No member path is re-resolved after its check.
- **M9, Enter could delete before the copy was read.** The removal review no longer moves focus onto **Remove newer plugin downloads**; focus stays where it was. Pinned in `test_restore_review_lists_newer_plugin_downloads_and_removes_them`; negative control (focus restored): the assertion fails.
- TASK-40308's set-aside folder moved out of `staging/` (review item I2), so a later restore never lists it as newer downloads.
**Review r14plg-rev2 (2026-11-13).** On case-insensitive APFS, `…/op/SNAPSHOT` named the `snapshot/` folder itself and was deleted. `remove_fetch_members` now refuses a member whose name casefolds to `snapshot`. After opening the folder it also refuses (`plugin_fetch_removal_invalid`) a member whose device and inode match `snapshot`'s, which catches any other alias.
- The unit test's refused cases include `added/"SNAPSHOT"`.
- Negative control, both guards removed: DID NOT RAISE.
- The reviewer's `probe2.py` item 4 now prints "SNAPSHOT refused".
<!-- SECTION:NOTES:END -->

