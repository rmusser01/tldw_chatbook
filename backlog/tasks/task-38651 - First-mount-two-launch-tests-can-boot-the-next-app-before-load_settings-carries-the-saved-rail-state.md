---
id: TASK-38651
title: First-mount two-launch tests can boot the next app before load_settings carries the saved rail state
status: Won't Do
assignee:
  - '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 06:50
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_first_chat_first_mount.py
- Tests/UI/test_console_collapsed_inspector_send_layout.py
dependencies:
- TASK-38644
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-38644 implementer (2026-11-03); no failure has been seen yet.

`Tests/UI/test_console_first_chat_first_mount.py` has tests that boot a second app on the same private profile after the first one saves rail state, for example `test_a_per_workspace_edit_leaves_the_global_rail_layout_at_its_defaults` (:252 on dev 18482c1cf7) and `test_untouched_workspace_first_use_survives_later_global_edit` (:456). Between launches they poll the config file on disk. The next app's config, though, is built from `load_settings()`, which is republished just after the file is written. So a short window remains in which the second app can boot with the settings from before the save.

TASK-38644 fixed this same mechanism in the late-opened Inspector parity test by waiting for `load_settings()` itself to carry the stored state (PR #3358).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file that boots another app after a save waits until `load_settings()` carries the saved state, not only the file on disk
- [ ] #2 The window is shown: under a temporary source edit that delays the republish, the old wait lets the next app boot stale (red) and the new wait passes
- [ ] #3 The file passes 10 runs with `-n 4`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. List every test that builds a second app after a save (`grep -n` on `_build_test_app` / `_first_mount_app`), and trace the config writer path: when the file shows a save, can `load_settings()` still return the settings from before it?
2. Temporary source edit that widens the republish window for the rail save worker's writes, plus a temporary probe assertion that the next app's config carries the first app's saved state. Run each two-launch test with its current (on-disk) wait under that edit.
3. If the old wait goes red: make each between-launch wait poll `load_settings()` (a republished dict carrying the state, as TASK-38644's `_leave_collapsed` does), show it green under the same edit, revert the edit, then 10 runs with `-n 4`. If the old wait cannot go red, stop and report NEEDS_CONTEXT with the evidence.
4. Preflight; task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Stopped, NEEDS_CONTEXT (2026-11-03, base 18482c1cf7).** AC#2's red cannot be produced. Once the file shows a save, `load_settings()` returns the saved settings, so the window this task describes does not exist. No test or source change is committed.

**Scope.** `grep -n _build_test_app` finds two tests that boot a second app after a save, the two named above. Every other test in the file builds one app.

**Why the next app cannot load the pre-save settings once the file shows the save:**
- A config write holds `_config_write_lock` from its read through its republish. That lock includes the process-wide `_settings_rebuild_lock`. `_commit_serialized_cli_config_unlocked` drops `load_settings()`'s cache right after it replaces the file. A `load_settings()` that misses waits on `_settings_rebuild_lock` and then returns the republished dict. Before the drop, the warm hit's posture check sees the replaced file's new inode and misses the same way.
- Both tests also save synchronously between launches (`rail_layout_scope`). That save takes the same lock, so it waits for any rail save still in flight and republishes before `_build_test_app` runs.
- The first launch of `test_untouched_workspace_first_use_survives_later_global_edit` writes no config (it asserts the file unchanged). Only its synchronous saves come between the launches.

**Evidence** (temporary edits, all reverted):
- A probe asserted that the second app's `app_config` carries the first launch's saved state. For the per-workspace test, that is the workspace record with `conversations_open = False` and scope `global`. For the untouched-workspace test, it is the shared record with `False` and scope `workspace`.
- Delay A put a 3 s sleep just before `_publish_runtime_config_unlocked` in `_apply_literal_settings_transaction_locked`, on worker threads only. Delay B put it between the file replace and `_invalidate_config_caches()`, the only interval in which the file and the cache disagree.
- With the current on-disk wait, the per-workspace test passed 5 of 5 under A and 5 of 5 under B. The untouched-workspace test passed 4 of 4 under each. All runs used `-n 4` at load 24-39 and showed 3 held rail saves per run.
- A `load_settings()` called right after the on-disk poll succeeded took 3.0-3.1 s, because it waited out the hold. It then returned the republished dict, carrying the record. Under B, the between-launch save returned 2.95 s after the first app exited.
- Positive control for the probe: Delay C slept 3 s before the worker's write began, and the poll was replaced by a 0.3 s pause. The probe failed ("next app booted stale"), and the second app had no workspace record. With the poll in place, Delay C passed 3 of 3. So the only stale boot is one where the save has not reached the file, and the on-disk poll already rules that out.
- At plain base, the whole file passed 4 of 5 runs under `-n 4` (load 30-38). Both two-launch tests passed all 5.
  - The one failure was in a single-launch test, `test_start_chatting_through_consoles_first_mount_warns_nothing`, with `WorkerFailed: KeyError('Unknown Console chat session: ...')` raised from `archive_recovery.py:166`.
  - There, `sync` captures `store.active_session_id`, awaits, and then reads `store.messages_for_session` for a session that is gone. This has nothing to do with rail state, and it was reported to the coordinator.

**One theoretical case remains.** On an event-loop thread, `_posture_unchanged` checks the path once per loop batch. So between the file replace and the cache drop, a read in the same batch as an earlier check can return the old dict. Neither test builds its next app in such a batch: the build follows awaits and a synchronous save that republishes.

**Why TASK-38644 differs.** That test paused a fixed 0.3 s, so its save had not reached the file at all. That is Delay C's case, not a file-to-cache lag.

**Closed as Won't Do (coordinator, 2026-11-03).** The window this task was filed for does not exist on dev 18482c1cf7. For these two tests, once the file on disk shows a save, `load_settings()` returns the saved settings, because the between-launch save takes the config write lock and waits for the republish. So the existing on-disk poll is already a sufficient wait. The theoretical same-batch case above does not arise in either test. The criteria assumed the window and stay unticked; no test or source changed. The investigation found a separate Console product race, filed as TASK-38652.
<!-- SECTION:NOTES:END -->
