---
id: TASK-36385
title: Console's mount saves its rail state into config.toml on an ordinary launch and strips the file's comments
status: Done
assignee:
  - '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 16:35
labels:
- follow-up
- console
- config
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/config.py
dependencies:
- TASK-35835
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35835 (PR #3216, 2026-10-22). Once TASK-35835 lands, a quit no longer rewrites config.toml. A whole boot and quit still does not leave the file byte-identical. When the Console mounts, it saves `console.rail_state` (chat_screen.py around :12145-12153) through the normal config writer. That write dumps the file, so on an ordinary launch it strips the comments a user or the template left. Its only skip today is `unchanged_selected_config`, which holds for recovered profiles only. A default rail state written on mount is not a user change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An ordinary launch that changes no setting leaves config.toml byte-identical
- [x] #2 A rail-state change the user makes is still persisted
- [x] #3 A test pins both
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Why the mount writes: `_ensure_console_rail_scope_seed` (a0d61d9957, scoped rail layouts, plan 2026-08-24) seeds an absent layout scope once. With a source record (the active workspace's `:layout`, its legacy `:global`, or the shared layout) it copies that record to the new key: a lossless one-time migration, which stays. With no source it serialized product defaults and saved them; only recovered profiles skipped that (`unchanged_selected_config`, TASK-32628).
2. Fix: never persist the no-source default seed. An absent record already reads back as those defaults, so the in-memory latch is enough. The recovered-profile check becomes redundant and goes. Real edits still go through `_set_console_rail_preference`, which writes.
3. Tests: the real-app boot + quit test (`Tests/ProductionApp/test_config_profile_isolation.py`) compares a commented config across the whole session; a census-listed mounted Console test pins that the first mount leaves a commented file byte-identical and a real rail toggle is persisted. Both red at dev where applicable. Update the two unit pins of the old default write.
4. List the other startup writes through the same writer; fix none unless the same guard applies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Why the mount wrote.** `_ensure_console_rail_scope_seed` (a0d61d9957, the scoped rail layouts plan of 2026-08-24) seeds an absent layout key once. When a source record exists (the active workspace's `:layout` record, its legacy `:global` record, or the shared layout), it copies it to the new key. That copy is the lossless one-time migration that replaced the old destructive fallback migration, and it still saves. When no source exists, it serialized product defaults and saved them. Only recovered profiles skipped that save (`unchanged_selected_config`, TASK-32628). On a fresh or template config, that defaults save was the boot's only config write. It dumped the file, which dropped the comments.

**Fix** (`tldw_chatbook/UI/Screens/chat_screen.py`): the seed saves only when it copied a source record. Defaults stay as the in-memory latch, since an absent record reads back as the same defaults. The recovered-profile check is now redundant, so it is removed (`config.py`'s shutdown persist still uses `unchanged_selected_config`). A real edit goes through `_set_console_rail_preference`, which saves as before. One edge case behaves differently. If the shared layout was never edited, it is never written, so a later switch from Per workspace back to Global copies the workspace's saved layout. That is what the spec describes for a missing shared record.

**Tests:**
- `Tests/ProductionApp/test_config_profile_isolation.py::test_real_app_quit_persists_only_the_effective_config` (AC#1). It covers a real `TldwCli` boot and quit. The sparse config now carries a leading comment and an inline one, and the bytes are compared across the whole session, no longer only around the shutdown persist. Red at dev: "the boot and quit rewrote the effective config".
- `Tests/UI/test_console_first_chat_first_mount.py::test_consoles_first_mount_leaves_config_alone_and_a_rail_toggle_saves` (AC#1 and AC#2, census-listed file, private profile). Console's real first mount leaves a commented config.toml byte-identical. A click on the Model section toggle then saves `model_open = true` under the shared layout key. Red at dev: "Console's mount rewrote config.toml". The toggle half was red under a temporary edit that made `_save_console_rail_preferences` return early ("condition was not met in time"), then reverted.
- Pins of the old defaults save, updated:
  - `test_normal_console_seed_persists_a_stored_source_but_not_defaults` (renamed from `..._still_persists_defaults_or_stored_source`; the `[None]` case now expects no write).
  - The seeding matrix in `test_console_rail_scope_seeding_is_lossless_one_time_and_responsive_safe`.
  - `test_console_rail_writes_cannot_finish_with_stale_seed`, which now seeds from a workspace record so the delayed seed write it races still happens.
- `test_start_chatting_through_consoles_first_mount_warns_nothing` asserted that the mount advanced the config generation. That no longer happens, so the assertion is dropped. `Tests/Wizards/test_first_run_setup_integration.py` still makes that write by hand; its docstring and the module docstring now say so. The boot-worker census comment is updated too.
- A temporary probe (not committed) booted and quit the real app twice: once on a commented sparse file and once on the full shipped template (splash and catalog refresh off, setup marked complete). Both files changed at dev and were byte-identical with the fix.

**Runs** (`-n 4`, through the scratch wrapper):
- Rail-seed group: 7 files, 146 tests. At HEAD, 10 failed and 136 passed. At dev, the same 10 failed plus the 2 new tests (12 failed, 134 passed). The 10:
  - 6 display assertions in `test_console_inspector_compact_access.py`;
  - 4 in `test_app_unhandled_exception_keepalive.py`, which the run includes so the ProductionApp file collects (run alone, its setup raises `RecoveryRequired`, the TASK-36381 trap).
- Other rail-persistence files: 9 files, 322 tests. At HEAD, 109 failed, 31 errors and 182 passed. At dev, 113 failed, 31 errors and 178 passed. No test fails only at HEAD. 4 failed only at dev, for unrelated reasons (a settings `KeyError` and a character-row `NoMatches`).
- Size ratchets (serial): 8 failed, the same set at dev, none in `chat_screen.py`.
- Preflight passed.

**Other startup writes through the same writers.** None fixed. On a template launch that lands on Console, the rail seed was the only write.
- `app.py` `_stamp_new_profile_library_lifecycle` runs only on a profile created in this run.
- `app.py` `_persist_env_key_notice_flag` writes once, when an env key silenced first run.
- Console's `_prune_console_rail_preferences` deletes orphaned rail keys at mount when any exist: a one-time cleanup that still rewrites the file.
- The Persona Buddy legacy selection migration in `app_service_wiring.py` runs at startup only when there is a legacy selection.
- Home and Library rail saves happen on user toggles only.

**Files:**
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/ProductionApp/test_config_profile_isolation.py`
- `Tests/UI/test_console_first_chat_first_mount.py`
- `Tests/UI/test_console_inspector_compact_access.py`
- `Tests/Backup_Recovery/test_profile_open.py`
- `Tests/Wizards/test_first_run_setup_integration.py`
- `Tests/Performance/test_boot_worker_census.py`
- `backlog/docs/lessons-testing-evidence.md`: a patched config writer made a probe's file look untouched.

No User Guide change: no page says that a launch rewrites config.toml.
<!-- SECTION:NOTES:END -->
