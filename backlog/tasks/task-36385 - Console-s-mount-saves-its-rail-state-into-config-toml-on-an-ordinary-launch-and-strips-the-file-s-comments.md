---
id: TASK-36385
title: Console's mount saves its rail state into config.toml on an ordinary launch and strips the file's comments
status: Done
assignee:
  - '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 18:10
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

**Fix** (`tldw_chatbook/UI/Screens/chat_screen.py`): the seed saves only when it copied a source record. Defaults stay in memory; review round 1 below makes the next rail save write them first. The recovered-profile check is now redundant, so it is removed (`config.py`'s shutdown persist still uses `unchanged_selected_config`). A real edit goes through `_set_console_rail_preference`, which saves as before. (Round 0 also said a never-written shared layout seeding from a workspace later matched the spec. It does not: the spec makes seeding one-time. Review round 1 fixes it.)

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
- Round 1: `tldw_chatbook/Chat/console_rail_state.py`, `Tests/Chat/test_console_rail_state.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `Docs/security/production-diagnostic-inventory.json`.

No User Guide change: no page says that a launch rewrites config.toml.

**Review round 1 (verdict "With fixes"):**
- **Seeding stays one-time (finding 1).** Round 0 never wrote a defaults seed, so a later record could become its source. The reviewer's probe:
  - Launch 1 in Global writes nothing.
  - The user switches to Per workspace and closes Conversations in workspace A, which saves A.
  - Launch 2 in Global seeds the shared layout from A, so Global flips with no action on it. At dev, Global stays at the defaults, and the spec (2026-08-24 rail authority, lines 140 and 250) says seeding happens once.
- **Fix:**
  - The seed records each defaults latch in `_console_rail_unsaved_seeds`.
  - Before any rail record is saved (an edit, or a seed copying a source), `Chat/console_rail_state.py::take_console_rail_seed_flush` queues the latched defaults that are still the seeded objects. When a workspace record is saved while the shared layout is absent from memory, it also holds and queues the shared defaults.
  - The save worker writes the queue first, under the same lock, so the saved record never reaches disk without them.
  - In a session that stayed in Global, an edit queues nothing: it saves the shared layout, which is the only latch. A Global edit after Per workspace was used in the same session also writes that session's held workspace defaults (`Tests/Chat/test_console_rail_state.py`, the shared-save case).
- **Two limits** (the latch is per session):
  - A workspace seeded with defaults in one session and left untouched is not written when the Global layout is edited in a later session, so that workspace still seeds from Global afterwards. At dev it stays at the defaults.
  - A profile that never opened Console in Global writes the shared defaults at its first per-workspace save. At dev, it would later seed Global from the active workspace. Global is the default scope, so on dev the shared record practically always exists.
- **Tests:**
  - New `test_console_first_chat_first_mount.py::test_a_per_workspace_edit_leaves_the_global_rail_layout_at_its_defaults` runs two real launches on one private profile. Red at a2d6db257c: `conversations_open` was False on launch 2 (`assert False is True`).
  - New `Tests/Chat/test_console_rail_state.py::test_seed_flush_writes_held_defaults_before_a_record_that_could_seed_them` covers both directions plus the identity check. Red when the shared-defaults branch is disabled (`assert [] == [(...)]`).
  - The first-mount toggle test now also pins that a Global edit saves only the shared key.
  - Both mounted tests press the toggle Button (its own Pressed path) instead of a pointer click, which missed in 1 of about 5 runs while the rail settled. The file then passed 12 of 12 runs, three of them in parallel.
- **Comments (finding 2):** the three "absent reads as defaults" comments now say the defaults are held for the next rail save.
- **Size ratchet:** `ChatScreen` is 24360 lines (+19); the row is raised to that measured value under the file's 2026-10-04 owner ruling, with no method added. The helper went to `console_rail_state.py` to keep the screen small.
- **Diagnostic inventory:** +1 call, a fixed-text `logger.warning`, checked with `--statements`, then rewritten with `--write`.
- **Runs** (`-n 4`):
  - Rail-seed group (7 files), final head: 9 failed and 139 passed. The 9 are all in the pre-existing set of 10; `test_sources_chip_opens_inspector_at_140_cols` passed this time.
  - Other rail-persistence files (9): no new failures except `test_character_mount_loads_groups_without_duplicate_cold_resume_worker`. It is flaky at dev too: 2 of 4 serial runs failed at dev, 1 of 3 at this head.
  - Size ratchets: the same 8 failures as dev.
  - Preflight passed.

**Review round 2 (verdict "With fixes"):**
- **The drain order is now pinned (finding 1).** Nothing failed if the save worker wrote its own record before the queued defaults; the reviewer showed that this order can put a held record the user edits back to the defaults. Two mounted tests sit in the census-listed `Tests/UI/test_console_first_chat_first_mount.py` and record every rail-state file write:
  - `test_held_shared_defaults_are_written_before_the_workspace_record`: the mount holds Global's defaults; a workspace edit in Per workspace then records the shared defaults first, then the workspace record twice (the seed's copy, then the edit).
  - `test_a_held_seed_the_user_edits_after_it_was_queued_keeps_the_edit`: the held shared record is queued, then edited and saved; the file ends at the edit.
  - Both were red with the drain temporarily moved after the record write: `writes[0]` was the workspace record, and `[True, False] == [False, True]`. The edit was reverted.
  - The recorder appends only after the real write returns. A first draft appended before the write and read the file too early.
- **Notes (finding 2):** the Global-edit line now says it holds for a session that stayed in Global, and the missing blank line before the round 1 heading is back.
- **Flush warning (recommendation 3):** it now logs `error_type` (the exception's type name). The diagnostic inventory row changed digest only; the new statement interpolates nothing but that type name. Checked with `--statements`, then rewritten with `--write`.
- **Size ratchet:** `ChatScreen` measures 24363 lines (+3 for the warning), and the row is raised to that value under the same ruling.
- **Runs** (`-n 4`; first-mount, rail_state, inspector compact access, profile_open, profile isolation, plus keepalive so the ProductionApp file collects):
  - First run: 10 failed, 132 passed. Two of them were `test_profile_open_requires_actual_mounted_local_reads[database_failure|read_failure]` (`backup_operation_failed`, at load average about 30). That test passed 12 of 12 run alone.
  - Second run: 8 failed, 134 passed. All 8 are in the pre-existing set that also fails at dev (4 keepalive, 4 compact_access display checks).
  - Screen ratchet: 4 passed.
  - Preflight passed.
<!-- SECTION:NOTES:END -->
