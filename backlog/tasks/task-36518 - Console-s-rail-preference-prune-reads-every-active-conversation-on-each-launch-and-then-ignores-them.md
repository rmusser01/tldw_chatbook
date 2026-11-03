---
id: TASK-36518
title: Console's rail-preference prune reads every active conversation on each launch and then ignores them
status: Done
assignee: ['@claude']
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 12:53
labels:
- follow-up
- console
- performance
- cleanup
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_rail_state.py
- Tests/Performance/test_boot_worker_census.py
- Tests/Chat/test_console_rail_state_prune.py
dependencies:
- TASK-36385
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking TASK-36385's list of startup config writes (PR #3225), on dev 5b5920b8e2.

After each launch's first Console sync (chat_screen.py:17684), `_dispatch_console_rail_preference_prune` (chat_screen.py:11870-11890) starts the `_prune_console_rail_preferences` worker. The boot worker census lists it (test_boot_worker_census.py:150). Whenever `[console.rail_state]` holds any key, which is true once a user has toggled a rail section, the worker pages through every active conversation, 1,000 rows at a time, under an owned connection (chat_screen.py:11922-11930). It builds a set of live ids from them.

`collect_prunable_console_rail_keys` then discards that set (`del live_scope_ids`, console_rail_state.py:316). Since TASK-718, prunability depends only on the key's shape: it drops the legacy per-conversation and per-session keys, and current code never writes those keys (`build_console_rail_preference_key`, :250-292). So the scan is dead work on every launch, and the write it guards is a one-time legacy cleanup.

TASK-34406 (In Progress) moves `_prune_console_rail_preferences` into its own `console-rail-prune` worker group and changes the boot-census row this task's AC#3 touches. Whichever lands second rebases onto the other.

The other startup writers on TASK-36385's list were checked too, and each writes once by design:
- `_stamp_new_profile_library_lifecycle` writes only on a profile this run created.
- `_persist_env_key_notice_flag` writes only while the notice flag is unset (`env_keys_that_silenced_first_run`, first_run_setup_state.py:982).
- The Persona Buddy migration writes only for a legacy `PersonaBuddySelection`, which a successful migration replaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a profile whose rail state holds no legacy per-conversation or per-session key, a launch that opens Console reads no conversation rows for the prune and starts no prune worker
- [x] #2 On a profile that holds such keys, exactly those keys are deleted, once, and the current layout keys are kept
- [x] #3 The boot worker census and the prune tests pass, updated where the worker no longer starts
<!-- AC:END -->

## Implementation Plan

1. Read the prune's history: the conversation scan fed a liveness filter (45113d6fae); TASK-718 (71e0e960b5) made prunability depend only on key shape and left the scan and the `live_scope_ids` parameter dead.
2. `_dispatch_console_rail_preference_prune` decides on the UI thread from the in-memory rail-state keys (`collect_prunable_console_rail_keys`, no liveness argument) and starts the worker only when a legacy key exists, passing exactly those keys; drop the store/DB readiness guards that existed only for liveness.
3. The worker only deletes the keys it was given from config.toml and drops them in memory; no database access.
4. Remove the dead `live_scope_ids` parameter; update its pure tests.
5. Tests (red at dev first): dispatch + real worker body on a profile without legacy keys reads no conversation rows and starts no worker; with legacy keys, exactly those keys are deleted once and current layout keys kept. Remove the boot census allowlist row (the census boot holds no legacy keys), red at dev. Retire the Backup_Recovery connection-retirement test whose subject (the worker's DB connection) no longer exists.
6. Covering tests, size ratchets, preflight.

## Implementation Notes

**Why the read existed.** 45113d6fae added the prune to bound `[console.rail_state]` to live scopes, so it needed every live conversation id. TASK-718 (71e0e960b5) keyed rail layout per workspace and made every per-conversation or per-session key prunable by shape alone. It kept the scan and the `live_scope_ids` parameter, and the parameter was then discarded (`del live_scope_ids`).

**Change.**
- `_dispatch_console_rail_preference_prune` latches once. On the UI thread it filters the in-memory rail-state keys through `collect_prunable_console_rail_keys`, reading them without materializing an empty table. It starts `_prune_console_rail_preferences` only when a legacy key exists, and passes exactly those keys. The store and database readiness guards existed only to establish liveness, so they are gone.
- The worker only deletes the keys it is given (`delete_settings_from_cli_config`) and drops them in memory on the UI thread. It no longer touches the database.
- `collect_prunable_console_rail_keys` loses its dead `live_scope_ids` parameter.
- After a successful delete the keys are gone from the file and from memory. The next launch finds none and starts no worker, so the cleanup runs once.

**Tests.**
- `Tests/Chat/test_console_rail_state_prune.py`:
  - Two new tests run the real dispatch with the real undecorated worker body inline, against a fake app whose `list_all_active_conversations` and config deleter record calls.
  - `test_launch_without_legacy_keys_reads_nothing_and_starts_no_worker` (AC#1) checks that a launch with no legacy keys reads nothing, starts no worker and deletes nothing.
  - `test_launch_with_legacy_keys_deletes_exactly_them_once` (AC#2) checks that exactly the legacy keys are deleted, the current keys stay, there are no reads, and a second launch starts no worker.
  - The pure tests drop the removed argument. The two tests that differed only in liveness input are merged into `test_prunes_conversation_keys`.
- `Tests/Performance/test_boot_worker_census.py`: the `("_prune_console_rail_preferences", "default")` allowlist row is replaced by a one-line comment. The census profile holds no legacy keys, so the worker must not start there (AC#3).
- `Tests/Backup_Recovery/test_console_backup_worker_retirement.py`: `test_rail_preference_prune_retires_finished_worker_connection` is removed. It pinned the worker's database-connection retirement, and the worker no longer opens a connection. Keeping it with the new signature would have called the real config deleter.

**Red/green evidence.**
- At dev 9bbd5c2515 both new tests failed on their assertions:
  - AC#1: `started == [(set(),)]` and `reads == [{'limit': 1000, 'offset': 0}]`.
  - AC#2: `reads == [{'limit': 1000, 'offset': 0}]`.
  - Both pass after the fix.
- The census with the row removed passed on dev locally. Under load the worker started after the 1 s window, though CI census JSONs from 2026-11-01 record it. A scratch probe that lengthened the window to 9.5 s recorded `('_prune_console_rail_preferences', 'default')` on dev. After the fix it recorded 28 pairs and no prune worker.

**Runs** (`-n 4`, through the scratch wrapper, load average 14-28):
- Prune tests, census and backup-retirement file: 79 passed.
- `test_console_rail_state.py`, `test_console_runtime_ownership.py`, `test_console_full_config_preparation.py`, `test_console_sync_maintenance.py`, `test_console_config_sync_lifetime.py`, `test_config_delete_settings.py` and `test_boot_worker_stagger_policy.py`: 14 failed, 279 passed, 1 xfailed.
  - The same 14 fail on dev source with the same signatures: `_sync_console_temporary_chip` missing on a stub, `ChatScreen.app_instance`, and subprocess timeouts. 14 failed, 26 passed for the three files that hold them.
- Size ratchets (`test_screen_size_ratchet.py` and `test_module_size_ratchet.py`, serial): 205 passed, with no re-pin.
- Ruff: no new findings.
- Preflight passed.

**Files:**
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Chat/console_rail_state.py`
- `Tests/Chat/test_console_rail_state_prune.py`
- `Tests/Performance/test_boot_worker_census.py`
- `Tests/Backup_Recovery/test_console_backup_worker_retirement.py`
- `backlog/docs/lessons-testing-evidence.md`: a census row looked dead locally because its worker started after the window.

No User Guide change: nothing visible changed.

**Review round 1 (2 Minor).**
- The second dispatch in `_launch` could not catch a missing latch: the inline worker had already deleted the keys. `test_failed_delete_starts_one_worker_per_launch_and_retries_next_launch` makes the fake deleter fail. It asserts one worker start across both dispatches, the legacy keys kept in memory, and the next launch deleting them. With the latch temporarily removed it failed `assert 2 == 1`; source reverted, 9 passed.
- `test_sanitized_conversation_keys_are_prunable` is folded into `test_prunes_conversation_keys`. The two `..._even_with_no_live_scopes` tests are renamed `test_keeps_global_scope_key` and `test_keeps_reserved_shared_layout_scope`; nothing else referenced the old names.
- Preflight passed.

**Coordination with TASK-34406.** TASK-34406 (In Progress) moves this worker into a `console-rail-prune` group with a finite preparation-read helper for its database connection. That custody work is moot now, because the worker reads no database. Whichever task lands second takes the other's shape.

