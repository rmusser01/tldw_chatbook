---
id: TASK-39118
title: "A media read outside transaction() can fail with ProgrammingError or DatabaseError if the quit-time idle close lands before its results are consumed"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 22:47
labels:
- follow-up
- database
- lifecycle
priority: low
references:
- tldw_chatbook/DB/Client_Media_DB_v2.py
- tldw_chatbook/DB/base_db.py
- Tests/DB/test_sync_log_retention.py
dependencies:
- TASK-39047
- TASK-39048
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-storage, which lands TASK-39047 and TASK-39048. Line numbers are dev's.

**What protects a media handle at quit on fu/r11-storage.**
- `MediaDatabase._get_thread_connection` holds an acquisition only while the getter itself runs.
- `SQLiteConnectionQuiescenceRegistry.close_if_idle` refuses to close during a barrier, active uses, acquisitions, an active core operation or an open transaction.
- Media takes no use token anywhere, while ChaChaNotes takes one per transaction.

**Where a read is left exposed.**
- `execute_query` (`DB/Client_Media_DB_v2.py:1315`; getter at `:1336`) is not a core operation; only `transaction()` is. The file has 22 such bare reads.
- So the quit close can land between a read's getter and its fetch:
  - before `execute`, the read fails as media `DatabaseError("Query execution failed: …")`, because `:1366` wraps `sqlite3.Error`;
  - after `execute`, the caller's fetch raises `ProgrammingError: Cannot operate on a closed database`. A stdlib probe shows a pending SELECT is not `in_transaction`.
- Writes go through `transaction()`, which the close respects.
- TASK-39048 AC#4 records the same window for ChaChaNotes only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bare media read in flight across `_close_main_databases` either completes, or makes the close skip media and log that media is still in use; a test is red on dev with TASK-39047 and TASK-39048 applied
- [x] #2 The TASK-39047 media quit tests stay green by name
- [x] #3 The notes record any window that remains
<!-- AC:END -->

## Implementation Plan

1. Test first (`Tests/DB/test_sync_log_retention.py`, already on the dreams-boundary sandboxed step): a bare `execute_query` read paused after its getter (before its statement) and after its statement (before its fetch), across `_close_main_databases`; it must return its rows, and the close must skip media with the "still in use" line. Show both red on base.
2. Give media handles ChaChaNotes' cursor-use tokens: open them with `_QuiescentSQLiteConnection` attached to the file's registry, so a cursor holds a use from its statement until its results are consumed and `close_if_idle` refuses meanwhile.
3. `execute_query` holds one registry use across its getter and statement, so the close cannot land between them.
4. Keep the TASK-39047 media quit tests green by name; record the windows that remain (direct `get_connection()` callers between getter and statement).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Media now has ChaChaNotes' cursor-use tokens, reused, not forked:
- `_get_thread_connection` opens with `factory=_QuiescentSQLiteConnection` and attaches the file's `SQLiteConnectionQuiescenceRegistry` before publishing the handle. Every cursor (including `conn.execute`) holds a registry use from its statement until its results are consumed (exhausting fetch, `fetchall`, close or GC), so `close_if_idle` refuses meanwhile and `_close_main_databases` skips media with "Media still in use at quit; its handles close with the process".
- `execute_query` holds one registry use across its getter and statement (body moved to `_execute_query`), so the close cannot land between the getter returning and the cursor taking its own use.
- The never-abort-a-write guarantee is unchanged: the change only adds refusals to `close_if_idle`.

**Red first (AC#1).** `test_app_unmount_store_close_never_breaks_a_bare_media_read[after-getter|before-fetch]` (`Tests/DB/test_sync_log_retention.py`) pauses a real `execute_query` read on a worker across `_close_main_databases`. On the pre-fix code: `after-getter` -> `DatabaseError: Query execution failed: Cannot operate on a closed database.`, `before-fetch` -> `ProgrammingError: Cannot operate on a closed database.`, and no "still in use" line. Green after: rows `[1, 2]` and exactly the one log line.

**AC#2.** The TASK-39047 media quit tests pass by name: `test_app_unmount_retires_a_worker_thread_handle_of_each_store[media]`, `test_app_unmount_store_close_never_aborts_a_straggler_write_in_each_store[media]`, `test_idle_close_refuses_while_a_getter_is_opening_a_handle[media]`, `test_app_unmount_store_close_skips_a_store_with_a_core_operation_running[media]`; the whole file 45 passed serially.

**AC#3, windows that remain.**
- A caller that takes `get_connection()` itself and runs its statement later (not through `execute_query`) is unprotected between the two, as ChaChaNotes is (TASK-39048 AC#4). Its cursor is protected from the statement on.
- A cursor that is never exhausted keeps its use until it is closed or collected; until then the quit skips media (handles close with the process). Same as ChaChaNotes.
- `execute_many` is unchanged: it is a write helper, and an open implicit transaction already makes the close refuse.

**Size.** `Client_Media_DB_v2.py` 9453 -> 9475 under the owner size waiver (dated row in `Tests/Architecture/test_module_size_ratchet.py`).

**Lane.** `Tests/DB/test_sync_log_retention.py` is the last target of dreams-boundary "Run Dreams and Guardian boundary contracts" (`--storage-lease-guard`); no pin changed. That step through `jobstep.sh`, three times, with this round's nodes: 1017 passed each, in 884.0 s, 607.7 s and 422.8 s at 1-minute load 48, 44 and 13. Its last CI run took 7.0 min (dev run 42296831798, job 11.1 min). No new node leaked a lease under the guard.

**Covering run.** 54 files (Backup_Recovery inventory, owner and data-group files, `Tests/UI/test_backup_data_groups.py`, the Workspace / Library Collections / notifications / store-kernel DB tests, `Tests/Media_DB`, the media and quiescence DB tests) at `-n 4` on this branch: 1379 passed, 27 failed, 3 skipped in 13 min (load ~40). Rerun per file on base `fu-base-r12`: 25 of the 27 fail there too. The other two were `test_core_owner_uses_literal_seam_and_preserves_connection_contract[media-first-create|media-harden]`, which pin media's exact connect kwargs; they now include the `factory`, as the ChaChaNotes case already does (TASK-39118). `test_every_durable_installed_adapter_has_a_named_selectable_group` was red on dev on `db.workspace_project_index` (missing from its non-selectable set), so it never reached the Personal Context owners; that row is added with them and the test passes.

**Files.** `tldw_chatbook/DB/Client_Media_DB_v2.py`, `Tests/DB/test_sync_log_retention.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Tests/DB/test_core_sqlite_owner_privacy.py`.
<!-- SECTION:NOTES:END -->
