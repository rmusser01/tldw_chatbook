---
id: TASK-39048
title: "The quit-time ChaChaNotes close ignores active storage core operations, so a straggler's raw handle can be closed between two statements"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 15:24
labels:
- follow-up
- database
- lifecycle
priority: low
references:
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/DB/base_db.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/participants.py
- Tests/DB/test_sync_log_retention.py
dependencies:
- TASK-38955
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on storage PR #3435 (head 3cc391fadc), which lands TASK-38955. Line numbers are dev's; ones marked #3435 are code that PR adds. Dev has no quit-time close of other threads' ChaChaNotes handles, so this gap opens with TASK-38955; its review notes document that "a handle fetched before the close fails loudly".

**What the close checks.**
- The quit close (`_close_main_databases`, #3435 `app_lifecycle.py:437`) calls `close_idle_connections` (#3435 `ChaChaNotes_DB.py:3258`), which calls `close_if_idle` (#3435 `base_db.py:355-384`).
- `close_if_idle` checks the barrier, active uses, acquisitions and transactions. It does not check the storage layer's active core operations (`storage._operations`, `Backup_Recovery/storage_admission.py:43`, added by `_repository_operation` at `:113-187`).
- The storage layer's own closers do refuse while an operation is active: `retire_current_thread_caches` (`participants.py:163`), `close_settled_core_cache` (`:220`) and `core_closing` (`:482-492`).

**What a straggler sees.**
- 18 sites in `ChaChaNotes_DB.py` fetch a raw connection and then run several statements on it (`conn = self.get_connection()`, plus `src_conn =` at `:3595`). Examples: the backup and integrity check (`:3595`, `:3641`), `execute_query` and `execute_many` (`:3710`, `:3838`), vacuum (`:22445`), and the flashcard methods (`:22566` to `:24779`). #3435 shifts each by 16 lines.
- A probe ran a worker inside `operation_owned_connection` and `core_operation`, holding a raw handle across the close:

  ```
  active core operations on ChaChaNotes: 1
  close_settled_core_cache -> False registered after: 2
  close_idle_connections -> True registered after: 0
  straggler second statement: sqlite3.ProgrammingError: Cannot operate on a closed database.
  straggler re-fetch: ok (reopened)
  ```
- Skipping the close while a core operation is active only narrows the window. A bare `asyncio.to_thread(db.<method>)` straggler holds no core operation (it only passes `core_access`), so it stays exposed between `get_connection()` and its first cursor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While any ChaChaNotes core operation is active, the quit close closes no ChaChaNotes handle and logs the existing "still in use" line; a test with a straggler inside `core_operation` holding a raw handle across the close is red before the fix (`ProgrammingError: Cannot operate on a closed database`) and green after
- [x] #2 The TASK-38955 quit tests in `Tests/DB/test_sync_log_retention.py` stay green by name
- [x] #3 The active-operation check is covered by a test that touches no private storage attribute, and the notes state the lock order between `close_if_idle`'s registry lock and the storage condition
- [x] #4 The notes record the window that remains for a bare `asyncio.to_thread` straggler that holds no core operation
<!-- AC:END -->

## Implementation Plan

1. Red test: add ChaChaNotes to TASK-39047's `test_app_unmount_store_close_skips_a_store_with_a_core_operation_running` (a worker inside the public `core_operation` holds a raw handle between two statements, no transaction); it reads the logged "still in use" line and touches no private storage attribute.
2. `CharactersRAGDB.close_idle_connections` passes itself to `close_if_idle`, which TASK-39047 taught to refuse while a counted core operation runs on the file, read under the registry lock.
3. TASK-38955's quit tests stay green by name; record the lock order and the window left for a bare `asyncio.to_thread` straggler.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `CharactersRAGDB.close_idle_connections` now passes the store to `close_if_idle` (`tldw_chatbook/DB/ChaChaNotes_DB.py`). TASK-39047 gave `SQLiteConnectionQuiescenceRegistry.close_if_idle(repository)` (`tldw_chatbook/DB/base_db.py`) one more refusal: any counted core operation on the file, read by `participants.core_operation_active` under the registry lock. Any same-file instance's operation counts, since the file's registry closes every instance's handles. A refused close closes nothing, and `_close_main_databases` logs the existing line, "ChaChaNotes still in use at quit; its handles close with the process".

**Test (AC#1, AC#3).** `Tests/DB/test_sync_log_retention.py::test_app_unmount_store_close_skips_a_store_with_a_core_operation_running[chachanotes]`. A worker enters the public `participants.core_operation(db)`, takes `db.get_connection()`, runs one statement, waits across `_close_main_databases`, then runs a second. The test asserts both reads ran and that the logger got the "still in use" line. It reads no private storage attribute. It was red before the fix: `['ProgrammingError: Cannot operate on a closed database.'] != ['read twice']`. Green after. The same test covers media, Dreams and Guardian for TASK-39047.

**TASK-38955's quit tests (AC#2)**, green by name: `test_app_pass_retires_the_handles_it_opens_on_its_worker_thread`, `test_app_unmount_retires_both_stores_on_every_thread`, `test_app_unmount_store_close_never_aborts_a_straggler_write`, `test_app_unmount_store_close_skips_a_busy_store_without_blocking`, `test_app_unmount_ui_close_bounds_its_checkpoint_wait`. The whole file passed serially with `--storage-lease-guard` (36 passed), and with `Tests/DB/test_chachanotes_connection_quiescence.py` (46 passed).

**Lock order (AC#3).** The registry's condition lock is always taken first, then `storage_admission._lock` (`_changed` shares it). `close_if_idle` reads the core operations under the registry lock, and each `connection.close()` it makes already took the storage lock the same way (`AdmittedConnection.close` in `private_sqlite.py`). An AST scan of every `with storage._lock` / `with storage._changed` block under `tldw_chatbook/` found no registry call, connection close or statement inside one, so nothing takes the two in the other order. One case is older than this task: a cursor or connection finalizer (`__del__`) that runs during garbage collection while its thread holds the storage lock takes the registry lock second.

**The window that remains (AC#4).** A bare `asyncio.to_thread(db.<method>)` straggler holds no core operation; `run_owned_db_call` does hold one. Such a straggler is still exposed between `get_connection()` returning and its first statement, and between two statements on a raw handle once the first has consumed its results. The registry acquisition covers the getter, and cursor-use tokens cover each statement until its results are consumed. If the close lands in the gap, the straggler's next statement raises `ProgrammingError: Cannot operate on a closed database`, and its next `get_connection()` reopens. An open transaction is never closed (`in_transaction`), so no committed or in-flight write is lost.

**Files.** `tldw_chatbook/DB/ChaChaNotes_DB.py`, `Tests/DB/test_sync_log_retention.py`.
<!-- SECTION:NOTES:END -->
