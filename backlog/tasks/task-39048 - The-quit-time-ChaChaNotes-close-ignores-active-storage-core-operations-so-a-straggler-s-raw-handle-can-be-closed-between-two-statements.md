---
id: TASK-39048
title: "The quit-time ChaChaNotes close ignores active storage core operations, so a straggler's raw handle can be closed between two statements"
status: To Do
created_date: 2026-11-06 10:09
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
- [ ] #1 While any ChaChaNotes core operation is active, the quit close closes no ChaChaNotes handle and logs the existing "still in use" line; a test with a straggler inside `core_operation` holding a raw handle across the close is red before the fix (`ProgrammingError: Cannot operate on a closed database`) and green after
- [ ] #2 The TASK-38955 quit tests in `Tests/DB/test_sync_log_retention.py` stay green by name
- [ ] #3 The active-operation check is covered by a test that touches no private storage attribute, and the notes state the lock order between `close_if_idle`'s registry lock and the storage condition
- [ ] #4 The notes record the window that remains for a bare `asyncio.to_thread` straggler that holds no core operation
<!-- AC:END -->
