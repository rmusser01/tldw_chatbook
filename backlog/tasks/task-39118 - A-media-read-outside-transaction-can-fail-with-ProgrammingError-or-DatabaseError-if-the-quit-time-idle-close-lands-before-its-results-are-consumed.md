---
id: TASK-39118
title: "A media read outside transaction() can fail with ProgrammingError or DatabaseError if the quit-time idle close lands before its results are consumed"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 06:37
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
- [ ] #1 A bare media read in flight across `_close_main_databases` either completes, or makes the close skip media and log that media is still in use; a test is red on dev with TASK-39047 and TASK-39048 applied
- [ ] #2 The TASK-39047 media quit tests stay green by name
- [ ] #3 The notes record any window that remains
<!-- AC:END -->

## Implementation Plan

1. Test first (`Tests/DB/test_sync_log_retention.py`, already on the dreams-boundary sandboxed step): a bare `execute_query` read paused after its getter (before its statement) and after its statement (before its fetch), across `_close_main_databases`; it must return its rows, and the close must skip media with the "still in use" line. Show both red on base.
2. Give media handles ChaChaNotes' cursor-use tokens: open them with `_QuiescentSQLiteConnection` attached to the file's registry, so a cursor holds a use from its statement until its results are consumed and `close_if_idle` refuses meanwhile.
3. `execute_query` holds one registry use across its getter and statement, so the close cannot land between them.
4. Keep the TASK-39047 media quit tests green by name; record the windows that remain (direct `get_connection()` callers between getter and statement).
