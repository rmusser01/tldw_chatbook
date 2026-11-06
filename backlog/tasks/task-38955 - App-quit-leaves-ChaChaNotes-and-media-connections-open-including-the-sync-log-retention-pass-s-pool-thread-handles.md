---
id: TASK-38955
title: "App quit leaves ChaChaNotes and media connections open including the sync-log retention pass's pool-thread handles"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- database
- lifecycle
priority: medium
references:
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/app_shutdown.py
- tldw_chatbook/DB/base_db.py
- Tests/DB/test_sync_log_retention.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-34389 (retention) and TASK-38714 (wrong-thread closes) are both Done, and neither covers a connection that is opened and never closed, or the missing close at shutdown.

**Retention pass.**
- `_perform_sync_log_retention` (`tldw_chatbook/app_lifecycle.py:1510-1555`) runs `asyncio.to_thread(db.delete_sync_log_entries_older_than, cutoff)` for both DBs (`:1543-1545`) and never retires the connection that call opens on the pool thread.
- `perform_media_cleanup` (`:1567-1576`) does it properly: it calls `close_connection()` in a `finally` on the same thread. The repo's helpers for this are `run_owned_db_call` (`tldw_chatbook/DB/base_db.py:140`) and `operation_owned_connection` (`:45`).
- The pass runs at startup and then every 24 h (`app_lifecycle.py:1433-1439`).
- A probe through the scratch-profile wrapper: one pass raised the ChaChaNotes `registered_connection_count()` from 1 to 2, and `private_sqlite._ordinary_connections` from 2 to 4 leases, all held on the default-executor thread.

**Quit.**
- `on_unmount` (`app_lifecycle.py:1022-1306`) and `tldw_chatbook/app_shutdown.py` close only the ingest-jobs store, the writing service and the research service.
- Nothing calls `quiesce_connections` or `close_connection` on `chachanotes_db`, or `close_connection` on `media_db`. A grep outside `DB/` finds no caller at shutdown.
- `Subscriptions_DB.py:266-295` records a measurement that a clean exit leaves no `-wal` file. So the criteria are about connections and leases being retired at a defined moment, not about WAL loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a sync-log retention pass, neither DB keeps a connection or lease that the pass opened; a test in Tests/DB/test_sync_log_retention.py compares registered connections and leases before and after and is red on dev (count 1 to 2, leases +2)
- [ ] #2 After app unmount, the ChaChaNotes and media DBs have no registered connection or ordinary lease left on any thread; a test is red on dev
- [ ] #3 The quit path stays bounded: the closes run after workers settle and never hold the exit past the shutdown watchdog
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-05, evidence added at filing (round 8, commit 3). Priority raised from low to medium because the leftover retention-pass lease also blocks a backup pause.

**Probe.** Run with the `Tests/DB/test_sync_log_retention.py` fixtures, one case per process, through the scratch-profile wrapper:
- One retention pass, then closes on the main thread: 2 storage leases were left on asyncio_0, and `_begin_local_pause().drain()` with a 1 s budget returned False after 1.00 s.
- Control, the same pass plus `close_connection` on the worker thread: 0 leases were left, and the drain returned True in 0.00 s.

**Effect during a real backup (inferred from code, not run in a mounted app).**
- `monitor_app` (`runtime_maintenance.py:698-731`) calls `retire_current_thread_caches` (`participants.py:113`), which closes only the loop thread's caches.
- It then retries `pause.drain` until its 30 s deadline and raises `RecoveryRequired("runtime_native_resources_not_settled")`.
- So a retention pass that ran since startup could make the next backup fail to settle.

The test-side leases of the same shape are filed separately as the dreams-boundary Guardian lease task.
<!-- SECTION:NOTES:END -->
