---
id: TASK-38972
title: "Mobile service shutdown closes ChaChaNotes after quiescing it, and logs a ProgrammingError for the dead thread-local handle"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- database
- web-server
- lifecycle
priority: low
references:
- tldw_chatbook/Web_Server/mobile_service.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/DB/base_db.py
- Tests/Web_Server/test_mobile_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The error.** A run of `Tests/Web_Server/test_mobile_service.py` (31 passed, 1 failed) logged 20 "Error during SQLite connection close/checkpoint" warnings across 19 tests, all on MainThread. The error is `ProgrammingError('Cannot operate on a closed database.')`, raised at `tldw_chatbook/DB/ChaChaNotes_DB.py:3505` (`conn.in_transaction`).

**The stack.** The test's `_shutdown` (`:90`) calls `mobile_service.py:1904` (`db.close()`), which calls `ChaChaNotes_DB.close` (`:22436`), which calls `close_connection` (`:3493`).

**The cause.** This is an already-closed handle, not a close on the wrong thread.
- `tldw_chatbook/Web_Server/mobile_service.py:1899-1905` runs `with db.quiesce_connections(timeout_seconds=5): pass` and then `db.close()`.
- Quiescence's `close_registered` (`ChaChaNotes_DB.py:3308-3315`) closes every registered handle, including the calling thread's cached `_local.conn`, but does not clear that cache.
- `close()` then works on the dead cached handle. Its exception path skips `registry.unregister` and the `local.conn = None` clear (`:3552-3560`), so the thread keeps a dead cached handle.

mobile_service is the only site in the repo that quiesces and then closes. TASK-38714 (Done) audited where these closes run, not this order. TASK-38955 (app shutdown leaving these DBs open) meets the same trap: a close added after quiescence hits the dead cached handle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mobile shutdown retires the ChaChaNotes handles once and logs no close ProgrammingError; a test that asserts no such warning is red on dev
- [ ] #2 After shutdown, no registered connection or lease remains and no thread-local cache holds a closed handle
- [ ] #3 The notes say which site was fixed (the shutdown sequence, or a close_connection that tolerates an already-retired cached handle) and why
<!-- AC:END -->
