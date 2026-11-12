---
id: TASK-38972
title: "Mobile service shutdown closes ChaChaNotes after quiescing it, and logs a ProgrammingError for the dead thread-local handle"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:21
updated_date: 2026-11-06 05:22
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
- [x] #1 Mobile shutdown retires the ChaChaNotes handles once and logs no close ProgrammingError; a test that asserts no such warning is red on dev
- [x] #2 After shutdown, no registered connection or lease remains and no thread-local cache holds a closed handle
- [x] #3 The notes say which site was fixed (the shutdown sequence, or a close_connection that tolerates an already-retired cached handle) and why
<!-- AC:END -->

## Implementation Plan

1. Red test in `Tests/Web_Server/test_mobile_service.py`: a real round trip and shutdown log no SQLite close error, and leave no registered handle, no ChaChaNotes lease and no cached handle on the shutdown thread.
2. Fix at the shared seam: `CharactersRAGDB.close_connection` forgets a cached handle the quiescence registry no longer holds (the getter's existing check), instead of closing it again.
3. Probe other threads' caches once with a temporary recorder.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Site fixed (AC#3).** The fix is `CharactersRAGDB.close_connection` (`tldw_chatbook/DB/ChaChaNotes_DB.py`), not the mobile shutdown sequence. When the thread's cached handle is no longer in the quiescence registry, `close_connection` now forgets it (clearing `conn` and the three per-connection authorizations) instead of closing it a second time. That is the same check `_get_thread_connection` already makes on that cache. Every cached handle is registered before it is cached, and only a retirement unregisters it, so "cached and unregistered" means already closed. Why this site: every quiesce-then-close caller goes through it. That includes the mobile shutdown and TASK-38955's new quit close. The getter and `close()` now agree, and `quiesce; close` is a correct sequence, so `mobile_service.py` is unchanged.

**Test.** New `test_shutdown_retires_chachanotes_handles_once_without_close_errors` in `Tests/Web_Server/test_mobile_service.py` does a real round trip and shutdown. It then asserts four things: no "SQLite connection close" warning, `registered_connection_count() == 0`, no live lease on the ChaChaNotes path, and an empty cache on the shutdown thread.
- Red on dev, at the warning assertion: `Error during SQLite connection close/checkpoint ... exception_type=ProgrammingError`.
- Green after the fix.

**Other threads' caches (AC#2).** A temporary, uncommitted probe recorded every thread's `_local` dict at open. After shutdown it reported `[('MainThread', True, None), ('asyncio_0', True, None)]`: no alive thread held a cached handle.

**Suite.** `Tests/Web_Server/test_mobile_service.py` on base: 20 close warnings, 31 passed, 1 failed. After the fix: 0 warnings, 32 passed, 1 failed. The failure is `test_serve_wiring_attaches_relay_only_when_broker_exists`, red at base with the same command.

**Size.** `ChaChaNotes_DB.py` went from 25854 to 25861 lines, raised under the owner waiver.
<!-- SECTION:NOTES:END -->

