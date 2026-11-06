---
id: TASK-38986
title: "Five dreams-boundary Guardian tests leave a storage lease open so a later local-pause test in the same pytest process fails"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- tests
- test-infra
- guardian
- backup
priority: high
references:
- Tests/Guardian/test_live_discovery_consent.py
- Tests/Guardian/test_daily_native_lifecycle.py
- Tests/Guardian/test_crisis_resources.py
- Tests/Guardian/test_check_pipeline.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/DB/store_kernel.py
- backlog/docs/lessons-testing-evidence.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38759 records this as a follow-up but does not fix it.

**How a leak shows.**
- `storage_admission._live_leases` (`tldw_chatbook/Backup_Recovery/storage_admission.py:42`) holds every live storage lease.
- `_LocalPause.drain()` (`:282-303`) succeeds only when `_live_leases` minus startups is empty, together with its other sets.
- So a lease that one test leaves behind makes a later `pause.drain(...)` in the same pytest process time out.
- No fixture or census checks for this. Any test that pauses or drains process-wide storage has to run first, which is why TASK-38759 moved `Tests/Backup_Recovery/test_async_file_participant_lifetimes.py` to the head of the dreams-boundary step.

**Reproduction.** An in-process plugin compared `_live_leases` before and after each test across the three Guardian files, run serially: 20 passed in 11.8 s at load 9.6-10.4. Two runs showed the same five leaks, one lease each:

| Node | Store | Thread | Where the connection opens |
|---|---|---|---|
| `test_live_discovery_consent.py::test_revocation_during_cycle_wait_prevents_next_external_call[provider]`, `[search]`, `[story]` | guardian.sqlite | asyncio_0, dead at teardown | The test's `revoke()` (`:86`, called from `resolve` `:108` and `reserve` `:95`) runs `GuardianDB.upsert_rule` (`Guardian_DB.py:452`) on Dreams' `to_thread` worker. `store_kernel.transaction` (`:277`) and `_held_connection` (`:207`) open it there. The test's `finally: guardian.close()` (`:119`) runs on the main thread. |
| `test_daily_native_lifecycle.py::test_daily_file_inbox_contention_keeps_ui_live_and_retires_only_new_handles[True]` | inbox.sqlite | ThreadPoolExecutor worker, dead at teardown | The borrowed branch runs `inbox._get_connection` through the patched `acquire` (`:228`). `ClientNotificationsDB._get_connection` (`client_notifications_db.py:108-111`) returns a raw connection that is never stored in `_thread_local`, so the teardown's `inbox.close()` on the pool (`:290`) cannot close it. |
| `test_crisis_resources.py::test_check_pipeline_uses_the_extracted_module` | guardian.sqlite | MainThread | `_fresh_db` (`Tests/Guardian/test_check_pipeline.py:45-51`), used at `test_crisis_resources.py:49`, is never closed. |

**Why closing does not help.** `SQLiteStoreKernel.close()` (`tldw_chatbook/DB/store_kernel.py:287-298`) reads only `self._thread_local.conn`, so it retires only the calling thread's connection. The `_on_connection_opened` hook (`:220-226`) could track connections across threads, but only `Dreams_DB` overrides it (`Dreams_DB.py:385`, with `close_all` at `:410`).

**Red-before control (load 5.8).** `test_async_file_participant_lifetimes.py`, as it stands on the TASK-38759 branch:
- alone: 26 passed in 9.7 s;
- after the three Guardian files in one serial process: 3 failed and 43 passed. The three failures are exactly TASK-38759's nodes, each at `assert False` on the drain.

**Production.** Dreams' own Guardian reads are covered:
- `read_guardian_topics` uses `operation_owned_connection` (`Dreams/profile_sources.py:178`);
- `run_finite_local_worker` (`Backup_Recovery/participants.py:42-110`) closes worker-acquired caches.

The production lease of this shape is the sync-log retention pass, which TASK-38955 owns. 38 other `asyncio.to_thread(<db>.<method>)` sites have the same shape and were not checked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the five nodes leaves storage_admission._live_leases as it found it; a per-test count shows +1 for each on dev and 0 after the fix
- [ ] #2 The three Guardian files followed by Tests/Backup_Recovery/test_async_file_participant_lifetimes.py in one serial pytest process pass 3 of 3 runs (dev: 3 failed, 43 passed)
- [ ] #3 In the dreams-boundary step, a test that leaves a new live storage lease fails and names itself; negative control: reintroduce one of the five leaks
<!-- AC:END -->
