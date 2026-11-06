---
id: TASK-38986
title: "Five dreams-boundary Guardian tests leave a storage lease open, so a later local-pause test in the same pytest process fails"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 05:00
labels:
- follow-up
- tests
- test-infra
- guardian
- backup
priority: medium
references:
- Tests/Guardian/test_live_discovery_consent.py
- Tests/Guardian/test_daily_native_lifecycle.py
- Tests/Guardian/test_crisis_resources.py
- Tests/Guardian/test_check_pipeline.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/DB/store_kernel.py
- backlog/docs/lessons-testing-evidence.md
dependencies:
- TASK-38759
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

**Red-before control.** `test_async_file_participant_lifetimes.py`, run after the three Guardian files in one serial process:
- on dev (29592ecc9b): 4 failed and 42 passed. Three are TASK-38759's drain nodes; the fourth is TASK-38759's own lock-order red;
- as it stands on the TASK-38759 branch (load 5.8): 3 failed and 43 passed. All three are TASK-38759's drain nodes, each at `assert False` on the drain. The same file alone passes 26 in 9.7 s.

The leaks are test-only, and today's step order hides them. The production lease of the same shape, TASK-38955, is medium.

**Production.** Dreams' own Guardian reads are covered:
- `read_guardian_topics` uses `operation_owned_connection` (`Dreams/profile_sources.py:178`);
- `run_finite_local_worker` (`Backup_Recovery/participants.py:42-110`) closes worker-acquired caches.

The production lease of this shape is the sync-log retention pass, which TASK-38955 owns. 38 other `asyncio.to_thread(<db>.<method>)` sites have the same shape and were not checked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the five nodes leaves storage_admission._live_leases as it found it; a per-test count shows +1 for each on dev and 0 after the fix
- [x] #2 With TASK-38759 merged, the three Guardian files followed by Tests/Backup_Recovery/test_async_file_participant_lifetimes.py in one serial pytest process pass 3 of 3 runs (red before: 4 failed, 42 passed on dev 29592ecc9b, the 4th being TASK-38759's own red; 3 failed, 43 passed on the TASK-38759 branch)
- [x] #3 In the dreams-boundary step, a test that leaves a new live storage lease fails and names itself; negative control: reintroduce one of the five leaks
<!-- AC:END -->

## Implementation Plan

1. Measure the per-test live-lease delta with a step-only guard: a `--storage-lease-guard` option in `Tests/conftest.py` registers an autouse fixture that fails the test naming itself when `storage_admission._live_leases` grew (AC#1 count, AC#3 guard).
2. Close each leak in its test: retire the worker-thread Guardian handle that `revoke()` opens (`operation_owned_connection`), close the borrowed raw inbox connection on its own pool thread, close `_fresh_db`'s store in `test_crisis_resources.py`.
3. Pass the flag in the dreams-boundary step and update both step pins; keep the TASK-38759 order (lifetimes file first).
4. Evidence: the guard on the three Guardian files (+1 each before, 0 after), the whole step under the guard, the AC#2 serial combo 3 of 3, and a negative control that reintroduces one leak.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Leaks closed (test-only; no product change).**
- `test_live_discovery_consent.py`: `revoke()` runs on Dreams' worker thread. It now wraps its `upsert_rule` in `operation_owned_connection(guardian)`, so the handle that write opens is retired on that same thread. The test's main-thread `guardian.close()` cannot reach it.
- `test_daily_native_lifecycle.py` `[True]`: the borrowed branch keeps the raw `inbox._get_connection()` it opens, and the teardown closes it on the pool thread before `inbox.close()`. The store never caches that handle.
- `test_crisis_resources.py`: the `_fresh_db` store is closed in a `finally`.

**Guard (AC#3).**
- `Tests/conftest.py` gains a `--storage-lease-guard` option. When set, `pytest_configure` registers `_StorageLeaseGuard`, whose autouse fixture compares `storage_admission._live_leases` before and after each test. A new lease fails the test at teardown with its node id and each lease's path and thread. Autouse fixtures are set up before the test's other function fixtures and torn down after them, so wider-scoped fixtures' leases are excluded.
- The guard lives in conftest rather than its own module. A conftest-imported module fails `test_perf_guard_runs_when_a_file_its_steps_load_changes` unless `perf-guard.yml` lists it, and that workflow is outside this brief.
- The dreams-boundary step passes the flag. `.github/workflows/derived-artifacts.yml` carries the flag and an updated comment. Both step pins are updated: `Tests/CI/test_response_rules_critical_gate.py`, and `Tests/CI/test_ci_queue_pressure_contract.py`, which now takes per-step flags.
- The TASK-38759 order is kept: the lifetimes file is still first.

**Evidence.**
- AC#1: the guard on the three Guardian files before the fixes flagged exactly the five nodes, `left 1 new live storage lease(s) open` each:
  - guardian.sqlite on asyncio_0, three times;
  - inbox.sqlite on ThreadPoolExecutor-4_0;
  - guardian.sqlite on MainThread.

  After the fixes: 20 passed, 0 leaks.
- Whole step: the step's 26 targets under the guard, before the fixes, gave 472 passed and 5 errors, exactly those five. No other test in the step leaks.
- AC#2: the three Guardian files followed by `test_async_file_participant_lifetimes.py`, serially in one process, passed 3 of 3 runs, 46 passed each (44.3 s, 54.9 s, 53.3 s at load 31-38). Red before, on the base worktree (e8eeae7db9) with the same command: 3 failed, 43 passed. The three TASK-38759 drain nodes failed.
- AC#3, negative control: dropping the `db.close()` from `test_crisis_resources.py` gave `ERROR Tests/Guardian/test_crisis_resources.py::test_check_pipeline_uses_the_extracted_module`, then reverted. That run also had one unidentified failure at load 86; the three full-step runs below were clean.
- Lane cost:
  - Recent PR runs: dreams-boundary job 6m14s to 7m04s of its 20 min, and this step 2m11s to 2m21s (5 runs, 2026-11-05).
  - Locally, the step's exact command with the flag, serially (my `stepguard.sh`, which parses the targets the way `dreamstep.sh` does): 472 passed in 340 s, 287 s and 316 s at load 34-51.
  - The unguarded, unfixed run was 326 s.
- CI pins: `Tests/CI` ran 410 passed, and the preflight lane-resolution pair passed 2.
<!-- SECTION:NOTES:END -->

