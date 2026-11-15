---
id: TASK-41035
title: "Backup participant-lifetime tests are red on dev, and their leaked leases fail test_merged_service_cache_retirement"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- backup-recovery
priority: medium
references:
- Tests/Backup_Recovery/test_operational_participant_lifetimes.py
- Tests/Backup_Recovery/test_participant_lifetimes.py
- Tests/Backup_Recovery/test_domain_owners.py
- Tests/Backup_Recovery/test_merged_service_cache_retirement.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three participant-lifetime files are red on dev ecad404bd4 (2026-11-15), both at `-n 4` and serially. They feed a fourth file's false failures.

- **`Tests/Backup_Recovery/test_operational_participant_lifetimes.py`** fails 32 nodes:
  - 14 LocalResearchService and LocalWritingService params read `owner._connection`. Those services keep `_connections` instead, so the params fail with AttributeError.
  - The `[allocation|setup]` params of `test_previous_raw_allocation_rechecks_participant_before_return` and `test_participant_pause_during_raw_allocation_refuses_return` (8 nodes) expect a `RecoveryRequired` that is not raised.
  - `test_pending_operation_remains_visible_until_refused_on_worker` fails for six stores, on a drain that returns False or a worker event that never fires.
  - `test_source_thread_retires_after_managed_work_and_raw_borrowers[WorkspaceDB|AgentRunsDB|ClientNotificationsDB]` expects a `sqlite3.ProgrammingError` that no longer comes.
  - `test_scheduled_schema_migrations_finish_in_original_admitted_scope` gets 7 where it pins 3.
- **`Tests/Backup_Recovery/test_participant_lifetimes.py`**
  - The `[EventStateRepository]` and `[SyncStateRepository]` params of `test_repository_schema_connection_is_closed_on_return`, `test_actual_worker_transaction_drains_after_commit_and_native_close` and `test_repository_transaction_rolls_back_and_closes_native_file` fail: no `ProgrammingError`, or a drain that returns False.
  - `test_pre_authority_attempt_stays_counted_until_return_and_cannot_allocate_late[authority]` also fails.
- **`Tests/Backup_Recovery/test_domain_owners.py`**: `test_real_file_owner_operations_retire_native_connections[*]` (4 params) expects a `sqlite3.ProgrammingError` that no longer comes.
- **`Tests/Backup_Recovery/test_merged_service_cache_retirement.py`** passes alone (16 of 16). It fails every node when an EventStateRepository test failed earlier in the same xdist worker. TASK-36948.1's review (2026-11-07) traced this to tests that fail before their `repository.close()` and so leave a live lease, which the merged file's `pause.drain()` then waits on. Which worker gets those tests depends on xdist scheduling, so the merged file's result swings from run to run, and a real regression in it can hide.

TASK-36948.1's review also found that EventStateRepository no longer closes on rollback or commit, so some of these expectations are stale rather than wrong product behaviour. None of these files runs on a PR lane (TASK-40307 covers lanes for this directory).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every node of test_operational_participant_lifetimes.py, test_participant_lifetimes.py and test_domain_owners.py passes on dev at `-n 4`; each red node first gets a named cause (test drift with its commit, or a product defect fixed with a test that was red)
- [ ] #2 A test that opens a repository closes it on every path, so a failing test leaves no live lease behind
- [ ] #3 test_merged_service_cache_retirement.py fails fast at setup, naming the leftover leases, when a lease is already live
- [ ] #4 test_merged_service_cache_retirement.py passes 3 of 3 runs at `-n 4` alongside the other three files
<!-- AC:END -->
