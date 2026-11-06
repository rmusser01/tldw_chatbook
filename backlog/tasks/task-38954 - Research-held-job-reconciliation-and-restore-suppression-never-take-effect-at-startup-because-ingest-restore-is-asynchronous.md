---
id: TASK-38954
title: "Research held-job reconciliation and restore suppression never take effect at startup, because ingest restore is asynchronous"
status: Done
assignee: ['@claude']
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 03:54
labels:
- follow-up
- library
- research
- bug
priority: medium
references:
- tldw_chatbook/app_ingest_queue.py
- tldw_chatbook/app.py
- tldw_chatbook/Library/library_ingest_jobs.py
- tldw_chatbook/Research_Workspace/source_association.py
- Tests/Library/test_library_ingest_runner.py
- Tests/App/test_submit_library_ingest_job.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38714 (Done, #3409) moved the held scan onto the UI thread (`app_ingest_queue.py:760`) but did not change the ordering described here.

**The held reconcile never runs in production.**
- `on_mount` (`tldw_chatbook/app.py:3368`) calls `_restore_ingest_jobs_and_schedule_research_sources` (`tldw_chatbook/app_ingest_queue.py:704-731`).
- That calls `_restore_ingest_jobs` (`:892-919`), which only starts a `thread=True` worker and returns.
- `_library_ingest_jobs_store` is assigned only in `_apply_ingest_job_restore` (`:1012`). The worker reaches that method through `call_from_thread` (`:972`), which cannot run while `on_mount` holds the loop.
- So the check at `:713-719` always sees `None`, and `_reconcile_research_source_held_jobs` (`:752`) is never scheduled.

**The restore suppression window covers nothing.**
- `_research_source_restore_in_progress` (`:707-712`) is set only while the worker starts.
- The real seeding happens later, through `merge_restored`, `restore` and `_notify_listeners` (`tldw_chatbook/Library/library_ingest_jobs.py:615-634`, `:636-686`), after the flag is reset.
- The listener (`app_ingest_queue.py:630-673`) can then schedule one `research_source_association` worker per restored terminal Research-linked job, up to `_MAX_PERSISTED_INGEST_JOBS = 500` (`:409`). That bypasses the bounded `resume_startup` (limit 50, `tldw_chatbook/Research_Workspace/source_association.py:561-570`).
- This second consequence was inferred from the code, not run.

**Why the tests miss it.**
- `Tests/Library/test_library_ingest_runner.py:865` (`test_real_app_wires_research_association_and_restores_before_startup_resume`) and `:1007` monkeypatch `_restore_ingest_jobs` with a synchronous stand-in.
- `Tests/App/test_submit_library_ingest_job.py:275-501` sets the store by hand and calls the reconcile directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a real start with a durable dispatch-held Research job and a pending operation, the held reconcile runs once the restored store attaches and links or cancels the job; a test drives the real threaded _restore_ingest_jobs (no synchronous stand-in) and is red on dev
- [x] #2 Seeding restored ingest history schedules no per-job research_source_association workers; restored terminal jobs resume only through the bounded resume_startup, shown by a test that restores more than 50 terminal Research-linked jobs through the real path and is red on dev
- [x] #3 The existing tests that stub _restore_ingest_jobs pass unchanged or are moved to the asynchronous path, with the reason recorded
<!-- AC:END -->

## Implementation Plan

1. Write the red tests first in `Tests/Library/test_library_ingest_runner.py`, on the real `_IngestRunnerHarness` App with the real threaded `_restore_ingest_jobs` (store path monkeypatched to tmp): a held Research job with a pending operation is linked and dispatched once the store attaches (AC#1); 60 restored terminal Research-linked jobs start no per-job association worker, at seeding or at the next mutation, and `resume_startup` runs once, after the registry is seeded (AC#2).
2. Move the two tests that stub `_restore_ingest_jobs` onto the asynchronous path (AC#3).
3. Fix in `app_ingest_queue.py`: the startup entry point only arms the startup work and starts the restore; the UI-thread restore completion (`_apply_ingest_job_restore`, or a marshalled call from the worker's failure paths) schedules the held reconcile, `resume_startup` and the staging sweep exactly once. Suppression wraps `merge_restored`, and while it is on, the listener marks restored terminal Research jobs as owned by the bounded `resume_startup` instead of starting a worker per job.
4. Allowlist the held reconcile in the boot worker census if it now starts during boot; run the covering tests.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix (`tldw_chatbook/app_ingest_queue.py`).** `_restore_ingest_jobs_and_schedule_research_sources` now only arms a one-shot `_research_source_startup_pending` flag and starts the restore. The held reconcile, `resume_startup` and the staging sweep moved into `_schedule_research_source_startup`, which runs on the UI thread when the restore finishes: at the end of `_apply_ingest_job_restore` (store attached, registry seeded), or, when the restore failed before or during the handoff, through `_queue_research_source_startup_from_thread`. The staging sweep and `resume_startup` read the registry too, so they had the same ordering bug. Restore suppression now wraps `merge_restored` itself. While it is on, the listener adds restored terminal Research jobs to `_research_source_terminal_jobs_scheduled` without starting a worker, so neither the seeding nor any later mutation starts one worker per restored job; they resume only through the bounded `resume_startup`. Retries mint a new job id, so this marking never blocks a retried job.

**Tests (`Tests/Library/test_library_ingest_runner.py`, all on the `_IngestRunnerHarness` App with the real threaded `_restore_ingest_jobs`; the jobs-DB path is monkeypatched to tmp).**
- AC#1, new `test_startup_restore_links_a_held_research_job_once_the_store_attaches`. Red on dev: `assert [] == ['ingest-job-1']`, because the held reconcile never ran. Green after the fix.
- AC#2: `test_real_app_wires_research_association_and_restores_before_startup_resume` was moved to the real path with 60 terminal Research-linked jobs. Red on dev at `groups == []`: the startup work was queued before the restore landed. With that check removed, it is red at 62 groups against 3, the extra 60 being per-job `research_source_association` workers from the seeding. After the fix: exactly one each of held, startup and staging; the 60 restored ids are owned; and a later live failure starts only its own worker.
- AC#3, reason recorded in each moved test. Both stubbed tests seeded the registry inside the startup call, which production never does, and that is how this bug hid. They are moved, not unchanged:
  - The wiring test keeps every wiring assertion. Its `terminal_jobs_scheduled == set()` became the 60 owned ids, plus the later-mutation check.
  - `test_restore_suppression_cleans_up_after_exception_and_empty_repeats` now takes its exception from `merge_restored` and does its repeats sequentially. It was already red on dev at its last assertion (`{'ingest-job-1'} == set()`: TASK-32804.5's early return skipped the prune during suppression). With the new "nothing queued before the restore lands" check, it is red on dev at that check. Its final set is `{live}`, because the real `merge_restored` keeps live jobs that the stub's `restore([])` wiped.
- The wiring test fails locally with `RecoveryRequired: raw_source_selection_changed` from `get_user_data_dir()` (TASK-38111), before and after. It was verified under a temporary `@private_profile_test` marker, 3 of 3 green at load 25, and the marker was then removed.

**Boot census.** `Tests/Performance/test_boot_worker_census.py` failed after the fix because `_reconcile_research_source_held_jobs` now starts during boot. It always should have, and `MiscWorkerHandler` and `test_worker_failure_event.py` already list its group. The row was added with its owner named.

**Size.** `app_ingest_queue.py` went from 5021 to 5057 lines, raised under the owner waiver with a dated row comment.

**Checks.** The covering files (`test_startup_init_hygiene.py`, `test_library_ingest_jobs_restore.py`, `test_submit_library_ingest_job.py`, `test_merged_startup_worker_retirement.py`, `test_worker_failure_event.py`, `test_library_ingest_runner.py`) were run with `-n 4` against the base worktree. They fail 122 locally on base and 120 here; the only difference is the two moved tests, and there are no new failures. The census and stagger policy passed 21, and the size ratchet passed 201.
<!-- SECTION:NOTES:END -->

