---
id: TASK-38954
title: "Research held-job reconciliation and restore suppression never take effect at startup, because ingest restore is asynchronous"
status: To Do
created_date: 2026-11-05 21:35
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
- [ ] #1 On a real start with a durable dispatch-held Research job and a pending operation, the held reconcile runs once the restored store attaches and links or cancels the job; a test drives the real threaded _restore_ingest_jobs (no synchronous stand-in) and is red on dev
- [ ] #2 Seeding restored ingest history schedules no per-job research_source_association workers; restored terminal jobs resume only through the bounded resume_startup, shown by a test that restores more than 50 terminal Research-linked jobs through the real path and is red on dev
- [ ] #3 The existing tests that stub _restore_ingest_jobs pass unchanged or are moved to the asynchronous path, with the reason recorded
<!-- AC:END -->
