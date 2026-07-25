---
id: TASK-634
title: >-
  RAG shared-service lock held across blocking model construction deadlocks the
  UI thread
status: Done
assignee: []
created_date: '2026-07-25 20:54'
updated_date: '2026-07-25 21:29'
labels:
  - followup
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT 2026-07-25 (scratchpad/uat-refix, refix-report.md): RAG -> Backfill -> Clone froze the app 6+ minutes at 0% CPU. sample(1) shows the main thread blocked in PyThread_acquire_lock_timed (waiting on a threading.Lock from inside an asyncio task step) while a worker thread sits in a raw select() on a stalled HuggingFace CloudFront socket (CLOSE_WAIT) for the whole sample window. get_shared_rag_service() holds _shared_service_lock across create_rag_service() (which can trigger blocking HF model-download I/O), and reset_shared_rag_service()/set_shared_rag_service() (called from the main-thread Backfill/Clone/Set-active path via active_config.set_active_profile and settings_rag_profile_adapter.save_rag_defaults_to_active_profile) acquire that same lock -- so a UI-thread caller blocks for as long as the stalled network read takes, appearing as a total app freeze.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared-service lock is never held across the blocking create_rag_service()/model-construction call
- [x] #2 A concurrent lock-taking caller (e.g. reset_shared_rag_service) completes promptly even while another thread is mid-construction, instead of blocking for the duration of a slow/stalled network read
- [x] #3 No double-construction leak: only one RAG service instance is ever installed as the shared singleton even when construction and a reset race
- [x] #4 Existing once-flag first-run-import ordering and non-reentrant reset-under-lock semantics are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause: get_shared_rag_service() holds _shared_service_lock across create_rag_service() (blocking model construction / HF download); reset_shared_rag_service()/set_shared_rag_service() acquire the same lock and are reachable from the main/UI thread (active_config.set_active_profile, settings_rag_profile_adapter.save_rag_defaults_to_active_profile) -> UI-thread deadlock when construction stalls on network I/O.
2. RED: add threading-based tests in Tests/RAG/test_ingestion_indexing.py simulating slow construction and asserting a concurrent reset completes promptly; verify they fail (block/timeout) against the current code.
3. Fix: split the single lock into _shared_service_lock (fast state read/write, used by reset/set, never held across construction) and _shared_service_build_lock (serializes actual construction attempts, preserving the task-249 "exactly one build" invariant) plus a _shared_service_generation counter so a build superseded by a concurrent reset is discarded at swap time instead of resurrecting stale config.
4. Verify GREEN: new tests pass; re-run the existing task-249 concurrency regression test (Tests/Library/test_library_local_rag_search_service.py::test_concurrent_rag_queries_initialize_one_shared_service) and the full Tests/RAG/ + Tests/UI/test_settings_rag_profile_region.py suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: get_shared_rag_service() held _shared_service_lock across the blocking create_rag_service() call (real HF model-download I/O), and reset_shared_rag_service()/set_shared_rag_service() -- reachable from the main/UI thread via active_config.set_active_profile() and the Settings Backfill/Clone/Set-active save path -- acquired that same lock, so a stalled construction froze the whole app (matches the UAT sample(1) capture: main thread in PyThread_acquire_lock_timed, worker thread in select() on a CLOSE_WAIT socket).

Fix: split the single lock into two. _shared_service_lock now guards ONLY state reads/writes and is always fast (reset/set never wait on construction). _shared_service_build_lock serializes actual construction ATTEMPTS so concurrent first-touch callers still build at most once (preserves the pre-existing task-249 invariant, verified against Tests/Library/test_library_local_rag_search_service.py::test_concurrent_rag_queries_initialize_one_shared_service, which an earlier "build fully outside any lock" draft broke). A new _shared_service_generation counter discards a build that finishes after a concurrent reset superseded it, instead of resurrecting stale config.

RED tests added in Tests/RAG/test_ingestion_indexing.py::TestSharedRagServiceLockDeadlock (reset-does-not-block, exactly-once-construction, reset-races-in-flight-build-and-discards-stale-result) all failed against the original code (one via a literal 5s block reproducing the deadlock) and pass after the fix.

Verified: Tests/RAG/ (567 passed, 8 skipped), Tests/UI/test_settings_rag_profile_region.py (120 passed), and the task-249 regression test, all green. A separate, pre-existing flaky test (Tests/Library/test_library_local_rag_search_service.py::TestLibraryRagAnswerRealRuntime::test_rag_answer_empty_real_store_reports_index_empty) was confirmed via git-stash baseline comparison to fail identically with and without this change when run in one specific multi-file combination -- unrelated pollution, out of scope.

Modified: tldw_chatbook/RAG_Search/ingestion_indexing.py (get_shared_rag_service/set_shared_rag_service/reset_shared_rag_service + module-level locks/generation counter); Tests/RAG/test_ingestion_indexing.py (new TestSharedRagServiceLockDeadlock class).
<!-- SECTION:NOTES:END -->
