---
id: TASK-634
title: >-
  RAG shared-service lock held across blocking model construction deadlocks the
  UI thread
status: Done
assignee: []
created_date: '2026-07-25 20:54'
updated_date: '2026-07-25 22:51'
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
- [x] #5 Round 2: the concurrent-first-touch race in config_profiles.get_profile_manager() (no lock guarding its lazy singleton) is closed -- two racing first-time callers construct exactly one ConfigProfileManager and share it
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

---

## Round 2 (final re-UAT): deadlock reproduced via a different trigger

Live re-UAT (scratchpad/uat-final/final-report.md, tip b4be6e6f3 including
48a042b57/7937e9d19/5e92b548e) confirmed task-635 and the settings.py None-guard
both hold, but found a NEW-SHAPE freeze: Backfill immediately followed by
opening the Clone-profile modal froze the app for 447s (kill -9 required,
unresponsive even to Ctrl+Q). `sample(1)` at t~62s and t~300s both pinned the
MAIN thread in `lock_PyThread_acquire_lock` -> `PyThread_acquire_lock_timed`
-> `_pthread_cond_wait` -- a genuine native lock/condition-variable wait, not
idle. The backfill worker's own logical work (per the app log) completed in
~2-3s; the freeze set in only afterward. Network to huggingface.co was
verified fast during the freeze (rules out a repeat of round 1's stalled
download); the embeddings/chromadb cache dirs showed nothing still
building.

### Investigation (root-cause-before-fix, per systematic-debugging)

Read both sample dumps in full. Native-frame-only symbolication (macOS
`sample(1)` does not resolve Python bytecode frames, only C-level ones) means
the EXACT Python call site can't be read directly off the dump -- worked
backward from code instead:

- Enumerated and read every `threading.Lock`/`RLock` reachable from the
  UAT's exact trigger path: `ingestion_indexing.py` (`_shared_service_lock`,
  `_shared_service_build_lock`, `_first_run_lock`, `_indexer_lock`,
  `_hook_lock`), `parallel_processor.ProgressTracker._lock` (per-run instance,
  brief in-memory-only critical sections), `simplified/circuit_breaker.py`
  (deliberately a `threading.Lock`, brief in-memory-only critical sections,
  well-documented), `simplified/simple_cache.py` (`RLock` + an `asyncio.Lock`
  that binds lazily on first `await`). `config_profiles.py` had **zero**
  locks at all (ruling out the reviewer's "config_profiles manager lock"
  hypothesis as literally nonexistent).
- Traced the exact call chain the report describes: the backfill worker's
  tail (`settings_screen.py::_rag_backfill_worker`) does 3
  `call_from_thread` calls, the last of which
  (`_refresh_library_rag_index_status`) synchronously starts ANOTHER
  `@work(exclusive=True, thread=True)` worker (`_rag_index_status_worker` ->
  `fetch_index_status()` -> `index_status()` -> a second `chromadb.
  PersistentClient()` for the same persist_directory the shared service
  already opened). Read Textual 8.2.7's actual source
  (`call_from_thread`/`_work_decorator.py`/`worker_manager.py`/`worker.py`)
  line by line: none of Textual's own worker-dispatch/cancel-group machinery
  touches a `threading.Lock` -- it's plain asyncio Task creation/cancellation.
- The Clone-modal-open handler itself
  (`_trigger_library_rag_profile_clone`) only calls `active_profile_info()`
  (profile-manager read) + `push_screen(...)` -- no RAG-service touch, and
  the modal (`RagProfileNameModal`) has no RAG logic in `compose`/`on_mount`.
  Since the freeze began BEFORE the user ever got to confirm the clone name,
  Clone's own CRUD/save path never actually ran in this scenario.
- Checked chromadb's `SharedSystemClient` (Python-level, 1.5.8): its
  `_create_system_if_not_exists` check-and-create has **no lock** guarding
  it (a real TOCTOU race), but the 14 `tokio-rt-worker` + 2
  `sqlx-sqlite-worker` threads visible in both samples were verified IDLE
  (waiting on ordinary pool condvars/semaphores for work), not stuck mid
  I/O -- ruling out a live Rust-side contention as the direct cause (a
  contended construction would show ACTIVE, not idle, Rust frames). The
  MAIN thread's own frames never entered `chromadb_rust_bindings.abi3.so` at
  all in either sample, which also rules out the MAIN thread itself being
  the one blocked inside chromadb's Rust layer.
- Verified `asyncio.run()`'s teardown (`_cancel_all_tasks` +
  `shutdown_asyncgens` + `shutdown_default_executor(timeout=300)` in
  CPython 3.12) as a candidate for the transient loop's own
  `select_select_impl` busy-loop (Thread_141102364 in the dump) -- the
  300s `THREAD_JOIN_TIMEOUT` is suspicious against a ~447s total freeze, but
  a grep of `RAG_Search/` found `run_in_executor` is only ever used with
  `self._executor`/a dedicated pool (rag_service.py's chunking) or would
  only fire for non-empty `_store_chunks` calls -- this UAT's backfill
  indexed 0 items, so that specific executor likely never got created on
  the transient loop. No `asyncio.create_task`/`ensure_future` exists
  anywhere in `RAG_Search/`, ruling out an immortal background task
  ignoring cancellation.
- Confirmed via `rag_service.py:198` that `_get_embedding_dimension()` runs
  EAGERLY inside `RAGService.__init__` (i.e. during `create_rag_service()`,
  before the "Created shared RAG service" log fires) -- this IS where the
  `all-MiniLM-L6-v2` HF 404 (missing the `sentence-transformers/` prefix,
  noted separately below) is hit, but the log timeline shows it resolved
  in ~2-3s in this run, not 447s, so it's not the direct cause here either.

### Two targeted empirical reproductions (real Textual, real locks)

1. A minimal real `textual.app.App` exercising the EXACT nested-worker
   shape (`@work` thread A doing 3 `call_from_thread` calls, the last
   starting `@work` thread B, racing a concurrent `push_screen`) completed
   cleanly with no hang.
2. The REAL `ingestion_indexing.get_shared_rag_service()`/its real locks,
   raced between a background-thread worker (slow fake `create_rag_service`,
   1s) and a coroutine on the real Textual main thread calling
   `get_shared_rag_service()` concurrently: resolved in ~0.95s (bounded by
   the construction time, not indefinite) -- confirming the task-634 (round
   1) two-lock design has no reentrant/indefinite-hang defect under real
   asyncio-thread interaction.

Neither reproduction hung, so the exact single native frame pinning the main
thread for the full 300s+ window could not be conclusively identified with
the tools available (no py-spy in this environment; `sample(1)` doesn't
resolve Python-level frames). This is disclosed honestly rather than
asserting false certainty.

### What WAS fixed (round 2)

Found and closed one concrete, provable, reproducible race in the exact
reachable path this UAT exercised: `config_profiles.get_profile_manager()`'s
lazy singleton (`_GLOBAL_PROFILE_MANAGER`) had **no lock** guarding its
check-and-create. The Clone-modal-open handler's `active_profile_info()`
(main thread) and the backfill worker's `create_rag_service()` ->
`get_profile_manager()` (worker thread) reach this function only ~7ms apart
on a genuinely fresh process (matching the UAT's fresh `uat_final` install)
-- both can observe `_GLOBAL_PROFILE_MANAGER is None` and each construct
their OWN `ConfigProfileManager`, breaking the module's own documented
invariant ("sharing one instance keeps writes visible to every default-dir
caller"): whichever call loses the race is left holding an orphaned manager
that never sees the winner's profile CRUD.

RED test: `Tests/RAG/test_config_profiles.py::
test_concurrent_first_touch_constructs_exactly_one_manager` -- two threads
race `get_profile_manager()` with a slowed `ConfigProfileManager.__init__`;
confirmed RED against the original code (`ConfigProfileManager` constructed
twice). Fix: `_profile_manager_lock` (double-checked locking, same shape as
`ingestion_indexing._shared_service_build_lock`) guards ONLY the
check-and-create -- safe to hold across `ConfigProfileManager.__init__()`
since nothing in it calls back into Textual/`call_from_thread` (unlike
`_shared_service_build_lock`, which must NOT be held across
`create_rag_service()` -- documented explicitly in the new code comment so
the two locks' different safety rules aren't confused later).

Verified GREEN: the new test, full `Tests/RAG/test_config_profiles.py` (43
passed), `Tests/RAG/test_ingestion_indexing.py` (31 passed),
`Tests/UI/test_settings_rag_profile_region.py` (121 passed), and the full
`Tests/RAG/` suite (568 passed, 8 skipped).

This fix is disclosed as the most concrete, provable defect found in the
exact reachable race window -- NOT a confirmed, closed-loop explanation of
the specific 447s main-thread lock-acquire signature. Residual follow-up
(further hardening the call_from_thread cascade, deeper Python-level
tracing with better tooling) is left to task-640, which already covers
"shrink the shared-service critical section" work in this same area.

### Separately noted (not fixed here, folded into task-640's description)

The builtin `hybrid_basic` profile's embedding model id `all-MiniLM-L6-v2`
404s against the Hugging Face Hub -- it is missing the
`sentence-transformers/` org prefix. Currently caught and gracefully
defaulted to dim=768 (see `rag_service.py::_get_embedding_dimension`), so it
is not fatal, but the builtin profile's semantic/embedding path is silently
degraded out of the box for every fresh install.
<!-- SECTION:NOTES:END -->
