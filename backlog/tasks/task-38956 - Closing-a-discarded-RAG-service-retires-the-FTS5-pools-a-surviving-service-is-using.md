---
id: TASK-38956
title: "Closing a discarded RAG service retires the FTS5 pools a surviving service is using"
status: Done
assignee: ['@claude']
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 05:33
labels:
- follow-up
- rag
- lifecycle
priority: low
references:
- tldw_chatbook/RAG_Search/simplified/db_connection_pool.py
- tldw_chatbook/RAG_Search/simplified/rag_service.py
- tldw_chatbook/RAG_Search/ingestion_indexing.py
- Tests/RAG_Search/test_keyword_leg_db_resolution.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38714 (Done, #3409) changed the thread the close runs on, not how many pools it closes.

**How the surviving pool gets closed.**
- `RAGService.close()` (`tldw_chatbook/RAG_Search/simplified/rag_service.py:4449-4466`) calls `close_all_pools()`.
- `close_all_pools()` (`tldw_chatbook/RAG_Search/simplified/db_connection_pool.py:117-128`) retires every pool in the module-wide `_connection_pools` (`:20`) and then clears the dict. Pools are keyed by db_path, not by service (`:68-98`).
- In production this is triggered by `_close_discarded_rag_service(built)` (`tldw_chatbook/RAG_Search/ingestion_indexing.py:226-255`, called at `:421`). When a build loses the race to a concurrent set or reset, closing the losing build also retires the pool the surviving shared service is using.

**Effect since #3409.**
- `close_all_pools` holds the lock while queued searches drain.
- An in-flight search gets `RuntimeError("media search pool closed")`, waits 0.1 s and retries on a rebuilt pool (`rag_service.py:1986-2014`). After two failed retries the keyword leg returns `[]`.
- The next search pays for constructing a fresh `MediaDatabase`.

The cost is churn and a rare empty keyword leg, not a leak.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Closing a RAGService that never searched (such as a discarded build) leaves another live service's pool open: that service's next search reuses the same connection with no retry and no rebuild; the test is red on dev
- [x] #2 Closing the last user of a pool still retires that pool's connection on its own thread; test_keyword_search_leaves_no_executor_connection_after_close and the racing close tests stay green
- [x] #3 Plain tests in Tests/RAG_Search/test_keyword_leg_db_resolution.py fail locally with RecoveryRequired (TASK-38111); the notes say how the new test was verified (the temporary private-profile marker workaround) and that the marker was removed
<!-- AC:END -->

## Implementation Plan

1. Red test in `Tests/RAG_Search/test_keyword_leg_db_resolution.py`: closing a never-searched service leaves the surviving service's pool and connection in use, with no retry or rebuild (verified under a temporary private-profile marker, TASK-38111).
2. Track the services that use each pool; `RAGService.close()` releases its own, and a pool retires (on its own thread) only when its last user releases it. `close_all_pools()` stays for explicit full teardown.
3. Keep the TASK-38714 close and racing-close tests green.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** In `tldw_chatbook/RAG_Search/simplified/db_connection_pool.py`, `_pool_users` keeps a `WeakSet` of services for each db_path.
- `get_connection_pool(..., user=)` registers its caller. The fast path stays lock-free once the caller is registered.
- `release_pools(user)` drops the user from its pools. It retires each pool it was the last user of, on that pool's own thread, through `_retire_pool` (factored out of `close_all_pools`).
- `close_all_pools()` still retires everything and also clears `_pool_users`.
- `RAGService` passes `user=self` on each keyword-search attempt, and `close()` calls `release_pools(self)` instead of `close_all_pools()`.
- So a discarded build that never searched releases nothing, while the last user's close still retires the pool on its thread.
- A service dropped without `close()` leaves the WeakSet by GC, and its pool stays open until another user releases it or `close_all_pools()` runs. That is the same as dev when nobody closes.

**Tests (`Tests/RAG_Search/test_keyword_leg_db_resolution.py`).**
- New `test_closing_a_discarded_service_keeps_the_survivors_pool`: the survivor searches, a never-searched service closes, and the survivor searches again. The test asserts (same pool, same connections, no "FTS5 search attempt" retry warning) and that every opened connection is closed after the survivor's own close.
- Red on dev: `(False, False, []) == (True, True, [])`. The pool and connection were rebuilt, without a retry. Green after.
- AC#2: `test_keyword_search_leaves_no_executor_connection_after_close` and both `test_a_search_racing_close_all_pools_leaves_no_connection_open` cases passed before and after (4 passed with the new test). `Tests/RAG_Search/test_db_connection_pool_thread_safety.py` passed 3.
- AC#3: plain tests in this file fail locally with `RecoveryRequired: raw_source_selection_changed` (TASK-38111). The new test and the three TASK-38714 tests were verified under a temporary `@private_profile_test` marker (with a temporary `request` parameter). Both were then removed; the committed file carries neither.

**Comparison, run against the base worktree.** Targets: `Tests/RAG_Search`, `Tests/RAG` and `Tests/Backup_Recovery/test_projection_generation.py`, with `-n 4`.
- Base: 346 failed, 1115 passed. Here: 347 failed, 1115 passed.
- The one extra failure is the new test, which fails without the marker for the same RecoveryRequired reason as every plain test in its file. Those plain tests fail on both trees.
- No other failure differs.

**Diagnostic inventory (reviewed before `--write`).** The legacy `db_path` debug statement in `close_all_pools` moved unchanged into `_retire_pool`. The call count is still 2, the text and arguments are the same, and the statement-level check shows no statement changed. Its path candidate stays `legacy_unreviewed`, as before.
<!-- SECTION:NOTES:END -->

