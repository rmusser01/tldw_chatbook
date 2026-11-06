---
id: TASK-38956
title: "Closing a discarded RAG service retires the FTS5 pools a surviving service is using"
status: To Do
created_date: 2026-11-05 21:35
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
- [ ] #1 Closing a RAGService that never searched (such as a discarded build) leaves another live service's pool open: that service's next search reuses the same connection with no retry and no rebuild; the test is red on dev
- [ ] #2 Closing the last user of a pool still retires that pool's connection on its own thread; test_keyword_search_leaves_no_executor_connection_after_close and the racing close tests stay green
- [ ] #3 Plain tests in Tests/RAG_Search/test_keyword_leg_db_resolution.py fail locally with RecoveryRequired (TASK-38111); the notes say how the new test was verified (the temporary private-profile marker workaround) and that the marker was removed
<!-- AC:END -->
