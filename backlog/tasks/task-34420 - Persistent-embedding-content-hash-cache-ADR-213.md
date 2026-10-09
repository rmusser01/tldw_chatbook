---
id: TASK-34420
title: Persistent embedding content-hash cache ADR-213
status: Done
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 09:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 / F6: no content-hash embedding cache exists - the wrapper cache check is dead code against a model-keyed dict so any re-index re-embeds everything and the skip check is N+1
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-213 written before code,embedding_cache table with migration and version bump,Unchanged-content re-ingest performs zero provider embed calls after restart,Batched skip check one query per batch,Hit-rate metric reports true values

  Deviation on "migration and version bump", recorded in ADR-223 §2: `RAG_Indexing_DB.py` has no schema-version chain (no `PRAGMA user_version`, no entry under `DB/migrations/` — that directory only serves DBs with version constants). The DB's own convention is idempotent `CREATE TABLE IF NOT EXISTS` in `_initialize_schema()` on every open, so the table was added there; the change is purely additive (old readers ignore the table, new readers create it on open) and a cache is never authoritative, so no migration/backfill exists to run. ADR numbered 223 (brief's 213 was a placeholder; 221/222 taken at authoring time).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 8 (T8)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR-223 (persistent embedding content-hash cache) + embedding_cache table in RAG_Indexing_DB (PK model_id+content_hash, created_at index, idempotent CREATE IF NOT EXISTS per that DB's no-version-chain convention — verified sound) + get_cached_embeddings/store_cached_embeddings API (chunked IN 500, one executemany INSERT OR REPLACE, transactional eviction by total-row cap, symmetric array('f') little-endian float32 codec with corrupt-guard). Wrapper: sha256 once per text, batch lookup before factory, misses-only embed guarded, order-preserving merge, store after; dead model-keyed _cache check deleted; embeddings_cache_hit_rate now counts real hits/misses (opt-in store attachment preserves no-store status quo and test hermeticity). Ingestion skip-check batched to one get_indexed_items_by_type read per item type with proven-identical semantics. Evidence: 100-chunk first index = 100 provider texts; reopened full re-index = 0 (pinned as regression test). 17/17 new hermetic tests; failure list byte-identical to baseline. Files: DB/RAG_Indexing_DB.py, RAG_Search/simplified/embeddings_wrapper.py, RAG_Search/ingestion_indexing.py, Tests/RAG/test_embedding_content_hash_cache.py. Report: .superpowers/sdd/task-8-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
