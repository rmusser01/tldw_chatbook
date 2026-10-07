---
id: TASK-34420
title: Persistent embedding content-hash cache ADR-213
status: Done
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 08:55
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
- **ADR**: `backlog/decisions/223-persistent-embedding-content-hash-cache.md`, written before any code. Key = `(model_id, sha256(full text))`; `model_id` is the wrapper's model name (the factory's `"default"` config slot is not a model identity).
- **DB** (`tldw_chatbook/DB/RAG_Indexing_DB.py`): `embedding_cache(model_id, content_hash, vector BLOB, created_at)` with PK `(model_id, content_hash)` and an index on `created_at` for the prune. Vectors are little-endian raw float32 bytes (stdlib `array`, byteswap-guarded; corrupt blobs decode to "miss", never crash indexing — this DB had no prior vector-blob convention). `get_cached_embeddings` dedups and reads chunked `IN` clauses of 500 (SQLite's default parameter ceiling is 999); `store_cached_embeddings` collapses in-batch duplicate hashes, one `executemany` of `INSERT OR REPLACE` in one transaction, and prunes the overflow oldest-first by `(created_at, rowid)` **in the same transaction** to `EMBEDDING_CACHE_MAX_ROWS = 200_000` (constant; per-construction override for tests). `clear_all()` now also wipes the cache (rebuildable state).
- **Wrapper** (`RAG_Search/simplified/embeddings_wrapper.py`): both `create_embeddings` and `create_embeddings_async` compute sha256 once per text, do one batched lookup, embed ONLY the misses through the unchanged circuit-breaker paths, merge in caller order, and store the misses. The dead md5-of-first-100-chars batch key (checked against the MODEL-keyed `EmbeddingFactory._cache`, so it could never hit) is deleted. `_cache_hits`/`_cache_misses` now count real per-TEXT store outcomes — `embeddings_cache_hit_rate` is true or honestly 0.0-over-zero-samples with no store attached; no synthetic hit/miss counters are emitted. Store is opt-in: `embedding_cache_db=` constructor kwarg or `set_embedding_cache_store()`; the wrapper never default-opens the user-data-dir DB (test hermeticity — RAGService/wrapper are constructed in many test contexts, ADR-223 §6).
- **Ingestion** (`RAG_Search/ingestion_indexing.py`): `index_entries` attaches the caller's `indexing_db` as the service wrapper's cache store (duck-typed, best-effort, re-attachable — that seam is the one place owning a RAGIndexingDB handle; worker and backfill both route through it, and query embeddings on the shared service hit the cache after the first ingest run). The per-entry `needs_reindexing` N+1 is replaced by ONE `get_indexed_items_by_type` read per distinct item type with identical semantics: unknown item → index; strictly newer `last_modified` → index; naive timestamps stamped UTC; any read/comparison failure → index that entry/type with a warning (fail open, as before).
- **Untouched by design**: `EmbeddingFactory._lock` (ADR-223 §8 — the cache removes most contending calls; lock redesign deferred) and every Chroma store path.
- **Tests** (`Tests/RAG/test_embedding_content_hash_cache.py`, 17 tests, hermetic: deterministic mock backend + tmp-file DBs, no network, nothing opens the default user-data DB): identical re-embed → zero factory work; changed text → exactly one embed; model isolation; DB-level roundtrip/600-hash chunked lookup/eviction under a tiny cap; persistence across close+reopen (8-text and the task's 100-chunk evidence scale); real hit-rate metric values; spy proving one `get_indexed_items_by_type` per batch with `needs_reindexing`/`get_indexed_item_info` never called; skip-semantics equivalence (older/equal → skip, newer → re-index with stale-chunk delete, unknown → index); mixed types → one read per type; full seam test — re-index after "restart" with moved mtime → 3 indexed, 0 provider texts.
- **Evidence**: standalone fake-factory run over a 100-chunk fixture: first index = 100 provider texts embedded; reopened DB, full re-index = 0; vectors identical (`np.allclose`), shape (100, 384).
- **Verification**: `pytest Tests/RAG Tests/RAG_Search Tests/DB/test_rag_indexing_db.py -k "embedding or indexing or cache"` — failure set byte-identical to the pre-change baseline (39 failed + 1 error, all pre-existing), passed 224 → 241 (the 17 new). `Tests/DB/test_rag_indexing_db.py` 15/15 green after the `clear_all` extension. `Tests/Library/test_library_rechunk_service.py` failures verified byte-identical to a scratch HEAD worktree (pre-existing). `ruff check --select E9,F,E1` clean on all touched files.
- **Files modified**: `tldw_chatbook/DB/RAG_Indexing_DB.py`, `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py`, `tldw_chatbook/RAG_Search/ingestion_indexing.py`, `tldw_chatbook/Library/library_rechunk_service.py` (docstring-only accuracy fix for the batched skip gate), `Tests/RAG/test_embedding_content_hash_cache.py` (new), `backlog/decisions/223-persistent-embedding-content-hash-cache.md` (new).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
