---
id: TASK-247
title: Index ingested content into the RAG vector store
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
  - ingest
dependencies:
  - TASK-246
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing in the app ever populates the RAG vector store: the only indexing entry point (index_documents_modular in Event_Handlers/Chat_Events/chat_rag_integration.py:300, which calls the non-existent `rag_service.embed_documents` method) has zero callers, and no app code invokes index_document/index_batch on any RAG service. Semantic and hybrid search therefore query an empty store and the vector leg always contributes nothing. Mirror the tldw_server design where chunking plus embedding plus storage happens at ingestion time via a background worker so semantic search has something to search. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Newly ingested media is chunked and embedded and indexed into the RAG vector store when embeddings deps are installed via a non-blocking background worker
- [x] #2 A semantic search for distinctive content of a newly ingested document returns that document
- [x] #3 A bulk backfill path exists to index pre-existing media/notes/conversations
- [x] #4 Indexing failures are logged and surfaced without breaking ingestion
- [x] #5 When embeddings deps are missing no indexing is attempted and ingestion is unaffected
- [x] #6 The broken/dead `index_documents_modular` entry point in `chat_rag_integration.py` is either removed or corrected to use the valid `RAGService` batch indexing API (e.g., `index_batch_optimized`)
<!-- AC:END -->

## Implementation Plan

1. **Ingestion seam — one post-commit hook at the DB layer.** All ingestion paths (Local_Ingestion processors, media_ingest_workers, tldw_api_events, subscription worker, reading service, chatbook import) converge on `MediaDatabase.add_media_with_keywords`. Rename its body to `_add_media_with_keywords_impl` and add a thin public wrapper that, after the transaction commits and a `media_id` is returned, dispatches module-level post-ingest callbacks (`register_media_post_ingest_callback`). Callback errors are swallowed+logged so ingestion is never broken (AC #4/#5). No per-processor edits.
2. **New module `RAG_Search/ingestion_indexing.py`:**
   - `semantic_indexing_available()`: gates on `optional_deps.embeddings_rag_deps_installed()` (cheap find_spec probe) plus an `[AppRAGSearchConfig.rag.indexing].enabled` config kill-switch (default on). When deps are missing nothing is attempted (AC #5).
   - `get_shared_rag_service(profile_name)`: process-wide singleton created via `create_rag_service` with the same profile resolution as the chat sidebar. **Decision: one shared RAGService instance for indexing and search** — `chat_rag_events.get_or_initialize_rag_service` is repointed at it. This guarantees indexer and searchers use the same collection/persist dir/embedding model, and avoids a second copy of the embedding model or a second Chroma client on one persist dir.
   - `IngestionIndexer`: lazily-started daemon consumer thread + `queue.Queue` (framework-free so it works from ingest worker threads and CLI backfill; satisfies the non-blocking background-worker AC without Textual worker exit_on_error hazards). Batches entries, consults `DB/RAG_Indexing_DB.RAGIndexingDB` (`needs_reindexing`/`mark_item_indexed`) for incremental skip, deletes stale chunks for updated docs, calls `index_batch_optimized`, records stats + last error, never lets an exception kill the thread (AC #4). `wait_until_idle()` for deterministic tests. Optional failure notifier for UI surfacing.
   - Document builders for media/notes/conversations with metadata contract `source_id`/`title`/`source_type` (+ `chunk_id` per chunk, see 4) so `library_local_rag_search_service._semantic_row` consumes them without rework (task-249).
   - `backfill_semantic_index(...)` (AC #3): batched, resumable walk of pre-existing media/notes/conversations using the same entry pipeline; plus a small `python -m tldw_chatbook.RAG_Search.backfill` CLI.
   - `install_media_ingest_hook()`: registers the DB callback; fetches the committed row on the ingesting thread (thread-local sqlite) and enqueues a self-contained payload.
3. **App wiring:** install the hook in `app.py::_init_media_db` (guarded), with a failure notifier that surfaces indexing errors via `notify` without touching the ingest path.
4. **Vector store:** add `delete_document(doc_id)` to `ChromaVectorStore` (where-filter delete) and `InMemoryVectorStore`, and inject `chunk_id` into per-chunk metadata in `rag_service.index_document`, `indexing_helpers.store_documents_batch`, `enhanced_indexing_helpers.store_documents_batch`. Without the delete, Chroma `add` silently keeps stale chunks for re-ingested/updated docs.
5. **AC #6:** delete `Event_Handlers/Chat_Events/chat_rag_integration.py` outright. Importer evidence: its only external importer is `Tools/rag_search_tool.py:89`, which imports `perform_modular_rag_search` purely as a dependency probe and never calls it (its `execute()` uses the real `RAGService.search` API directly); `index_documents_modular` and the other functions have zero callers. Drop that dead import; `rag_search_tool` stays functional.
6. **TDD:** failing tests first in `Tests/RAG/test_ingestion_indexing.py` — post-ingest hook contract; deps-missing gate (monkeypatched probe); document builders; indexer worker success/failure/skip/reindex with a fake service; end-to-end ingest→worker→semantic-search with the deterministic mock embedding backend (memory store + chromadb-gated persistent variant per `test_vector_store_selection.py` pattern); backfill full + incremental runs. Then run Tests/RAG/, Tests/DB/, and an `import tldw_chatbook.app` smoke.

## Implementation Notes

Ingestion-time semantic indexing is live: a post-commit hook on the media DB seam feeds a background worker that chunks -> embeds -> upserts into the RAG vector store, plus a bulk backfill path for pre-existing content.

- **Seam**: `MediaDatabase.add_media_with_keywords` was split into a thin public wrapper + `_add_media_with_keywords_impl`; the wrapper dispatches module-level post-ingest callbacks post-commit whenever a media_id is returned (creates/updates, not duplicate skips). One hook covers every ingest path (Local_Ingestion processors, ingest workers, tldw_api events, subscriptions, reading service, chatbook import) with zero per-processor edits. Callback errors are swallowed+logged.
- **New `RAG_Search/ingestion_indexing.py`**: availability gate (`embeddings_rag_deps_installed()` probe + `[AppRAGSearchConfig.rag.indexing].enabled` kill switch), `IngestionIndexer` (daemon thread + queue; batches, incremental skip via `DB/RAG_Indexing_DB`, stale-chunk delete before re-index, stats + last_error + failure notifier; worker never dies), document builders emitting the `source_id`/`title`/`source_type` (+ per-chunk `chunk_id`) metadata contract `_semantic_row` reads, and `backfill_semantic_index()` for media/notes/conversations (resumable/incremental). CLI: `python -m tldw_chatbook.RAG_Search.backfill`.
- **Shared service decision**: one process-wide RAG service (`get_shared_rag_service`), used by both the indexer and `chat_rag_events.get_or_initialize_rag_service` — indexing writes to exactly the collection/persist-dir/embedding model searches read, and no second embedding model or second Chroma client on one dir. `circuit_breaker._lock` switched asyncio.Lock -> threading.Lock since the shared breaker is now exercised from two event loops concurrently (its critical sections are await-free counter updates).
- **Vector stores**: added `delete_document(doc_id)` to Chroma + in-memory stores (Chroma `add` silently keeps existing IDs, so updated docs would otherwise retain stale chunks).
- **AC #6**: deleted `chat_rag_integration.py` outright. Its only external importer was `Tools/rag_search_tool.py:89`, importing `perform_modular_rag_search` purely as a dependency probe and never calling it; that import was removed and the tool keeps using the real `RAGService.search` API. Bonus fix: `EnhancedRAGServiceV2.index_batch_optimized`'s "parallel" branch imported two nonexistent helpers (ImportError for any profile with `processing_config`) and mismatched the storage helper's `doc_chunk_info` shape — it now delegates to the working base batch path (regression-tested).
- **App wiring**: `app.py::_init_media_db` installs the hook with a `call_from_thread(notify)` failure notifier.
- **Tests**: `Tests/RAG/test_ingestion_indexing.py` (27 tests) — hook contract, deps-missing/config gates, builders, worker success/skip/reindex/failure-survival/non-blocking, end-to-end ingest->worker->semantic-search with the deterministic mock embedding backend (memory + persistent-Chroma round-trip + V2 parallel-profile regression), backfill full+incremental. Verified end-to-end through the real `ingest_local_file` pipeline. Suites: Tests/RAG + Tests/RAG_Search 318 passed/19 skipped; Tests/Media_DB + Tests/DB 189 passed/6 skipped; Tests/Chatbooks 138 passed; Tests/Library 535 passed; `import tldw_chatbook.app` OK.
