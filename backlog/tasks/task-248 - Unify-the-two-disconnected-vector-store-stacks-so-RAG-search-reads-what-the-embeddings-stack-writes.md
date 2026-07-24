---
id: TASK-248
title: >-
  Unify the two disconnected vector-store stacks so RAG search reads what the
  embeddings stack writes
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
dependencies:
  - TASK-246
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The codebase has two parallel Chroma stacks: Embeddings/Chroma_Lib.py (ChromaDBManager, used by the embeddings management surfaces and RAG_Admin local service) and RAG_Search/simplified/vector_store.py (ChromaVectorStore, which RAGService._semantic_search actually queries). They use different clients and collection conventions, so embeddings created via one path are invisible to RAG search. Converge on a single store and collection layout so every embedding created in the app is searchable. tldw_server uses one ChromaDB with per-user collections as the reference model. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Embeddings created through any app path are queryable by RAG semantic search
- [x] #2 Only one Chroma client / collection convention remains (the other stack is removed or delegates to the survivor)
- [x] #3 A migration or documented reset path exists for pre-existing collections
- [x] #4 Tests cover write-via-one-path read-via-search
<!-- AC:END -->

## Implementation Plan

Re-scoped against current dev (post PR #667 shared RAG service, post PR #669 SearchWindow/embeddings-UI removal):

1. Inventory every `Chroma_Lib` / `ChromaDBManager` consumer with git grep and classify live vs orphaned.
   Findings: (a) `RAG_Admin/local_rag_admin_service.py` — lazy `_build_chroma_manager`, its collection surface
   is reachable only through `RAGAdminScopeService`, which app.py constructs but nothing reads anymore;
   (b) `Event_Handlers/embeddings_events.py` — `register_embeddings_events` has zero callers, and its only
   importer (`event_dispatcher.py`) is itself never instantiated and references a nonexistent
   `EMBEDDINGS_BUTTON_HANDLERS` attribute; (c) tests referencing the legacy stack.
2. Repoint `LocalRAGAdminService`'s collection surface (list/detail/export/delete) at the survivor store:
   the shared RAG service's `ChromaVectorStore` (`RAG_Search/ingestion_indexing.get_shared_rag_service()`),
   i.e. the exact store RAG semantic search reads. Delete the dead in-file helpers
   (`_get_media_for_embedding`, `_word_chunks_for_reprocess`, `_embedding_ids_for_media`,
   `_record_local_media_job` — zero callers, two latent bugs). Simplify the app.py construction only as far
   as the removed parameters require (lazy construction stays task-254).
3. Delete `Event_Handlers/embeddings_events.py` (orphaned) and drop its import/spread from
   `event_dispatcher.py`.
4. Delete `Embeddings/Chroma_Lib.py` — zero importers remain after 2–3 (AC #2 by removal).
   `Embeddings/Embeddings_Lib.py` (EmbeddingFactory) stays: live dependency of the RAG embeddings wrapper.
5. Tests: rework `Tests/RAG_Admin/test_local_rag_admin_service.py` to the vector-store seam; add a
   chromadb-gated integration test proving write-via-RAG-service → visible to both semantic search and the
   admin surface (AC #1/#4, complementing the existing
   `Tests/RAG/test_ingestion_indexing.py::TestEndToEndSemanticSearch` e2e); delete
   `test_chroma_lib_graceful_failure`; remove the never-requested `mock_chroma_manager` fixture.
6. AC #3: document the orphaned legacy persist dir (`<USER_DB_BASE_DIR>/<user>/chroma_storage`) and the
   reset path (delete dir, re-index via `python -m tldw_chatbook.RAG_Search.backfill`) in
   `backlog/docs/` + Implementation Notes; no automated migration (embeddings are model-dependent and
   regenerable).

## Implementation Notes

**Re-scoping against current dev.** The task was filed when the ChromaDBManager stack still had a UI.
Since then PR #667 (task-247) routed every RAG write/read through the single shared service
(`RAG_Search/ingestion_indexing.get_shared_rag_service()`), and PR #669 (task-253) deleted the entire
legacy embeddings-management UI — the main ChromaDBManager consumers. So AC #1 was mostly satisfied by
deletion of the divergent *write* paths; this task's remaining job was to remove/repoint the leftover
ChromaDBManager consumers and delete the legacy stack itself.

**Consumer inventory (git grep evidence):**
- `RAG_Admin/local_rag_admin_service.py` — the one live-constructed consumer (eager in `app.py`), lazy
  `_build_chroma_manager()`. Its collection surface is reachable only via `RAGAdminScopeService`, which
  nothing outside `app.py` reads anymore. **Repointed** (not stripped): the collection surface now
  resolves the shared RAG service's vector store, so the local/server admin seam survives for the future
  Console-parity admin UI and — key unification property — administers the *same* collections RAG search
  queries. Chose "delegates to the survivor" over stripping because stripping would leave
  `RAGAdminScopeService`'s local collection routes raising `AttributeError`.
- `Event_Handlers/embeddings_events.py` — **orphaned, deleted.** `register_embeddings_events` had zero
  callers; its only importer (`event_dispatcher.py`) is never instantiated by app code and referenced a
  nonexistent `EMBEDDINGS_BUTTON_HANDLERS` attribute (would have been an `AttributeError` if ever run).
- `Embeddings/Chroma_Lib.py` — zero importers after the above → **deleted** (AC #2 by removal; the sole
  remaining Chroma client/collection convention is `RAG_Search/simplified/vector_store.ChromaVectorStore`).
  `Embeddings/Embeddings_Lib.py` (EmbeddingFactory) intentionally untouched — live dependency of the RAG
  embeddings wrapper.
- Also removed four zero-caller helpers inside `local_rag_admin_service.py`
  (`_get_media_for_embedding`, `_word_chunks_for_reprocess`, `_embedding_ids_for_media`,
  `_record_local_media_job`) — two carried latent bugs (`re` never imported; `_local_media_jobs` never
  initialized), proof they never ran. `app.py` construction simplified only by dropping the now-unused
  `app_config` kwarg; lazy construction remains task-254.

**AC #3 (reset path, no migration):** documented in
`backlog/docs/rag-legacy-chroma-storage-reset-2026-07.md`. Legacy data lives at
`<USER_DB_BASE_DIR>/<user_id>/chroma_storage` (default `~/.local/share/tldw_cli/<user_id>/chroma_storage`)
and is orphaned — delete it and rebuild via `python -m tldw_chatbook.RAG_Search.backfill`. No automated
migration: embeddings are model/dimension-dependent derived artifacts, fully regenerable from SQLite.

**AC #4 tests:** the write-via-ingestion → read-via-search e2e already exists
(`Tests/RAG/test_ingestion_indexing.py::TestEndToEndSemanticSearch`, incl. persistent-Chroma round trip).
Added `Tests/RAG_Admin/test_local_rag_admin_service.py::test_write_via_rag_service_is_visible_to_search_and_admin`
(index via `RAGService` → found by semantic search AND by the admin collection surface over the same
store) plus seam tests for shared-store resolution, name-only `list_collections` coercion (chromadb
version drift), and loud unavailability errors. Deleted `test_chroma_lib_graceful_failure` and the
never-requested `mock_chroma_manager` fixture (tested only the deleted stack).

**Files:** deleted `tldw_chatbook/Embeddings/Chroma_Lib.py`,
`tldw_chatbook/Event_Handlers/embeddings_events.py`; modified
`tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/Event_Handlers/event_dispatcher.py`,
`tldw_chatbook/app.py`, `Tests/RAG_Admin/test_local_rag_admin_service.py`,
`Tests/Utils/test_optional_deps.py`, `Tests/UI/Embeddings/test_base.py` (+README),
`Tests/RAG_Search/conftest.py`, `Tests/Performance/test_app_import_weight.py` (docstring); added
`backlog/docs/rag-legacy-chroma-storage-reset-2026-07.md`.

**Verification (local, CI intentionally cancelled):** `import tldw_chatbook.app` OK;
Tests/RAG_Admin 28 passed; Tests/RAG + Tests/Utils/test_optional_deps.py + Tests/UI/Embeddings +
Tests/RuntimePolicy + Tests/Character_Chat/test_dead_attach_removed.py: 593 passed / 8 skipped;
Tests/RAG_Search 54 passed / 11 skipped; Tests/Performance/test_app_import_weight.py 6 passed;
Tests/Event_Handlers 96 passed / 26 skipped.
