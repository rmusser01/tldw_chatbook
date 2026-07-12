# RAG Module Audit — 2026-07-12

**Scope:** Does the chatbook Search/RAG surface actually perform semantic (embeddings + vector) retrieval as designed in tldw_server's RAG module, or is it plain-text FTS5 search in practice?
**Verified against:** `dev` @ `6a9c624a`. Reference design: `tldw_server2` @ local checkout of the same date.
**Follow-up work:** backlog tasks **196–206** (each task carries its own evidence pointers).

---

## Verdict

The chatbook RAG module **copies the shape** of tldw_server's design — semantic/hybrid modes, an embeddings wrapper, vector-store classes, citations, reranking — but at runtime **every retrieval a user can trigger resolves to SQLite FTS5 keyword search**. The semantic machinery is fully coded and almost entirely dead, because:

1. **Nothing ever populates the vector store.** The only indexing entry point (`index_documents_modular` → `rag_service.embed_documents`, `Event_Handlers/Chat_Events/chat_rag_integration.py:300-325`) has **zero callers**. No app code calls `index_document` / `index_batch` on any RAG service. (task-197)
2. **The default vector store is in-memory** (`RAG_Search/simplified/config.py:48`, `type: str = "memory"`), and the `hybrid_basic` runtime profile doesn't override it — so even if something indexed, it would start empty every launch. (task-196)
3. **The RAG runtime itself is only initialized by one lazy path** — the chat sidebar (`chat_rag_events.py:339` is the sole caller of `get_or_initialize_rag_service`). Every other surface that offers a "semantic" mode either never initializes it or silently returns nothing when it's absent.

So "semantic search" today means: embed the query (if the `embeddings_rag` extra is even installed), query an empty in-memory store, return zero rows.

## Per-surface findings

### Library Search canvas (the current flagship surface)
Wired to `LibraryLocalRagSearchService` (`app.py:3128`, service in `Library/library_local_rag_search_service.py`).

- **`search` mode — real and working, but keyword-only.** Fans out FTS5 over the notes / media / conversations seams (with the task-185 plural/singular MATCH widening via `library_fts_query.py`). Runtime backend label: `local-fts`. This is honest FTS, not RAG.
- **`rag` mode ("RAG Answer") — never functional.** Delegates to `app._rag_service` (`library_local_rag_search_service.py:239`); the Library screen never initializes it, so users always get the "RAG unavailable" recovery state. Even after a chat semantic search creates the service, the index is empty (point 1 above), so it would return zero rows. (task-199)

### Chat sidebar RAG
`get_rag_context_for_chat` → pipelines in `RAG_Search/pipeline_builder_simple.py`:

- `plain` = pure FTS5 (`pipeline_functions_simple.py:19-173`).
- `semantic` = returns `[]` with only a log warning if `_rag_service` is unset (`pipeline_functions_simple.py:184-186`); with it set, queries the empty store.
- `hybrid` = FTS5 legs + a vector leg that contributes nothing.
- On a default install (no `embeddings_rag` extra), semantic falls back to plain (`chat_rag_events.py:341-342`).

### Standalone Search window (`TAB_SEARCH`, `UI/Views/RAGSearch/search_rag_window.py`)
Default mode is `plain` (`:171`). It never initializes `_rag_service`, so "contextual" silently returns nothing and "hybrid" degrades to FTS-only with no indication (task-200). Its **"Start Indexing" button (`:474`) has no event handler anywhere** — a dead control (task-201).

## Divergence from the tldw_server reference design

| tldw_server (reference) | chatbook (actual) |
|---|---|
| Single `unified_rag_pipeline` entry; `search_mode ∈ {fts, vector, hybrid}` | Three disconnected surfaces; only FTS legs ever return results |
| Hybrid = RRF (k=60) + alpha-weighted blend, alpha=0.7 vector-weighted (`database_retrievers.py:2044-2092`) | Ad-hoc weighted merge; vector leg empty in practice (task-206) |
| ChromaDB default store, per-user collections; indexing at ingestion via worker pipeline (chunk → embed → upsert) | In-memory store default; **no indexing path wired at all** (tasks 196, 197) |
| `all-MiniLM-L6-v2` @ 384 dims default; provider auto-resolution | Embeddings wrapper exists; only ever embeds queries against empty stores |
| Reranking ON by default (flashrank) | Reranker module is imported by `enhanced_rag_service_v2` but that service only runs on the dead-in-practice semantic path |
| Profiles fast / balanced / accuracy | Profiles exist (`config_profiles.py`) but can't change the empty-store reality |

## Second store, invisible to search

There are **two disconnected Chroma stacks**: `Embeddings/Chroma_Lib.py` (`ChromaDBManager`, used by RAG_Admin and the legacy embeddings-management UI) and `RAG_Search/simplified/vector_store.py` (`ChromaVectorStore`, the one RAG search actually queries). Different clients, different collection conventions — embeddings created via one path are invisible to the other. (task-198)

## Dead / unreachable surface (misleads reviewers)

- `RAG_Search/late_chunking_service.py`, `late_chunking_integration.py`, `context_assembler.py`, `query_expansion.py` — zero importers outside each other; the `enable_late_chunking` config keys and the query-expansion Settings handler (`app.py:7679`, a stub that stores a string) never reach these modules. (task-202)
- `UI/SearchWindow.py` — imported by nothing; it is the **only** mount point for `SearchEmbeddingsWindow`, `Embeddings_Management_Window`, the embeddings wizards, and the chunking-template widgets, so the entire manual-embeddings UI is unreachable. Plus `SearchRAGWindow.py.bak`. (task-203)
- `RAG_Admin` services — eagerly constructed on every launch (`app.py:2545-2559`) while all their UI consumers are unreachable. (task-204)
- `research` route — registered (`screen_registry.py:90`) with no navigation path to it. (task-205)

Not dead (verified): `reranker.py` and `parallel_processor.py` are imported by `enhanced_rag_service_v2.py:20-21`; `chunking_service.py` is live in media ingestion.

## Recommended sequencing

Foundations: **196** (persistent Chroma default) → **197** (ingestion-time indexing + backfill) unlock everything. Then **198** (single store), **199** (Library RAG-answer mode), **200** (no silent-empty semantic), **206** (server-parity RRF fusion). Cleanup in parallel: **201–205**.

The honest alternative — if local semantic RAG is *not* a goal — is to drop the pretense: ship FTS5 as "Search", remove the semantic modes and the ~20 dead files, and route "RAG Answer" through the tldw server API instead. That decision should be made explicitly before anyone picks up 196/197.

## Method note

Initial findings were traced on a stale checkout and then **re-verified line-by-line against `origin/dev` @ `6a9c624a`**; two claims changed in the interim (Library `search` mode got a real FTS backend via `LibraryLocalRagSearchService`; the reranker/parallel-processor modules gained a real importer) and are reflected above. If you re-audit, check assignment sites (`git grep "_rag_service ="`), indexing callers (`git grep "embed_documents\|index_document"`), and the store default (`simplified/config.py`) first — those three facts decide whether anything semantic is real.
