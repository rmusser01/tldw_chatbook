# RAG pipeline: chunking, embedding, retrieval, grounded answers

This document describes the RAG stack: the shared RAG service (embeddings + vector store + chunking + cache + search legs), hybrid retrieval and reranking, the Library's four-seam keyword search and grounded answer generation, incremental indexing off ingestion, and the failure/honesty model.

## Authoritative files

| File | Role |
| --- | --- |
| `RAG_Search/ingestion_indexing.py` | The only constructor of the shared service (`get_shared_rag_service`), the daemon `IngestionIndexer`, post-ingest hooks, `backfill_semantic_index()` |
| `RAG_Search/simplified/rag_service.py` | `RAGService` — `index_batch_optimized`, `search`, `_semantic_search`, `_keyword_search` (per-DB sub-legs), `_hybrid_search`, `_fuse_hybrid_results`, `_chunk_document` |
| `RAG_Search/simplified/enhanced_rag_service.py` / `_v2.py` | `EnhancedRAGServiceV2` — the runtime class; adds reranking + experiment tracking |
| `RAG_Search/simplified/rag_factory.py` | `create_rag_service(profile, config)` — must receive the full profile or reranking config is lost |
| `RAG_Search/simplified/config.py` | `RAGConfig` (embedding/vector-store/chunking/search sections); hybrid constants |
| `RAG_Search/simplified/vector_store.py` | `VectorStore` Protocol, `ChromaVectorStore`, `InMemoryVectorStore`, `create_vector_store` |
| `RAG_Search/simplified/embeddings_wrapper.py` + `Embeddings/Embeddings_Lib.py` | `EmbeddingsServiceWrapper` over `EmbeddingFactory` — exactly two providers: local HuggingFace transformers and OpenAI API |
| `RAG_Search/chunking_service.py` | `ChunkingService` — the only door from RAG into `Chunking/` |
| `DB/RAG_Indexing_DB.py` | `RAGIndexingDB` — incremental-indexing **state** DB (not the vector store) |
| `RAG_Search/reranker.py` | Reranker families incl. `CrossEncoderReranker` (the only local/offline strategy) |
| `RAG_Search/fusion.py` | RRF + interleaving used by the Library keyword merge |
| `Library/library_local_rag_search_service.py` | The Library Search/RAG canvas backend: four keyword seams + semantic leg |
| `Library/library_rag_answer_service.py` | `generate_library_rag_answer()` — grounded answer generation |

## Search flow (user query → cited answer)

1. **Library canvas**: the user picks Search vs RAG Answer mode and source scopes (notes / media / conversations / prompts).
2. **Search mode (keyword)**: the Library service runs four FTS seams (media DB FTS, ChaCha notes FTS, conversations FTS, prompts FTS). Each seam reports `SeamState` AVAILABLE/UNAVAILABLE/FAILED; failed seams are excluded **and disclosed** ("N seams failed — results exclude them") rather than silently returning zero. The seams merge via `fusion.interleave_rankings`.
3. **RAG Answer mode**: retrieval follows the active RAG profile's `default_search_mode` (plain → keyword path; semantic → vector; hybrid → both legs with scope pushed into both).
4. **Engine**: `EnhancedRAGServiceV2.search` → `RAGService.search`: scope allowlist frozen; incompatible scoping kwargs raise `ValueError` rather than being ignored; cache lookup keyed on query/type/top_k/filters/scope **and the resolved fusion params** (else different `rrf_k` values share stale entries).
5. **Semantic leg**: async query embedding, then vector search with citations (a multi-entry allowlist becomes one store query per entry, merged by score).
6. **Hybrid leg**: each leg fetches `top_k × hybrid_pool_multiplier`; the keyword leg's per-DB sub-legs use strict implicit-AND FTS MATCH with per-row prefix/OR fallback (row-stamped `fts_match` provenance); fusion is RRF (default alpha 0.7 vector, k=5).
7. **Rerank (optional)**: reranking can never fail a search — construction failure, exception, or degraded outcome tags the result `reranking_skipped`/`reranking_degraded` and returns the unreranked base. Outcome counts come from this call, not the shared reranker singleton.
8. **Answer**: the canvas renders the Answer region **above** the evidence rows. `generate_library_rag_answer` builds an evidence bundle; with no citable references it returns `no_evidence` **without any provider call**; otherwise exactly one `chat_api_call` (max_tokens 1200, 6000 for reasoning-typed models, temperature 0.2) under an honesty-contract system prompt; citations validate against the evidence grammar shared with Console cited answers; the whole body never raises into the calling worker.
9. **Unavailability is honest**: "RAG unavailable" surfaces a recovery state with the `pip install "tldw_chatbook[embeddings_rag]"` instruction; "Index empty" points at ingest/backfill.

## Indexing flow (document → chunk → embed → store)

1. Ingestion fires a registered **post-commit** hook (`add_media_with_keywords` and the notes/conversation equivalents); a cheap availability gate (find-spec probe + the `[…rag.indexing] enabled` kill switch) runs first; hook errors are swallowed — ingestion is never affected.
2. Entry builders produce documents `{id: "media_<id>", content, title, metadata}` for media, notes, and conversations.
3. The daemon `IngestionIndexer` drains a queue in batches on one reused event loop; every batch failure is contained.
4. Per item: skip when `RAGIndexingDB.needs_reindexing` is false (last-modified unchanged); best-effort `delete_document` first (Chroma `add` upserts-by-id-keeping-old); then `index_batch_optimized`; on success only the search cache clears (the embedding cache survives); the whole batch is marked indexed in one transaction.
5. Inside the service: chunking runs on a dedicated thread pool → `ChunkingService` → `Chunk_Lib.improved_chunking_process` → the vendored `Chunking/engine/` `Chunker` (legacy validation: size>0, 0≤overlap<size; flat `start_char/end_char/word_count/chunk_index` contract with rich metadata).
6. Embed: local transformers (cache dir `<user_data_dir>/models/embeddings`) or OpenAI API (model prefixed `openai/`); circuit breaker and memory metrics wrap it.
7. Store: Chroma collection names are config-fingerprinted (an embedding-model or config change opens a different collection); a legacy `default` collection is adopted.
8. Incremental state lives in `<user_data_dir>/rag_indexing.db` (`indexed_items` keyed `(item_id, item_type)` with `last_modified`; WAL, thread-local connections). Vectors live in ChromaDB; source content stays in the media/ChaCha DBs.
9. Backfill reconciles tracked-but-deleted media, then batches all sources, resumable via the same incremental skip.

## Rerankers

Families: cross-encoder (local, offline — loads with `local_files_only=True`; a missing model degrades to retrieval order rather than downloading mid-search; a chat-model name in the config falls back to a measured default), plus pointwise/pairwise/listwise (each spends one provider call per search). Construction is via `create_reranker_from_config`; the profile must carry the reranking section or rerankers silently disappear (see gotchas).

## Config keys

- `[rag.service] profile` — default `hybrid_basic`; profiles: bm25_only / vector_only / hybrid_basic / hybrid_enhanced / hybrid_full.
- `[AppRAGSearchConfig.rag.<section>]` — `vector_store` (type/persist_directory, `auto` resolves chroma-with-deps else in-memory), `indexing` (`enabled` kill switch), `retriever` (`hybrid_alpha`), search/cache keys.
- `[Embeddings]` — `embedding_provider` (default openai), `embedding_model` (default text-embedding-3-large; dataclass default for local profiles `mxbai-embed-large-v1`), `onnx_model_path`, `chunk_size`/`overlap` (defaults 400/100 words), api url/key.
- Dataclass defaults: top_k 10, inline citations on, cache TTL 3600 s, `hybrid_alpha` 0.7, `rrf_k` 5, pool multiplier 2.

## Boundaries

- `RAG_Search/simplified/` owns the engine; `RAG_Search/chunking_service.py` is the only door into `Chunking/`; `DB/RAG_Indexing_DB.py` owns incremental bookkeeping, not vectors.
- `ingestion_indexing.py` is the **only** constructor of the shared service (two-lock + generation design so a frozen UI can never hold the fast lock during a build).
- The Library service owns the four-seam keyword path and UI recovery states; answer generation deliberately lives outside it as a pure seam with an injected `chat` callable.
- `RAG_Search/search_service.py` exposes MCP-facing search that falls back to keyword FTS when the RAG runtime is unavailable (fallback rows carry `score: None`, never a fabricated 1.0).

## Governing decisions and docs

ADR-005 (invest in local RAG mirroring tldw server), ADR-013 (plain-text vs FTS match boundary), ADR-024 (citation provenance and source resolution), ADR-003/030 (library RAG defaults, derived-index lifecycle). User guide: `Docs/User_Guide/library/search-and-rag.md`. QA evidence: `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/` (first-query stall — model load), `2026-08-17-cross-encoder/`, `2026-08-18-*` (merge tiering, granularity, prompts seam).

## Verified gotchas

1. `create_rag_service` must receive the whole profile; a bare `RAGConfig` silently disabled every reranker.
2. "Answer-first" means two different things in the docs: the QA dir name refers to the first-query stall; the UI meaning is the Answer panel rendered above Evidence.
3. Chroma `add` keeps existing ids — re-index paths must delete first.
4. Hybrid cache keys must include the **resolved** fusion params.
5. `SeamState` enum members are all truthy — compare with `is`, never truthiness.
6. Cross-encoder is the only local/offline rerank strategy; the LLM-based ones cost a provider call per search.
