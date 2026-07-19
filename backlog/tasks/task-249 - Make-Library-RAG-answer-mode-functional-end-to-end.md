---
id: TASK-249
title: Make Library RAG-answer mode functional end-to-end
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - library
dependencies:
  - TASK-247
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library Search canvas is wired to LibraryLocalRagSearchService (app.py:3128): search mode runs real FTS5 over the notes/media/conversations seams and works today. But rag mode (RAG Answer) delegates to app._rag_service (library_local_rag_search_service.py:239), which is only ever created by the chat sidebar path (chat_rag_events.py get_or_initialize_rag_service, its sole caller is chat_rag_events.py:339). The Library screen never initializes it, so RAG Answer always returns the RAG-unavailable recovery state; and even when a prior chat semantic search created the service, the vector index is empty (see task-247) so it returns zero rows. Initialize the RAG runtime from the Library path when embeddings deps are present and return real semantic results. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting RAG Answer mode with embeddings deps installed initializes the RAG runtime instead of returning the RAG-unavailable recovery state
- [x] #2 With indexed content present a RAG Answer query returns semantic results with citations in the evidence rows
- [x] #3 Without embeddings deps the existing recovery state still routes the user to setup
- [x] #4 When the RAG runtime is available but the index is empty the outcome says so instead of showing a bare zero-results state
<!-- AC:END -->

## Implementation Plan

1. Failing tests first (Tests/Library/test_library_local_rag_search_service.py, conventions of the existing suite):
   - AC #1: rag mode on an app without `_rag_service` lazily resolves the runtime via `get_shared_rag_service()` (monkeypatched fake), caches it on `app._rag_service`, and returns semantic rows instead of the RAG-unavailable recovery state.
   - AC #3: with `embeddings_rag_deps_installed()` returning False, rag mode returns the existing `_rag_mode_unavailable_recovery_state` and never calls `get_shared_rag_service()` (update the existing no-service test to pin the deps gate explicitly).
   - AC #4: runtime available, search returns zero rows, `vector_store.get_collection_stats()` reports `count == 0` -> outcome status `empty` with distinct "index empty" recovery copy (stable selector `library-rag-empty-state`); `count > 0` with zero rows keeps the generic zero-results path; stats errors/missing stats fall back to the generic path.
   - Race: two concurrent rag queries against the real `get_shared_rag_service` with a slow counting `create_rag_service` fake -> exactly one construction, both queries share the instance.
   - AC #2 (integration, skipif no embeddings deps, mock embedding model per Tests/RAG/test_ingestion_indexing.py): index a real document via `RAGService.index_batch_optimized`, publish it via `set_shared_rag_service`, run a Library rag query end-to-end through `run_library_rag_search` -> evidence rows carry real citations (`LibraryRagResultRow.citations` labels) and metadata contract fields; empty real store variant proves the index-empty outcome against the real stats API.
2. Implement in `Library/library_local_rag_search_service.py`:
   - `_resolve_rag_runtime()`: return `app._rag_service` when usable; else gate on `embeddings_rag_deps_installed()` (cheap probe, before any heavy work); else construct via `await asyncio.to_thread(get_shared_rag_service)` (blocking model load off the UI event loop; the shared factory is process-lock idempotent) and cache on `app._rag_service`.
   - `_search_semantic()`: use the resolved runtime; on zero raw results probe `vector_store.get_collection_stats()` (in a thread) and return an honest index-empty `LibraryRagSearchOutcome(status="empty", recovery_state=...)` distinct from generic zero-results.
3. Open the Library UI run gate: `library_screen.py` `provider_ready` currently requires a pre-existing `app._rag_service`, which would make lazy init unreachable; make it deliberately-always-ready like `dependencies_ready`/`index_ready` (service double-guards and owns recovery copy). Update the two UI tests pinning the old gate (`test_library_shell.py` mode-toggle gate test, gate16 persistent-recovery test) to pin the new end-to-end behavior: rag mode Run stays enabled, and without deps the service-level "RAG unavailable" recovery renders under `#library-rag-service-error`.
4. Run: new tests, Tests/Library/, Tests/RAG/, the two updated UI suites, `python -c "import tldw_chatbook.app"`.

## Implementation Notes

Followed the plan; the "RAG Answer" gap closes at the retrieval-service seam, so both the Library Search canvas and the Console `run-library-rag` action (same `run_library_rag_search` funnel) inherit the fix.

- **Lazy runtime initialization** (`Library/library_local_rag_search_service.py`): new `_resolve_rag_runtime()` — an existing usable `app._rag_service` always wins; otherwise the `embeddings_rag_deps_installed()` probe short-circuits before any heavy work (AC #3 keeps the exact `_rag_mode_unavailable_recovery_state` copy); otherwise the runtime is created via `await asyncio.to_thread(get_shared_rag_service)` so the first-time embedding-model load never runs on the UI event loop, then cached on `app._rag_service` for every other RAG surface. Concurrency is owned by `get_shared_rag_service`'s double-checked process lock (task-247); a dedicated race test proves two concurrent Library rag queries construct exactly one service.
- **Honest empty-index outcome (AC #4)**: on zero raw semantic results, `_semantic_index_is_empty()` probes `vector_store.get_collection_stats()` in a thread; only a trustworthy `count == 0` (no error payload) returns the new `LibraryRagSearchOutcome(status="empty", recovery_state=_rag_index_empty_recovery_state())` ("Index empty" / next action: ingest content — auto-indexed — or run a semantic backfill, stable selector `library-rag-empty-state`). Unverifiable stats, populated stores, and scope-filtered-to-zero results keep the generic zero-results path.
- **UI run gate** (`UI/Screens/library_screen.py`): `provider_ready` was `app._rag_service is not None`, which made the lazy path unreachable from the Run button; it now joins `dependencies_ready`/`index_ready` as deliberately-always-ready — the retrieval service double-guards and owns recovery copy (a disabled button could not carry the "Install embeddings support" routing).
- **Citations verified end-to-end (AC #2)**: a with-deps integration test (mock embedding backend, per Tests/RAG conventions) indexes a real document through `RAGService.index_batch_optimized`, resolves the runtime lazily from an uninitialized app, and asserts the `SearchResultWithCitations.citations` objects survive `_semantic_row` → `LibraryRagResultRow` normalization into evidence-row citation labels/badges.
- **Tests**: 12 new/updated tests in `Tests/Library/test_library_local_rag_search_service.py` (61 pass); rewrote the two UI tests pinning the retired provider gate (`Tests/UI/test_library_shell.py`, `Tests/UI/test_product_maturity_gate16_library_search_rag.py`). Full runs: Tests/Library 547 passed; Tests/RAG 264 passed/8 skipped; gate16 16 passed; test_library_shell 256 passed + 1 unrelated flake (`note_conflict_during_preview`, passes in isolation); `import tldw_chatbook.app` OK.
- **Modified files**: `tldw_chatbook/Library/library_local_rag_search_service.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/Library/test_library_local_rag_search_service.py`, `Tests/UI/test_library_shell.py`, `Tests/UI/test_product_maturity_gate16_library_search_rag.py`.
