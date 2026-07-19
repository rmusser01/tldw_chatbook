---
id: TASK-250
title: >-
  Semantic and hybrid pipeline search must not silently return empty when the
  RAG service is uninitialized
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_semantic returns an empty list whenever app._rag_service is unset (RAG_Search/pipeline_functions_simple.py:184-186, logged only as a warning). The standalone SearchRAGWindow never initializes the service, so its contextual mode always returns nothing and hybrid quietly degrades to FTS-only; the chat sidebar initializes it lazily but other callers do not. Users cannot distinguish no-matches from not-initialized from missing-deps. The Library canvas already surfaces honest recovery states for this (library_local_rag_search_service.py); the chat and standalone Search surfaces should reach the same bar. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting semantic or contextual or hybrid mode in chat or standalone Search either initializes the RAG service or visibly reports why semantic retrieval is unavailable (missing deps or uninitialized runtime or empty index)
- [x] #2 Hybrid results clearly indicate when they are FTS-only because the vector leg was unavailable
- [x] #3 No code path remains where a user-triggered semantic search silently returns an empty result due to an uninitialized service
<!-- AC:END -->

## Implementation Plan

1. New small shared module `RAG_Search/semantic_availability.py`: reason codes + user-facing copy (deps missing / init failed / search error / index empty, wording consistent with the Library recovery states), `resolve_semantic_rag_service(app)` (existing `app._rag_service` wins -> `embeddings_rag_deps_installed()` cheap gate -> `get_shared_rag_service` in `asyncio.to_thread`, cache on app; mirrors Library `_resolve_rag_runtime` without touching it), and `semantic_index_is_empty(rag_service)` (trustworthy-count `get_collection_stats` probe, same semantics as Library's). Library behavior/files unchanged.
2. Root fix in `pipeline_functions_simple.search_semantic`: lazily initialize via the resolver instead of bailing on a missing `app._rag_service`; grow an optional `diagnostics` dict param and record WHY (`unavailable`+reason, `empty_index`, or `ok`) instead of only logging. `PipelineContext` gains a `diagnostics` key; `execute_pipeline` seeds it and accepts a caller-provided dict so the reason rides out without changing the `(results, context)` return shape.
3. Fix the retrieve-step params-splat cousin bug in `pipeline_builder_simple._execute_retrieve_step` exactly the way task-256/PR #681 fixed the parallel step (limit=top_k + `score_threshold`/`filter_metadata` whitelist); same fix for the registry `parallel_search` helper; the parallel step also records a `search_error` semantic diagnostic when the vector leg raises (today gather() swallows it).
4. Chat sidebar (`chat_rag_events.py`): `get_or_initialize_rag_service` delegates to the resolver (off the event loop); `get_rag_context_for_chat` notifies the user with the honest reason when semantic mode falls back to plain, and after hybrid/custom searches surfaces "keyword-only (FTS)" / "index empty" notifications from the pipeline diagnostics. `perform_plain/full/hybrid_rag_search` + `perform_search_with_pipeline` forward the optional `diagnostics` dict.
5. Standalone Search window: repair `_perform_plain/contextual/hybrid_search` (they call the pipeline entry points with a long-dead signature and drop the results tuple), pass sources from the filter checkboxes, thread the diagnostics dict, and surface the semantic-leg state: warning notification + "keyword-only" marker on the results count for hybrid, a recovery Static (instead of the generic empty state) for contextual when semantic is unavailable/index-empty; fix the missing `Static` import on the error path in `search_event_handlers.py`.
6. TDD: new `Tests/RAG/test_semantic_honest_states.py` (failing first) covering: uninitialized->init happens (mock `get_shared_rag_service`), deps-missing->honest marker + zero heavy work, init-failure->marker not crash, empty-index->distinct marker only on a trustworthy zero, hybrid FTS-only diagnostics + chat notification, retrieve-step splat regression with a strict-signature stub service (mirror of #681's parallel-step regression test); standalone-window seam tests in `Tests/UI/test_search_rag_window.py`.
7. Verify: new tests, Tests/RAG (incl. test_fusion.py, test_ingestion_indexing.py), Tests/RAG_Search, Tests/Library, Tests/Event_Handlers, `python -c "import tldw_chatbook.app"`.

## Implementation Notes

Semantic/contextual/hybrid searches now initialize the shared RAG runtime lazily and, when they cannot, say WHY -- everywhere a user can trigger them.

**Shared availability module -- `RAG_Search/semantic_availability.py` (new):** single home for the reason codes (`deps_missing` / `init_failed` / `search_error` / empty-index), the user-facing copy (kept consistent with the Library task-249 recovery wording: "Install embeddings support...", "The semantic index has no content yet. Ingest content... or run a semantic index backfill"), `resolve_semantic_rag_service(app)` (existing `app._rag_service` wins -> cheap `embeddings_rag_deps_installed()` gate BEFORE any heavy work -> `get_shared_rag_service` in `asyncio.to_thread`, cached on the app, requires a callable `search`), and `semantic_index_is_empty()` (trustworthy-count `get_collection_stats` probe: only an error-free integer 0 counts). Library's own seams are byte-identical/untouched.

**Root fix -- `pipeline_functions_simple.search_semantic`:** no longer bails with only a log warning when `app._rag_service` is unset; it resolves/initializes through the shared factory and records the semantic-leg state (`ok` / `unavailable`+reason / `empty_index`) into an optional `diagnostics` dict. `PipelineContext` gained a `diagnostics` key; `execute_pipeline` seeds it and accepts a caller-provided dict, so the WHY rides out without changing the `(results, context)` return shape. All `perform_*` entry points in `chat_rag_events.py` forward `diagnostics`.

**Retrieve-step params-splat cousin bug (flagged in task-256's notes):** `_execute_retrieve_step` splatted the full pipeline params into `search_semantic`, duplicating `top_k`/`include_citations` inside the RAG service call -> TypeError -> the pure `semantic` pipeline could never return vector results. Fixed exactly like #681's parallel-step fix (`limit=top_k` + `score_threshold`/`filter_metadata` whitelist), same for the registry `parallel_search` helper; regression-tested with the REAL `search_semantic` against a strict-signature (no-`**kwargs`) stub service. The parallel step also records `search_error` when the vector leg raises mid-search (gather() used to swallow it silently).

**Chat sidebar (`chat_rag_events.py`):** semantic mode with an unavailable runtime still falls back to plain, but now notifies the user with the reason + "Using keyword (FTS) search instead."; after any search, `_notify_semantic_leg_state` surfaces "Hybrid RAG context is keyword-only (FTS): <reason>" or the empty-index copy from the diagnostics. `get_or_initialize_rag_service` now delegates to the shared resolver, which also moves first-time construction (embedding-model load, seconds) off the UI event loop -- it previously ran the factory synchronously on the loop.

**Standalone Search window:** `_perform_plain/contextual/hybrid_search` called the pipeline entry points with a long-dead signature (`api_choice`/`filters`/`collection_name`, no app/sources) and fed the `(results, context)` tuple into a list-only formatter -- every mode was broken (TypeError caught as "Search failed"), and the error path itself crashed on an unimported `Static` (worker exit_on_error). Repaired to the real contract (sources from the filter checkboxes), threaded the diagnostics dict, fixed the `Static` import, and surfaced the state: warning notification, a "keyword-only (semantic unavailable/index empty)" marker on the results count for hybrid, and a recovery Static replacing the generic no-results empty state for contextual with zero results.

**Tests:** new `Tests/RAG/test_semantic_honest_states.py` (28 tests: resolver order incl. deps-gate-before-factory, trustworthy-count probe truth table, lazy-init-happens, deps-missing/init-failure/empty-index markers, legacy no-diagnostics call, retrieve-step + `parallel_search` splat regressions with strict stub, parallel-step `search_error` recording, hybrid FTS-only end-to-end, chat-sidebar notification tests for fallback/hybrid/empty-index); `Tests/UI/test_search_rag_window.py` +4 seam tests (sources mapping, hybrid/contextual pipeline contract + diagnostics capture, notice copy per state). `Tests/RAG/test_ingestion_indexing.py`'s `FakeRAGService` gained a trivial `search` seam because the resolver validates search-capability (identity assertion unchanged).

**Results:** Tests/RAG + Tests/RAG_Search 402 passed / 19 skipped; Tests/Library 548 passed; Tests/Event_Handlers 96 passed / 26 skipped; Tests/UI/test_search_rag_window.py 23 passed; adjacent UI suites (search handoffs, ux-audit smoke, phase1/phase5, navigation, focus, tooltips, chat-first handoffs) green except `test_unified_shell_phase5_recovery_taxonomy.py::test_service_backed_policy_destinations_use_async_workers_without_asyncio_run`, pre-existing on origin/dev (no diff in the flagged file). `import tldw_chatbook.app` OK.

**Observed, not fixed (out of AC scope):** `search_notes_fts5` still had the three-dot relative import (`from ...Notes.DB...`) that task-260 fixed for the conversations seam -- since fixed upstream by task-295 (landed on dev while this PR was in review; rebase picked it up).

**PR #692 review round (qodo, all four findings addressed):** (1) Google-style Args/Returns added to `record_semantic_empty_index` / `record_semantic_ok` / `semantic_index_is_empty`. (2) The chat notification wording now keys off the produced results instead of the mode string -- custom pipeline IDs ride through `search_mode` verbatim, so a hybrid-like custom pipeline with FTS results + unavailable semantic leg reads "RAG context is keyword-only (FTS): ..." (and hybrid with zero results correctly reads "returned no context"); covered by new custom-pipeline-id and hybrid-zero-results tests. (3) The empty-index probe now requires a genuine integer 0 (`isinstance(count, int) and not isinstance(count, bool)`) -- `0.0`/`False`/`"0"` are rejected, matching the stated contract; Library's own probe keeps its `int(...)` coercion (deliberately untouched, noted on the thread). (4) `search_semantic` now catches a raising `rag_service.search`, records `search_error`, and returns [] so the direct semantic/retrieve path renders the curated honest state instead of raw exception text; direct-path + end-to-end pipeline tests added. New-suite count: 35 tests.

**Modified:** `RAG_Search/semantic_availability.py` (new), `RAG_Search/pipeline_functions_simple.py`, `RAG_Search/pipeline_builder_simple.py`, `RAG_Search/pipeline_types.py`, `Event_Handlers/Chat_Events/chat_rag_events.py`, `UI/Views/RAGSearch/search_rag_window.py`, `UI/Views/RAGSearch/search_event_handlers.py`, `Tests/RAG/test_semantic_honest_states.py` (new), `Tests/UI/test_search_rag_window.py`, `Tests/RAG/test_ingestion_indexing.py`.
