---
id: TASK-256
title: >-
  Align hybrid retrieval fusion with the tldw_server design (RRF plus alpha
  weighting)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:12'
labels:
  - rag
dependencies:
  - TASK-247
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_server fuses hybrid results with Reciprocal Rank Fusion (k=60) followed by an alpha-weighted blend of the FTS and vector RRF scores, with alpha=0.7 weighting the vector side (tldw_server2 database_retrievers.py:2044-2092). The chatbook hybrid pipeline is an ad-hoc weighted merge whose vector leg has always been empty in practice. Once indexing makes the vector leg real (task-247), align the fusion math and defaults with the server so hybrid result quality and behavior match the reference design. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hybrid mode fuses FTS and vector rankings via RRF with an alpha-weighted blend matching server defaults (k=60, alpha=0.7 vector-weighted)
- [x] #2 Fusion is covered by unit tests using known input rankings
- [x] #3 Alpha is exposed in config with a server-consistent default and documented semantics (0 = FTS only, 1 = vector only)
<!-- AC:END -->

## Implementation Plan

1. Add a shared pure fusion helper `RAG_Search/simplified/fusion.py`:
   - `reciprocal_rank_fusion(fts_results, vector_results, key=..., alpha=0.7, rrf_k=60, max_results=None)` implementing the exact server math (`database_retrievers.py:2047-2105`): per-leg RRF score `1/(k + rank)` with rank starting at 1 in each leg's returned order, docs missing from a leg contribute 0, final = `(1-alpha)*fts_rrf + alpha*vector_rrf`. Returns fused entries carrying both leg items, leg ranks/RRF contributions (provenance), and the fused score; deterministic tie-break.
   - `interleave_rankings(...)` to collapse the pipeline's multiple per-source FTS5 lists into one rank-fair FTS leg before fusion.
   - `resolve_hybrid_alpha(...)`: explicit value -> `[AppRAGSearchConfig.rag.retriever] hybrid_alpha` -> default 0.7.
2. Rewire `RAGService._hybrid_search` (`RAG_Search/simplified/rag_service.py`) to fuse the keyword (FTS) and semantic (vector) legs via the helper, replacing the ad-hoc citation-count-weighted / max-score merges. Citations still merge when a doc appears in both legs; fused RRF score replaces leg scores; leg provenance stored in result metadata. Both enhanced services inherit this via `super().search()`.
3. Rewire the `hybrid` builtin pipeline (`RAG_Search/pipeline_builder_simple.py`): replace the `weighted_merge` [0.25 x4] step with a new `rrf_merge` parallel-step merge that partitions legs by function (`*_fts5` lists interleaved = FTS leg; `search_semantic` = vector leg) and fuses via the same helper. `perform_hybrid_rag_search` (`chat_rag_events.py`) exposes `hybrid_alpha` (legacy `bm25_weight`/`vector_weight` mapped to alpha for back-compat).
4. TOML pipelines: fold standalone `type = "merge"` steps into the preceding parallel step at load time (they are silently skipped today), and ship `hybrid_v2` in `Config_Files/rag_pipelines.toml` with `rrf_merge`.
5. Config: single authoritative knob `[AppRAGSearchConfig.rag.retriever] hybrid_alpha` -> `RAGConfig.search.hybrid_alpha`; default migrated 0.5 -> 0.7 (server parity, deliberate) in `simplified/config.py`, `config.py` (DEFAULT_RAG_SEARCH_CONFIG + config template), `rag_config_example.toml`, Settings fallbacks (`settings_library_rag_defaults.py`, `Tools_Settings_Window.py`); semantics documented as 0 = FTS only, 1 = vector only.
6. Unit tests `Tests/RAG/simplified/test_fusion.py` with hand-computed rankings (alpha in {0, 0.5, 0.7, 1}, empty legs, ties, k, max_results, provenance), plus coverage of the `RAGService` merge path and the pipeline `rrf_merge` step; update existing tests asserting the 0.5 default.

## Implementation Notes

Server-parity RRF fusion is now the single hybrid merge everywhere, implemented once as a pure module and consumed by both hybrid sites.

**Shared fusion helper — `tldw_chatbook/RAG_Search/fusion.py`** (moved up from the planned `simplified/fusion.py`: the `simplified` package `__init__` eagerly imports the embeddings stack and can `ImportError` without the `embeddings_rag` extra, and `pipeline_builder_simple` must import fusion unconditionally). Contains:
- `reciprocal_rank_fusion(fts, vector, key=..., alpha=0.7, rrf_k=60, max_results=None)` — exact server math (`database_retrievers.py:2047-2105`): per-leg `1/(k + rank)` with 1-based ranks in each leg's returned order, missing leg contributes 0, `final = (1-alpha)*fts_rrf + alpha*vector_rrf`. Returns `FusedResult` entries carrying both leg items, per-leg ranks/RRF contributions (provenance), fused score; deterministic tie-break (score desc, then FTS rank, then vector rank). Two deliberate strictness improvements over the server: duplicates within a leg keep the best rank (server overwrites with the worse one), and ordering is fully deterministic (server sorts over a set).
- `interleave_rankings(...)` — rank-fair round-robin collapse of the pipeline's three FTS5 source lists (their raw scores are constant 1.0 / not comparable) into one FTS leg.
- `resolve_hybrid_alpha(explicit)` — explicit -> `[AppRAGSearchConfig.rag.retriever] hybrid_alpha` -> 0.7; invalid/out-of-range values warn and fall back.

**Fusion sites unified:**
1. `RAGService._hybrid_search` (`simplified/rag_service.py`) — replaced the citation-count-weighted / max-score merges (`_merge_results_with_citations`/`_merge_basic_results`, deleted) with `_fuse_hybrid_results` using the shared helper and `config.search.hybrid_alpha`. Citations still merge when a chunk appears in both legs; leg provenance stored in `metadata['hybrid_fusion']`. Enhanced services inherit via `super().search()`.
2. Builtin `hybrid` pipeline (`pipeline_builder_simple.py`) — parallel step's `weighted_merge` [0.25 x4] replaced with `rrf_merge` (FTS5 legs interleaved vs `search_semantic` leg); `perform_hybrid_rag_search` (`chat_rag_events.py`) resolves alpha (new `hybrid_alpha` param; legacy `bm25_weight`/`vector_weight` map to `vector/(bm25+vector)`) and now deep-copies the builtin config (the old shallow copy mutated `BUILTIN_PIPELINES` across calls).
3. TOML pipelines — standalone `type = "merge"` steps were silently skipped by `execute_pipeline` (no such step type); they now fold into the preceding parallel step at load time, and shipped `hybrid_v2` uses `rrf_merge`. Shipped/legacy pipeline params dropped `bm25_weight/vector_weight = 0.5` (would have pinned alpha to 0.5 past the knob); `technical_docs` and the custom example carry `hybrid_alpha = 0.4` preserving their old 0.6/0.4 intent. `weighted_merge` remains supported for user TOML.

**Drive-by fix required for AC#1 to be real on the pipeline path:** the parallel step called `search_semantic(app, query, sources, **config)` with the full pipeline param soup; the duplicated `top_k` raised `TypeError` inside the leg, `gather(return_exceptions=True)` swallowed it, and the hybrid pipeline's vector leg was ALWAYS silently empty (independent of task-247's indexing). The parallel step now passes `limit` plus a whitelist (`score_threshold`, `filter_metadata`). Regression-tested with the real `search_semantic` against a strict-signature stub service (no `**kwargs` fake). The equivalent bug in the `retrieve`-step path (pure `semantic` pipeline) is task-250's scope and was left untouched.

**Config knob story (deliberate default migration 0.5 -> 0.7):** one authoritative knob `[AppRAGSearchConfig.rag.retriever] hybrid_alpha` -> `RAGConfig.search.hybrid_alpha` (dataclass default now `DEFAULT_HYBRID_ALPHA` = 0.7). Aligned every other surface that advertised 0.5: `config.py` `DEFAULT_RAG_SEARCH_CONFIG` + `[rag.retriever]` template, `rag_config_example.toml`, `Tools_Settings_Window` fallback, `SettingsLibraryRagDefaults` dataclass default, and the `EXAMPLE_TOML_CONFIG` docstring (which had `hybrid_alpha` under `.rag.search`, a section the loader never reads — moved to `.rag.retriever`). `Docs/Development/RAG/RAG-Documentation.md` had the semantics inverted ("semantic (0.0) and keyword (1.0)") and the wrong section — fixed. Semantics documented everywhere as 0 = FTS only, 1 = vector only.

**Tests** (`Tests/RAG/test_fusion.py`, 36 tests): hand-computed A/B/C rankings for alpha in {0, 0.5, 0.7, 1} with exact scores, empty-leg/both-empty cases, deterministic ties, `rrf_k`, `max_results`, in-leg dupes, provenance, no-mutation, rank-not-score invariance; `RAGService._fuse_hybrid_results` (citation merge, cap, alpha=0); pipeline `rrf_merge` (hand-computed interleaved-leg scores, vector-leg failure degrades to FTS ordering, live-vector-leg regression, `perform_hybrid_rag_search` end-to-end smoke incl. builtin-non-mutation). Updated `Tests/UI/test_settings_library_rag_defaults.py` default assertion to 0.7.

**Results:** Tests/RAG + Tests/RAG_Search: 354 passed, 19 skipped. Tests/UI settings suites (library_rag_defaults + configuration_hub): 255 passed. `import tldw_chatbook.app` OK.

**Modified:** `RAG_Search/fusion.py` (new), `RAG_Search/simplified/rag_service.py`, `RAG_Search/simplified/config.py`, `RAG_Search/pipeline_builder_simple.py`, `Event_Handlers/Chat_Events/chat_rag_events.py`, `config.py`, `Config_Files/rag_pipelines.toml`, `Config_Files/pipeline_configs/custom_pipelines_example.toml`, `RAG_Search/rag_config_example.toml`, `UI/Screens/settings_library_rag_defaults.py`, `UI/Tools_Settings_Window.py`, `Docs/Development/RAG/RAG-Documentation.md`, `Tests/RAG/test_fusion.py` (new), `Tests/UI/test_settings_library_rag_defaults.py`.
