# RAG Server-Port Programme — P0 Foundations (design)

Date: 2026-08-07
Status: approved-pending-user-review
Programme: port retrieval improvements from tldw_server2's RAG pipeline into tldw_chatbook
Phase: P0 of 5

## Background

A full mapping of both RAG stacks (tldw_server2 `app/core/{RAG,Chunking,Embeddings}`,
~85k LOC, vs tldw_chatbook `RAG_Search/` + Library/Console surfaces) established:

- Chatbook's **plumbing and honesty layers are stronger** than the server's
  (incremental ingestion indexing, profiles UI, named degradation states,
  abstention + citation validation).
- Chatbook's **live retrieval is weaker than what it already owns**: the Library
  `rag` mode issues a single unranked vector query, while the engine underneath
  (`RAG_Search/simplified/rag_service.py`) already implements hybrid RRF
  (k=60 + alpha, built for tldw_server parity under ADR-005) and keyword search,
  and `rag_factory` already wires profile-driven reranking through
  `EnhancedRAGServiceV2`. None of it is reachable from the live path.
- The native Console send path performs **no RAG injection at all** (TASK-406);
  the only wired injection path is the legacy, unreachable chat-events path.
- The server's own orchestrator (`unified_pipeline.py`, one 5,800-line function)
  must be treated as a **spec, never lifted** — its stage list informs later
  phases; no server code is copied wholesale in any phase.

## Programme skeleton (context — each later phase gets its own spec)

- **P0 Foundations** (this spec): make the retrieval chatbook already owns
  reachable, honestly.
- **P1 Eval harness**: golden query set over a fixture corpus; port the server's
  `retrieval_metrics.py` (P/R/MRR/NDCG), `regression.py` (JSON metric
  baselines), `quality_gating.py` (thresholds). Pure python. After P1, every
  retrieval change must move measured numbers.
- **P2 Retrieval intelligence**: query expansion + synonyms registry + rewrite
  cache, HyDE, PRF, clarification gate, granularity router, sibling/parent
  inclusion, semantic (similarity-keyed) cache; extend scope allowlists and the
  keyword leg to all four source seams. Each feature admitted only if it
  improves P1's baselines.
- **P3 Answer trust**: document grader → rewrite-retry loop, groundedness /
  utility graders (heuristic fallbacks), hard per-sentence citations, numeric
  fidelity, faithfulness scoring.
- **P4 Chunking**: auto-planner, `structure_aware` / `code_ast` / `propositions`
  strategies, template auto-classifier. Forces re-indexing, hence last.

Sequencing honesty: P0 flips default retrieval behavior (hybrid) **before** the
P1 harness exists. This is deliberate: ADR-005 committed to server parity, the
default profile (Hybrid Basic) already *promises* hybrid to the user in
Settings, and P0 is reachability-not-tuning. The spec states this rather than
hiding it.

## P0 scope

Two workstreams. TASK-657 (dep-gate never runs) was found already fixed
(2026-07-25) and is out of scope.

### Workstream A — profile-honoring retrieval on the live Library path

`Library/library_local_rag_search_service.py::_search_semantic` hardcodes
`search_type="semantic"`. Change it to resolve the **active RAG profile**:

- `default_search_mode` (profile vocabulary: `"plain"` / `"semantic"` /
  `"hybrid"`) maps semantic→`"semantic"` and hybrid→`"hybrid"` on
  `rag_service.search`. **Plain-profile routing**: `"plain"` does NOT use the
  engine's media-only keyword leg — a BM25 Only user in `rag` mode would get a
  strictly worse version of the Library's own `search` mode (one guessed-path
  seam vs four authorized, scope-aware seams). Instead, plain-profile `rag`
  queries route through the existing four-seam `_search_keyword` path,
  honestly labeled. The engine's keyword leg therefore runs **only as
  hybrid's FTS leg** in P0, never as a standalone mode. This also resolves
  the scoped+plain conflict for free (the seam path is already scope-aware,
  so a BM25 Only profile is never silently forced onto vectors).
- Reranking needs **no new plumbing** (`rag_factory.py:65` already sets
  `enable_reranking = profile.reranking_config is not None`; V2 applies it
  post-search). P0 verifies it and fixes the load-time hazard (below).

Constraints, each handled by disclosure rather than silence:

1. **Scope allowlists are semantic-only** — `RAGService.search` raises
   `ValueError` for hybrid/keyword with a non-empty `metadata_allowlist`.
   Scoped queries (conversation/workspace scope active) therefore stay on the
   semantic path in P0, disclosed via the existing coverage-note vocabulary.
   Extending allowlists to the FTS leg is P2.
2. **The keyword leg is replaced, not patched** (data-safety findings, verified
   in code):
   - Today it *guesses* DB paths, including opening `chacha_notes.db` (the
     ChaChaNotes DB) with media-schema SQL (`FROM Media m JOIN media_fts`),
     and on a total miss it **creates a new MediaDatabase as a side effect of
     a search**. Both behaviors are removed: the media DB path is injected
     explicitly from config; the create-on-miss branch is deleted; a failed
     leg is a disclosed degradation, never a write.
   - Keyword rows carry no `source_type` provenance, so the Library layer's
     canonicalizing post-filter would drop them silently (hybrid degenerating
     to semantic with zero error). P0 stamps `source_type` on keyword rows and
     pins "hybrid returns keyword-leg rows through the Library seam" with a
     RED-first test.
   - The leg covers **media only** in P0; provenance and the coverage note say
     so. Four-seam keyword coverage (notes/conversations/prompts) is P2.
   - When "media" is not among the selected source types, hybrid **skips the
     keyword leg** (it could only contribute rows the post-filter drops) and
     the coverage note discloses that the search ran semantic-only.
3. **Score-kind-aware presentation** (ship-breaking defect found in review):
   the Library match bands are calibrated for cosine similarity
   (strong ≥ 0.5, moderate ≥ 0.2 — `library_rag_state.py`), but RRF-fused
   scores max out at ~`1/(rrf_k+1) ≈ 0.016`, and cross-encoder reranker scores
   are unbounded logits. Without this item, every hybrid result bands
   "weak (0.02)" and `library_rag_all_matches_weak` fires on every search.
   Therefore:
   - `_fuse_hybrid_results` preserves the original per-leg scores in the
     `hybrid_fusion` metadata block (ranks + RRF contributions are already
     there).
   - Banding becomes score-kind-aware: band on the vector-leg similarity when
     present; FTS-only rows render as "keyword match" (never a fabricated
     similarity); reranked rows disclose their score kind instead of being
     banded on cosine thresholds.
   - `library_rag_all_matches_weak` only speaks over vector-similarity score
     kinds.
   - The backend label becomes mode-truthful (`rag-hybrid` / `rag-keyword` /
     `rag-semantic`).
4. **Reranker model load must not block the event loop**: if the reranker
   model loads lazily at first `.rerank()`, the first search under a reranking
   profile freezes the UI (task-641's scar class: a model download froze the
   app for 6+ minutes). The plan verifies load timing and moves model load to
   off-thread service construction if needed.
5. **Zero-results honesty under hybrid**: the "Index empty" recovery state
   (vector count == 0) must not fire when the keyword leg returned rows; the
   converse case discloses "semantic leg empty — keyword-only results".

### Workstream B — Console native send-path injection (TASK-406)

The native send path (`ConsoleChatController.submit_draft`) performs no RAG
injection. Design:

- An **"Auto-retrieve on send" toggle** in the existing Console RAG chip modal,
  default **OFF**, persisted as a global config key (the modal toggles it).
  This consciously re-homes TASK-406's assumed enablement from the legacy
  sidebar checkbox; the task's AC is edited accordingly before implementation
  (per backlog rules: update the AC first, then implement).
- Auto-retrieve fires only for **plain user text sends** — never for slash
  commands, tool approvals, or regenerations.
- When ON at send: the retrieval query is the **outgoing draft text,
  length-capped** (cap value decided in the plan); scope is
  resolved via `resolve_effective_scope_for_chat`; retrieval runs through the
  same profile-driven Library service as Workstream A; results route through
  the **existing staged-evidence pipeline** — strip shows "Evidence sent · N",
  consumption reuses PR-4's consume-on-send predicate. Never invisible
  injection.
- If evidence is already manually staged, auto-retrieve **skips** (no double
  retrieval/spend).
- Because Console retrieval is conversation-scoped, it inherits Workstream A's
  constraint 1: with an active scope allowlist it runs the semantic path, not
  hybrid, until P2 extends allowlists to the FTS leg.
- EMPTY scope short-circuits with the shared notice copy (task-406 AC #2).
- Explicit **5s timeout** with a visible "Retrieving…" strip state; on failure
  or timeout the send proceeds without evidence plus a quiet notice. The
  notice distinguishes "RAG service still initializing" (first-use model
  load can take minutes) from "retrieval failed", reusing the existing
  recovery-state vocabulary. A send is never blocked on retrieval. Retrieval
  runs in an **exclusive worker** so a double-send cannot double-retrieve.
- The Console chip's existing **manual run inherits the same profile-driven
  service**: its hardcoded `top_k=5` (chat_screen.py) is replaced by the
  active profile's `default_top_k`, so manual and auto retrieval cannot
  disagree about depth.
- Legacy path untouched (task-406 AC #3).

## Non-goals (declared, with follow-ups filed)

- At P0 ship time, MCP `perform_rag_search` and the agent-side `RAGSearchTool`
  did **not** honor profiles — Library and MCP briefly disagreed about what a
  "rag search" meant. A combined follow-up task was filed at ship time
  (adjacent to open TASK-694 / TASK-1077) rather than diverging silently.
  **Current disposition:** when direct Library tools are off,
  `LibraryRagToolProvider.search_library_rag` owns fallback agent RAG
  retrieval; when direct Library tools are on, `LibraryToolProvider` owns
  direct `library_search_notes`. TASK-3500 is narrowed to MCP
  `perform_rag_search` only, which is not yet aligned.
- No retrieval-quality tuning (alpha, rrf_k, thresholds, rerank models) — that
  is P1+P2 work, done against measured baselines.
- No new server-ported features in P0 (expansion, HyDE, PRF, etc. are P2).
- Four-seam keyword leg and hybrid-compatible scope allowlists are P2.

## Error handling / degradation

All existing named-degradation states are unchanged (deps missing, index
empty, scope empty). New rules:

- Keyword leg failing (DB missing/locked/query error) → fall back to the
  semantic leg **with a disclosed coverage note**, never silently narrower.
- Reranker unavailable (model/dep missing) → skip reranking; provenance says
  so; the search itself never fails because of reranking.
- Console auto-retrieve failure/timeout → quiet notice, send proceeds.

## Testing

- RED-first tests per behavior; mutation checks on the mode-mapping and the
  consume-on-send predicate (each mutation reddens exactly its own test).
- Pinned regression tests for: keyword rows surviving the Library post-filter;
  score-kind-aware banding (a fused 0.016 score must not band "weak"; an
  FTS-only row must not display a similarity); no-write-on-search (the deleted
  create-on-miss branch stays deleted); "Index empty" suppressed when keyword
  rows exist.
- Targeted suites with collection-count arithmetic (house rule: read the
  passed count; "no tests ran" is a failed gate).
- Live TUI walkthrough on a scratch profile with copied real DBs (the proven
  PR-2 recipe: scratch `TLDW_CONFIG_PATH`, `[first_run]` pre-set, real
  ChaChaNotes + media DBs copied in, config verified untouched after).
- P0 makes no retrieval-quality claims; behavioral verification only.

## Plan-phase verification items (carried into writing-plans)

1. Real media DB filename/path on a standard install vs the keyword leg's
   guess list — determine whether the leg has ever returned rows in the wild.
2. Reranker model load timing (construction vs first `.rerank()`), and whether
   `create_reranker` downloads models on the event loop.
3. Exact metadata shape needed for keyword rows to pass
   `_SEMANTIC_SOURCE_TYPE_MAP` canonicalization.
4. Where the Answer path (`build_library_rag_evidence_bundle` + coverage note)
   consumes scores, so fused/reranked score kinds don't leak into answer copy
   as "similarity".
5. `SimpleRAGCache` keys include search type — confirm no stale-cache hazard
   when the default mode changes.
6. Confirm no layer (engine, Library service, UI state, Answer bundle)
   applies similarity-scale thresholds (`score_threshold`, min-score
   filters) to fused or reranker score kinds.
