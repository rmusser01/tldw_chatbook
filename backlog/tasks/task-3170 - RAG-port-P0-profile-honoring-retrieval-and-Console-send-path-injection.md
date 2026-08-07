---
id: TASK-3170
title: RAG-port P0 profile-honoring retrieval and Console send-path injection
status: In Progress
assignee: []
created_date: '2026-08-07 14:19'
labels:
  - rag
  - console
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P0 of the RAG server-port programme. Chatbook's live retrieval is weaker than the profile-driven hybrid/reranking engine it already owns (RAG_Search/simplified/rag_service.py, rag_factory.py): Library/library_local_rag_search_service.py's rag mode hardcodes semantic-only search, and the native Console send path (ConsoleChatController.submit_draft) performs no RAG context injection at all -- the only wired injection path (get_rag_context_for_chat) is the legacy chat_events route, which is unreachable in the live app (see task-406). This task makes the retrieval tldw_chatbook already owns reachable, honestly: the Library path resolves the active RAG profile's search mode with data-safety and score-kind-honesty fixes to the underlying engine, and Console's native send path gains opt-in, visibly-staged retrieval injection. Full design: Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library rag-mode search resolves the active RAG profile's default_search_mode (plain/semantic/hybrid) instead of hardcoding semantic search; plain-profile queries route through the existing four-seam _search_keyword path (honestly labeled), never through the engine's standalone keyword leg.
- [ ] #2 The engine's keyword leg no longer guesses DB paths or opens chacha_notes.db with media-schema SQL: the media DB path is injected explicitly from config, the create-a-new-MediaDatabase-on-miss side effect is deleted, keyword rows carry source_type provenance so they survive the Library's canonicalizing post-filter, and a failed leg degrades to a disclosed coverage note rather than performing a write.
- [ ] #3 Match-strength presentation is score-kind-aware: _fuse_hybrid_results preserves per-leg scores in hybrid_fusion metadata, RRF-fused and reranker scores are never banded on the cosine strong/moderate/weak thresholds, FTS-only rows render as 'keyword match' rather than a fabricated similarity, library_rag_all_matches_weak fires only over vector-similarity score kinds, and the backend label is mode-truthful (rag-hybrid/rag-keyword/rag-semantic).
- [ ] #4 The reranker factory is fixed so a reranking-enabled profile actually constructs a reranker; model loading is verified off the event loop (moved to service construction if it loads lazily at first .rerank()); an unavailable reranker (missing model/dep) skips reranking with disclosed provenance instead of failing the search.
- [ ] #5 Zero-results honesty under hybrid: the 'Index empty' recovery state does not fire when the keyword leg returned rows, and when the semantic leg is empty but the keyword leg is not, the search discloses 'semantic leg empty -- keyword-only results' instead of the generic empty state.
- [ ] #6 An 'Auto-retrieve on send' toggle in the Console RAG chip modal controls native send-path injection, defaults to OFF, and persists as a global config key, re-homing enablement away from the legacy sidebar checkbox.
- [ ] #7 Auto-retrieve fires only for plain user text sends -- never for slash commands, tool approvals, or regenerations -- and skips automatically when evidence is already manually staged (no double retrieval/spend).
- [ ] #8 When auto-retrieve fires, retrieved results route through the existing staged-evidence pipeline (strip shows 'Evidence sent · N', consumption reuses the consume-on-send predicate) rather than invisible prompt injection; the Console chip's manual run is switched from its hardcoded top_k=5 to the active profile's default_top_k so manual and auto retrieval never disagree about depth.
- [ ] #9 An EMPTY resolved scope short-circuits auto-retrieve with the same shared notice copy as the manual path (task-406 AC #2).
- [ ] #10 Auto-retrieve runs in an exclusive worker with a visible 'Retrieving...' strip state and an explicit 5s timeout; on failure or timeout the send proceeds without evidence and shows a quiet notice that distinguishes 'RAG service still initializing' from 'retrieval failed', and a send is never blocked on retrieval.
- [ ] #11 The legacy chat_events RAG injection path is untouched by this work (task-406 AC #3).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-08-07-rag-port-p0-foundations.md (spec: Docs/superpowers/specs/2026-08-07-rag-port-p0-foundations-design.md).
1. Backlog filing + task-406 AC edit (this task).
2. Fusion preserves original leg scores.
3. Replace the keyword leg's DB resolution (no guessing, no writes).
4. Fix the reranker factory seam.
5. Library service honors the profile's search mode.
6. Score-kind-aware bands (UI state + Answer bundle).
7. Console auto-retrieve toggle (config + modal).
8. Console send-path injection (TASK-406).
9. Chip manual run inherits profile top_k.
10. Docs, follow-up filing, backlog closure.
11. Gates + live TUI walkthrough.
<!-- SECTION:PLAN:END -->
