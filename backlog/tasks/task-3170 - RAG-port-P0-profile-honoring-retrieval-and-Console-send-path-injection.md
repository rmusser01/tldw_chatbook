---
id: TASK-3170
title: RAG-port P0 profile-honoring retrieval and Console send-path injection
status: Done
assignee: []
created_date: '2026-08-07 14:19'
updated_date: '2026-08-12'
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
- [x] #1 Library rag-mode search resolves the active RAG profile's default_search_mode (plain/semantic/hybrid) instead of hardcoding semantic search; plain-profile queries route through the existing four-seam _search_keyword path (honestly labeled), never through the engine's standalone keyword leg.
- [x] #2 The engine's keyword leg no longer guesses DB paths or opens chacha_notes.db with media-schema SQL: the media DB path is injected explicitly from config, the create-a-new-MediaDatabase-on-miss side effect is deleted, keyword rows carry source_type provenance so they survive the Library's canonicalizing post-filter, and a failed leg degrades to a disclosed coverage note rather than performing a write.
- [x] #3 Match-strength presentation is score-kind-aware: _fuse_hybrid_results preserves per-leg scores in hybrid_fusion metadata, RRF-fused and reranker scores are never banded on the cosine strong/moderate/weak thresholds, FTS-only rows render as 'keyword match' rather than a fabricated similarity, library_rag_all_matches_weak fires only over vector-similarity score kinds, and the backend label is mode-truthful (rag-hybrid/rag-keyword/rag-semantic).
- [x] #4 The reranker factory is fixed so a reranking-enabled profile actually constructs a reranker; model loading is verified off the event loop (moved to service construction if it loads lazily at first .rerank()); an unavailable reranker (missing model/dep) skips reranking with disclosed provenance instead of failing the search.
- [x] #5 Zero-results honesty under hybrid: the 'Index empty' recovery state does not fire when the keyword leg returned rows, and when the semantic leg is empty but the keyword leg is not, the search discloses 'semantic leg empty -- keyword-only results' instead of the generic empty state.
- [x] #6 An 'Auto-retrieve on send' toggle in the Console RAG chip modal controls native send-path injection, defaults to OFF, and persists as a global config key, re-homing enablement away from the legacy sidebar checkbox.
- [x] #7 Auto-retrieve fires only for plain user text sends -- never for slash commands, tool approvals, or regenerations -- and skips automatically when evidence is already manually staged (no double retrieval/spend).
- [x] #8 When auto-retrieve fires, retrieved results route through the existing staged-evidence pipeline (strip shows 'Evidence sent · N', consumption reuses the consume-on-send predicate) rather than invisible prompt injection; the Console chip's manual run is switched from its hardcoded top_k=5 to the active profile's default_top_k so manual and auto retrieval never disagree about depth.
- [x] #9 An EMPTY resolved scope short-circuits auto-retrieve with the same shared notice copy as the manual path (task-406 AC #2).
- [x] #10 Auto-retrieve runs in an exclusive worker with a visible 'Retrieving...' strip state and an explicit 5s timeout; on failure or timeout the send proceeds without evidence and shows a quiet notice that distinguishes 'RAG service still initializing' from 'retrieval failed', and a send is never blocked on retrieval.
- [x] #11 The legacy chat_events RAG injection path is untouched by this work (task-406 AC #3).
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 11 ACs are complete: unit/integration tests (RED-first, mutation-checked
per task -- see the SDD ledger at
.superpowers/sdd/2026-08-07-rag-port-p0-foundations/progress.md and each
task's own report) PLUS the live TUI walkthrough Task 11 owed, recorded at
the end of these notes.

Approach: nine sequential tasks, each RED-first with mutation checks, each
independently reviewed (most needed one fix round; task-406/AC#1's Console
injection took two). Two merge-order gates were tracked explicitly in the
ledger and both cleared before the branch could merge: Task 5's I3 (hybrid
match-strength honesty depended on Task 6 landing) and the general
task-406-untouched-until-toggle-exists ordering (task 6/7 before task 8).

Headline findings (the "why this task existed" evidence):
- Reranking had NEVER activated in production. A double-`strategy` keyword
  TypeError meant every reranking-enabled profile silently failed to
  construct a reranker (task 4); even after that fix, the real production
  factory (`rag_factory.create_rag_service`) separately dropped
  `reranking_config` before passing it to the constructor, so reranking
  still could not have run via the app's actual RAG-service entry point
  until both bugs were fixed together.
- Library's rag-mode search hardcoded semantic-only retrieval regardless of
  the active profile's `default_search_mode` -- a user on the "BM25 Only"
  profile got vector-only search with nothing on screen saying so (task 5).
- The engine's keyword leg had THREE independent never-worked bugs (guessed
  DB paths, opened chacha_notes.db with media-schema SQL, a nonexistent
  Media.tags column) plus an inverted FTS5 ordering (`ORDER BY rank` on a
  `-rank` alias returns worst-first) -- it could never have returned a
  correct row (task 3).
- Hybrid search's fused RRF scores (max ~0.016) were banded against the
  cosine similarity thresholds meant for vector-only scores, so every
  hybrid hit rendered "match: weak" regardless of actual relevance --
  score-kind-aware banding fixed this and also gave FTS-only rows an
  honest "keyword match" label instead of a fabricated similarity (task 6).
- The native Console send path (`ConsoleChatController.submit_draft`) had
  ZERO RAG context injection; the only wired path (`get_rag_context_for_chat`)
  was the legacy `chat_events` route, unreachable in the live app. Auto-
  retrieve (opt-in, default OFF, staged-evidence pipeline, 5s timeout, never
  blocks a send) closes that gap (tasks 7-8), and the manual chip run's
  hardcoded `top_k=5` now matches the active profile's `default_top_k`
  (task 9) -- disclosed to users in the docs updated by this task.

Plain-profile routing (task 5): `rag`-mode search now dispatches on the
active profile's search mode -- `plain` routes through the Library's own
four-seam, scope-aware keyword path (never the engine's media-only keyword
leg, which would be a strictly worse search than `search` mode already
gives); `hybrid` runs the engine's fused hybrid only when unscoped AND
Media is selected (scoped or Media-deselected hybrid falls back to
semantic, disclosed, because the engine's allowlist pushdown and FTS leg
are semantic-only/media-only in P0); `semantic` is unchanged. Every
divergence from "what you'd expect from a hybrid profile" is disclosed via
a quiet route-note line, not applied silently.

Files touched (by task; full list in each task-N-report.md):
RAG_Search/simplified/rag_service.py (fusion leg-score preservation, task 2);
RAG_Search/{reranker.py,simplified/enhanced_rag_service_v2.py,simplified/
rag_factory.py,config_profiles.py} (reranker factory + degradation tags,
task 4); Library/library_local_rag_search_service.py (keyword leg DB
resolution task 3; profile-mode routing task 5); RAG_Search/
local_citation_capture.py, Library/library_rag_score_kinds.py (new),
Library/library_rag_state.py, Widgets/Library/library_search_rag_panel.py,
UI/Views/RAGSearch/search_handoff.py (score-kind bands, task 6);
tldw_chatbook/config.py, Widgets/Console/console_rag_settings_modal.py,
UI/Screens/chat_screen.py (auto-retrieve toggle task 7, send-path injection
task 8, chip top_k parity task 9); Docs/User_Guide/{library/search-and-rag.md,
console/context-and-rag.md,settings/rag.md} (this task).

Deviations from the plan: none load-bearing. Two Important review findings
were fixed per task (task 4 reranker cache-mutation + silent-failure
disclosure; task 5 stale-profile-cache + zero-row-disclosure-dropped; task 7
draft-discard coupling + UI-thread write; task 8 unpinned placeholder
staging) -- all documented in their own task reports, not repeated here.

Follow-ups filed by this task (all reference TASK-3170): TASK-3500 (filed for
combined MCP perform_rag_search + agent RAGSearchTool profile-driven-retrieval
parity, the declared P0 non-goal; current disposition: `LibraryRagToolProvider`
satisfies the agent side and TASK-3500 is MCP-only), TASK-3501
(pipeline_builder_simple.py's hybrid
merge has the same leg-score aliasing bug task 2 fixed in rag_service.py),
TASK-3502 (reranker provider/model selection + cost surface, plus two
re-review residuals), TASK-3503 (config.load_settings cache-miss race can
return None to worker threads), TASK-3504 (Console auto-retrieve zero-result
outcome is fully silent).
<!-- SECTION:NOTES:END -->

### Task 11 gates + live TUI walkthrough (2026-08-07)

**Targeted battery.** All 11 test files this branch created or modified, one
run: **496 passed, 0 failed** (4:58). That equals the sum of their individual
collected counts, so nothing was skipped or deselected.

**Collection arithmetic** (`--collect-only -q` over `Tests/`, HEAD vs. a
detached worktree at the merge-base `65c743d1e`): baseline **31,931**, HEAD
**32,038**, delta **+107**. Accounted for exactly: five new files contribute
16 + 2 + 4 + 12 + 23 = 57, and six modified files contribute 25 + 2 + 2 + 13
+ 5 + 3 = 50. 31,931 + 57 + 50 = 32,038. No pre-existing test was lost.

**Live walkthrough.** Real TUI in tmux at 235x52, scratch profile via
`TLDW_CONFIG_PATH` with `users_name = verify_ragp0`, holding a COPY of the
real ChaChaNotes + media DBs and the real `chromadb/` directory (453 vectors)
placed before first launch; real OpenAI (Library RAG Answer) and real
Anthropic (Console) credentials. The live `~/.config/tldw_cli/config.toml`
was byte-identical afterwards (SHA-256 unchanged, mtime unchanged).

1. **Hybrid Basic (default), Library rag mode** -- rows banded on real cosine
   similarity: "match: moderate" x2, then "weak (0.18)" / "weak (0.17)".
   The pre-fix wall of "weak (0.02)" (banding an RRF score whose ceiling is
   ~0.016) is gone. Zero-row sources disclosed: "Semantic search found
   nothing from: Notes, Conversations."
2. **BM25 Only** -- "Set active" repointed `[rag.service] profile` and the
   next run disclosed "Profile 'BM25 Only': keyword search (no vectors)."
   and returned a note row AND a **media** row -- the four-seam keyword path
   including the media seam that AC#2 repaired. The route note also rendered
   on a zero-row query, which is the AC#5 disclosure-survives-empty rule.
3. **Hybrid Full** -- the service constructed and the search completed (the
   pre-AC#4 double-strategy `TypeError` would have killed it). With no index
   for that profile's embedding model it degraded honestly: "Semantic leg
   empty -- keyword-only results", and its FTS-leg row rendered "keyword
   match" rather than a fabricated similarity (AC#3).
4. **Console** -- see TASK-406's notes: toggle persists at flip time,
   "Auto-retrieving..." -> `RAG: on / Sources: 1 staged` -> "Evidence sent
   with this message - 15 sources", and the model quoted the injected
   `[S1]..[S15]` block back. A slash-command send retrieved nothing.
5. **Scoped run** -- deselecting Media routed to semantic and said so:
   "Media excluded -- semantic only."

**Defect found live and fixed forward (RED-first, own commit).** Under
Hybrid Full the one FTS-leg row rendered as "1. Untitled source | keyword
match" while its own citation line read "Citations: meeting_notes". The
vector leg gets `title` for free (indexing spreads document metadata into
every chunk); the keyword leg builds metadata from scratch and stamped only
`doc_title`, which `library_local_rag_search_service._semantic_row` -- the
mapper every engine-backed Library row passes through -- does not read. The
symptom was unreachable before this branch, because that leg could not
return a row at all. Both keyword-leg metadata blocks now stamp `title`; the
new test asserts through `_semantic_row` AND `LibraryRagResultRow.from_result`
so it pins the rendered title, not a key name. Re-verified live after
relaunch: the row now reads "1. meeting_notes | keyword match".

Evidence captures (one file per checklist item) live in the session
scratchpad under `ragp0-evidence/`, and the full gate report is at
`.superpowers/sdd/2026-08-07-rag-port-p0-foundations/task-11-report.md`.
