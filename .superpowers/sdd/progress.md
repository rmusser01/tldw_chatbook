# Startup-perf follow-up ledger (task 163)
Plan: Docs/superpowers/plans/2026-07-11-followups-startup-perf.md
Branch: claude/followups-startup-perf off dev b8344550
Baseline: import tldw_chatbook.app ~4.8s/6518 modules; torch 1026 + transformers 256 + scipy 526 + sklearn 190 + pandas 293 + nltk 240 present.
Root causes (meta-path traced): (1) app.py:182 RAG_Admin -> Chroma_Lib -> Embeddings_Lib:53-55 module-scope get_safe_import(numpy/torch/transformers) [REAL __import__]; (2) Article_Extractor_Lib:99 eager Summarization_General_Lib -> Chunk_Lib -> nltk stack.

## Outcome (see .superpowers/sdd/startup-perf-report.md for full detail)
Fixed (in scope, Embeddings/Web_Scraping): Embeddings_Lib.py (torch/transformers/numpy
lazy via _ensure_*()), Article_Extractor_Lib.py (analyze deferral + pandas find_spec
deferral), Chroma_Lib.py (chunk_for_embedding + analyze deferral -- 2nd eager edge
found), WebSearch_APIs.py (analyze deferral -- 3rd eager edge found, separate boot
path via Tools/web_search_tool.py).
Result: torch/transformers fully eliminated from `import tldw_chatbook.app`
(6519->4659 modules, ~5.65s->~4.11s, fresh-dir apples-to-apples).
Commit 2 (scope extension AUTHORIZED 2026-07-11): Chunking/Chunk_Lib.py -- deferred
`import nltk` behind _ensure_nltk() + find_spec-based NLTK_AVAILABLE; removed the
module-scope ensure_nltk_data() call (which did a punkt NETWORK DOWNLOAD at import)
and made it idempotent + lazy; guarded _adaptive_chunk_size_nltk + _semantic_chunking
use sites. nltk-transitive scipy/sklearn/pandas removed automatically.
Result: ALL 8 HEAVY_MODULES absent at boot. 6519 -> 3291 modules (-49.5%), ~5.65s -> ~1.48s.
AC #1 SATISFIED. Full-set perf test promoted from xfail to hard assertion (RED-verified).
Feature regression: 404 passed / 20 skipped / 0 failed across Chunking/RAG/RAG_Admin/
RAG_Search/Web_Scraping/startup-polish/optional_deps. numpy stays (chromadb/pymupdf, allowed).
Pre-existing punkt vs punkt_tab nltk-5.x bug in semantic chunking documented as follow-up
(reproduces on original code, out of scope).

## Task 163 — COMPLETE (2026-07-11)
- Commit b5970b19: torch/transformers/numpy + summarization stack deferred (Embeddings_Lib, Chroma_Lib, Article_Extractor_Lib, WebSearch_APIs)
- Commit fd79452a: nltk deferred in Chunk_Lib → scipy+sklearn+pandas (all nltk-transitive) also gone
- Whole-branch review (opus): APPROVE, 2 Minor non-blocking thread-window notes
- Gate: perf test + Chunking + RAG + RAG_Admin + RAG_Search + embeddings-datatable = 319 passed / 20 skipped / 0 failed; app import OK
- Boot: 6,519 → 3,291 modules (−49.5%), ~5.65s → ~1.48s; torch/transformers/nltk/scipy/sklearn/pandas/docling/torchvision all absent
- Follow-up logged: pre-existing nltk 5.x punkt/punkt_tab semantic-chunk resource mismatch (reproduces on original code)

---
# SDD ledger — agent-runtime Plan A (engine + persistence) [2026-07-13]
Plan: Docs/superpowers/plans/2026-07-13-agent-runtime-plan-a-engine.md | Spec: Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md | Branch: claude/agent-runtime-spec (worktree .claude/worktrees/agent-runtime)
Task 1: complete (commit 45c4107e, review Approved first pass — byte-exact transcription, purity verified, 7 tests)
Task 2: complete (commits ce642ddd+09eb8b3d; review Needs-fixes->Approved. Important fix: fence tag now requires clean line boundary — ```tool_calls/```tool_call_schema no longer mis-parse; split scans forward past bad tags; bare FENCE_OPEN now 'undecided'. RED-verified 4-fail pre-fix. Minor for final review: sniffer/parser whitespace-class mismatch on / + bare-CR tag line — fails toward text/None, safe direction)
Task 3: complete (commit 769df545; review Approved first pass — all 8 binding semantics traced in shipped code, ternary precedence verified via dis, truncation asymmetry + infinite-loop safety confirmed. Minors for final review: unused ToolCatalogEntry import (plan-inherited), load_tools no-room branch untested, multi-native-call batch untested)
Task 4: complete (commit 35271edd; review Approved first pass — byte-exact, asyncio.run-under-running-loop degrades safely (verified live), calculator content shape verified. Minors for final review: _owner_and_id per-lookup re-listing (no caching — revisit when MCP provider lands, else N network calls per invoke), raw.get('error') branch dead code today, brief '11 tests' prose miscount (file has 10))
Task 5: complete (commit 1c826b13; review Approved first pass — append_steps single-transaction RMW verified, COALESCE preserves result, no BaseDB connection-caching hazard (plain connect per call, verified by full base_db read), BaseDB.__init__ auto-inits schema at :70. Minors: duplicate PRAGMA/row_factory, both harmless pattern-mirrors)
Task 6: complete (commit 541e03ff; review Approved first pass — all 6 binding behaviors + adversarial checks verified LIVE (recursive-spawn rejection, disclosed-names closure mutation, child protocol excludes SPAWN); 59-test full suite independently re-run; byte-exact transcription. Brief prose miscounts (11/60 vs real 10/59) noted, non-issues. ALL Plan-A implementation tasks complete)
Final whole-branch review (opus): APPROVE WITH FIXES — 0 Critical/High, 1 Medium (F1 gate-vs-active-cap desync, repro-confirmed, dormant in slice) FIXED commit 15167461 + verified by reviewer (60 tests). Residual Minor F1-b (duplicate re-load desync: loop active list vs disclosed set dedupe) DEFERRED to task-201/200 gate w/ T3b (loop-side dedupe, no signature change) — documented in plan-a-final-review.md addendum. Minor triage: 3 drop / 5 defer / 0 block. Plan-B handoff MUSTs recorded: populate allowed_tools (fail-closed default), UI markup-escaping, worker thread (asyncio.run in provider), optional on_step hook.
LIVE SMOKE (informal, not a gate): AgentService vs REAL llama.cpp Qwen-27B — model emitted a perfectly-formed fence FIRST TRY, real calculator executed (379*6421=2433559 correct), result fed back, final answer, persisted done, 4 steps, 44s. Fence-first text protocol validated against the actual QA model.
PLAN A COMPLETE: 6 tasks + 2 fix waves, 60 tests green, commits 45c4107e..15167461 on claude/agent-runtime-spec.
