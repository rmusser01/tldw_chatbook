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
