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
BLOCKED: nltk/scipy/sklearn/pandas still load via RAG_Admin/local_rag_admin_service.py
-> Chunking/__init__.py (package init) -> Chunking/Chunk_Lib.py:31 `import nltk` --
outside authorized scope (Chunking/, RAG_Admin/, not Embeddings/Web_Scraping/LLM_Calls).
Tracked via Tests/Performance/test_app_import_weight.py::test_app_import_does_not_load_full_heavy_dependency_set
(xfail, strict=True). Task 163 should stay open pending a follow-up scoped to Chunking/.
