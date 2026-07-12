---
id: TASK-163
title: Slim remaining eager heavy imports at app startup
status: Done
assignee: []
created_date: '2026-07-11 22:02'
updated_date: '2026-07-12 04:04'
labels:
  - follow-up
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F3 dropped the ingest chain from ~5.5s to ~0.9s, but app boot still eagerly loads nltk/torch/transformers via Web_Scraping/Article_Extractor_Lib (eager Summarization_General_Lib import) and Utils/optional_deps (eager check_dependency probe). Defer these to shed the remaining startup weight (~1s+).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 import tldw_chatbook.app no longer loads nltk/torch/transformers at boot
- [x] #2 A subprocess regression test pins the absence
- [x] #3 No feature regression from the deferral
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Defer torch/transformers/numpy eager imports in Embeddings_Lib.py via _ensure_*() lazy accessors.
2. Defer eager analyze/chunk_for_embedding imports in Article_Extractor_Lib.py, Chroma_Lib.py, WebSearch_APIs.py into their call sites.
3. Add a subprocess regression test pinning module absence.
4. Verify no feature regression via existing test suites + a real end-to-end embedding run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE. All three ACs satisfied. `import tldw_chatbook.app` no longer loads torch/transformers (commit b5970b19: lazy _ensure_torch/_ensure_transformers/_ensure_numpy in Embeddings_Lib.py + deferred analyze/chunk_for_embedding/pandas imports in Chroma_Lib/Article_Extractor_Lib/WebSearch_APIs) NOR nltk/scipy/sklearn/pandas (commit fd79452a, authorized scope extension: deferred `import nltk` + removed the import-time punkt network download in Chunking/Chunk_Lib.py; scipy/sklearn/pandas are nltk-transitive and dropped with it). AC #1: all 8 HEAVY_MODULES absent at boot, 6519->3291 modules (-49.5%), ~5.65s->~1.48s. AC #2: Tests/Performance/test_app_import_weight.py (subprocess, fresh interpreter) hard-asserts the full absent set + lazy-accessor availability, RED-verified against pre-fix code. AC #3: 404 passed/20 skipped/0 failed across Chunking/RAG/RAG_Admin/RAG_Search/Web_Scraping/optional_deps + real end-to-end embedding & chunk runs. numpy intentionally still loads (chromadb/pymupdf dep, light, allowed). Full detail: .superpowers/sdd/startup-perf-report.md. Documented follow-up (NOT an AC, pre-existing, reproduces on original code): semantic chunking hits an nltk-5.x punkt vs punkt_tab resource-name mismatch -- update the two 'punkt' strings in ensure_nltk_data()/_semantic_chunking to 'punkt_tab'.
<!-- SECTION:NOTES:END -->
