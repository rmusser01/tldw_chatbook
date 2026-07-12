---
id: TASK-163
title: Slim remaining eager heavy imports at app startup
status: In Progress
assignee: []
created_date: '2026-07-11 22:02'
updated_date: '2026-07-12 03:44'
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
- [ ] #1 import tldw_chatbook.app no longer loads nltk/torch/transformers at boot
- [ ] #2 A subprocess regression test pins the absence
- [ ] #3 No feature regression from the deferral
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
PARTIAL: torch/transformers fully eliminated from app boot (verified via subprocess sys.modules check, 6519->4659 modules, ~5.65s->~4.11s fresh-dir apples-to-apples). AC #1 NOT fully met: nltk/scipy/sklearn/pandas still load via a chain outside the authorized fix scope: RAG_Admin/local_rag_admin_service.py -> Chunking/__init__.py (package init) -> Chunking/Chunk_Lib.py:31 eager `import nltk`. AC #2 met (Tests/Performance/test_app_import_weight.py, subprocess-based, RED-verified). AC #3 met (full feature-regression sweep green: RAG_Search, Chunking, RAG_Admin, RAG, Web_Scraping, Tools_Interop, optional_deps, embeddings UI -- all passing, plus a real end-to-end HF embedding creation).

Full trace, before/after numbers, and a recommended follow-up (fix Chunking/__init__.py or Chunk_Lib.py's eager nltk import, same _ensure_*() pattern) are in .superpowers/sdd/startup-perf-report.md. Left as In Progress rather than Done pending that follow-up (or an explicit AC narrowing) per Definition of Done.
<!-- SECTION:NOTES:END -->
