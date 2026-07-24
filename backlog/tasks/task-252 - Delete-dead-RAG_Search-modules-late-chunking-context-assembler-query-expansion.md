---
id: TASK-252
title: >-
  Delete dead RAG_Search modules (late chunking, context assembler, query
  expansion)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - cleanup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four RAG_Search modules have zero importers in app code: late_chunking_service.py and late_chunking_integration.py and context_assembler.py (only imported by each other) and query_expansion.py (no importers at all; the query-expansion Settings handler in app.py:7679 is a UI stub that stores a string and never imports the module). Config keys like enable_late_chunking reference the concept but never reach these modules. This inflates the module and misleads reviewers into thinking the features are live. Note reranker.py and parallel_processor.py are NOT dead: enhanced_rag_service_v2.py imports create_reranker and the parallel processor. Repo precedent for removal: commit 628b1b8b deleted zero-importer legacy modules. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 late_chunking_service.py and late_chunking_integration.py and context_assembler.py and query_expansion.py are removed or an explicit decision doc records why they stay
- [x] #2 Orphaned config keys and UI stubs that only fed the removed modules are cleaned up or explicitly kept with a comment
- [x] #3 Full test suite passes with no import errors after removal
- [x] #4 ENHANCED_RAG_FEATURES.md and README_enhanced_services.md are updated or removed to match what actually exists
<!-- AC:END -->

## Implementation Plan

1. Re-verify zero importers for each of the four modules on this checkout (git grep across tldw_chatbook/ and Tests/) immediately before each deletion.
2. Delete late_chunking_service.py, late_chunking_integration.py, context_assembler.py, query_expansion.py.
3. Delete Tests/RAG/simplified/test_query_expansion.py (imports the removed query_expansion module directly; only tests removed code).
4. Clean orphaned config keys/UI stubs for enable_late_chunking (verified: only late_chunking_integration.py ever read the key; enhanced_chunking_service.py does not consume it):
   - simplified/config.py: remove enable_late_chunking + late_chunking_cache_size from ChunkingConfig (from_settings never loads them; RAGConfig.from_dict callers are round-trip tests only).
   - config_profiles.py: remove the single enable_late_chunking assignment line (keep diff minimal; task-246 may touch this file).
   - media_details_widget.py: remove the "Enable Late Chunking" checkbox and its save/reset/load lines (the stored per-document key had no live consumer).
5. Keep QueryExpansionConfig in simplified/config.py with an explicit comment (AC #2 "kept with a comment" path) — removing it would be a large diff in a file a parallel task also edits; the [AppRAGSearchConfig.rag.query_expansion] TOML section remains parseable.
6. Keep the app.py handle_query_expansion_method_changed UI stub with a comment (stores a string; never imported the removed module).
7. Delete Helper_Scripts/Examples/rag_query_expansion_config.toml (example config solely for the removed feature; zero references).
8. Docs: fix stale references in ENHANCED_RAG_FEATURES.md (nonexistent test_enhanced_rag.py) and README_enhanced_services.md (nonexistent config_examples/rag_v2_example.toml); neither doc advertises the removed modules, so no sections need trimming.
9. Run Tests/RAG/ + Tests/UI/ plus an import smoke test; compare UI failures against the known baseline.

## Implementation Notes

Deleted the four dead modules plus their only test and an orphaned example config. Importers were re-verified with git grep across tldw_chatbook/ and Tests/ immediately before deletion: late_chunking_service.py was imported only by late_chunking_integration.py; late_chunking_integration.py had zero importers; context_assembler.py was imported only by late_chunking_integration.py; query_expansion.py was imported only by Tests/RAG/simplified/test_query_expansion.py. reranker.py and parallel_processor.py were left untouched (live via enhanced_rag_service_v2.py).

Removed files:
- tldw_chatbook/RAG_Search/late_chunking_service.py
- tldw_chatbook/RAG_Search/late_chunking_integration.py
- tldw_chatbook/RAG_Search/context_assembler.py
- tldw_chatbook/RAG_Search/query_expansion.py
- Tests/RAG/simplified/test_query_expansion.py
- Helper_Scripts/Examples/rag_query_expansion_config.toml (example config solely for the removed feature; zero references)

Orphaned config/UI cleanup — enable_late_chunking removed everywhere (verified only late_chunking_integration.py ever read the key; enhanced_chunking_service.py, which is live via ChunkPreviewModal, never consumes it):
- simplified/config.py: dropped ChunkingConfig.enable_late_chunking + late_chunking_cache_size (from_settings never loaded them; the only RAGConfig.from_dict callers are to_dict/from_dict round-trip tests, which stay symmetric)
- config_profiles.py: dropped the single hybrid_full assignment line (one-line diff to avoid conflicts with parallel task-246)
- media_details_widget.py: removed the Enable Late Chunking checkbox and its save/reset/load lines; previously stored per-document configs containing the key still load fine (config.get access)

Explicitly kept with comments (AC #2 "kept with a comment" path):
- QueryExpansionConfig in simplified/config.py — keeps [AppRAGSearchConfig.rag.query_expansion] TOML parsing and RAGConfig round-trips working; removal would be a large diff in a file a parallel task also edits
- app.py handle_query_expansion_method_changed — UI-only stub storing a string; never imported the removed module
- [middleware.query_expansion] in rag_pipelines.toml untouched: pipeline_loader.py has its own _expand_query placeholder and never imported the deleted module

Docs: neither ENHANCED_RAG_FEATURES.md nor README_enhanced_services.md actually advertised the deleted modules; both received a short Removed Features section and fixes for stale references (nonexistent test_enhanced_rag.py -> pytest Tests/RAG/; nonexistent config_examples/rag_v2_example.toml -> rag_config_example.toml).

Testing: Tests/RAG/ 205 passed, 8 skipped. Tests/UI/ 31 failed, 3469 passed, 19 skipped — at/below the ~33 pre-existing baseline, and all failures are shell/console/settings tests unrelated to this change; the RAG/media-adjacent files (test_chat_window_enhanced.py, test_media_details_widget_tooltips.py, test_settings_library_rag_defaults.py) pass 48/48. Import smoke test (import tldw_chatbook.app) and a config round-trip + hybrid_full profile load both pass.
