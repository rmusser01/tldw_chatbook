---
id: TASK-202
title: >-
  Delete dead RAG_Search modules (late chunking, context assembler, query
  expansion)
status: To Do
assignee: []
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
- [ ] #1 late_chunking_service.py and late_chunking_integration.py and context_assembler.py and query_expansion.py are removed or an explicit decision doc records why they stay
- [ ] #2 Orphaned config keys and UI stubs that only fed the removed modules are cleaned up or explicitly kept with a comment
- [ ] #3 Full test suite passes with no import errors after removal
- [ ] #4 ENHANCED_RAG_FEATURES.md and README_enhanced_services.md are updated or removed to match what actually exists
<!-- AC:END -->
