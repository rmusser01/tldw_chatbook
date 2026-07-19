---
id: TASK-260
title: RAG conversation search: batch fetch + skip image BLOBs
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, rag]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
pipeline_functions_simple.search_conversations_fts5 (96-115) issues one get_messages_for_conversation per matched conversation (≤20 queries/search) and every call SELECTs image_data BLOBs only to build text snippets. get_messages_for_conversations_batch (ChaChaNotes_DB 5870-5929) exists unused. Add an include_image_data=False variant for text-only callers (also mindmap_integration.py:88). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One batched query replaces the per-conversation loop
- [ ] #2 Text-only callers no longer fetch BLOB columns
- [ ] #3 RAG search results unchanged (tests)
<!-- AC:END -->
