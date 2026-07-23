---
id: TASK-407
title: Fix media search results dedup collapse from missing content key
status: To Do
assignee: []
created_date: '2026-07-21 09:48'
labels:
  - rag
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing bug confirmed twice during the RAG scope program (Task 5 review traced it end-to-end): search_media_db returns no content key, so pipeline_functions_simple.search_media_fts5 builds SearchResult(content="") for every media hit and deduplicate_results (content[:200] key) collapses ALL unscoped multi-media results to one. Any unscoped chat-RAG media search returns at most one media result regardless of matches. Fix by having the media leg populate content (fetch/attach snippet or use title+snippet fallback) or dedup by (source, id) instead of content prefix; add a multi-media regression test (n>=3 matches all surviving).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An unscoped media FTS search with 3+ distinct matches returns all of them through the pipeline
- [ ] #2 Dedup still collapses true duplicates
<!-- AC:END -->
