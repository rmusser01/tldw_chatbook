---
id: TASK-21126
title: >-
  Library Search/RAG panel runs an unindexed full-table GROUP BY on the event loop per panel mount
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - library
  - rag
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21126).

`RAG_Admin/local_rag_admin_service.py:592-596` runs `SELECT chunk_engine_version,
COUNT(DISTINCT media_id) ... GROUP BY chunk_engine_version` over `UnvectorizedMediaChunks`
(rows-per-chunk; no index on chunk_engine_version; temp B-tree for the DISTINCT). The
`_maybe_await` seam (rag_admin_scope_service.py:81-84) evaluates this sync call ON the loop,
and the panel remounts per destination switch (its own docstring,
library_search_rag_panel.py:95-98). Full scan + sort per Library navigation click at realistic
chunk counts (1e5-1e6 rows).

## Acceptance Criteria

- [ ] The census query runs off the loop and its result is cached per session (invalidated on ingest/re-chunk)
- [ ] An index or maintained count removes the full-scan; EXPLAIN QUERY PLAN before/after recorded in the task
- [ ] Panel content unchanged
