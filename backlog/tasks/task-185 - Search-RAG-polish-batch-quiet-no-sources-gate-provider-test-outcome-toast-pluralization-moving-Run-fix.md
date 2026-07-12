---
id: TASK-185
title: >-
  Search/RAG polish batch: quiet no-sources gate, provider-test outcome toast,
  pluralization, moving-Run fix
status: Done
assignee: []
created_date: '2026-07-12 02:48'
updated_date: '2026-07-12 06:30'
labels:
  - ux
  - library
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11 findings across Search and adjacent surfaces: (1) the no-sources state stacks an 8-line Why/Next/Recovery/Owner dump plus three overlapping guidance lines with internal jargon ('Owner: Library source index') - regressing the quiet-gate principle; (2) Settings 'Provider test finished.' toast reports no pass/fail; (3) 'Matched conversation - 1 messages'; (4) the Run and Start ingest primary buttons shift 30-40px when gate helper lines collapse, breaking muscle memory; (5) keyword search misses plural forms (loops vs loop).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-sources state shows one quiet gate line plus a single recovery action
- [x] #2 Provider test reports success or failure with a reason
- [x] #3 Result counts pluralize correctly
- [x] #4 Primary action buttons keep a stable position across gate-state changes
- [x] #5 Keyword search matches simple plural/singular variants or documents the limitation inline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Quiet no-sources gate, outcome-bearing provider test toast, pluralization, and stable Run/Start-ingest button positions all fixed and live-verified on PR #604. Remaining AC: keyword-search plural/singular matching (or an inline limitation note) — untouched; live network reachability probe split to task-191.

AC #5 shipped: `Library/library_fts_query.py` widens each alphabetic keyword term (>=3 chars) into a quoted FTS5 OR-group of naive plural/singular variants (s/es add-remove, ies<->y) joined by explicit AND, wired through all three keyword seams (media/conversations directly; notes via a new optional pass-through `fts_match_query` on NotesScopeService -> NotesInteropService -> CharactersRAGDB.search_notes, plus a verbatim branch for caller-built MATCH strings in Client_Media_DB_v2.search_media_db); covered by unit, FTS5-injection, and real-DB end-to-end tests in Tests/Library/test_library_local_rag_search_service.py.
<!-- SECTION:NOTES:END -->
