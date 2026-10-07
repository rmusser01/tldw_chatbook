---
id: TASK-407
title: Fix media search results dedup collapse from missing content key
status: Done
assignee: ['@claude']
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
- [x] #1 An unscoped media FTS search with 3+ distinct matches returns all of them through the pipeline
- [x] #2 Dedup still collapses true duplicates
<!-- AC:END -->

## Implementation Plan

Premise re-verified at branch base `cddc89d3e7` (2026-10-06, real in-memory
`MediaDatabase`): `search_media_db` returns no `content` key, the media leg
builds `SearchResult(content="")` for every hit, and `deduplicate_results`
(content[:200] key) collapses 3 distinct matches to 1.

1. Add a batched content-prefix fetch to `MediaDatabase`
   (`fetch_content_prefixes_for_media_batch`, SQL-side `substr` so full
   transcripts are never hauled) mirroring `fetch_keywords_for_media_batch`.
2. In `pipeline_functions_simple.search_media_fts5`, fetch prefixes for the
   matched ids in one threaded call and populate `content` with the prefix,
   falling back to the title when content is empty/null (task's
   "title+snippet fallback").
3. Red-first regression tests (`Tests/RAG/test_pipeline_media_dedup.py`):
   3+ distinct unscoped matches all survive the pipeline including
   `deduplicate_results`; true duplicates (identical content, different ids)
   still collapse; media-leg results carry non-empty content.
4. Targeted runs: the new file plus the existing suites that exercise the
   media leg (`Tests/RAG/test_scope_pipeline_enforcement.py`,
   `Tests/RAG/test_fusion.py`, `Tests/RAG/test_local_citation_capture.py`).

## Implementation Notes

Premise CONFIRMED at base `cddc89d3e7` (2026-10-06): with a real tmp-path
`MediaDatabase` seeded 3 distinct media, `search_media_db`'s row has no
`content` key, `search_media_fts5` returned 3 hits all with `content=''`,
and `deduplicate_results` collapsed them to 1 (repro script output captured
in the transcript; red-first tests below pin the same).

Approach — both mechanisms the description named, composed:

1. **Media leg populates content** (`pipeline_functions_simple.py`): new
   `_MEDIA_SNIPPET_PREFIX_CHARS = 500` (> the dedup key's 200 chars); after
   the search the leg makes one batched thread call to the new
   `MediaDatabase.fetch_content_prefixes_for_media_batch`
   (`Client_Media_DB_v2.py`) — a SQL-side `substr(content, 1, ?)` over the
   matched ids so multi-megabyte transcripts are never hauled — and builds
   each `SearchResult.content` from the prefix, falling back to the title
   when a row's content is empty/NULL. `search_media_db` itself and the
   unscoped call shape are byte-identical (zero drift preserved).
2. **Dedup key scoped by source** (`deduplicate_results`): key is now
   `(source, content[:200])`, not a bare content prefix. Required by the
   existing E2E suite once media hits carry real content:
   `test_scope_pipeline_enforcement.py` seeds a media doc and a note with
   IDENTICAL text; a bare content key made the media hit eat the note
   (red with fix-1 alone: `Note 0` vanished from the built context).
   Within one source, identical content is still a true duplicate (same
   document found by two legs, e.g. FTS + vector both `source="media"`).

Evidence (venv `3.12.13`, `-p no:xdist`):

- Red first: `Tests/RAG/test_pipeline_media_dedup.py` → 3 failed / 3
  passed pre-fix (failures = the collapse `1 == 4`, empty content, title
  fallback); post-fix **7 passed** (`test_three_plus_distinct_matches_
  survive_the_pipeline`, `test_media_leg_populates_nonempty_content`,
  `test_contentless_media_falls_back_to_title`, premise pin
  `test_search_media_db_row_has_no_content_key`, and both AC2 pins).
- Repro script post-fix: leg returns 3 hits with distinct contents; dedup
  keeps all 3.
- Regression sweep, my-change vs HEAD (file-swap A/B, no stash): identical
  results — `test_scope_pipeline_enforcement.py` 4 failed / 73 passed
  BOTH sides (`TestActiveConsoleSessionRealGlue` quartet, pre-existing);
  `test_fusion.py` + `test_local_citation_capture.py` 163 passed;
  `Tests/RAG_Search/test_pipeline_notes_search.py` +
  `test_conversation_search_batch.py` + `test_pipeline_middleware_contract.py`
  2 failed / 21 passed BOTH sides (`RecoveryRequired:
  raw_source_selection_changed`, pre-existing);
  `Tests/Chat/test_scope_picker_listers.py` +
  `Tests/Media/test_local_media_reading_service.py` 13 failed / 93 passed
  BOTH sides (pre-existing); `Tests/UI/test_library_media_trash.py` 92
  setup errors BOTH sides (pre-existing asyncio-fixture environment issue).

ADR required: no — routine bug fix (dedup-key correction + one read-only
helper method); no schema, sync, boundary, or contract change.
Files: `tldw_chatbook/RAG_Search/pipeline_functions_simple.py`,
`tldw_chatbook/DB/Client_Media_DB_v2.py`,
`Tests/RAG/test_pipeline_media_dedup.py` (new).
