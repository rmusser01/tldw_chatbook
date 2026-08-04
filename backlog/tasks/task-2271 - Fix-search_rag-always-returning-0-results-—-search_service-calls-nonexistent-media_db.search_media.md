---
id: TASK-2271
title: >-
  Fix search_rag always returning 0 results — search_service calls nonexistent
  media_db.search_media()
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-04 21:30'
updated_date: '2026-08-04 18:47'
labels:
  - rag
  - mcp
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovered during the PR-5 live check (`.superpowers` live-check report, 2026-08-04) and PRE-EXISTING — not introduced by PR-5 or by the PR #1226 coroutine fix: the RAG search service (observed at `search_service.py:118` on `feat/rag-v2-mcp-guardrails` @ a953e4c1e) calls `media_db.search_media()`, a method that does not exist on the media DB (`search_media_db` does). The resulting `AttributeError` is swallowed by a broad exception handler, so the search silently falls back to an empty result — the MCP `search_rag` tool (and potentially other RAG surfaces routed through the same service) returns an honest-looking "0 results" for EVERY query against a real profile.

This is exactly the dishonesty class RAG-49 exists to prevent (a crash masquerading as an empty result), but it sits upstream of the tool boundary, so the tool's own error shape (`[{"error": ...}]`) never fires. During the live check, every `search_rag` query against a seeded real-profile copy returned 0 results while `search_notes` (a different service path) returned real hits — that contrast is the reproduction.

Verify the call-site line against the current file before fixing (the live-check agent identified it empirically; line numbers drift).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `search_rag` returns real results against a profile whose media DB contains matching content.
- [x] #2 A failure inside the search service surfaces as an error (the tool's error shape and/or a logged error with context), never as a silent empty-success.
- [x] #3 A regression test pins the media-DB method name actually called (a call-path test against the real DB API, not a mock that would accept any name).
- [x] #4 Audit of the same service for other swallowed-exception fallbacks that convert crashes into empty results, with findings fixed or filed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Root cause verified 2026-08-04 on `fix/task-2271-search-media` @ b3a123d25: `RAG_Search/simplified/search_service.py:118` (`keyword_search`) calls `self.media_db.search_media(query=, limit=, media_types=)`; the real API is `search_media_db(search_query=..., media_types=..., results_per_page=...) -> Tuple[List[Dict], int]` (Client_Media_DB_v2.py:1774). The blanket `except Exception → return []` swallows the AttributeError. `semantic_search` FALLS BACK to `keyword_search` (line 94) so both modes break. Blast radius: the simplified service is imported only by `MCP/tools.py`, whose `perform_rag_search` already catches into `[{"error": ...}]` (RAG-49 renders it honestly).

1. TDD against a real in-memory MediaDatabase: `keyword_search` returns real rows for seeded content (pin row-shape mapping against actual `search_media_db` row keys); tuple correctly unpacked; `results_per_page`/`media_types` threaded.
2. Replace the swallow: on failure, log with context and RAISE (typed or re-raise) from `keyword_search`/`semantic_search`; end-to-end test that a raising media_db surfaces as the `[{"error": ...}]` shape from `perform_rag_search`, not `[]`.
3. Audit the module for remaining crash→empty conversions (AC#4); fix or document each.
4. Targeted gates + collect sweep; live verification on a scratch profile (search_rag returns real results); ship per merge-when-verified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed exactly as diagnosed: keyword_search called self.media_db.search_media(query=, limit=, media_types=) -- no such method exists on MediaDatabase. The AttributeError was swallowed by `except Exception: return []`, so every search_rag query silently returned 0 results. semantic_search falls back to keyword_search when no enhanced RAG service is configured, so both modes were broken, and semantic_search had the identical swallow.

Fix: keyword_search now calls the real `search_media_db(search_query=, media_types=, results_per_page=limit)` and unpacks the (rows, total) tuple. Real row keys (verified against search_media_db's SELECT and Media table schema) are id/uuid/url/title/type/author/ingestion_date/transcription_model/transcription_provenance_json/is_trash/trash_date/chunking_status/vector_processing/content_hash/last_modified/version/client_id/deleted -- notably NO `content` (search_media_db's projection omits the large text column by design, same as the existing search_media_by_keyword_for_embedding "second query" pattern) and NO `local_path` (Media has no such column; `url` is the only source-reference field). Fixed mapping: content is now batch-fetched via get_media_by_ids_for_embedding(ids) in one extra query; url maps from item["url"]; file_path is set to None (documented -- there is no distinct local-path column, so duplicating url would be dishonest); metadata keys trimmed to real columns (dropped the invented "created_at", kept ingestion_date/author/transcription_model). Outer shape (id/title/content/media_type/url/file_path/score/metadata) kept stable for MCP/tools.py.

Both keyword_search's and semantic_search's blanket excepts now log-then-raise instead of returning []. perform_rag_search (MCP/tools.py) already catches into the honest `[{"error": ...}]` shape, so callers get a real error instead of a fake empty success.

Module audit (AC#4): only one other except block exists in search_service.py -- __init__'s `except Exception: self.rag_service = None` around create_rag_service(). Left as-is: it's a constructor-time degrade-gracefully mode selection (falls back to keyword search, already logged), not a crash-during-search converted into an empty result, so it's a different category from the bug this task targets.

TDD: Tests/RAG/simplified/test_search_service.py (new, 8 tests) uses a REAL in-memory MediaDatabase (no DB mocks). Verified RED against the original code (all 8 assertions failed / didn't raise, with the exact `'MediaDatabase' object has no attribute 'search_media'` error logged), then GREEN after the fix. Covers: real row-key mapping, media_types filter, limit, a raising media_db propagating through keyword_search AND through perform_rag_search's error shape, semantic_search's fallback-to-keyword_search happy path, and semantic_search re-raising on failure. Tests/MCP/test_rag_search_tool.py (protected oracle) required no changes -- it stubs at the SimplifiedRAGSearchService interface level, not media_db, so it never exercised the buggy call path; ran unmodified and still passes.

Gates run: Tests/RAG/simplified/test_search_service.py + Tests/MCP/test_rag_search_tool.py (10 passed), Tests/MCP/ full directory (417 passed, includes a real-DB MCPTools construction test), `pytest Tests/ --collect-only -q` (29915 collected, 0 errors).

AC#1 ("search_rag returns real results against a profile...") has a unit-level equivalent (test_end_to_end_perform_rag_search_returns_real_results, calling MCPTools.perform_rag_search against a real in-memory MediaDatabase) but no live-app/live-profile verification was run in this worktree -- left unchecked for the controller to verify live if desired, per task instructions.

Files changed: tldw_chatbook/RAG_Search/simplified/search_service.py (keyword_search rewritten, both excepts now raise); Tests/RAG/simplified/test_search_service.py (new).
<!-- SECTION:NOTES:END -->
