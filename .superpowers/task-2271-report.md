# task-2271: Fix search_rag always returning 0 results

## Root cause (confirmed)

`RAG_Search/simplified/search_service.py`'s `keyword_search` called
`self.media_db.search_media(query=, limit=, media_types=)` — this method does
not exist on `MediaDatabase`. The real API is
`Client_Media_DB_v2.MediaDatabase.search_media_db(search_query=..., media_types=...,
results_per_page=..., page=...) -> Tuple[List[Dict], int]`. The `AttributeError`
was swallowed by `except Exception: return []`, so every `search_rag` query
silently returned "0 results".

`semantic_search` falls back to `keyword_search` when no enhanced RAG service
is configured (the common case for the MCP-only path), so the fallback hit
the same broken call, and `semantic_search` had an identical swallow of its
own — both modes were broken, not just keyword mode.

Verified with a RED run: reverted the file to the original committed content
(`git show HEAD:...` → temp file → copied over the working file, no `git
stash` used), ran the new test suite, got 7/7 failures with the log line
`Error in keyword_search: 'MediaDatabase' object has no attribute
'search_media'` on every one. Restored the fix from a saved copy; reran — 7/7
(later 8/8) green.

## Real row keys found (vs. the imagined ones the old code used)

`search_media_db`'s `base_select_parts` (Client_Media_DB_v2.py ~1884-1903)
projects: `id, uuid, url, title, type, author, ingestion_date,
transcription_model, transcription_provenance_json, is_trash, trash_date,
chunking_status, vector_processing, content_hash, last_modified, version,
client_id, deleted` (plus `relevance_score` when FTS relevance sort is
active).

Two things the old code invented that don't exist:
- **`content`** — not selected by `search_media_db` at all (the row
  projection intentionally omits the large text column). Confirmed by
  `search_media_by_keyword_for_embedding` (Client_Media_DB_v2.py ~7250-7284),
  which does its own explicit "second query" via
  `get_media_by_ids_for_embedding` to fetch content after a
  `search_media_db` call — same pattern I reused.
- **`local_path`** — Media table (schema at Client_Media_DB_v2.py ~279-301)
  has no such column. The only source-reference column is `url`.

## Mapping fixes applied

- `keyword_search` now calls `search_media_db(search_query=query,
  media_types=media_types, results_per_page=limit)` and unpacks the
  `(rows, total)` tuple.
- `content`: batch-fetched in one extra query via
  `get_media_by_ids_for_embedding(media_ids)` (avoids N+1; mirrors the
  existing embedding-path precedent) and merged into each result by id.
- `url`: mapped from the real `item["url"]` (previously correct by luck).
- `file_path`: set to `None` with a comment explaining there is no distinct
  local-path column — duplicating `url` into it would be dishonest, and the
  outer key is kept only because `MCP/tools.py`'s `perform_rag_search`
  formatter reads `result.get("url") or result.get("file_path", "")`.
- `metadata`: trimmed to real columns (`author`, `ingestion_date`,
  `transcription_model`); dropped the old `created_at` key, which mapped to
  a column that doesn't exist on Media (only `ingestion_date` does).
- Outer result shape (`id`/`title`/`content`/`media_type`/`url`/`file_path`/
  `score`/`metadata`) kept byte-stable — `MCP/tools.py` consumes exactly
  these keys.

## Swallow fix

Both `keyword_search`'s and `semantic_search`'s `except Exception: ...
return []` became `except Exception: logger.error(...); raise` (plain
re-raise, no new exception type introduced — the module has none). Since
`semantic_search`'s fallback call to `keyword_search` sits inside its own
`try`, a raised error from the fallback now also propagates correctly through
`semantic_search`'s except → re-raise.

`MCP/tools.py:perform_rag_search` already catches into
`[{"error": str(e)}]` (task's stated blast-radius/design constraint), so this
lands directly in the existing honest error-rendering path — verified
end-to-end in a test.

## Module audit (AC#4)

Only one other `except` block exists in `search_service.py`:
`__init__`'s `except Exception: self.rag_service = None` around
`create_rag_service(profile_name=...)`, with a logged
"Falling back to basic search for MCP integration" message.

**Disposition: left as-is, not a bug in this class.** This is a
constructor-time capability-selection fallback (semantic RAG service
unavailable → degrade to keyword-only mode), not a crash-during-search
converted into a fake empty result. It already logs the failure and the
degraded mode is functional (keyword_search, now fixed, still returns real
results). This is a different category from the swallow this task targets,
so I did not touch it.

No swallows found outside this module were in scope per the task's stated
blast radius (only `MCP/tools.py` imports this service, and its own catch is
the intended honest-error boundary — not touched).

## TDD evidence

New file: `Tests/RAG/simplified/test_search_service.py` (8 tests), using a
real in-memory `MediaDatabase(":memory-equivalent tmp_path", client_id=...)`
— no DB mocks, per project policy.

- RED: reverted `search_service.py` to the original committed content (via
  `git show HEAD:... > /tmp/...`, plain file copy — `git stash` never used),
  ran `pytest Tests/RAG/simplified/test_search_service.py -q`:
  **7 failed** (the 8th test — the extra happy-path end-to-end one — was
  added after the RED/GREEN cycle for the other 7, see below), each failure
  showing the exact `'MediaDatabase' object has no attribute 'search_media'`
  error log line.
- GREEN: restored the fixed file; reran: **7 passed**. Then added one more
  test (`test_end_to_end_perform_rag_search_returns_real_results`, covering
  AC#1's happy path through the real MCP entry point) and reran the full
  file: **8 passed**.
- Protected oracle `Tests/MCP/test_rag_search_tool.py`: run unmodified,
  **2 passed** both before and after — it stubs at the
  `SimplifiedRAGSearchService` interface level (its own `_StubRAGSearchService`
  class), not at `media_db`, so it never exercised the buggy call path. **No
  stub extension was needed or made.**
- `Tests/MCP/` full directory: **417 passed** (includes
  `test_mcp_tools_constructs_against_real_dbs`, confirming the real
  `SimplifiedRAGSearchService.__init__` + `MediaDatabase` construction path
  still works after the change).
- `pytest Tests/ --collect-only -q`: **29915 tests collected, 0 errors**.

## Files changed

- `tldw_chatbook/RAG_Search/simplified/search_service.py` — `keyword_search`
  rewritten (real API call, tuple unpack, real-key mapping, batch content
  fetch); both `keyword_search` and `semantic_search` now re-raise instead
  of swallowing.
- `Tests/RAG/simplified/test_search_service.py` — new, 8 tests.
- `backlog/tasks/task-2271 - ...md` — status In Progress (already set by
  controller before this session), AC#2/#3/#4 checked, Implementation Notes
  added. AC#1 left unchecked (unit-level equivalent exists via
  `test_end_to_end_perform_rag_search_returns_real_results`, but no
  live-app/live-profile run was done in this worktree).

## Concerns / left for controller

- AC#1 wording ("against a profile") reads as a live-verification
  requirement; I did not launch the app (pytest-only per standing rules).
  The unit test covers the same code path end-to-end through
  `MCPTools.perform_rag_search`, but if live verification is required for
  merge, that's still open.
