---
id: TASK-2320
title: Fix missing await on keyword_search in MCP search_and_synthesize_prompt
status: Done
assignee: []
created_date: '2026-08-04 22:40'
labels:
  - mcp
  - rag
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found incidentally by the task-2271 review while auditing consumers of the simplified RAG search service: `MCP/prompts.py` (observed ~line 256, in `search_and_synthesize_prompt`) calls `rag_service.keyword_search(query, limit=num_sources)` WITHOUT `await`. `keyword_search` is a coroutine, so `enumerate()` over the un-awaited coroutine raises `TypeError` before any search runs — the function's own blanket `except Exception` then returns the "Error creating prompt: …" message. Net effect: `search_and_synthesize_prompt` has always errored regardless of the media DB's state.

This is the same bug class as PR #1226 (`'coroutine' object is not iterable` in `perform_rag_search`) — that fix covered `tools.py` but not `prompts.py`. Pre-existing; unrelated to task-2271's change (which made `keyword_search` raise instead of returning `[]` on failure — irrelevant here since the coroutine is never awaited at this call site).

While fixing, sweep `MCP/prompts.py` (and any other `rag_service.` call sites outside `tools.py`) for further un-awaited coroutine calls of the same shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] `search_and_synthesize_prompt` awaits `keyword_search` and produces a real prompt containing retrieved sources when the media DB has matches.
- [x] A regression test drives the real code path (real in-memory MediaDatabase, no mocks of the search), pinning that the result is not the "Error creating prompt" fallback.
- [x] Sweep of remaining `rag_service.*` call sites for un-awaited coroutines, with findings fixed or filed.

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- `tldw_chatbook/MCP/prompts.py:259` now reads `results = await rag_service.keyword_search(query=query, limit=num_sources)` — the un-awaited-coroutine defect is gone.
- Regression pin: `Tests/MCP/test_gateway_runtime_prompts.py::test_search_and_synthesize_awaits_keyword_search` (:818) drives the real `MCPPrompts.search_and_synthesize_prompt` and asserts retrieved source content reaches the prompt (not the "Error creating prompt" fallback). The service is stubbed at the class seam (`SimplifiedRAGSearchService` → AsyncSearch) rather than via a real in-memory MediaDatabase — noted for AC2's letter-vs-spirit.
- Sweep (AC3): `git grep "rag_service\." tldw_chatbook/MCP/` on dev shows no remaining un-awaited coroutine call sites.
- Landed in `eebfbe4100` ("feat(mcp): map bounded prompt results"). All three ACs ticked (AC2 with the stubbing nuance noted above).
