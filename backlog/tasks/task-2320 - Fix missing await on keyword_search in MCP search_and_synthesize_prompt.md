---
id: TASK-2320
title: Fix missing await on keyword_search in MCP search_and_synthesize_prompt
status: To Do
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

- [ ] `search_and_synthesize_prompt` awaits `keyword_search` and produces a real prompt containing retrieved sources when the media DB has matches.
- [ ] A regression test drives the real code path (real in-memory MediaDatabase, no mocks of the search), pinning that the result is not the "Error creating prompt" fallback.
- [ ] Sweep of remaining `rag_service.*` call sites for un-awaited coroutines, with findings fixed or filed.
