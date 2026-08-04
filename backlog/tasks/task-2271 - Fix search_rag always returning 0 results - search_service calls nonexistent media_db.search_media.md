---
id: TASK-2271
title: Fix search_rag always returning 0 results — search_service calls nonexistent media_db.search_media()
status: To Do
assignee: []
created_date: '2026-08-04 21:30'
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

- [ ] `search_rag` returns real results against a profile whose media DB contains matching content.
- [ ] A failure inside the search service surfaces as an error (the tool's error shape and/or a logged error with context), never as a silent empty-success.
- [ ] A regression test pins the media-DB method name actually called (a call-path test against the real DB API, not a mock that would accept any name).
- [ ] Audit of the same service for other swallowed-exception fallbacks that convert crashes into empty results, with findings fixed or filed.
