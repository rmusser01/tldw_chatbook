---
id: TASK-16304
title: Isolate MCP conversation search from embedding download
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 01:03'
updated_date: '2026-08-14 01:05'
labels:
  - test-health
  - mcp
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep real MCP conversation-search accessor tests offline by replacing their unused semantic RAG dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation-search tests do not attempt an embedding-model download.
- [x] #2 The real MCP tool, FTS database accessors, preview query, and character filter remain exercised.
- [x] #3 The complete module, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this isolates two tests at an unused constructor dependency without changing MCP or embedding runtime behavior.

1. Preserve the two swallowed Hugging Face download attempts as RED evidence.
2. Replace the unused RAG service only for the two MCPTools search tests.
3. Run the complete module, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an opt-in fixture used only by the two `MCPTools.search_conversations` tests to replace their unused semantic RAG constructor. The tests continue through the real tool methods, SQLite FTS conversation/message accessors, preview selection, and character filter. Before the repair both tests made swallowed Hugging Face attempts; afterward all 14 module tests and all 480 chunk-25 tests passed offline. Ruff lint/format and diff checks passed.
<!-- SECTION:NOTES:END -->
