---
id: TASK-16219
title: Isolate MCP import smoke tests from embedding download
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:58'
updated_date: '2026-08-14 01:01'
labels:
  - test-health
  - mcp
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep MCP import and real-database smoke tests focused on tool wiring without downloading an unrelated embedding model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP tool construction and real database operations run without external network access.
- [x] #2 The tests continue to exercise the real MCP tools, delegate, and database implementations under review.
- [x] #3 The complete module, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this isolates a test module at an existing constructor dependency; no MCP or embedding runtime behavior changes.

1. Preserve the four swallowed Hugging Face download attempts as RED evidence.
2. Replace only the unused RAG service dependency in this smoke-test module.
3. Run the complete module, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a module-scoped fixture that replaces only the unused `SimplifiedRAGSearchService` constructor while leaving the real MCP tools, runtime delegate, and temporary SQLite databases in place. Before the repair, four tests each made swallowed Hugging Face embedding-model attempts; afterward all eight module tests ran offline. Final chunk 24 passed 790 tests. Ruff lint/format and diff checks passed.
<!-- SECTION:NOTES:END -->
