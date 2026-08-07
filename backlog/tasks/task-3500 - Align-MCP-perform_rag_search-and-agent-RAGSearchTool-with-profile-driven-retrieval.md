---
id: TASK-3500
title: >-
  Align MCP perform_rag_search and agent RAGSearchTool with profile-driven
  retrieval
status: To Do
assignee: []
created_date: '2026-08-07 20:34'
labels:
  - rag
  - mcp
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG-port P0 programme (TASK-3170) made Library's rag-mode search honor the active RAG profile's search mode (plain/semantic/hybrid), score-kind-aware match bands, a fixed reranker factory, and a rebuilt keyword leg -- but declared this a non-goal for the MCP surface: MCP's perform_rag_search tool and the agent runtime's RAGSearchTool still search however they always have, independent of the active profile. Library and MCP therefore currently disagree about what a 'rag search' means and can return different results, different match semantics, and different reranking behavior for the same query depending on which surface issued it. This task aligns the MCP and agent-tool retrieval paths with the profile-driven engine Library now uses, closing that gap. Related: TASK-694, TASK-1077 (both track other MCP/agent retrieval-parity gaps).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP perform_rag_search resolves and honors the active RAG profile's default_search_mode the same way Library's rag-mode search does, instead of a hardcoded search behavior
- [ ] #2 The agent runtime's RAGSearchTool resolves and honors the active RAG profile the same way
- [ ] #3 A query issued through MCP/RAGSearchTool and the same query issued through Library's Search/RAG canvas return results with the same match-strength semantics (score-kind-aware bands, not a fabricated similarity) for the same active profile
- [ ] #4 Reranking-enabled profiles apply reranking on the MCP/agent path exactly as they do on the Library path, with the same skip-on-unavailable disclosure rather than a hard failure
- [ ] #5 Existing MCP/agent RAG search callers continue to work without a breaking API change
<!-- AC:END -->
