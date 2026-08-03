---
id: TASK-1337
title: Add direct local Library tools for Console agents and MCP
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 03:30'
updated_date: '2026-08-03 03:41'
labels:
  - library
  - agents
  - mcp
  - privacy
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md
  - backlog/decisions/030-local-library-agent-tool-boundary.md
documentation:
  - Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console agents and local MCP clients safe read-only access to factual local Library inventory and content without requiring semantic retrieval so users can count find and inspect their own Library items predictably.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console agents and local MCP expose list get and lexical search for Media Notes Prompts Skills Conversations and Collections
- [ ] #2 List and search return bounded pages with exact distinct totals stable IDs deterministic pagination and keyword matches
- [ ] #3 Get requires a returned stable ID and chunks large text with revision-aware continuation below 32 KiB
- [ ] #4 Direct tools never use RAG embeddings or semantic similarity and never return binary data or filesystem paths
- [ ] #5 The Console direct-tools setting defaults on and off mode exposes bounded Library RAG with visible cloud-model privacy copy
- [ ] #6 Console mode cannot be bypassed by built-in MCP overlaps while MCP client compatibility remains unchanged
- [ ] #7 Automated tests cover contracts trust boundaries MCP bootstrap Console integration and settings behavior
- [ ] #8 ADR-030 design and implementation documentation are linked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/030-local-library-agent-tool-boundary.md
Reason: durable cross-module Console/MCP read contract, direct-versus-RAG runtime boundary, stable-ID continuation, and cloud-model privacy behavior.

Implementation plan: Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md

1. Implement descriptor, ID, cursor, validation, and byte-bound contracts with tests.
2. Add exact text-only query seams for all six Library types.
3. Implement the shared 18-operation LocalLibraryToolService.
4. Integrate Console direct/RAG providers and the global privacy setting.
5. Add Console-only MCP overlap filtering and descriptor-backed MCP registration/delegation.
6. Verify cross-runtime parity, compatibility, documentation, and Definition of Done.
<!-- SECTION:PLAN:END -->
