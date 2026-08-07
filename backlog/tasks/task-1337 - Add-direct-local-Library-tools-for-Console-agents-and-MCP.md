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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**2026-08-06 progress (@kimi, branch feat/hub-local-agent-tools):** Plan Task 1 (shared contracts) landed: `tldw_chatbook/Library/library_tool_contract.py` — the 18-descriptor table (names, trust/read-only description tails, bounded input schemas, `type.operation` routes), opaque stable-ID codec (`type:<base64url>`, ASCII, ≤128 B, fail-closed parse with wrong-type/malformed/path-like rejection), versioned continuation-cursor codec (SHA-256-checked canonical payload, tamper → `invalid_argument`, revision mismatch → `content_changed` + fresh-start hint), the six spec error codes with JSON-safe `to_payload()`, page/max-chars/query validation, control-character display normalization (160 B cap, 32 B floor), and the 32 KiB byte fitting (`fit_page_payload` with the fixed keywords→preview→metadata trim order + `fit_text_segment` largest-prefix fitting with skip/repeat-free offsets). Tests: `Tests/Library/test_library_tool_contract.py` (~30 cases). `Library/__init__.py` exports the descriptor table and error type. Remaining: plan Tasks 2–10 (storage query seams per type, `LocalLibraryToolService`, Console provider + privacy setting, MCP registration/delegation, parity/docs). No ACs ticked yet — service behavior does not exist until Task 5.
<!-- SECTION:NOTES:END -->
