---
id: TASK-31686
title: Refresh MCP gateway fixtures for current provider and RAG seams
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:27'
updated_date: '2026-09-05 18:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep public-profile schema compilation and private keyword-search failure tests exercising the current local tool catalog and shared RAG acquisition seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All public protocol profiles compile the exact current local provider schemas with no excluded tools.
- [x] #2 Keyword search failure reaches the real gateway with the private canary redacted through the existing fallback boundary.
- [x] #3 Both complete gateway test files pass with scoped static checks and no production changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the six failures and verify current external tool registration and ADR084 shared RAG acquisition. 2. Pin the exact current20tool schema names (not a vacuous count) across every protocol profile. 3. Patch get_shared_rag_service at its actual use seam to return unavailable, retaining the real keyword/gateway fault translation and all privacy assertions. 4. Run both complete gateway files and static checks. ADR required: no. ADR path: backlog/decisions/084-mcp-profile-driven-rag-search-contract.md (existing). Reason: test-only updates to already shipped interfaces; no catalog/runtime authority change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced obsolete17count with exact20name catalog equality across all5protocol profiles, including the current fs_patch and Watchlists receipt tools. Changed only the private-search injection from retired create_rag_service to current get_shared_rag_service, keeping real keyword fallback, gateway translation, exact call and private-canary assertions. Baseline6failed137passed; final143passed20.95s (/private/tmp/tldw-review-mcp-gateway-final-20260905.xml). Ruff, changed-range format, diff whitespace and self-review pass. Existing ADR084 shared-runtime contract; no new ADR/runtime authority.
<!-- SECTION:NOTES:END -->
