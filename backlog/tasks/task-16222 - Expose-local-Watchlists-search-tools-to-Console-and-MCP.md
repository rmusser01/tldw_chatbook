---
id: TASK-16222
title: Expose local Watchlists search tools to Console and MCP
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 14:18'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-14-watchlists-agent-search-tools-design.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users ask the Console agent and approved external MCP clients evidence-backed questions about recent Watchlists items, sources, and collections without leaving the agent workflow. Reuse the local Watchlists corpus search and preserve the existing local-tool permission boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console and approved external MCP clients can search local Watchlists items with optional full-text terms, collection/source scope, statuses, date floor, bounded page size, and continuation; absent or incompletely backfilled FTS uses the complete literal LIKE fallback.
- [ ] #2 Search results are newest-first, source-linked, collection-aware, date-explicit, byte-bounded valid JSON with match-centered snippets and clearly delimited untrusted evidence.
- [ ] #3 Users can retrieve bounded detail for one canonical local Watchlists item returned by search.
- [ ] #4 Human-readable source and collection scopes resolve exact and unique partial matches and return bounded disambiguation candidates when ambiguous.
- [ ] #5 Continuation uses stable keyset ordering and rejects malformed or mismatched cursors without claiming full snapshot isolation.
- [ ] #6 Server Watchlists mode returns an explicit non-retryable unsupported response and performs no local search.
- [ ] #7 Both tools reuse the existing local-tool permission, kill-switch, approval, definition-hash, and MCP exposure gates and perform no Watchlists domain mutations.
- [ ] #8 Automated and isolated live verification cover retrieval, validation, safety labeling, Console registration, MCP registration, and local/server behavior.
- [ ] #9 Standalone external MCP opens only an existing subscriptions database through a registered read-only SQLite path and never creates the database file, writes rows/schema, or runs migrations; failed lazy candidates close before retry.
- [ ] #10 Tool output is field-allowlisted, emits only absolute HTTP(S) URLs after stripping userinfo/query/fragment, labels URL transformation, and never exposes raw exception payloads, auth/header fields, or extracted raw records; preserved URL paths remain permission-gated Watchlists metadata rather than being claimed credential-free.
<!-- AC:END -->
