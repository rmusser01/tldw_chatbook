---
id: TASK-16222
title: Expose local Watchlists search tools to Console and MCP
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 14:18'
updated_date: '2026-08-14 22:41'
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
- [x] #1 Console and approved external MCP clients can search local Watchlists items with optional full-text terms, collection/source scope, statuses, date floor, bounded page size, and continuation; absent or incompletely backfilled FTS uses the complete literal LIKE fallback.
- [x] #2 Search results are newest-first, source-linked, collection-aware, date-explicit, byte-bounded valid JSON with match-centered snippets and clearly delimited untrusted evidence.
- [x] #3 Users can retrieve bounded detail for one canonical local Watchlists item returned by search.
- [x] #4 Human-readable source and collection scopes resolve exact and unique partial matches and return bounded disambiguation candidates when ambiguous.
- [x] #5 Continuation uses stable keyset ordering and rejects malformed or mismatched cursors without claiming full snapshot isolation.
- [x] #6 Server Watchlists mode returns an explicit non-retryable unsupported response and performs no local search.
- [x] #7 Both tools reuse the existing local-tool permission, kill-switch, approval, definition-hash, and MCP exposure gates and perform no Watchlists domain mutations.
- [x] #8 Automated and isolated live verification cover retrieval, validation, safety labeling, Console registration, MCP registration, and local/server behavior.
- [x] #9 Standalone external MCP opens only an existing subscriptions database through a registered read-only SQLite path and never creates the database file, writes rows/schema, or runs migrations; failed lazy candidates close before retry.
- [x] #10 Tool output is field-allowlisted, emits only absolute HTTP(S) URLs after stripping userinfo/query/fragment, labels URL transformation, and never exposes raw exception payloads, auth/header fields, or extracted raw records; preserved URL paths remain permission-gated Watchlists metadata rather than being claimed credential-free.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the registered read-only SubscriptionsDB construction path and prove it cannot initialize, migrate, create the database, or write the main database file, schema, or rows. A SQLite WAL reader may create or update SQLite-managed `-wal`/`-shm` coordination sidecars; byte-stable sidecars are not part of this logical read-only contract.
2. Extend existing Watchlists DB search/detail/resolution seams for literal FTS with complete LIKE fallback, keyset continuation, batched memberships, and bounded scope resolution.
3. Build the shared synchronous WatchlistsToolService with authoritative validation, runtime-source handling, canonical IDs, output allowlisting, cursor encoding, URL sanitization, untrusted-evidence labels, and strict byte-bounded JSON.
4. Register both read-only specs through LocalToolProvider and prove normal agent discovery plus existing permission gates.
5. Inject the app-owned database in Console and a lock-protected lazy read-only view in external MCP, including exposure-gate and gateway pass-through tests.
6. Correct settings/UI/operator documentation for the expanded workspace, web, and Watchlists permission boundary.
7. Run focused, regression, isolated subprocess live, static/format, full-suite, privacy, and self-review checks before completing TASK-16222.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 owns the amended private Watchlists-data permission and external MCP boundary; ADR-030 supplies the local domain-tool precedent.

Detailed plan: Docs/superpowers/plans/2026-08-14-watchlists-agent-search-tools.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a registered read-only subscriptions-database view, bounded Watchlists search/detail/scope queries, and the shared `WatchlistsToolService`; composed both tools into the Console provider and the external MCP gateway without adding a parallel permission path.
- Search uses stable newest-first keyset continuation, complete literal fallback when FTS is absent or incomplete, batched collection membership, canonical IDs, centered snippets, and strict valid-JSON byte ceilings. Detail and disambiguation use the same bounded, field-allowlisted envelope.
- Security and privacy boundaries include explicit untrusted-evidence labeling, terminal-control stripping, sanitized absolute HTTP(S) URLs with userinfo/query/fragment removed, generic error envelopes, no raw records or auth/header fields, explicit server-mode unsupported results, and existing-file-only read-only external database opening with failed-candidate cleanup.
- Expanded the shared local-tool consent label and operator documentation so filesystem, web, and Watchlists tools remain governed by the existing kill switch, approval, definition-hash, runtime-policy, and MCP exposure gates described by [ADR-032](../decisions/032-local-agent-tool-permission-boundary.md). [ADR-030](../decisions/030-local-library-agent-tools.md) supplies the local-domain-tool precedent; no new ADR was needed.
- Production changes span `DB/{base_db,Subscriptions_DB,private_sqlite}.py`, `Tools/watchlists_tool_service.py`, the local provider and Console/MCP composition seams, config and MCP workbench permission copy; corresponding DB/service/provider/Console/MCP/UI tests and user/operator documentation were added or updated.
- Fresh impacted verification on final HEAD: focused eight-module suite `498 passed` in 16.07s; four-module regression suite `349 passed, 1 skipped` in 37.92s (the skip is the expected Windows-only functional posture); isolated clean-process live QA `1 passed` in 1.77s; post-fix service/provider/workbench suites `159 passed`, `154 passed`, and `32 passed`; scoped Ruff check passed, Ruff format reported all 19 files already formatted, and `git diff --check` passed. The whole-branch reviewer reported Ready: Yes with no findings.
- Live QA set config and XDG paths before child imports, proved every resolved path stayed under a scratch root, exercised the real Console provider and external gateway under explicit Allow, and covered search/continuation/detail/disambiguation/source switching/server no-open/distinct timestamps/semantic equivalence. It also proved C1 sanitization, preserved and labeled hostile evidence, redacted URLs, logical read-only behavior, and unchanged real runtime-policy/subscriptions fingerprints; the temporary harness was deleted afterward.
- Plan deviation: the user explicitly limited final verification to touched and impacted functionality. The in-progress full-repository pytest run was stopped at 17%, its unrelated failures were intentionally not classified, and it is excluded from completion evidence; all requested impacted gates passed.
<!-- SECTION:NOTES:END -->
