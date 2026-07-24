---
id: TASK-557
title: Integrate citation foundation with current dev schema
status: In Progress
assignee: []
created_date: '2026-07-24 22:25'
updated_date: '2026-07-24 22:25'
labels:
  - rag
  - citations
  - integration
dependencies:
  - TASK-553.4
  - TASK-553.12
  - TASK-556
references:
  - Docs/superpowers/plans/2026-07-24-rag-citation-foundation-dev-integration.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the completed citation provenance foundation with current dev so its database migration, persistence seams, shared test setup, and generated styles are merge-ready for a pull request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Citation provenance migrates current dev schema v26 to schema v27 without overwriting message-generation metadata or Console rewind migration ownership
- [x] #2 Combined chat persistence and database initialization preserve both current dev behavior and citation atomicity
- [x] #3 Shared test setup and generated CSS contain both branches' intended behavior with no conflict markers or stale bundle state
- [x] #4 Citation foundation, migration, database, UI maturity, static, and qualification gates pass on the integrated branch
- [x] #5 The branch is pushed and a ready pull request targets dev with accurate verification and limitation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Merge current origin/dev once with --no-commit and preserve the five aggregate conflict sites for deliberate resolution.
2. Keep dev schema v24→v25 message-generation metadata and v25→v26 conversation summaries, then renumber citation provenance to v26→v27 across migration SQL, database dispatch, tests, and documentation.
3. Combine current dev chat persistence and test-environment behavior with citation atomicity and test-database isolation; regenerate the CSS bundle from its merged source modules.
4. Run conflict-focused RED/GREEN tests, the citation foundation and DB gates, UI maturity regressions, qualification, static checks, and an independent review.
5. Commit the dev integration, push the feature branch, and create a ready pull request against dev with accurate verification and limitations.

ADR required: no new ADR
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: ADR-024 already owns the citation storage and persistence contract; advancing its migration to the next free version and combining current dev behavior is an anticipated mechanical integration, not a new architecture decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged `origin/dev` at `e00b2930` into the citation foundation and resolved the
five preflighted aggregate conflicts. Current `dev` retains ownership of schema
v25 message-generation metadata and schema v26 Console summaries; canonical
citation provenance now migrates v26→v27 through its transaction-safe standalone
SQL runner. Migration dispatch restores a real SQLite transaction after legacy
`executescript` steps, so chained v24→v27 failures cannot leak citation DDL.
Chat message creation now combines generation metadata with sealed
citation writes atomically and validates both on uncertain retry. Shared test
isolation retains call-time HOME handling, lazy RAG pre-arm, and database/prompt
singleton cleanup, while the CSS bundle was regenerated from merged sources.

Verification:
- Citation foundation: 768 passed.
- ChaChaNotes/DB: 355 passed.
- Persistence and environment isolation: 156 passed.
- CSS/UI source and parity gates: 138 passed; CSS bundle guards: 10 passed.
- UI maturity: 87 passed.
- Qualification: eligible with `overall_pass=true`.
- Ruff: passed for Chat, Chatbooks, DB, and their citation/performance tests.
- Independent review: approved with no blocking findings; its focused suite
  passed 82 tests. Suggested rollback-sidecar coverage and stale comments were
  addressed, with 16 affected tests passing afterward.

An additional repository-wide pytest run was deferred because another checkout
has had a long-running full-suite process active throughout this integration;
the earlier branch-wide attempt was not counted because concurrent runs produced
non-reproducible setup failures. The scoped foundation, database, persistence,
CSS, and UI gates above are clean.

ADR decision: no new ADR required. Existing ADR-024 defines the citation storage,
identity, governance, and atomic persistence boundaries used by this integration.

Publication:
- Pushed `codex/rag-citation-provenance-foundation` to `origin`.
- Opened ready pull request
  [#853](https://github.com/rmusser01/tldw_chatbook/pull/853) against `dev`.
- The task remains In Progress because repository Definition of Done requires an
  uncontaminated full-suite pass; that gate remains deferred for the concurrent
  full-suite reason documented above.
<!-- SECTION:NOTES:END -->
