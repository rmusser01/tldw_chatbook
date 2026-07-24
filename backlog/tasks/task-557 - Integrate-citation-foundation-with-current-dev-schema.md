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
- [ ] #1 Citation provenance migrates current dev schema v26 to schema v27 without overwriting message-generation metadata or Console rewind migration ownership
- [ ] #2 Combined chat persistence and database initialization preserve both current dev behavior and citation atomicity
- [ ] #3 Shared test setup and generated CSS contain both branches' intended behavior with no conflict markers or stale bundle state
- [ ] #4 Citation foundation, migration, database, UI maturity, static, and qualification gates pass on the integrated branch
- [ ] #5 The branch is pushed and a ready pull request targets dev with accurate verification and limitation notes
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
