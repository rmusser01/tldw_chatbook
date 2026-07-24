---
id: TASK-553.11
title: Make legacy world-book migration fixtures pre-v25 accurate
status: Done
assignee: []
created_date: '2026-07-24 16:36'
updated_date: '2026-07-24 16:39'
labels: []
dependencies: []
parent_task_id: TASK-553
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep older ChaChaNotes migration tests valid after the citation-provenance schema addition by ensuring simulated pre-v25 databases do not retain future citation tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 World-book v20 and v21 migration fixtures remove the citation-provenance schema before version rollback
- [x] #2 Citation migration remains fail-closed for pre-existing or partial provenance tables
- [x] #3 Focused world-book and citation migration tests pass
- [x] #4 DB verification distinguishes confirmed base failures from branch regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the two failing legacy world-book migration fixtures. 2. Reuse a shared test helper to remove future citation tables before rolling schema versions back. 3. Re-run focused world-book and citation migration tests plus the DB slice. 4. Document ADR rationale, verification, and confirmed base-only residual failures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the simulated v20 and v21 world-book databases to drop all twelve v25 citation-provenance tables in dependency-safe order before rolling back the recorded schema version. This keeps the fixtures historically accurate while preserving the production v24-to-v25 migration's deliberate fail-closed handling of pre-existing or partial tables. The established inline test pattern was copied locally instead of adding a cross-test helper or production abstraction. ADR required: no; ADR path: N/A; Reason: test-only fixture correction with no change to schema, migration policy, storage, or runtime boundaries. TDD evidence: both focused tests failed before the change because rag_identity_context already existed, then passed after the fixture correction. Verification: focused tests 2 passed; all requested world-book and citation-provenance migration tests 18 passed; Ruff lint and format checks passed; git diff --check passed. The three confirmed merge-base legacy-conversation fixture failures were intentionally not modified or included in this scoped gate.
<!-- SECTION:NOTES:END -->
