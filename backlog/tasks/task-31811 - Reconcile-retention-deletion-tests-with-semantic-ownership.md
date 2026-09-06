---
id: TASK-31811
title: Reconcile retention deletion tests with semantic ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 03:12'
labels:
  - tests
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three deletion probes predate the semantic mutation guard and fail before
their retention assertions. Preserve the historical cascade guarantee and
verify current authorized deletion and fail-closed raw deletion separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current coordinated message hard deletion removes its sync-log body and entries durably, while raw deletion remains rejected.
- [x] #2 Rejected raw conversation deletion preserves the conversation, messages and sync-log proof after reopening.
- [x] #3 A genuine schema-46 database still verifies conversation-to-message cascade and sync-log purge; upgrading it does not restore deleted content.
- [x] #4 Complete affected files and related guard/migration/coordinator tests pass, with scoped static checks and independent review; production guards remain unchanged.
- [x] #5 Historical v44 retention-migration seeds use version-compatible writes while retaining original purge, rollback and FTS assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both complete files (3 failed/50 passed) and trace the ADR-097 coordinator and raw-SQL guards. Read TASK-19564 retention rationale and existing historical-bootstrap and coordinated-delete tests.
2. Route the current message deletion probe through SemanticRevisionCoordinator after proving raw deletion is denied. Change current conversation raw-delete probes to assert atomic preservation, not an unauthorized cascade. Retain the original cascade/purge evidence in a genuine schema-46 fixture and verify the purge survives current upgrade.
3. Run complete retention, properties, retention-migration, v57 guard and exchange-sidecar files. Mutation-check the historical purge assertion by removing only its fixture-owned retention trigger. Run lint, changed-region formatting, diff checks and independent review; update the review checkpoint.
4. The first six-file run exposed three v44 seed failures (118 passed/3 failed): current soft-delete now requires a semantic graph table absent in v44. Reconcile only historical fixture writes, including edit seeds that also route through current semantic coordination; keep the migration's original assertions intact and repeat the complete selection.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Test-only reconciliation with the existing authorized deletion contract; no new deletion API, persistence policy, schema or authority bypass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current message hard-delete coverage now uses the existing semantic revision
coordinator after proving raw deletion is rejected. Reopening verifies the
message, sync-log body and entries are gone. Current raw conversation-delete
coverage verifies exact durable parent, child and sync-proof preservation.
The original raw FK cascade/purge regression remains in a genuine schema-46
fixture and verifies that current-schema upgrade does not restore deleted data.
The historical probe uses SQL compatible with that schema, not today's getter.

The broader run exposed three additional historical-v44 seed failures: current
soft-delete requires the later semantic graph table. A test-only, schema-44
checked tombstone seed and versioned edit SQL now exercise real historical
triggers without invoking later semantic ownership. All original purge,
transaction rollback and FTS assertions are retained. No production code,
guard, schema or diagnostic change was necessary; ADR-097 remains authoritative.

Evidence: original two-file baseline 3 failed/50 passed
(`/private/tmp/tldw-retention-delete-baseline.xml`); initial expanded selection
118 passed/3 historical-seed failures (`/private/tmp/tldw-retention-delete-final.xml`).
Removing only the historical fixture's message-retention trigger made its
body-purge assertion fail (`/private/tmp/tldw-retention-cascade-mutation.xml`);
that negative control was restored before final verification. The complete
retention-migration file then passed all 7 tests. Independent review found no
actionable issues in both the original repair and historical-seed follow-up.

Final six complete retention, properties, retention-migration, v57 guard,
message-exchange and semantic-coordinator files: 121 passed in 51.94 seconds,
with two existing dependency warnings
(`/private/tmp/tldw-retention-delete-verified.xml`). Full-file Ruff lint,
changed-region formatting and diff checks pass. Formatting is scoped to avoid
unrelated baseline drift. Broader review failures remain open; this is not a
full-suite or merge-readiness claim.
<!-- SECTION:NOTES:END -->
