---
id: TASK-31906
title: Separate historical migration assertions from current schema upgrades
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:50'
updated_date: '2026-09-05 19:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep migration regressions precise after newer conversation and workspace schema versions land.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The v27-to-v28 authority test asserts its exact one-column change at v28 and verifies authority values survive the subsequent current-schema upgrade.
- [x] #2 Historical Workspace upgrades reach every current migration version while preserving existing data and strict receipt schema assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce full-file migration failures and identify later-schema additions. 2. Use the existing historical bootstrap context for the exact v28 contract, then reopen current and recheck authority; update Workspace end-to-end version assertions to the current constant and contiguous complete version history. 3. Run both complete files and historical-bootstrap tests; lint/format changed regions and review. ADR required:no. ADR path:N/A. Reason:test-only use of existing schema bootstrap and migration chains, no schema or migration behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four complete-file baseline failures (4failed/57passed) were outdated schema targets. The authority test now runs exact v27→v28 through the existing historical bootstrap, retaining the one nullable TEXT column, exact proven/local versus unproven/server backfill and no-sync assertions, then reopens current and rechecks values/no sync. Workspace historical upgrades assert current target and the entire contiguous migration ledger, preserving all receipt-column/payload, quarantine, rollback and unrelated-row checks. Three complete files including historical bootstrap:122passed32.08s, XML /private/tmp/tldw-31743-migration-current-target-fixed.xml. No migration/runtime change or new ADR. Full-file lint, changed-region formatting and whitespace checks pass; self-reviewed exact intermediate and final-stage assertions.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31743 was renumbered to TASK-31906 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
