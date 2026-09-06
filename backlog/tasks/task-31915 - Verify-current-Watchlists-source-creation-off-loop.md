---
id: TASK-31915
title: Verify current Watchlists source creation off loop
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 00:53'
labels:
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The off-loop regression observes a retired post-insert getter. Prove the current
atomic batch lookup, insert and result materialization stay off the event loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real file-backed creation and exact-source reuse execute the batch transaction off loop and return the durable source.
- [x] #2 An inline-offload mutation fails the updated guard, without loosening nonempty thread assertions.
- [x] #3 Complete affected Watchlists files and scoped lint pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the reproduced missing-getter failure and inspect the exact batch owner.
2. Observe the real batch and transaction boundary for new and existing sources;
   keep insertion and returned identity/data assertions, with test-owned cleanup.
3. Run complete off-loop and service files, process-local inline mutation, and review.

ADR required: no
ADR path: N/A
Reason: Test-only repair reflecting the existing atomic source-creation contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the retired add/get expectation with real batch, transaction and result-materialization thread observations for both creation and exact-source reuse. Keeps positive work assertions, verifies the actual source_id projection against the durable row and checks no insert on reuse. Function-scoped loop executor shutdown allows thread-owned SQLite cleanup before asserting zero remaining handles. Original complete two-file baseline:41 passed/2 failed. Final five complete off-loop/service/project-context/atomic-promotion files:143 passed,2 existing dependency warnings in28.90s. XML:/private/tmp/tldw-offloop-promotion-final.xml. A process-local inline-offload mutation failed both new variants at the off-loop assertion. Whole changed-test Ruff/format and diff checks pass; independent review clear. No production Watchlists changes or new ADR; follows existing exact batch contract.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31796 was renumbered to TASK-31915 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
