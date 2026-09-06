---
id: TASK-31903
title: Close agent-swap fixture-owned runtime and database resources
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:35'
updated_date: '2026-09-05 19:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent real agent-swap test fixtures from retaining SQLite descriptors after teardown while preserving real controller and durable-store behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every fixture-created controller reaches its supported shutdown boundary before its exact owned databases are closed.
- [x] #2 The representative real send leaves zero fixture-file descriptors after teardown instead of five, and the complete agent-swap file no longer emits the descriptor-growth warning.
- [x] #3 Existing agent-swap assertions, real constructors and class identities remain unchanged, with scoped static checks and independent review complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review the already-verified root regeneration diff before touching the shared agent-swap test file.
2. Add module-local async ownership tracking for fixture-created real controllers and databases, with controller shutdown followed by exact-file ChaChaNotes quiescence and AgentRuns close.
3. Measure attributed descriptor cleanup in the representative send and run the complete agent-swap file; preserve leak thresholds and existing assertions.
4. Run scoped static checks, record the descriptor and teardown-order incident in the existing testing lesson, and obtain independent review before a scoped commit.
ADR required: no
ADR path: N/A
Reason: Test-only lifecycle repair uses existing runtime/database ownership APIs; no production boundary, policy, schema, or dependency change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a module-local async lifecycle fixture that tracks real constructor-created controllers and exact temporary-root database files without replacing canonical class identities. Teardown awaits controller shutdown, quiesces all registered ChaChaNotes handles with an immediate zero-count assertion, closes fixture-thread AgentRuns handles, and releases tracking references before the existing cleanup fixture. No production changes, extra GC calls, threshold changes, or foreign-file drain. Existing testing lesson updated with the measured incident; no ADR required for this test-only lifecycle repair.
Evidence: the representative real send retained five fixture descriptors before repair and zero afterward; the root-owned regeneration fixture correction independently measured four to zero. Complete agent-swap file: 47 passed in 26.10 seconds, only the existing Requests dependency warning, no descriptor-growth warning; /private/tmp/tldw-31737-agent-swap-full.xml. Both descriptor-probed targets passed together. Whole-file Ruff check and format plus diff whitespace pass. Root independently reviewed the scoped fixture and approved it with no actionable findings.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31737 was renumbered to TASK-31903 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
