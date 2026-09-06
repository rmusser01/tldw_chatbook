---
id: TASK-31914
title: Watchlists failure-policy test bypasses the live check coordinator
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 00:46'
updated_date: '2026-09-05 01:24'
labels:
  - watchlists
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Watchlists unexpected check-failure UI test hermetic and discriminating after local Check now moved behind the operation coordinator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unexpected-failure test intercepts the current local Check now execution boundary.
- [x] #2 The test records a warning-or-higher failure without attempting network egress.
- [x] #3 The complete Watchlists check-now failure-policy module passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the CI teardown error and trace the request from the button handler to the current local execution owner.
2. Update the test to inject its failure at the operation-coordinator boundary and assert the legacy controller is not reached.
3. Run the focused regression, the complete failure-policy module, Ruff, and git diff checks.
4. Record implementation evidence and complete the task.

ADR required: no
ADR path: N/A
Reason: hermetic test maintenance for an existing Watchlists execution boundary; no runtime architecture, storage, security, service contract, or long-lived UX decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Patched the current `watchlists_operation_coordinator.accept_checks` boundary and asserted the accepted source IDs, replacing a stale patch of the bypassed legacy controller method.
- This keeps the unexpected-exception policy test hermetic; CI had recorded real CloudFront connection attempts during teardown when the stale patch allowed the seeded URL through.
- Evidence: the exact regression passes and the complete module passes 27/27; Ruff and diff checks pass.
- ADR required: no; runtime behavior is unchanged.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31758 was renumbered to TASK-31914 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
