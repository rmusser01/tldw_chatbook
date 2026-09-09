---
id: TASK-32016
title: Bound unused fleet event retention across turns
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 03:23'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
FleetCoordinator produces started and finished events but no production consumer drains them; pruning all handles leaves the event history growing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated finish and prune cycles do not retain events for pruned handles.
- [x] #2 Events for surviving handles remain ordered and drain once; ordinary within-turn event behavior remains unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the scoped test-first steps in Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md for TASK-32016. Run the named targeted suites, self-review the final diff, and record fresh verification before completion.

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted fleet reliability decision; no additional ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Terminal pruning now removes those handles' undrained lifecycle events in the same critical section. The 100-cycle regression leaves only the surviving live handle's event, which drains once. Existing coordinator event tests pass. No durable run or bridge fanout semantics changed.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no additional ADR. Implemented backlog/decisions/129-fleet-mailbox-and-wake-reliability.md without broadening orchestration authority.
<!-- SECTION:NOTES:END -->
