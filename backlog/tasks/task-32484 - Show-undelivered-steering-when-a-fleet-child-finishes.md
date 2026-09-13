---
id: TASK-32484
title: Show undelivered steering when a fleet child finishes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:02'
updated_date: '2026-09-08 03:23'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A message accepted during the final model call can remain unread while the queue indicator disappears.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A terminal child with unread steering visibly reports the unread count and whether continuation can recover it.
- [x] #2 A completed final turn is not automatically restarted; normal delivered steering and approval behavior remain unchanged.
- [x] #3 A real runtime boundary probe and rendered row test distinguish delivered from unread steering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the scoped test-first steps in Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md for TASK-32484. Run the named targeted suites, self-review the final diff, and record fresh verification before completion.

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted fleet reliability decision; no additional ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Terminal snapshots now preserve the unread steering count and compute continuation availability from current retention. Fleet rows show the unread count and recovery guidance; queue acceptance copy explains that delivery needs another model turn. No automatic continuation was added. Real-loop final-call tests cover retained and unretained outcomes, and compositor tests verify the complete recovery text within its own widget region.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no additional ADR. Implemented backlog/decisions/129-fleet-mailbox-and-wake-reliability.md without broadening orchestration authority.
<!-- SECTION:NOTES:END -->
