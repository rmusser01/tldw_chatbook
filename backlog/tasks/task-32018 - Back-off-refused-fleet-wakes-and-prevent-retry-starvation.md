---
id: TASK-32018
title: Back off refused fleet wakes and prevent retry starvation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 03:23'
labels:
  - agents
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An unavailable provider retried 183 times in 150 ms; in a two-session probe an unavailable first session retried 143 times while the ready second session received no attempt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persistent refusal has a bounded automatic retry rate and preserves pending results and delivery marks.
- [x] #2 A refused conversation cannot starve a ready conversation behind it.
- [x] #3 Existing manual-send priority, approval boundaries, serialization, retry-on-recovery, and disposed-controller behavior remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the scoped test-first steps in Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md for TASK-32018. Run the named targeted suites, self-review the final diff, and record fresh verification before completion.

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted fleet reliability decision; no additional ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refused or raised wake submissions now receive a one-second per-conversation retry deadline and a single delayed timer. Ready conversations can proceed during another conversation's delay. Real-controller tests pin one attempt before the delay, eventual recovery without an external poke, ready-second-session progress, and no retry after disposal. Existing wake safety, manual-send precedence, serialization, ledger, staleness, and view-mark tests pass. Added the observed retry-storm testing lesson.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no additional ADR. Implemented backlog/decisions/129-fleet-mailbox-and-wake-reliability.md without broadening orchestration authority.
<!-- SECTION:NOTES:END -->
