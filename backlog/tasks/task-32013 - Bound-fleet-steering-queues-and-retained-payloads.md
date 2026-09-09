---
id: TASK-32013
title: Bound fleet steering queues and retained payloads
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
Valid steering messages can accumulate without an aggregate bound and unread entries bypass the retained transcript size cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queue admission atomically enforces entry and character limits without losing accepted messages or splitting FIFO ordering.
- [x] #2 Retention measures the complete transcript plus unread steering and refuses an oversized payload without truncating native tool pairs.
- [x] #3 Both producers explain a full queue accurately and preserve a refused user draft.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the scoped test-first steps in Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md for TASK-32013. Run the named targeted suites, self-review the final diff, and record fresh verification before completion.

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted fleet reliability decision; no additional ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded pending steering to 32 entries and 64,000 text characters under the coordinator lock. Retention now measures transcript plus unread steering. Full queues return an honest service refusal; the Console preserves the draft and reveals the refusal note after layout. Updated coordinator, agent service, steering widget/handler, concurrent mailbox tests, regression tests, and user guide.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no additional ADR. Implemented backlog/decisions/129-fleet-mailbox-and-wake-reliability.md without broadening orchestration authority.
<!-- SECTION:NOTES:END -->
