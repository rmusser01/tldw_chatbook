---
id: TASK-32487
title: Isolate nested retained fleet transcripts from callers
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
Retained transcript copies are shallow; modifying a nested native tool call through a returned snapshot corrupts stored history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mutating nested input messages after retention cannot change stored history.
- [x] #2 Mutating nested data returned from get_retained cannot change a later snapshot or continuation seed.
- [x] #3 Native tool call IDs and ordering remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the scoped test-first steps in Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md for TASK-32487. Run the named targeted suites, self-review the final diff, and record fresh verification before completion.

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted fleet reliability decision; no additional ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained transcripts are deeply copied at admission and read boundaries, preserving nested native tool-call IDs and ordering. Self-review added a failing test for uncopyable provider content; failed copying now refuses retention before claiming unread steering. Input mutation, returned-snapshot mutation, and failed-copy regressions pass with the existing continuation/mailbox suite.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no additional ADR. Implemented backlog/decisions/129-fleet-mailbox-and-wake-reliability.md without broadening orchestration authority.
<!-- SECTION:NOTES:END -->
