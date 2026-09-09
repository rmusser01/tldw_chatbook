---
id: TASK-32015
title: Restore fleet wake safety test setup
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 05:29'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three wake safety tests stop at project binding preflight because the imported controller helper does not bring its module-scoped legacy-session fixture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three existing wake safety tests reach their original approval and authority assertions and pass without a command-line fixture patch.
- [x] #2 Production project-instruction defaults and preflight behavior remain unchanged.
- [x] #3 The headless wake dispatch/budget test reaches both original manual and automatic dispatch assertions using an explicit legacy-session setup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Preserve the completed safety-module fixture repair. For the newly reproduced headless dispatch precondition failure, first verify that the existing legacy-session fixture alone restores the unchanged assertions, then set that one test session to legacy-disabled project instructions. Run the headless and wake-safety modules, compare scoped lint/format, and append evidence. ADR required: no. ADR path: N/A. Reason: test setup repair; no production preflight or authority behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Imported the existing legacy-session fixture into the wake safety module so the helper-created sessions reach the original approval and authority checks. All three original tests pass without a command-line fixture workaround. Production defaults and project-instruction preflight were not changed.

Verification: combined targeted fleet/runtime/continuation/wake/ledger/steering UI run: 295 passed in 101.23s. After final defensive-copy refinement: 84 affected coordinator/continuation/mailbox tests passed. Stronger compositor checks: 3 passed. Counts overlap. New regression and safety modules pass Ruff lint, new modules pass formatting, edited legacy modules add no Ruff diagnostics, scoped diff whitespace checks and diagnostic-inventory checks pass. Self-review completed. Existing unrelated lint findings were preserved. No full suite or live provider run. Expanded-review semaphore failures are documented separately in backlog/docs/agent-orchestration-review-2026-09-07.md.

Documentation: backlog/docs/agent-orchestration-review-2026-09-07.md and Docs/superpowers/plans/2026-09-07-fleet-reliability-repairs.md. User-facing mailbox and retry behavior is documented in Docs/User_Guide/console/agent-runs-and-tools.md.

ADR required: no. ADR path: N/A. Test-fixture repair only.

2026-09-08 follow-up: the headless agent-dispatch/budget test failed at its manual-send precondition with binding_unavailable. Loading only the existing legacy-session fixture diagnostically restored the original assertions. The test now explicitly selects ProjectInstructionControlState.legacy_disabled() for its own recording-bridge session. All manual/wake dispatch, cancellation, authority, and budget assertions are preserved; production defaults/preflight are unchanged.

The final combined targeted budget/fleet/headless/safety run passes 168 tests in 46.70 seconds, including all 16 headless/safety cases. The modified legacy module adds no Ruff findings versus its starting tree; its changed range is formatted and scoped whitespace checks pass. Updated the orchestration review ledger. No full suite or live provider used. ADR required: no. ADR path: N/A. This is a test-setup correction within the existing task.
<!-- SECTION:NOTES:END -->
