---
id: TASK-31701
title: Align saved trace-admission expectation with recoverable pause contract
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:46'
updated_date: '2026-09-05 18:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve durable-authority coverage while recognizing the recoverable manual-send trace admission pause introduced by the current controller contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save-and-Send preserves a new attempt and durable library authority before trace admission
- [x] #2 A pre-dispatch trace reservation failure remains retryable without entering the provider or duplicating the user row
- [x] #3 The full controller exchanges file passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only repair for the established trace-call recovery state machine.
1. Reproduce the failure and inspect the failure handler.
2. Align the expected preparation state and visible recovery choice while retaining durable-authority and side-effect assertions.
3. Run the complete exchanges file and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the stale accepted-state expectation to the explicit recoverable TRACE_CALL pause for a manual pre-dispatch reservation failure. Retained new-attempt, durable-authority, no-provider-entry and exactly-one-user-row assertions; added pause-kind assertion. No production change. The original node failed with PAUSED vs ACCEPTED; full exchanges file now 47 passed in 2.53s. Ruff lint and changed-region formatter checks passed; self-review complete. ADR not required (test-only existing recovery contract).
<!-- SECTION:NOTES:END -->
