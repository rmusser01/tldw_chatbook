---
id: TASK-37144
title: Keep Console typing census independent of elapsed draft-pause timers
status: In Progress
created_date: 2026-10-28 04:00
assignee:
- '@codex'
labels:
- ci
- tests
- performance
priority: high
references:
- TASK-36390
- TASK-33260
- TASK-33005.2
- Tests/Performance/test_console_keystroke_work_census.py
- backlog/decisions/097-boot-budget-ratchets.md
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
updated_date: 2026-10-28 04:03
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR3283 ready-event Perf run40610052389 counted79 readiness builds for24 keys and exceeded the unchanged three-per-key bound, while its synchronize run passed on the identical merge b8e2b1e735e0a4d5dfef6d58f0e1a2c3b036b11b. Native caller tracing shows one build per key plus three for each elapsed draft-pause refresh. Keep typing metrics independent of runner speed while retaining paused-work coverage. This is a distinct residual phase-attribution gap after completed TASK-36390, not a new product performance feature or duplicate of TASK-37000.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The native per-key census remains within its existing positive and upper bounds under controlled elapsed pause intervals between real Pilot key dispatches; the new regression fails before the fix without setup/teardown errors or skips.
- [ ] #2 The existing storage-unit typing-pause, idle and revisit work remains exercised with unchanged callbacks and ceilings; phase separation does not discard its coverage.
- [ ] #3 Production behavior, dependencies and existing guard caps/timeouts/shard counts are unchanged; the controlled regression is enrolled in the existing Perf Guard step and targeted checks plus scoped Superpowers review qualify the repair. Actual current-head CI remains a separate PR delivery gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the false attribution with a controlled clock around native Textual timers: retain actual Console, coordinator callbacks and Pilot dispatch; inspect actual child XML. 2. Apply the existing storage-census typing/pause separation to all typing variants without raising the three-per-key bound or caching live readiness. 3. Rerun the new regression and two existing Perf nodes; prove the storage_units=True path/callbacks and its existing hosted pause/idle guard stay unchanged. 4. Enroll the new native control in the existing Perf step and run focused style/CI checks. 5. Obtain Superpowers scoped review and inspect every current-head hosted run; record exact evidence and limitations. ADR required: no. ADR path: N/A. Reason: bounded test-harness phase attribution and existing-lane enrollment only, no production/dependency/runtime policy change. Existing ADRs: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
