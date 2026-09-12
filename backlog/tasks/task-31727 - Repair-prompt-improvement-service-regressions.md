---
id: TASK-31727
title: Repair prompt-improvement service regressions
status: Done
assignee: []
created_date: '2026-09-05 04:47'
updated_date: '2026-09-05 04:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore prompt-improvement regression coverage against the current auxiliary gateway and routing contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced prompt-improvement failures pass
- [x] #2 Affected prompt-improvement test module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current prompt-improvement module failures. 2. Classify failures as production regressions or stale test-double contracts and make the smallest justified correction. 3. Run focused regressions and the full affected module. ADR required: no. ADR path: N/A. Reason: this is a localized regression repair within an existing service contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the prompt-improvement gateway protocol and its test double with the current
keyword-only auxiliary request route. The contract test now verifies that `route` is
keyword-only and defaults to `None`. The full prompt-improvement service module passes:
103 tests passed. ADR required: no; this corrects type and harness drift within the
existing auxiliary completion boundary.
<!-- SECTION:NOTES:END -->
