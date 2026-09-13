---
id: TASK-31590
title: Repair Console side-chat modal regressions
status: Done
assignee: []
created_date: '2026-09-05 05:23'
updated_date: '2026-09-05 05:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair current Console side-chat streaming, cancellation, and provider-error regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced side-chat modal failures pass
- [x] #2 Console side-chat modal module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and inspect the side-chat provider failures against the current auxiliary gateway contract. 2. Correct stale harness seams or production behavior with the smallest justified change. 3. Run focused regressions and the full side-chat modal module. ADR required: no. ADR path: N/A. Reason: this is localized regression repair within an existing modal/provider boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned both side-chat gateway doubles with the current keyword-only provider route
contract. The headless service and modal suites pass together with 38 tests. ADR
required: no; production side-chat behavior was already correct and only the test
harness contract had drifted.
<!-- SECTION:NOTES:END -->
