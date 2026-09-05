---
id: TASK-31729
title: Repair Console local citation boundary regressions
status: Done
assignee: []
created_date: '2026-09-05 04:51'
updated_date: '2026-09-05 05:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Console local citation-boundary coverage against current routing and message contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced citation-boundary failures pass
- [x] #2 Affected citation-boundary module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current local citation-boundary failures. 2. Classify contract drift versus production defects and make the smallest justified correction. 3. Run focused regressions and the full affected module. ADR required: no. ADR path: N/A. Reason: this is localized regression maintenance for an existing Console boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the citation gateway doubles with the current streaming route, capture, and
dispatch keyword contract. Updated the pre-dispatch privacy test to patch the current
memory-preflight seam, and made checking-state cancellation synchronization event-driven
instead of scheduler-speed-dependent. The full module passes with 94 tests and its one
documented xfail. ADR required: no; all changes maintain existing Console behavior and
test boundaries.
<!-- SECTION:NOTES:END -->
