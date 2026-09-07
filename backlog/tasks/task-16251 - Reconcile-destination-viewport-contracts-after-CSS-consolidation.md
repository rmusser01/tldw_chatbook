---
id: TASK-16251
title: Reconcile destination viewport contracts after CSS consolidation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 13:03'
updated_date: '2026-08-14 13:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the destination parity checkpoint by correcting the Scheduling and MCP viewport regressions introduced by CSS consolidation and by making the Settings dirty-state interaction exercise the visible control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schedules workbench remains fully inside default and compact viewports.
- [x] #2 MCP inspector remains constrained inside the production workbench during blocked loading.
- [x] #3 Settings dirty-state parity test visibly toggles the intended control.
- [x] #4 Destination parity module and checkpoint 53 pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a narrow CSS sizing and test-interaction correction within existing destination layout contracts.

1. Preserve the six failing destination-parity cases as the RED baseline and isolate each failure to production layout or test interaction.
2. Size the Scheduling workbench from remaining tab-pane space and restore the missing production MCP inspector height constraint.
3. Scroll the Settings toggle into view before exercising its dirty-state behavior and use the Scheduling-specific start threshold already implied by its taller status strip.
4. Run focused destination tests, relevant CSS/static checks, and the full checkpoint-53 file set.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Constrained the Schedules shell and inner workbench to remaining vertical space, and consolidated its duplicate resize handlers so compact mode updates after terminal resizing.
- Restored the production MCP inspector height constraint in the source stylesheet and regenerated the committed CSS bundle.
- Updated the parity test to scroll the Settings checkbox into view, prove the checkbox actually toggled, honor Schedules' taller status strip, and assert the existing narrow-screen inspector-hide notice.
- Verified 113 destination-parity tests; 23 CSS synchronization/consolidation tests; and checkpoint 53 with 783 passed, 1 expected skip, and 2 dependency/deprecation warnings. Ruff and diff checks pass. The repository's pre-existing Ruff-format drift in the two touched Python files was characterized against HEAD and left unchanged outside the task hunks.
- ADR check: no ADR required; existing destination layout and responsive-degradation boundaries are unchanged.
<!-- SECTION:NOTES:END -->
