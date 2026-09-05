---
id: TASK-31591
title: Repair Console modal dismissal regressions
status: Done
assignee: []
created_date: '2026-09-05 05:23'
updated_date: '2026-09-05 05:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair current Console modal inventory, dismissal, and focus-restoration regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced modal dismissal failures pass
- [x] #2 Console modal dismissal module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare the modal inventory and dismissal contracts with current Console launch sites and modal APIs. 2. Update stale contract data or repair production behavior with the smallest justified change. 3. Run focused regressions and the full modal-dismissal module. ADR required: no. ADR path: N/A. Reason: this is localized regression maintenance for existing modal lifecycles.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Console modal inventory and launch graph for current library, fork, terminal, capture-policy, privacy, export, and markdown-save flows. Extended AST launch detection for lazy modal constructors and typed modal factories. Refreshed the model-popover factory and settings picker interaction expectations. ADR required: no; this is test maintenance for existing modal lifecycles. Verification: Tests/UI/test_console_modal_dismissal.py — 119 passed; Ruff and git diff checks passed.
<!-- SECTION:NOTES:END -->
