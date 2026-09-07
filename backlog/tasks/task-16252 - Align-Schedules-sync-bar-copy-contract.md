---
id: TASK-16252
title: Align Schedules sync-bar copy contract
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 13:26'
updated_date: '2026-08-14 13:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Schedules UX regression module by asserting the current server-identity and clear-action copy that the production sync bar has rendered since July.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync-bar test proves the visible active server identity.
- [x] #2 Sync-bar test proves the current clear-action label and tooltip.
- [x] #3 Schedules UX and workbench suites pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-only reconciliation with the existing Schedules copy contract.

1. Preserve the stale sync-bar assertions as the RED baseline and verify production history.
2. Assert the current visible server identity and the current concise action labels/tooltips.
3. Run the Schedules UX/workbench modules, Ruff, and diff hygiene checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced assertions that were stale when introduced with the production sync bar's established visible server identity and concise Clear action copy.
- Kept exact tooltip assertions so the shorter labels retain explanatory context, and removed an unused test import exposed by the focused lint gate.
- Verified 52 Schedules workbench/UX/CSS tests with one existing dependency warning; Ruff and diff checks pass.
- ADR check: no ADR required; production behavior and copy are unchanged.
<!-- SECTION:NOTES:END -->
