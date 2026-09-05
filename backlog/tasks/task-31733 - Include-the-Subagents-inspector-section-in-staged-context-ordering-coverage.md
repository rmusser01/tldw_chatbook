---
id: TASK-31733
title: Include the Subagents inspector section in staged-context ordering coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:28'
updated_date: '2026-09-05 19:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the inspector ordering contract after upstream moved Subagents into the right rail before staged context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inspector body explicitly orders Environment, Tasks, Subagents, then staged context
- [x] #2 Staged context remains above run and source-readiness content with mounted geometry assertions unchanged
- [x] #3 The full Console session settings file and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only reconciliation with the established upstream right-rail topology; no new UI or ownership decision.
1. Reproduce the full-file ordering failure and read right_rail.py composition.
2. Add the exact Subagents child expectation and preserve relative and painted geometry assertions.
3. Run the targeted topology test and full settings file with static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the inspector fixture to assert the upstream right rail exact sequence: Environment, Tasks, Subagents, staged context. Relative staged-before-run/readiness assertions and all mounted geometry remain unchanged. Original full-file RED: staged tray expected at child2 but the intentional Subagents section occupied that slot. Targeted GREEN passed; complete session-settings file passed 416 tests in 282.54s with RuntimeWarning escalated. Ruff lint/changed-region format passed; root reviewed the scope. Test-only current-topology repair, no ADR required; self-review complete.
<!-- SECTION:NOTES:END -->
