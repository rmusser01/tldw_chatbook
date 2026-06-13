---
id: TASK-11.6
title: 'Phase 4.6: Schedules and Workflows run control'
status: Done
assignee: []
created_date: '2026-05-12 00:00'
labels:
  - product-maturity
  - phase-4-agent-execution
  - schedules
  - workflows
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make schedule and workflow runs controllable with clear state approval retry and Console-follow readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schedules and Workflows display selected run state consistently across list detail inspector and action controls.
- [x] #2 Approval retry pause blocked and no-active-run states show clear recovery actions.
- [x] #3 Console follow or launch uses existing active-work and Console handoff seams.
- [x] #4 QA walkthrough and focused regression evidence prove the Schedules and Workflows flows are usable in the running app.
<!-- AC:END -->

## Implementation Plan

1. Add mounted regressions for Schedules failed-run recovery and Workflows pending-approval recovery.
2. Derive detail pane status, inspector status, run-control summaries, and next-action copy from the same selected active-work item.
3. Add disabled recovery controls for retry, pause, and approval review with explicit tooltips until the underlying run-control services exist.
4. Run focused Schedules/Workflows/Home adapter regressions and capture actual running-app screenshots for approval.
5. Update Phase 4 QA evidence and task tracking after visual approval.

## Implementation Notes

- Added mounted Schedules and Workflows regressions for failed-run retry state, pending-approval state, and no-active-run empty states.
- Updated Schedules and Workflows panes so list detail inspector and disabled run-control actions derive visible state from the same active-work item.
- Preserved existing Console handoff seams while keeping retry pause and approval controls disabled with explicit service-not-wired recovery copy.
- Captured approved textual-web screenshots for Schedules and Workflows and recorded Phase 4.6 QA evidence.
