---
id: TASK-16257
title: Reconcile retained Library canvas transitions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 15:45'
updated_date: '2026-08-14 16:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore state, recovery, and footer synchronization across retained Library canvas transitions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation error retry and off-page deep-link ownership survive initial snapshot reconciliation.
- [x] #2 Created/discarded Notes immediately reconcile the retained list canvas.
- [x] #3 Entering Notes sync immediately publishes sync-local footer guidance.
- [x] #4 The five deterministic Library shell regressions and nearby retained-canvas tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This restores existing UI lifecycle contracts at the retained-canvas synchronization boundary.

1. Preserve the five deterministic RED regressions exposed by the checkpoint sweep.
2. Use the public pre-mount navigation context in the stale conversation-error fixture.
3. Preserve pending deep-link selection during the initial page load.
4. Reconcile Notes list and footer state through the retained canvas after transitions.
5. Run focused and nearby tests, lint, formatter characterization, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reworked the conversation retry fixture to enter through the mounted public rail and preserved pending deep-link selection during page reconciliation.
- Added one awaited Notes list reconciliation seam for create/discard transitions, retained flat rows until the folder-tree root arrives, and kept the create-recovery receipt ahead of folder-loading copy.
- Published Notes sync footer state at the transition boundary. The five regressions, 51 nearby folder/canvas tests, Ruff lint, and diff checks pass; Ruff format remains the unchanged two-file baseline failure present at `HEAD`.
<!-- SECTION:NOTES:END -->
