---
id: TASK-16261
title: Repair duplicate Library Notes task identity
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 17:04'
updated_date: '2026-08-14 17:08'
labels:
  - backlog
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore unique backlog task identity after the independently landed Library Notes recompose task collided with the earlier canonical Moonshot TASK-16074.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library Notes recompose task has a unique task ID and filename.
- [x] #2 The canonical Moonshot TASK-16074 and its references remain unchanged.
- [x] #3 The unique-ID architecture test and backlog references pass.
- [x] #4 Task notes record the collision provenance and repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm both task histories and identify the later colliding task.
2. Rename only the later Library Notes task to the next unused ID and update exact references.
3. Run the unique-ID test and focused backlog searches.
4. Record the collision repair and verification.

ADR required: no

ADR path: N/A

Reason: backlog metadata correction only; no product or architecture change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Confirmed the Moonshot TASK-16074 history predates the independently landed Library Notes task: the canonical Moonshot task began in commit `46139fc20`, while the colliding Library task arrived later in `1ad1b7c2c`.
- Preserved Moonshot TASK-16074 and all of its spec, plan, ADR, and lesson references unchanged. Renamed only the later Library Notes task to TASK-16262 and updated its frontmatter ID.
- The product-maturity unique-frontmatter-ID regression passes, and focused searches find no stale reference to the old Library filename or identity.
- This is a backlog metadata correction only; no production behavior or ADR changed.
<!-- SECTION:NOTES:END -->
