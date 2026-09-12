---
id: TASK-31658
title: Release cancelled Media entry-focus restoration guards
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:35'
updated_date: '2026-09-05 17:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent a superseded Media canvas synchronization from leaving focus traversal permanently classified as programmatic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A superseded or suppressed Media entry-focus callback releases its restoration guard without moving focus to stale children.
- [x] #2 Arrow traversal after initial Media entry selects and loads the destination row, with existing Reader behavior and assertions preserved.
- [x] #3 Focused regression tests, Reader characterization and architecture checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the stranded Media entry guard and identify the callback supersession or suppression path.
2. Add focused regression coverage for cancelled focus cleanup, preserving stale-focus vetoes.
3. Implement bounded cleanup at the owning Media synchronization seam, separate from the decomposition commits.
4. Verify Reader characterization, focus lifecycle regressions, static checks and unchanged architecture ceilings.
ADR required: no
ADR path: N/A
Reason: Routine correction of existing Media entry-focus lifecycle and cancellation behavior, preserving current ownership and callback contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A traced real-app reproduction showed facet settlement queueing guarded entry focus, then page settlement replacing that callback before recomposition: 3 of 5 runs stranded the guard indefinitely. Added three deterministic regressions using the real latest-wins callback queue; all three failed before implementation. Each Media browse sync now derives guard ownership from its current entry intent and releases it on failed/suppressed synchronization, matching the established Trash precedent without replaying stale callbacks.
Verification so far: 15 targeted tests passed (three new regressions plus the twelve previously order-sensitive match/navigation/no-change tests). Parent review found no actionable issue. Reader full selection is being rerun after independent mounted-Find harness readiness repair. Screen measures 41,324 lines / 1,301 methods, below unchanged pin.
ADR: no new ADR; routine repair of existing focus cancellation ownership. Incident documented in Library decomposition recipe; root coordinates shared testing lesson.

Final verification: exact Reader five-file selection68passed181.50s; Library screen/module architecture35passed3deselected3.36s. New regression module Ruff+format, legacy harness E9/F and diffcheck pass. No behavioral assertions or timeout ceilings changed. Task31649 recorded the separate extraction; runtime cleanup remains a separate commit.
<!-- SECTION:NOTES:END -->
