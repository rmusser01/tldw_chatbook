---
id: TASK-21664
title: Preserve Library Media reader recovery and focus contracts
status: Done
assignee: []
created_date: '2026-08-24 15:31'
updated_date: '2026-08-24 16:15'
labels:
  - library
  - media
  - tui
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md
  - backlog/decisions/084-library-media-reader-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep single-item deletion, Undo, reading progress, keyboard focus, Escape graduation, and footer hints correct inside the permanent Library Media reader shell.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single-item delete selects the following row, then the previous row, then settles Reader empty, and uses the existing shared receipt and Undo seam.
- [x] #2 Undo reconciles restored items against the active scope, reselecting only when visible and otherwise reporting restoration outside the filter.
- [x] #3 Reading progress is stored and restored by loaded local identity; stale detail and external server detail cannot write local progress.
- [x] #4 Escape gives transients first refusal, then graduates Reader to Items to Library to the existing screen-back behavior while skipping hidden panes.
- [x] #5 Hidden panes have no focusable descendants, pane grips remain reachable, deferred focus cannot override newer user focus, and footer hints advertise only working actions.
- [x] #6 Existing bulk selection, bulk delete, export, and bulk Undo contracts remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for delete adjacency, shared receipt/Undo, and scope-aware restore.
2. Adapt the existing mutation seam to reconcile the permanent reader shell.
3. Add red tests and implement progress keyed only by loaded local identity.
4. Add red tests for transient-first Escape, effective-pane focus graduation, hidden-pane focusability, and truthful footer hints.
5. Implement one outward Escape handler and state-driven footer refresh.
6. Run focused tests, required mutation inverses, static checks, and self-review.

ADR required: no new ADR
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: ADR-084 already defines mutation, progress, adaptive pane, and permanent Reader contracts; this task directly implements it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Preserved the shared mutation interlock, receipt, and Undo worker while making single delete traverse the pre-delete ordering (following, previous, empty) and making one-item Undo reselect only after controller reconciliation when the active query/type scope admits it.
- Added loaded-local-identity progress fetch, cache, write, and restore fencing; external detail and stale requests cannot use the local progress seam.
- Added transient-first Escape graduation across effective Reader, Items, and Library roles, disabled hidden pane descendants while leaving grips reachable, and refreshed state-driven footer copy as focus changes.
- Fixed first-layout hysteresis so the synthetic zero-width startup state cannot strand panes collapsed, restored the viewer's delete confirmation controls, and corrected two Trash harness fixtures that never entered the populated Library profile.
- Verification: 136 focused/adjacent tests passed; Ruff, compileall, and `git diff --check` passed. Required mutation inverses for delete order, selected-vs-loaded progress identity, and hidden-pane focusability each failed their target test and were restored.
- ADR: existing [ADR-084](../decisions/084-library-media-reader-ia.md) applies; no new ADR was required.
<!-- SECTION:NOTES:END -->
