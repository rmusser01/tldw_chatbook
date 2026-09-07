---
id: TASK-15602
title: Separate File Notes Delete from primary creation actions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 03:21'
labels: []
dependencies:
  - TASK-15601
documentation:
  - Docs/User_Guide/library/file-notes.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce accidental File Notes deletion by separating Delete spatially and sequentially from New and the routine editor actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 New remains the first File Notes editor action and Delete becomes the final action in DOM and keyboard order
- [x] #2 In a wide editor toolbar Delete is anchored at the far-right edge with flexible separation from routine actions
- [x] #3 In the compact action layout Delete occupies its own final full-width row rather than sitting beside New or another routine action and every action label remains fully readable at 40 columns
- [x] #4 Delete retains its existing two-activation confirmation copy focus behavior stale-state checks and recovery behavior
- [x] #5 Mounted 40x20 and 120x40 tests verify rendered placement keyboard order and unchanged confirmation behavior and focused static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a local action-layout and tab-order safety improvement that changes no mutation, confirmation, recovery, or authority protocol.

1. Add mounted layout assertions that fail against the adjacent New/Delete ordering.
2. Move Delete to the final DOM position and use a flexible separator on wide layouts plus an isolated final row on compact layouts; use a single compact column at 40 columns so long recovery labels remain readable.
3. Re-run the existing destructive confirmation tests and rendered compact/wide checks.
4. Update the File Notes guide and record closeout evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Moved Delete to the final toolbar and keyboard position, using a flexible wide-layout separator and an isolated final compact row. At 40 columns the compact toolbar now uses one column so recovery labels remain complete. The existing two-activation delete, stale-state, and restore flow was left unchanged and verified through the real service test.

- Updated the File Notes workspace, mounted layout coverage, and File Notes guide.
- Verified 40x20 and 120x40 rendered labels, geometry, and DOM order; the real create/move/delete/restore flow; and adjacent action-disclosure layouts.
- Ruff passes for the focused File Notes implementation and tests; `git diff --check` passes.
- ADR required: no. No storage, authority, confirmation, or recovery contract changed.
