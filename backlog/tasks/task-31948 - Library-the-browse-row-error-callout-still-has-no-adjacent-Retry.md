---
id: TASK-31948
title: Library - the browse-row error callout still has no adjacent Retry
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:34'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR G Task 3 concern (2026-09-05): PR G gave the Media load path a failure callout with a Retry beside it, but #library-canvas-error still paints a message with no action next to it, so the only recovery from a failed browse row is to leave the surface and come back.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A browse error a refetch can clear carries a Retry adjacent to the message
- [x] #2 Retry re-runs the same fetch and clears the callout on success
- [x] #3 A painted pin covers the callout and its Retry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted pin: a refetch-clearable browse error paints a callout with a Retry beside the message; Retry re-runs the same fetch and clears the callout on success. 2. One builder for all three `#library-canvas-error` paint sites; a sync so a repeat failure repaints.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_library_canvas_error_widget()` serves the reconcile, the notes-compact compose and the route-content compose, returning the hub's `ds-recovery-callout` with the existing `#library-source-retry` when the hub's own `_library_source_load_failure()` predicate says a refetch can clear it (deadline and hard failures); a policy denial or a runtime with no source services keeps the bare Static (no inert Retry). `_sync_library_canvas_error()` repaints a repeat failure (the reconcile never remounts an existing error node). Retry runs `_refresh_local_source_snapshot()` — the same call the initial load makes. Live: a broken-db profile paints the callout; Retry re-ran (attempt 2, 3). Batch 2 of the same PR shares the callout builder with PR G's surfaces and refreshes tint and shape on repaint.
<!-- SECTION:NOTES:END -->
