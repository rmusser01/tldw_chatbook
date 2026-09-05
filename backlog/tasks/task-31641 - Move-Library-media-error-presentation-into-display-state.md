---
id: TASK-31641
title: Move Library media error presentation into display state
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:27'
updated_date: '2026-09-05 16:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the media browse controller size contract by placing pure error-copy projection beside the existing media display-state contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The media browse controller fits its existing 371-line ceiling.
- [x] #2 Retry, page-load, and filter failure messages remain unchanged.
- [x] #3 Affected media behavior tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run existing controller, media-state, and mounted multiselect characterization tests.
2. Move pure retry and failed-page copy projection into `Library/library_media_state.py`.
3. Route controller and bulk-restore failure handling through the display-state helpers; remove the private sync trampoline whose callers are all internal.
4. Verify unchanged behavior, measure the controller, and tighten its existing ratchet if applicable.
5. Run Ruff, changed-range formatting, and diff checks.

ADR required: no
ADR path: N/A
Reason: pure presentation extraction into an existing display-state boundary, with unchanged copy and runtime ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved unchanged retry and load-failure copy into the existing media display-state module, and routed browse and bulk-restore callers through it. Removed the internal sync trampoline. Controller reduced from 410 to 371 lines, preserving its existing ceiling. All 207 affected media and module-ratchet checks passed after formatting; scoped Ruff and diff checks passed. The large Library screen retains its pre-existing Ruff diagnostics, with no new diagnostic types/messages introduced by the import move. ADR required: no; routine presentation extraction, unchanged behavior and boundaries.
<!-- SECTION:NOTES:END -->
