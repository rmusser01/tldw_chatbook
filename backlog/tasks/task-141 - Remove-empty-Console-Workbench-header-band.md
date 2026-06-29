---
id: TASK-141
title: Remove empty Console Workbench header band
status: Done
assignee:
  - '@codex'
created_date: '2026-06-29 20:08'
updated_date: '2026-06-29 20:14'
labels:
  - ui
  - textual
  - workbench
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the visually empty horizontal band left by the Console Workbench destination header after its copy was collapsed. The Console should keep visible controls and recovery state without reserving passive header space.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console no longer displays an empty destination header band.
- [x] #2 Visible mode, action, and control strips remain available.
- [x] #3 Visual snapshots prove the blank band is gone.
- [x] #4 Targeted Console Workbench tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md

Reason: this is a corrective implementation of the existing Workbench UI System decision; no new architectural boundary or long-lived UX model is being introduced.

1. Add a failing Console Workbench contract test that requires the empty destination header band to have no visible layout cost.
2. Hide the Console DestinationHeader as a compatibility seam while preserving sync selectors for existing state refreshes.
3. Rebuild CSS if needed and regenerate visual proof screenshots.
4. Run targeted Console Workbench tests, visual snapshot checks, and diff hygiene.
5. Update TASK-141 acceptance criteria and implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the visible Console `DestinationHeader` layout cost while preserving the mounted `#console-workbench-header` seam for state sync and compatibility selectors. The header now has `display: none` and zero height in both the composing helper and generated TCSS, so the visible mode/action strip starts directly below the global navigation border instead of leaving a passive blank band.

Regenerated SVG and PNG visual evidence for normal, compact, focus, and command-palette states. The focused regression first failed against the visible header band, then passed after the seam was hidden. Targeted Console Workbench contract and visual snapshot tests passed.
<!-- SECTION:NOTES:END -->
