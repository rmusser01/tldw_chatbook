---
id: TASK-1335
title: Stack collapsed Console rail labels vertically
status: In Progress
assignee: []
created_date: '2026-08-03 03:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console context and Inspector handles read vertically so they consume less horizontal space while remaining understandable and keyboard accessible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collapsed Context handle label reads top-to-bottom.
- [ ] #2 Collapsed Inspector handle label reads top-to-bottom.
- [ ] #3 Collapsed Console handles use a stable three-cell width without changing the Personas workbench handles.
- [ ] #4 Expanded rail headers and handle tooltips remain horizontal and descriptive.
- [ ] #5 Rail badges remain legible and retain their full tooltip text.
- [ ] #6 Targeted Console rail tests pass.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This is a reversible presentation refinement that preserves the shared
widget boundary, Console behavior, and persisted rail state.

1. Add failing widget and mounted-rail assertions for opt-in vertical labels,
   three-cell sizing, badges, tooltips, and the unchanged horizontal default.
2. Add an explicit vertical presentation option to `ConsoleRailHandle` and use
   it only from the two Console collapsed-handle call sites.
3. Add the vertical handle rules to the component TCSS and regenerate the
   bundled stylesheet.
4. Run the focused widget, Console rail, Personas, and CSS integrity checks;
   self-review the diff and record any pre-existing harness failures separately.
