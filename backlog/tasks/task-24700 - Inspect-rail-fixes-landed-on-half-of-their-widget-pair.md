---
id: TASK-24700
title: Inspect rail fixes landed on half of their widget pair
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 06:17'
updated_date: '2026-08-30 06:24'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-24605 fold-height fix lowered #console-right-rail to min-height 12 but left .console-inspector-rail-handle at 20 - and the handle is the rail's SHIPPING DEFAULT state. Separately, TASK-24605 made the outer fold hint render but never gave .console-inspector-outer-scroll-hint a style rule, so the MORE important fold indicator paints as plain body text while the less important local one is muted+italic. Both found by re-critique after the fixes shipped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The collapsed rail handle's min-height does not exceed what the smallest supported terminal yields
- [ ] #2 The outer fold hint is visually distinguishable from body text and consistent with the local fold hint
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-24605 lowered '#console-right-rail' to min-height 12 because a 24-row terminal allots the rail 13; the COLLAPSED form is a different widget and kept 20. The collapsed handle is the rail's SHIPPING DEFAULT, so the half users meet first still over-claimed rows on exactly the terminals the original fix was written for. Now 12, matching the rail it stands in for.

Second half: the outer fold hint TASK-24605 made render had no style rule at all, while the LOCAL fold hint is muted + italic -- so the more important of the two indicators painted as plain body text and read louder than the less important one. It now shares that treatment, minus the 'display: none' (this hint's visibility is owned by the rail's fold reconcile, not by CSS).

Verified live at 100x24: the handle renders centred in a 14-row rail region.

Modified: css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_inspector_focus_visibility.py.
<!-- SECTION:NOTES:END -->
