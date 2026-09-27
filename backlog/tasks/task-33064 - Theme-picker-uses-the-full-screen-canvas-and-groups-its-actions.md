---
id: TASK-33064
title: Theme picker uses the full-screen canvas and groups its actions
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. At 211x44 and 235x52 the theme list is capped at 22 rows (of ~95) leaving 14-19 rows empty, and the card's 9-12 actions are one ungrouped full-width stack with Revert in the middle of the file actions. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 211x44 and 235x52 the list fills the available height (no empty band below it)
- [x] #2 Switching actions (Use, Try, Revert), creation actions (Clone, New, Import) and your-theme actions (Edit, Rename, Export, Delete) are visibly grouped, laid out horizontally when the card is wide enough
- [x] #3 At 80x24 every control stays reachable and the list shows at least 5 rows
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Picker pane gets a picker-mode class; in the wide layout the pane/picker/list column/list take 1fr of the detail viewport (list min 8) instead of the 24-row cap; compact keeps the auto chain.
2. Compose the card actions as three groups: switching (Use, Try + Revert), creation (Clone, New, Import), your-theme (Edit, Rename, Export, Delete); horizontal rows in wide, vertical stack in compact.
3. Geometry tests at 211x44 and 235x52; keep the 80x24 reachability test green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ThemePane carries `-picker` while the picker shows (set in __init__ since ContentSwitcher's initial bypasses watch_current, then by a watch_current override); CSS makes pane -> picker -> columns -> list column -> list 1fr of the detail body viewport (pane min-height 24, list min 8) instead of the 24-row max-height; the editor keeps height:auto and compact restores the auto chain. Own-width switches in ThemePane.on_resize (after Lane C's finding that the workbench compact class only fires at <=100 cols): below 96 cols `-stacked` puts the card under the list (fix round, review M-2: now the shared `THEME_STACK_BELOW = 100`, measured on the Settings detail body's width so the scrollbar stacking adds cannot hold `-stacked` past it; test_picker_and_editor_stack_together_at_one_threshold) (min-height 32), and a card under 48 cols gets `-narrow-card` (chips full width, as before). Card chips are composed as three groups: [Use][Try] + a Revert row, [Clone][New][Import], yours [Edit][Rename][Export][Delete] (group row hidden for non-yours); chips share their row (1fr, min-width 0); the export result stays vertical. Measured: list 28 rows at 211x44 and 36 at 235x52, ending on the pane's last row; test_picker_fills_the_full_screen_canvas_and_groups_actions pins it; test_every_picker_control_is_reachable now also runs at 120x36 and 150x40.
<!-- SECTION:NOTES:END -->
