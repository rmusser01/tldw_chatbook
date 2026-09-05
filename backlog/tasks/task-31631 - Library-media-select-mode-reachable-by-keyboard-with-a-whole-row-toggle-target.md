---
id: TASK-31631
title: >-
  Library media select mode - reachable by keyboard with a whole-row toggle
  target
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 06:18'
updated_date: '2026-09-05 15:34'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: after pressing s from the rail, F6, Down and Space all no-op with no focus indicator painted anywhere; only a mouse click on the one-cell ☐ glyph seeds focus, after which the keys work. Row-title clicks do nothing in select mode, and Done takes the exact slot sort: occupied. Cause: focus sits on the pane grip after the recompose that enters select mode (task-31567 is the general fix; this task is the select-mode contract).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering select mode by key or click puts focus on the selected or first row with a visible focus ring, so Down and Space work immediately
- [x] #2 Clicking anywhere on a row toggles its selection in select mode
- [x] #3 Done does not occupy the position sort: held in browse mode
- [x] #4 Painted tests at 235x52 and 100x30 cover the keyboard path from the rail
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Entry focuses a row via the armed list-entry seam; Done on its own row. 2. Whole-row toggle: the row is one Button; Textual drops a click inside the previous press's 0.2 s -active flash → active_effect_duration = 0 on media rows. 3. SDD reviews, live at 235x52/100x30.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Select-mode entry used to focus the Items pane, so Space's row guard never held (F6/Down/Space no-op, nothing painted). Entry now uses _arm_library_list_entry_focus (the only seam that survives the background recompose). Done moved to its own row #library-media-select-done (the summary row already uses 33 of the 36-cell floor) — one extra toolbar row in select mode, pinned clear of every browse toolbar slot at both sizes. Title clicks did route to the row handler already; the lost second click was Textual Button._on_click dropping a click inside the previous press's 0.2 s -active flash — media rows set active_effect_duration = 0 (siblings filed as a rider). Live at 235x52 and 100x30. Files: library_screen.py, library_media_canvas.py, tests in test_library_multiselect_media.py / test_library_media_render_fixes.py.
<!-- SECTION:NOTES:END -->
