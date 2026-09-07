---
id: TASK-31959
title: >-
  Library sibling canvases - select-mode Export selected shifts two cells when
  it enables
status: Done
assignee: []
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 20:20'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J Task 2 review I3: _apply_library_row_toggle patches the conversations, notes and prompts row buttons in place, so their select-mode 'Export selected' label moves two cells when the enabled state flips (library_conversations_canvas.py ~199, library_notes_canvas.py ~721 and ~1507). PR J padded only the Media select-mode row buttons, via library_disabled_action_label(align=True).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The sibling canvases' select-mode action labels hold their column across an enabled-state flip
- [x] #2 Painted column pins cover each sibling canvas that carries the row toggle
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted column pins per sibling canvas: `Export selected` column before and after the first selection. 2. `library_disabled_action_label(align=True)` on the Conversations, Notes and Prompts select-mode actions; the in-place patcher honours the align flag; Prompts (recomposes) carries it in compose.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Conversations, Notes and Prompts select-mode actions now pad their enabled labels to the `○ ` marker width via `library_disabled_action_label(align=True)`; `_apply_library_row_toggle` in canvas_sync.py rebuilds with the align flag; Prompts recomposes and carries it in compose. Pins measured the exact two-cell shifts before the fix (71→69, 65→63, 45→43 ×2, 62→61) and hold the column after. Two exact-label pins updated (enabled labels carry the reserved width); Prompts' `Select page` padded the same way at review. Inherent: the Conversations 9-cell button now clips `Exp` where it clipped `Expor` (noted in the guide). Finding (rider): Notes' select-mode toolbar overflows the 36-cell pane at every width, so its `Export selected` is never painted on the real screen — the Notes pin drives the mounted canvas.
<!-- SECTION:NOTES:END -->
