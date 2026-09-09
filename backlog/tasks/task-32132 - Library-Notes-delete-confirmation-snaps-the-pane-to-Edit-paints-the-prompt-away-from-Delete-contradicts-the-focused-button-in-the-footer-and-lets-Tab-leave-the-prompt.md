---
id: TASK-32132
title: >-
  Library Notes delete confirmation snaps the pane to Edit, paints the prompt
  away from Delete, contradicts the focused button in the footer, and lets Tab
  leave the prompt
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:57'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the design assessor and reproduced by the parent: pressing Delete in Info switches the pane back to Edit and paints 'Delete this note?' below the body editor, 14 rows lower; the footer reads 'enter confirm delete | esc cancel delete' while Cancel holds focus, and Enter cancels; eight Tabs walk focus out of the prompt into a pane grip with the prompt still open. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The confirmation renders in the Info Danger section where Delete was pressed, without changing mode
- [x] #2 The footer describes the focused button (Enter cancels while Cancel is focused)
- [x] #3 Tab and Shift+Tab cycle between Cancel and Delete while the prompt is open
- [x] #4 Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two bugs shared one cause: the canvas's show_context formula unconditionally excluded Info whenever confirming_delete was true (forcing the editor pane visible), and the controller ALSO redundantly forced _library_note_context/_library_note_preview to False when entering confirm. Delete is only reachable from Info in the live UI (the Edit pane's own Delete button lives in library-note-wide-utilities, which is unconditionally display=False) -- so relaxed show_context to stay true for the context region even while confirming (Preview still yields, since it never hosts a delete flow), and stopped the controller from stomping the mode. The shared confirmation Vertical is a single DOM node positioned after the Info scroll region, not literally nested in the Danger section's border, but it now renders immediately below Info with no mode change or scroll jump -- confirmed live at 235x52. Footer: extended _library_focus_enter_label with the two confirm-button ids so the confirming_delete footer branch names whichever button is actually focused. Tab-trap: new branch in LibraryScreen.on_key mirroring the existing emergency_tab pattern, cycling only #library-note-delete-cancel/-confirm while confirming_delete is set. Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py, tldw_chatbook/UI/Screens/library_screen.py. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (3 new tests). Live-verified at 235x52: prompt in place, Cancel focused with matching footer, Tab cycles Cancel<->Delete, Escape restores Info with Delete refocused.
<!-- SECTION:NOTES:END -->
